import streamlit as st
import pandas as pd
import geopandas as gpd
import xarray as xr
import numpy as np
import requests
import gzip
import shutil
import tempfile
import os
import zipfile
import pydeck as pdk
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import rioxarray
import time
import base64
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

# =============================
# 1) PAGE CONFIG
# =============================
st.set_page_config(layout="wide", page_title="CNR Radar Portal", initial_sidebar_state="expanded")

# =============================
# 2) CSS
# =============================
st.markdown(
    """
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"]{
  height:100vh !important; width:100vw !important;
  margin:0 !important; padding:0 !important;
}
body { background:#000 !important; }
.main .block-container{
  padding:0 !important; margin:0 !important;
  max-width:100vw !important;
}
header, footer, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"]{
  display:none !important; height:0 !important; visibility:hidden !important;
}
[data-testid="stSidebar"]{
  position:fixed !important;
  left:0 !important; top:0 !important;
  height:100vh !important;
  min-width:400px !important;
  max-width:400px !important;
  width:400px !important;
  background:rgba(17,17,17,0.95) !important;
  border-right:1px solid rgba(255,255,255,0.08);
  z-index:5000 !important;
}
[data-testid="stSidebarResizer"],
[data-testid="collapsedControl"],
button[title="Collapse sidebar"]{
  display:none !important;
}
[data-testid="stSidebarContent"]{
  height:100vh !important;
  overflow:auto !important;
  padding:10px 12px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# =============================
# 3) STATE
# =============================
if "radar_cache" not in st.session_state: st.session_state.radar_cache = {}
if "time_list" not in st.session_state: st.session_state.time_list = []
if "active_gdf" not in st.session_state: st.session_state.active_gdf = None
if "basin_vault" not in st.session_state: st.session_state.basin_vault = {}
if "map_view" not in st.session_state: st.session_state.map_view = pdk.ViewState(latitude=40.7, longitude=-74.0, zoom=9)
if "img_dir" not in st.session_state: st.session_state.img_dir = tempfile.mkdtemp(prefix="radar_png_")
if "is_playing" not in st.session_state: st.session_state.is_playing = False
if "current_time_index" not in st.session_state: st.session_state.current_time_index = 0
if "processing_msg" not in st.session_state: st.session_state.processing_msg = ""
if "ws_bbox" not in st.session_state: st.session_state.ws_bbox = None 

RADAR_COLORS = ["#76fffe", "#01a0fe", "#0001ef", "#01ef01", "#019001", "#ffff01", "#e7c001", "#ff9000", "#ff0101"]
RADAR_CMAP = ListedColormap(RADAR_COLORS)

MRMS_S3_BASE = "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/MultiSensor_QPE_01H_Pass2_00.00"
RO_S3_BASE   = "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/RadarOnly_QPE_15M_00.00"

# =============================
# 4) HELPERS
# =============================
def csv_download_link(df: pd.DataFrame, filename: str, label: str):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(csv_bytes).decode()
    href = f"data:text/csv;base64,{b64}"
    st.markdown(f'<div class="output-link">📄 <a download="{filename}" href="{href}">{label}</a></div>', unsafe_allow_html=True)

def local_naive_to_utc(dt_local_naive: datetime) -> datetime:
    aware_local = dt_local_naive.replace(tzinfo=NY_TZ)
    aware_utc = aware_local.astimezone(UTC_TZ)
    return aware_utc.replace(tzinfo=None)

def ceil_to_hour(dt: datetime) -> datetime:
    dt0 = dt.replace(minute=0, second=0, microsecond=0)
    return dt0 if dt == dt0 else (dt0 + timedelta(hours=1))

def normalize_to_latlon(da: xr.DataArray) -> xr.DataArray:
    if da is None: return None
    # CRITICAL: Round coordinates to prevent float precision alignment crashes
    da = da.assign_coords(latitude=da.latitude.round(4), longitude=da.longitude.round(4))
    drop_coords = [c for c in da.coords if c not in ("latitude", "longitude")]
    da = da.drop_vars(drop_coords, errors="ignore")
    for d in list(da.dims):
        if d not in ("latitude", "longitude"): da = da.isel({d: 0})
    return da.sortby("latitude", ascending=False).sortby("longitude")

def normalize_to_tlatlon(da: xr.DataArray) -> xr.DataArray:
    if da is None: return None
    da = da.assign_coords(latitude=da.latitude.round(4), longitude=da.longitude.round(4))
    drop_coords = [c for c in da.coords if c not in ("time", "latitude", "longitude")]
    da = da.drop_vars(drop_coords, errors="ignore")
    for d in list(da.dims):
        if d not in ("time", "latitude", "longitude"): da = da.isel({d: 0})
    return da.sortby("latitude", ascending=False).sortby("longitude")

def subset_bbox_rect(da: xr.DataArray, bbox):
    if bbox is None: return da
    minx, miny, maxx, maxy = bbox
    return da.sel(latitude=slice(maxy, miny), longitude=slice(minx, maxx))

def nearest_index(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    src, tgt = np.asarray(src).astype("float64"), np.asarray(tgt).astype("float64")
    descending = src[0] > src[-1]
    src_work = src[::-1] if descending else src
    pos = np.clip(np.searchsorted(src_work, tgt), 1, len(src_work) - 1)
    choose_right = (np.abs(tgt - src_work[pos]) < np.abs(tgt - src_work[pos-1]))
    idx = pos - 1 + choose_right.astype(int)
    return (len(src) - 1) - idx if descending else idx

def align_ro_to_mrms_grid_nearest(ro: xr.DataArray, mrms0: xr.DataArray) -> xr.DataArray:
    ro = normalize_to_tlatlon(ro)
    mrms0 = normalize_to_latlon(mrms0)
    lat_idx = nearest_index(ro.latitude.values, mrms0.latitude.values)
    lon_idx = nearest_index(ro.longitude.values, mrms0.longitude.values)
    aligned = ro.values[:, lat_idx[:, None], lon_idx[None, :]]
    return xr.DataArray(aligned, dims=("time", "latitude", "longitude"),
                        coords={"time": ro.time.values, "latitude": mrms0.latitude.values, "longitude": mrms0.longitude.values})

def load_precip(file_type: str, dt_local_naive: datetime, bbox=None) -> xr.DataArray | None:
    dt_utc = local_naive_to_utc(dt_local_naive)
    ts, ymd = dt_utc.strftime("%Y%m%d-%H%M00"), dt_utc.strftime("%Y%m%d")
    base = RO_S3_BASE if file_type == "RO" else MRMS_S3_BASE
    fn = f"MRMS_{'RadarOnly_QPE_15M' if file_type=='RO' else 'MultiSensor_QPE_01H_Pass2'}_00.00_{ts}.grib2.gz"
    url = f"{base}/{ymd}/{fn}"
    tmp_path = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False).name
    try:
        r = requests.get(url, stream=True, timeout=35)
        if r.status_code != 200: return None
        with gzip.GzipFile(fileobj=r.raw) as gz, open(tmp_path, "wb") as f: shutil.copyfileobj(gz, f)
        with xr.open_dataset(tmp_path, engine="cfgrib", backend_kwargs={"indexpath": ""}) as ds:
            da = ds[list(ds.data_vars)[0]].clip(min=0).load()
            da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180).sortby("longitude")
            da = da.rio.write_crs("EPSG:4326")
            da = normalize_to_latlon(da)
            return subset_bbox_rect(da, bbox).astype("float32")
    except: return None
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

def drop_time_coord(da: xr.DataArray) -> xr.DataArray:
    return da.drop_vars("time", errors="ignore")

def save_frame_png(da: xr.DataArray, dt_local_naive: datetime) -> tuple[str, list]:
    data = da.values.astype("float32")
    data[data < 0.1] = np.nan
    img_path = os.path.join(st.session_state.img_dir, f"radar_{dt_local_naive.strftime('%Y%m%d_%H%M')}.png")
    plt.imsave(img_path, data, cmap=RADAR_CMAP, vmin=0.1, vmax=15.0)
    return img_path, [float(da.longitude.min()), float(da.latitude.min()), float(da.longitude.max()), float(da.latitude.max())]

def watershed_mean_inch(da_full: xr.DataArray, ws_gdf: gpd.GeoDataFrame) -> float:
    try:
        sub = da_full.rio.write_crs("EPSG:4326")
        clipped = sub.rio.clip(ws_gdf.geometry, ws_gdf.crs, drop=True)
        return float(clipped.mean().values) / 25.4
    except: return float(da_full.mean().values) / 25.4

# =============================
# 5) SIDEBAR UI
# =============================
with st.sidebar:
    st.title("CNR GIS Portal")
    s_date = st.date_input("Start Date", value=datetime.now().date())
    e_date = st.date_input("End Date", value=datetime.now().date())
    c1, c2 = st.columns(2)
    hours = [f"{h:02d}:00" for h in range(24)]
    s_time = c1.selectbox("Start Time", hours, index=19)
    e_time = c2.selectbox("End Time", hours, index=21)
    bbox_buf = st.slider("BBox Buffer (degrees)", 0.25, 5.0, 1.5, 0.25)
    up_zip = st.file_uploader("Watershed Boundary (ZIP)", type="zip")
    basin_name = up_zip.name.replace(".zip", "") if up_zip else "Default_Basin"

    if up_zip:
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(up_zip, "r") as z: z.extractall(td)
            shps = list(Path(td).rglob("*.shp"))
            if shps:
                st.session_state.active_gdf = gpd.read_file(shps[0]).to_crs("EPSG:4326")
                b = st.session_state.active_gdf.total_bounds
                st.session_state.ws_bbox = (b[0]-bbox_buf, b[1]-bbox_buf, b[2]+bbox_buf, b[3]+bbox_buf)
                st.session_state.map_view = pdk.ViewState(latitude=(b[1]+b[3])/2, longitude=(b[0]+b[2])/2, zoom=11)

    if st.session_state.time_list:
        st.divider()
        if st.button("▶ Play" if not st.session_state.is_playing else "⏸ Pause", use_container_width=True):
            st.session_state.is_playing = not st.session_state.is_playing; st.rerun()
        idx = st.slider("Timestep", 0, len(st.session_state.time_list)-1, int(st.session_state.current_time_index))
        if idx != st.session_state.current_time_index:
            st.session_state.current_time_index, st.session_state.is_playing = idx, False; st.rerun()
        st.caption(st.session_state.time_list[st.session_state.current_time_index])

    if st.button("Run Processing", use_container_width=True):
        if st.session_state.active_gdf is None: st.error("Upload ZIP first"); st.stop()
        start_dt = datetime.combine(s_date, datetime.strptime(s_time, "%H:%M").time())
        end_dt = datetime.combine(e_date, datetime.strptime(e_time, "%H:%M").time())
        ro_times = list(pd.date_range(start_dt + timedelta(minutes=15), end_dt, freq="15min"))
        mrms_times = list(pd.date_range(ceil_to_hour(start_dt + timedelta(minutes=45)), end_dt.replace(minute=0), freq="1H"))
        
        pb, msg = st.progress(0.0), st.empty()
        ro_list, ro_kept = [], []
        for t in ro_times:
            da = load_precip("RO", t.to_pydatetime(), bbox=st.session_state.ws_bbox)
            if da is not None: ro_list.append(da); ro_kept.append(t.to_pydatetime())
        
        mrms_list, mrms_kept = [], []
        for t in mrms_times:
            da = load_precip("MRMS", t.to_pydatetime(), bbox=st.session_state.ws_bbox)
            if da is not None: mrms_list.append(da); mrms_kept.append(t.to_pydatetime())

        if len(ro_list) >= 4 and mrms_list:
            ro = xr.concat(ro_list, dim="time").assign_coords(time=ro_kept)
            mrms = xr.concat(mrms_list, dim="time").assign_coords(time=mrms_kept)
            ro = align_ro_to_mrms_grid_nearest(ro, mrms.isel(time=0))
            
            cache, stats = {}, []
            for Tdt in mrms_kept:
                block = ro.sel(time=slice(Tdt - timedelta(minutes=45), Tdt))
                if block.sizes.get("time") == 4:
                    m_val = drop_time_coord(mrms.sel(time=Tdt))
                    r_sum = drop_time_coord(block.sum("time"))
                    scale = xr.where(r_sum > 0.01, m_val / r_sum, 0.0).clip(0, 50)
                    for t_sub in pd.date_range(Tdt-timedelta(minutes=45), Tdt, freq="15min"):
                        t_sub_py = t_sub.to_pydatetime()
                        if t_sub_py in ro_kept:
                            da_sc = (drop_time_coord(ro.sel(time=t_sub_py)) * scale).astype("float32")
                            img_path, bounds = save_frame_png(da_sc, t_sub_py)
                            label = t_sub_py.strftime("%Y-%m-%d %H:%M")
                            cache[label] = {"path": img_path, "bounds": bounds}
                            stats.append({"time": t_sub_py, "rain_in": watershed_mean_inch(da_sc, st.session_state.active_gdf)})

            st.session_state.radar_cache, st.session_state.time_list = cache, sorted(list(cache.keys()))
            st.session_state.basin_vault[basin_name] = pd.DataFrame(stats)
            st.rerun()

    if st.session_state.basin_vault:
        st.divider()
        for name, df in st.session_state.basin_vault.items():
            csv_download_link(df, f"{name}.csv", f"{name}.csv")

# =============================
# 6) ANIMATION & MAP
# =============================
if st.session_state.time_list and st.session_state.is_playing:
    st.session_state.current_time_index = (st.session_state.current_time_index + 1) % len(st.session_state.time_list)
    time.sleep(0.5); st.rerun()

layers = []
if st.session_state.time_list:
    curr = st.session_state.radar_cache[st.session_state.time_list[st.session_state.current_time_index]]
    layers.append(pdk.Layer("BitmapLayer", image=curr["path"], bounds=curr["bounds"], opacity=0.7))
if st.session_state.active_gdf is not None:
    layers.append(pdk.Layer("GeoJsonLayer", st.session_state.active_gdf.__geo_interface__, stroked=True, filled=False, get_line_color=[255, 255, 255], line_width_min_pixels=3))

st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=st.session_state.map_view, 
                         map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"), height=1000)
