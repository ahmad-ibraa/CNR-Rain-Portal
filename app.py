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

NY_TZ  = ZoneInfo("America/New_York")
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
  overflow:hidden !important;
}
body { background:#000 !important; }

.main .block-container{
  padding:0 !important; margin:0 !important;
  max-width:100vw !important;
}

header, footer, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"]{
  display:none !important; height:0 !important; visibility:hidden !important;
}

/* Sidebar locked */
[data-testid="stSidebar"]{
  position:fixed !important;
  left:0 !important; top:0 !important;
  height:100vh !important;
  min-width:400px !important;
  max-width:400px !important;
  width:400px !important;
  background:rgba(17,17,17,0.95) !important;
  backdrop-filter: blur(10px);
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
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{ gap:0.28rem !important; }

/* DeckGL fullscreen */
#deckgl-wrapper{
  position:fixed !important;
  inset:0 !important;
  width:100vw !important;
  height:100vh !important;
  z-index: 1 !important;
}

/* Floating control bar */
.control-bar{
  position: fixed !important;
  left: 420px !important;
  right: 18px !important;
  bottom: 18px !important;
  z-index: 12000 !important;
  background: rgba(15,15,15,0.92) !important;
  padding: 12px 16px !important;
  border-radius: 999px !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  backdrop-filter: blur(10px);
}
.control-bar .stButton button{
  border-radius:999px !important;
  width:44px !important;
  height:44px !important;
}
[data-testid="stSidebar"] .stButton button{
  width:100% !important;
  border-radius:10px !important;
  height:44px !important;
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

RADAR_COLORS = ["#76fffe", "#01a0fe", "#0001ef", "#01ef01", "#019001", "#ffff01", "#e7c001", "#ff9000", "#ff0101"]
RADAR_CMAP = ListedColormap(RADAR_COLORS)

MRMS_S3_BASE = "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/MultiSensor_QPE_01H_Pass2_00.00"
RO_S3_BASE   = "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/RadarOnly_QPE_15M_00.00"

# =============================
# 4) HELPERS
# =============================
def normalize_grid(da: xr.DataArray) -> xr.DataArray:
    """Ensure coordinates are consistently float64 and sorted."""
    da = da.assign_coords(
        latitude=da.latitude.astype("float64"),
        longitude=da.longitude.astype("float64")
    )
    return da.sortby("latitude", ascending=False).sortby("longitude")

def csv_download_link(df: pd.DataFrame, filename: str, label: str):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(csv_bytes).decode()
    href = f"data:text/csv;base64,{b64}"
    st.markdown(f'<div class="output-link">📄 <a download="{filename}" href="{href}">{label}</a></div>', unsafe_allow_html=True)

def normalize_to_latlon(da: xr.DataArray) -> xr.DataArray:
    if da is None: return None
    drop_coords = [c for c in da.coords if c not in ("latitude", "longitude")]
    da = da.drop_vars(drop_coords, errors="ignore")
    allowed = {"latitude", "longitude"}
    for d in list(da.dims):
        if d not in allowed: da = da.isel({d: 0})
    return da.sortby("latitude", ascending=False).sortby("longitude")

def normalize_to_tlatlon(da: xr.DataArray) -> xr.DataArray:
    if da is None: return None
    drop_coords = [c for c in da.coords if c not in ("time", "latitude", "longitude")]
    da = da.drop_vars(drop_coords, errors="ignore")
    allowed = {"time", "latitude", "longitude"}
    for d in list(da.dims):
        if d not in allowed: da = da.isel({d: 0})
    return da.sortby("latitude", ascending=False).sortby("longitude")

def nearest_index(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    src = np.asarray(src).astype("float64")
    tgt = np.asarray(tgt).astype("float64")
    descending = src[0] > src[-1]
    src_work = src[::-1] if descending else src
    pos = np.searchsorted(src_work, tgt)
    pos = np.clip(pos, 1, len(src_work) - 1)
    left, right = src_work[pos - 1], src_work[pos]
    choose_right = (np.abs(tgt - right) < np.abs(tgt - left))
    idx = pos - 1 + choose_right.astype(int)
    return (len(src) - 1) - idx if descending else idx

def align_ro_to_mrms_grid_nearest(ro: xr.DataArray, mrms0: xr.DataArray) -> xr.DataArray:
    ro = normalize_to_tlatlon(ro)
    mrms0 = normalize_to_latlon(mrms0)
    ro_lat, ro_lon = ro.latitude.values, ro.longitude.values
    mr_lat, mr_lon = mrms0.latitude.values, mrms0.longitude.values
    lat_idx = nearest_index(ro_lat, mr_lat)
    lon_idx = nearest_index(ro_lon, mr_lon)
    aligned = ro.values[:, lat_idx[:, None], lon_idx[None, :]]
    return xr.DataArray(aligned, dims=("time", "latitude", "longitude"),
                        coords={"time": ro.time.values, "latitude": mr_lat, "longitude": mr_lon})

def local_naive_to_utc(dt_local_naive: datetime) -> datetime:
    return dt_local_naive.replace(tzinfo=NY_TZ).astimezone(UTC_TZ).replace(tzinfo=None)

def ceil_to_hour(dt: datetime) -> datetime:
    dt0 = dt.replace(minute=0, second=0, microsecond=0)
    return dt0 if dt == dt0 else (dt0 + timedelta(hours=1))

def load_precip(file_type: str, dt_local_naive: datetime) -> xr.DataArray | None:
    dt_utc = local_naive_to_utc(dt_local_naive)
    ts, ymd = dt_utc.strftime("%Y%m%d-%H%M00"), dt_utc.strftime("%Y%m%d")
    base = RO_S3_BASE if file_type == "RO" else MRMS_S3_BASE
    filename = f"MRMS_{'RadarOnly_QPE_15M' if file_type=='RO' else 'MultiSensor_QPE_01H_Pass2'}_00.00_{ts}.grib2.gz"
    url = f"{base}/{ymd}/{filename}"
    tmp_path = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False).name
    try:
        r = requests.get(url, stream=True, timeout=25)
        if r.status_code == 404: return None
        r.raise_for_status()
        with gzip.GzipFile(fileobj=r.raw) as gz, open(tmp_path, "wb") as f:
            shutil.copyfileobj(gz, f)
        with xr.open_dataset(tmp_path, engine="cfgrib", backend_kwargs={"indexpath": ""}) as ds:
            da = ds[list(ds.data_vars)[0]].load().squeeze()
        da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180).sortby("longitude")
        return normalize_to_latlon(da.rio.write_crs("EPSG:4326"))
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
    if ws_gdf is None: return float(da_full.mean().values) / 25.4
    try:
        sub = da_full.rio.write_crs("EPSG:4326", inplace=False)
        clipped = sub.rio.clip(ws_gdf.geometry, ws_gdf.crs, drop=True)
        return float(clipped.mean().values) / 25.4 if clipped.size > 0 else float(da_full.mean().values) / 25.4
    except: return float(da_full.mean().values) / 25.4

# =============================
# 5) SIDEBAR & PROCESSING
# =============================
with st.sidebar:
    st.title("CNR GIS Portal")
    s_date, e_date = st.date_input("Start Date"), st.date_input("End Date")
    c1, c2 = st.columns(2)
    s_time = c1.selectbox("Start Time", [f"{h:02d}:00" for h in range(24)], index=19)
    e_time = c2.selectbox("End Time", [f"{h:02d}:00" for h in range(24)], index=21)
    up_zip = st.file_uploader("Watershed Boundary (ZIP)", type="zip")
    basin_name = up_zip.name.replace(".zip", "") if up_zip else "Default_Basin"

    if up_zip:
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(up_zip, "r") as z: z.extractall(td)
            shps = list(Path(td).rglob("*.shp"))
            if shps:
                st.session_state.active_gdf = gpd.read_file(shps[0]).to_crs("EPSG:4326")
                b = st.session_state.active_gdf.total_bounds
                st.session_state.map_view = pdk.ViewState(latitude=(b[1]+b[3])/2, longitude=(b[0]+b[2])/2, zoom=11)

    if st.button("Run Processing", use_container_width=True):
        try:
            if st.session_state.active_gdf is None:
                st.error("Upload watershed ZIP first"); st.stop()
            
            start_dt = datetime.combine(s_date, datetime.strptime(s_time, "%H:%M").time())
            end_dt = datetime.combine(e_date, datetime.strptime(e_time, "%H:%M").time())
            ro_times = list(pd.date_range(start_dt + timedelta(minutes=15), end_dt, freq="15min"))
            mrms_times = list(pd.date_range(ceil_to_hour(start_dt + timedelta(minutes=45)), end_dt.replace(minute=0), freq="1H"))

            pb = st.progress(0.0); msg = st.empty()
            ro_list, ro_kept = [], []
            for i, t in enumerate(ro_times):
                msg.info(f"RO 15m: {i+1}/{len(ro_times)} • {t:%H:%M}")
                da = load_precip("RO", t.to_pydatetime())
                if da is not None:
                    ro_list.append(da.astype("float32"))
                    ro_kept.append(t)
            
            if len(ro_list) < 4: raise RuntimeError("Insufficient RO frames")
            ro = xr.concat(ro_list, dim="time").assign_coords(time=ro_kept)

            mrms_list, mrms_kept = [], []
            for j, t in enumerate(mrms_times):
                msg.info(f"MRMS 1h: {j+1}/{len(mrms_times)} • {t:%H:%M}")
                da = load_precip("MRMS", t.to_pydatetime())
                if da is not None:
                    mrms_list.append(da.astype("float32"))
                    mrms_kept.append(t)
            
            if not mrms_list: raise RuntimeError("No MRMS frames")
            mrms = xr.concat(mrms_list, dim="time").assign_coords(time=mrms_kept)
            
            msg.info("Aligning Grids...")
            ro = align_ro_to_mrms_grid_nearest(ro, mrms.isel(time=0))
            
            ro_hourly, v_times = [], []
            for T in mrms.time.values:
                Tdt = pd.to_datetime(T).to_pydatetime()
                block = ro.sel(time=slice(Tdt - timedelta(minutes=45), Tdt))
                if block.sizes["time"] == 4:
                    ro_hourly.append(block.sum(dim="time"))
                    v_times.append(Tdt)
            
            mrms = mrms.sel(time=v_times)
            ro_h_da = xr.concat(ro_hourly, dim="time").assign_coords(time=v_times)
            scaling = xr.where(ro_h_da > 0.01, mrms / ro_h_da, 0.0).clip(0, 50)

            ro_scaled_list, ro_scaled_times = [], []
            for t in ro.time.values:
                tdt = pd.to_datetime(t).to_pydatetime()
                hour_end = next((T for T in v_times if T >= tdt and tdt >= T - timedelta(minutes=45)), None)
                if hour_end:
                    scale_raster = drop_time_coord(scaling.sel(time=hour_end))
                    ro_scaled_list.append(normalize_grid(drop_time_coord(ro.sel(time=tdt)) * scale_raster))
                    ro_scaled_times.append(tdt)

            ro_scaled_da = xr.concat(ro_scaled_list, dim="time").assign_coords(time=ro_scaled_times)
            
            cache, stats = {}, []
            for tdt in ro_scaled_da.time.values:
                dt_l = pd.to_datetime(tdt).to_pydatetime()
                da = ro_scaled_da.sel(time=tdt)
                path, bnds = save_frame_png(da, dt_l)
                lbl = dt_l.strftime("%Y-%m-%d %H:%M")
                cache[lbl] = {"path": path, "bounds": bnds}
                stats.append({"time": dt_l, "rain_in": watershed_mean_inch(da, st.session_state.active_gdf)})

            st.session_state.radar_cache, st.session_state.time_list = cache, list(cache.keys())
            st.session_state.basin_vault[basin_name] = pd.DataFrame(stats)
            msg.success("Complete!"); st.rerun()

        except Exception as e:
            st.error(f"Error: {e}"); st.exception(e); st.stop()

# =============================
# 6) RENDER
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

st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=st.session_state.map_view, map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"), height=1000)

if st.session_state.time_list:
    st.markdown('<div class="control-bar">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 10, 2])
    with c1:
        if st.button("⏸" if st.session_state.is_playing else "▶"):
            st.session_state.is_playing = not st.session_state.is_playing; st.rerun()
    with c2:
        idx = st.select_slider(" ", options=range(len(st.session_state.time_list)), value=st.session_state.current_time_index, format_func=lambda i: st.session_state.time_list[i], label_visibility="collapsed")
        if idx != st.session_state.current_time_index:
            st.session_state.current_time_index = idx; st.session_state.is_playing = False; st.rerun()
    with c3: st.markdown(f"**{st.session_state.time_list[st.session_state.current_time_index]}**")
    st.markdown("</div>", unsafe_allow_html=True)
