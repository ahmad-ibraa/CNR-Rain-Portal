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
import gc  # Added for memory management
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
# 2) CSS (FULL RESTORATION + SLIDER FIX)
# =============================
st.markdown(
    """
<style>
/* --- 1. GLOBAL LAYOUT --- */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"]{
  height:100vh !important; width:100vw !important;
  margin:0 !important; padding:0 !important;
  overflow:hidden !important;
  background:#000 !important;
}

.main .block-container{
  padding:0 !important; margin:0 !important;
  max-width:100vw !important;
}

header, footer, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"]{
  display:none !important; height:0 !important; visibility:hidden !important;
}

/* --- 2. SIDEBAR (Locked & Styled) --- */
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
[data-testid="stSidebarResizer"], [data-testid="collapsedControl"], button[title="Collapse sidebar"]{
  display:none !important;
}
[data-testid="stSidebarContent"]{
  height:100vh !important;
  overflow:auto !important;
  padding:10px 12px !important;
}

/* Sidebar Button Styling */
[data-testid="stSidebar"] .stButton button{
  width:100% !important;
  border-radius:10px !important;
  height:44px !important;
  font-weight:650 !important;
}

/* --- 3. MAP OVERLAY --- */
#deckgl-wrapper{
  position:fixed !important;
  inset:0 !important;
  z-index: 0 !important;
}

/* --- 4. FLOATING CONTROL BAR (The Slider Fix) --- */
/* The main container must allow clicks to pass through to the map, 
   but we RE-ENABLE them for the controls. */
   #deckgl-wrapper, #deckgl-wrapper canvas {
  pointer-events: auto !important;
}

[data-testid="stSidebar"], .control-bar-wrapper, .stButton, .stSlider, .stSelectSlider {
    pointer-events: auto !important;
}

.control-bar-wrapper {
  position: fixed !important;
  left: 420px !important;
  right: 18px !important;
  bottom: 18px !important;
  z-index: 1000000 !important; /* Extremely high to stay on top */
  background: rgba(15,15,15,0.92) !important;
  padding: 12px 16px !important;
  border-radius: 999px !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  backdrop-filter: blur(10px);
}

/* Round Play/Pause Button */
.control-bar-wrapper .stButton button {
  border-radius: 999px !important;
  width: 44px !important;
  height: 44px !important;
  padding: 0 !important;
  font-size: 18px !important;
}

/* --- 5. MISC --- */
.output-link a{ text-decoration:none !important; font-weight:600 !important; }
.output-link a:hover{ text-decoration:underline !important; }

</style>
""",
    unsafe_allow_html=True,
)
st.markdown("""
<style>
/* Ensure map stays clickable */
#deckgl-wrapper, #deckgl-wrapper canvas{
  pointer-events:auto !important;
}

/* FLOATING CONTROL BAR (match ancestor even if nested) */
div:has(#control_bar_anchor){
  position: fixed !important;
  left: 420px !important;
  right: 18px !important;
  bottom: 18px !important;
  z-index: 1000000 !important;
  background: rgba(15,15,15,0.92) !important;
  padding: 12px 16px !important;
  border-radius: 999px !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  backdrop-filter: blur(10px);
  pointer-events: auto !important;
}

/* Make sure widgets inside are clickable */
div:has(#control_bar_anchor) *{
  pointer-events: auto !important;
}

/* Round Play/Pause Button */
div:has(#control_bar_anchor) .stButton button{
  border-radius: 999px !important;
  width: 44px !important;
  height: 44px !important;
  padding: 0 !important;
  font-size: 18px !important;
}
</style>
""", unsafe_allow_html=True)


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
def csv_download_link(df: pd.DataFrame, filename: str, label: str):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(csv_bytes).decode()
    href = f"data:text/csv;base64,{b64}"
    st.markdown(
        f'<div class="output-link">📄 <a download="{filename}" href="{href}">{label}</a></div>',
        unsafe_allow_html=True,
    )

def normalize_grid(da: xr.DataArray) -> xr.DataArray:
    if da is None: return None
    return da.assign_coords(latitude=da.latitude.round(4), longitude=da.longitude.round(4))

def normalize_to_latlon(da: xr.DataArray) -> xr.DataArray:
    if da is None: return None
    da = normalize_grid(da)
    drop_coords = [c for c in da.coords if c not in ("latitude", "longitude")]
    da = da.drop_vars(drop_coords, errors="ignore")
    allowed = {"latitude", "longitude"}
    for d in list(da.dims):
        if d in allowed: continue
        da = da.isel({d: 0})
    da = da.sortby("latitude", ascending=False).sortby("longitude")
    return da

def normalize_to_tlatlon(da: xr.DataArray) -> xr.DataArray:
    if da is None: return None
    da = normalize_grid(da)
    drop_coords = [c for c in da.coords if c not in ("time", "latitude", "longitude")]
    da = da.drop_vars(drop_coords, errors="ignore")
    allowed = {"time", "latitude", "longitude"}
    for d in list(da.dims):
        if d in allowed: continue
        da = da.isel({d: 0})
    da = da.sortby("latitude", ascending=False).sortby("longitude")
    return da

def nearest_index(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    src = np.asarray(src).astype("float64")
    tgt = np.asarray(tgt).astype("float64")
    descending = src[0] > src[-1]
    src_work = src[::-1] if descending else src
    pos = np.searchsorted(src_work, tgt)
    pos = np.clip(pos, 1, len(src_work) - 1)
    left = src_work[pos - 1]
    right = src_work[pos]
    choose_right = (np.abs(tgt - right) < np.abs(tgt - left))
    idx = pos - 1 + choose_right.astype(int)
    if descending: idx = (len(src) - 1) - idx
    return idx

def align_ro_to_mrms_grid_nearest(ro: xr.DataArray, mrms0: xr.DataArray) -> xr.DataArray:
    ro = normalize_to_tlatlon(ro)
    mrms0 = normalize_to_latlon(mrms0)
    ro_lat, ro_lon = np.asarray(ro.latitude.values, dtype="float64"), np.asarray(ro.longitude.values, dtype="float64")
    mr_lat, mr_lon = np.asarray(mrms0.latitude.values, dtype="float64"), np.asarray(mrms0.longitude.values, dtype="float64")
    lat_idx = nearest_index(ro_lat, mr_lat)
    lon_idx = nearest_index(ro_lon, mr_lon)
    data = ro.values.astype("float32")
    aligned = data[:, lat_idx[:, None], lon_idx[None, :]]
    return xr.DataArray(
        aligned, dims=("time", "latitude", "longitude"),
        coords={"time": ro.time.values, "latitude": mr_lat, "longitude": mr_lon}
    )

def local_naive_to_utc(dt_local_naive: datetime) -> datetime:
    aware_local = dt_local_naive.replace(tzinfo=NY_TZ)
    aware_utc = aware_local.astimezone(UTC_TZ)
    return aware_utc.replace(tzinfo=None)

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
            var = list(ds.data_vars)[0]
            da = ds[var].clip(min=0).load()
        da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180).sortby("longitude")
        da = da.rio.write_crs("EPSG:4326")
        return normalize_to_latlon(da)
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
    bounds = [float(da.longitude.min()), float(da.latitude.min()), float(da.longitude.max()), float(da.latitude.max())]
    return img_path, bounds

def watershed_mean_inch(da_full: xr.DataArray, ws_gdf: gpd.GeoDataFrame) -> float:
    if ws_gdf is None: return float(da_full.mean().values) / 25.4
    try:
        sub = da_full.rio.write_crs("EPSG:4326", inplace=False)
        clipped = sub.rio.clip(ws_gdf.geometry, ws_gdf.crs, drop=True)
        return float(clipped.mean().values) / 25.4 if clipped.size > 0 else float(da_full.mean().values) / 25.4
    except: return float(da_full.mean().values) / 25.4

# =============================
# 5) SIDEBAR UI
# =============================
with st.sidebar:
    st.title("CNR GIS Portal")
    s_date = st.date_input("Start Date", value=datetime.now().date())
    e_date = st.date_input("End Date", value=datetime.now().date())
    c1, c2 = st.columns(2)
    s_time = c1.selectbox("Start Time", hours := [f"{h:02d}:00" for h in range(24)], index=19)
    e_time = c2.selectbox("End Time", hours, index=21)
    up_zip = st.file_uploader("Watershed Boundary (ZIP)", type="zip")
    basin_name = up_zip.name.replace(".zip", "") if up_zip else "Default_Basin"

    if up_zip:
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(up_zip, "r") as z: z.extractall(td)
            if shps := list(Path(td).rglob("*.shp")):
                st.session_state.active_gdf = gpd.read_file(shps[0]).to_crs("EPSG:4326")
                b = st.session_state.active_gdf.total_bounds
                st.session_state.map_view = pdk.ViewState(latitude=(b[1]+b[3])/2, longitude=(b[0]+b[2])/2, zoom=11)

    if st.session_state.processing_msg: st.caption(st.session_state.processing_msg)

    if st.button("Run Processing", use_container_width=True):
        try:
            if st.session_state.active_gdf is None:
                st.session_state.processing_msg = "Upload a watershed boundary ZIP first."; st.rerun()

            # 1. Setup spatial bounds (CROP TO BASIN)
            b = st.session_state.active_gdf.total_bounds
            # Buffer by ~0.1 degrees to ensure no edge issues
            BUFFER_DEG = 0.35  # ~35-40 km-ish
            lon_min, lat_min, lon_max, lat_max = b[0]-BUFFER_DEG, b[1]-BUFFER_DEG, b[2]+BUFFER_DEG, b[3]+BUFFER_DEG


            start_dt = datetime.combine(s_date, datetime.strptime(s_time, "%H:%M").time())
            end_dt = datetime.combine(e_date, datetime.strptime(e_time, "%H:%M").time())
            ro_times = list(pd.date_range(start_dt + timedelta(minutes=15), end_dt, freq="15min"))
            mrms_times = list(pd.date_range(ceil_to_hour(start_dt + timedelta(minutes=45)), end_dt.replace(minute=0, second=0, microsecond=0), freq="1H"))

            pb, msg = st.progress(0.0), st.empty()
            
            # 2. Download and CROP RO Data
            ro_list, ro_kept = [], []
            for i, t in enumerate(ro_times):
                msg.info(f"RO → {i+1}/{len(ro_times)}")
                da = load_precip("RO", t.to_pydatetime())
                if da is not None:
                    # SUBSET IMMEDIATELY
                    da = da.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
                    ro_list.append(da.astype("float32"))
                    ro_kept.append(t.to_pydatetime())
            
            if len(ro_list) < 4: raise RuntimeError("Insufficient RO data.")
            ro = xr.concat(ro_list, dim="time").assign_coords(time=ro_kept)
            del ro_list; gc.collect()

            # 3. Download and CROP MRMS Data
            mrms_list, mrms_kept = [], []
            for j, t in enumerate(mrms_times):
                msg.info(f"MRMS → {j+1}/{len(mrms_times)}")
                da = load_precip("MRMS", t.to_pydatetime())
                if da is not None:
                    # SUBSET IMMEDIATELY
                    da = da.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
                    mrms_list.append(da.astype("float32"))
                    mrms_kept.append(t.to_pydatetime())
            
            if not mrms_list: raise RuntimeError("No MRMS data found.")
            mrms = xr.concat(mrms_list, dim="time").assign_coords(time=mrms_kept)
            del mrms_list; gc.collect()
            
            # 4. Alignment (Now much faster because arrays are tiny)
            msg.info("Aligning grids...")
            ro = align_ro_to_mrms_grid_nearest(ro, mrms.isel(time=0))
            gc.collect()

            # 5. Bias Scaling
            msg.info("Calculating Bias Scaling...")
            ro_hourly, v_times = [], []
            for T in mrms.time.values:
                Tdt = pd.to_datetime(str(T)).to_pydatetime()
                block = ro.sel(time=slice(Tdt - timedelta(minutes=45) - timedelta(seconds=1), Tdt))
                if block.sizes.get("time", 0) == 4:
                    ro_hourly.append(block.sum(dim="time").astype("float32"))
                    v_times.append(Tdt)
            
            if not v_times: raise RuntimeError("No matching RO/MRMS windows found.")
            
            mrms_subset = mrms.sel(time=pd.to_datetime(v_times))
            ro_hr_da = xr.concat(ro_hourly, dim="time").assign_coords(time=pd.to_datetime(v_times))
            scaling = xr.where(ro_hr_da > 0.01, mrms_subset / ro_hr_da, 0.0).clip(0, 50).astype("float32")
            del ro_hr_da, ro_hourly, mrms_subset; gc.collect()

            # 6. Final Scaling and PNG Generation
            msg.info("Rendering frames...")
            cache, stats = {}, []
            m_ends = [pd.to_datetime(str(x)).to_pydatetime() for x in mrms.time.values]
            
            for t in ro.time.values:
                tdt = pd.to_datetime(str(t)).to_pydatetime()
                h_end = next((T for T in m_ends if T >= tdt), None)
                if h_end and tdt >= h_end - timedelta(minutes=45):
                    h_idx = m_ends.index(h_end)
                    # Apply spatial scaling
                    s_frame = (drop_time_coord(ro.sel(time=tdt)) * drop_time_coord(scaling.isel(time=h_idx))).astype("float32")
                    
                    img, bnds = save_frame_png(s_frame, tdt)
                    lbl = tdt.strftime("%Y-%m-%d %H:%M")
                    cache[lbl] = {"path": img, "bounds": bnds}
                    stats.append({"time": tdt, "rain_in": watershed_mean_inch(s_frame, st.session_state.active_gdf)})

            st.session_state.radar_cache, st.session_state.time_list = cache, list(cache.keys())
            st.session_state.basin_vault[basin_name] = pd.DataFrame(stats)
            msg.success("Complete."); st.rerun()
        except Exception as e:
            st.error(f"Error: {e}"); st.exception(e); st.stop()

# =============================
# 6) ANIMATION & MAIN DISPLAY
# =============================
if st.session_state.time_list and st.session_state.is_playing:
    st.session_state.current_time_index = (st.session_state.current_time_index + 1) % len(st.session_state.time_list)
    time.sleep(0.5)
    st.rerun()

# --- MAP DISPLAY ---
layers = []
if st.session_state.time_list:
    curr = st.session_state.radar_cache[st.session_state.time_list[st.session_state.current_time_index]]
    layers.append(pdk.Layer("BitmapLayer", image=curr["path"], bounds=curr["bounds"], opacity=0.70))
if st.session_state.active_gdf is not None:
    layers.append(pdk.Layer("GeoJsonLayer", st.session_state.active_gdf.__geo_interface__, stroked=True, filled=False, get_line_color=[255, 255, 255], line_width_min_pixels=3))

st.pydeck_chart(pdk.Deck(
    layers=layers, 
    initial_view_state=st.session_state.map_view, 
    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
), use_container_width=True, height=1000)

# =============================
# 7) FLOATING CONTROLS
# =============================
if st.session_state.time_list:
    with st.container():
        # anchor that lets CSS "grab" THIS container and make it fixed
        st.markdown('<div id="control_bar_anchor"></div>', unsafe_allow_html=True)

        col_play, col_slider, col_txt = st.columns([1, 10, 3])

        with col_play:
            btn_icon = "⏸" if st.session_state.is_playing else "▶"
            if st.button(btn_icon, key="play_btn"):
                st.session_state.is_playing = not st.session_state.is_playing
                st.rerun()

        with col_slider:
            idx = st.select_slider(
                "Timeline",
                options=list(range(len(st.session_state.time_list))),
                value=int(st.session_state.current_time_index),
                format_func=lambda i: st.session_state.time_list[i],
                label_visibility="collapsed",
                key="timeline_slider"
            )
            if idx != st.session_state.current_time_index:
                st.session_state.current_time_index = idx
                st.session_state.is_playing = False
                st.rerun()

        with col_txt:
            ts = st.session_state.time_list[st.session_state.current_time_index]
            st.markdown(
                f'<p style="color:#01a0fe; margin:0; font-family:monospace; font-size:14px; font-weight:bold; line-height:44px;">{ts}</p>',
                unsafe_allow_html=True
            )

# =============================
# 8) OUTPUTS (STYLIZED BAR CHART)
# =============================
with st.sidebar:
    if st.session_state.basin_vault:
        st.divider()
        st.subheader("Rainfall Analysis")
        
        df = st.session_state.basin_vault[basin_name]
        
        fig, ax = plt.subplots(figsize=(5, 3.5))
        # Use a calculated width for bars (approx 10 mins in day-units)
        ax.bar(df["time"], df["rain_in"], color="#01a0fe", width=0.006, edgecolor="white", linewidth=0.3)
        
        # Highlight current time in the bar chart
        curr_time = pd.to_datetime(st.session_state.time_list[st.session_state.current_time_index])
        ax.axvline(curr_time, color="#ff0101", linestyle="--", alpha=0.8, lw=1)
        
        ax.set_ylabel("Inches", color="gray")
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        ax.tick_params(colors='gray', labelsize=8)
        
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        csv_download_link(df, f"{basin_name}_rain.csv", f"Export {basin_name} Data")




