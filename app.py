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

# -----------------------------
# 1. PAGE CONFIG
# -----------------------------
st.set_page_config(layout="wide", page_title="CNR Radar Portal", initial_sidebar_state="expanded")

# -----------------------------
# 2. COMPLETE CSS FIX - NO TOP OFFSET
# -----------------------------
st.markdown("""
<style>
/* --- GLOBAL: no scroll, no padding --- */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    height: 100vh !important;
    width: 100vw !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}

/* Main area must not add padding */
.main .block-container {
    padding: 0 !important;
    margin: 0 !important;
    max-width: 100vw !important;
}

/* Hide streamlit chrome */
header, footer, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {
    display: none !important;
    height: 0 !important;
    visibility: hidden !important;
}

/* --- SIDEBAR: fixed width, not resizable, not collapsible --- */
[data-testid="stSidebar"] {
    position: fixed !important;
    left: 0 !important;
    top: 0 !important;
    height: 100vh !important;

    min-width: 400px !important;
    max-width: 400px !important;
    width: 400px !important;

    background: rgba(17,17,17,0.95) !important;
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255,255,255,0.08);
    z-index: 1000 !important;
}

/* Remove resizer + collapse button */
[data-testid="stSidebarResizer"],
[data-testid="collapsedControl"],
button[title="Collapse sidebar"] {
    display: none !important;
}

/* Sidebar internal padding/gap fixes (THIS removes the “big empty space”) */
[data-testid="stSidebar"] > div {
    padding-top: 10px !important;
    padding-bottom: 10px !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.35rem !important;   /* reduce vertical gaps between widgets */
}
[data-testid="stSidebar"] .block-container {
    padding: 0 14px !important; /* reduce left/right padding */
}

/* Keep sidebar content scrollable if it ever overflows */
[data-testid="stSidebarContent"] {
    height: 100vh !important;
    overflow: auto !important;
}

/* --- MAP: truly edge-to-edge, behind sidebar --- */
/* Make pydeck wrapper fixed fullscreen */
.stPydeckChart, .stPydeckChart > div {
    position: fixed !important;
    inset: 0 !important;        /* top/right/bottom/left = 0 */
    height: 100vh !important;
    width: 100vw !important;
    margin: 0 !important;
    padding: 0 !important;
    z-index: 0 !important;
}

/* Target the actual pydeck iframe */
iframe[title="pydeck.io"],
iframe[title="streamlit_pydeck.pydeck_chart"] {
    position: fixed !important;
    inset: 0 !important;
    height: 100vh !important;
    width: 100vw !important;
    border: 0 !important;
    z-index: 0 !important;
}

/* --- FLOATING CONTROLS AT BOTTOM (your play/slider row) --- */
/* Float the ENTIRE Streamlit element-container that contains our anchor */
div[data-testid="element-container"]:has(#controls-anchor) {
    position: fixed !important;
    bottom: 18px !important;
    left: 420px !important;   /* sidebar(400) + 20 */
    right: 18px !important;
    z-index: 3000 !important;

    background: rgba(15, 15, 15, 0.92) !important;
    padding: 12px 20px !important;
    border-radius: 999px !important;
    border: 1px solid rgba(255,255,255,0.12);
    backdrop-filter: blur(10px);

    display: block !important;
    pointer-events: auto !important;
}

/* Make sure widgets inside are visible */
div[data-testid="element-container"]:has(#controls-anchor) * {
    visibility: visible !important;
}

/* Circular play button */
div[data-testid="element-container"]:has(#controls-anchor) .stButton button {
    border-radius: 999px !important;
    width: 44px !important;
    height: 44px !important;
    padding: 0 !important;
    font-size: 18px !important;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* ---- DECK.GL ROOT: force fullscreen, kill offsets ---- */
#deckgl-wrapper {
  position: fixed !important;
  inset: 0 !important;          /* top/right/bottom/left = 0 */
  left: 0 !important;
  top: 0 !important;
  right: 0 !important;
  bottom: 0 !important;
  width: 100vw !important;
  height: 100vh !important;
  margin: 0 !important;
  padding: 0 !important;
  transform: none !important;
}

/* ---- Ensure all deck/map wrappers inherit full viewport ---- */
#view-default-view,
#view-default-view > div,
#view-default-view .mapboxgl-map,
#view-default-view .mapboxgl-canvas-container,
#view-default-view .overlays {
  position: absolute !important;
  inset: 0 !important;
  width: 100% !important;
  height: 100% !important;
}

/* ---- Force BOTH canvases to fill (map + deck overlay) ---- */
canvas#deckgl-overlay,
canvas.mapboxgl-canvas {
  position: absolute !important;
  inset: 0 !important;
  width: 100% !important;
  height: 100% !important;
}

/* ---- If anything still tries to size itself to px dims, kill it ---- */
#deckgl-wrapper * {
  max-height: none !important;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# 3. STATE MANAGEMENT
# -----------------------------
if 'radar_cache' not in st.session_state: st.session_state.radar_cache = {}
if 'time_list' not in st.session_state: st.session_state.time_list = []
if 'active_gdf' not in st.session_state: st.session_state.active_gdf = None
if 'basin_vault' not in st.session_state: st.session_state.basin_vault = {} 
if 'map_view' not in st.session_state:
    st.session_state.map_view = pdk.ViewState(latitude=40.7, longitude=-74.0, zoom=9)
if "img_dir" not in st.session_state:
    st.session_state.img_dir = tempfile.mkdtemp(prefix="radar_png_")
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False
if 'current_time_index' not in st.session_state:
    st.session_state.current_time_index = 0

RADAR_COLORS = ['#76fffe', '#01a0fe', '#0001ef', '#01ef01', '#019001', '#ffff01', '#e7c001', '#ff9000', '#ff0101']
RADAR_CMAP = ListedColormap(RADAR_COLORS)

# -----------------------------
# 4. RADAR ENGINE
# -----------------------------
def get_radar_image(dt_utc):
    ts_str = dt_utc.strftime("%Y%m%d-%H%M00")
    url = f"https://noaa-mrms-pds.s3.amazonaws.com/CONUS/RadarOnly_QPE_15M_00.00/{dt_utc.strftime('%Y%m%d')}/MRMS_RadarOnly_QPE_15M_00.00_{ts_str}.grib2.gz"
    tmp_grib = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False).name
    try:
        r = requests.get(url, stream=True, timeout=15)
        if r.status_code != 200: return None, 0, None
        with gzip.GzipFile(fileobj=r.raw) as gz, open(tmp_grib, "wb") as f:
            shutil.copyfileobj(gz, f)
        with xr.open_dataset(tmp_grib, engine="cfgrib") as ds:
            da = ds[list(ds.data_vars)[0]].load()
            da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180).sortby("longitude")
            subset = da.sel(latitude=slice(42.5, 38.5), longitude=slice(-76.5, -72.5))
            
            site_mean = 0.0
            if st.session_state.active_gdf is not None:
                sub = subset.rio.write_crs("EPSG:4326", inplace=False)
                clipped = sub.rio.clip(st.session_state.active_gdf.geometry, st.session_state.active_gdf.crs, drop=False)
                if not clipped.isnull().all():
                    site_mean = float(clipped.mean().values)

            data = subset.values.astype("float32")
            data[data < 0.1] = np.nan
            img_path = os.path.join(st.session_state.img_dir, f"radar_{dt_utc.strftime('%Y%m%d_%H%M')}.png")
            plt.imsave(img_path, data, cmap=RADAR_CMAP, vmin=0.1, vmax=15.0)
            bounds = [float(subset.longitude.min()), float(subset.latitude.min()), 
                      float(subset.longitude.max()), float(subset.latitude.max())]
            return img_path, max(0.0, site_mean / 25.4), bounds
    except: return None, 0, None
    finally:
        if os.path.exists(tmp_grib): os.remove(tmp_grib)

# -----------------------------
# 5. SIDEBAR
# -----------------------------
with st.sidebar:
    st.title("CNR GIS Portal")
    tz_mode = st.radio("Timezone", ["Local (EST/EDT)", "UTC"])
    s_date = st.date_input("Start Date", value=datetime.now().date())
    e_date = st.date_input("End Date", value=datetime.now().date())

    c1, c2 = st.columns(2)
    hours = [f"{h:02d}:00" for h in range(24)]
    s_time = c1.selectbox("Start", hours, index=19)
    e_time = c2.selectbox("End", hours, index=21)

    up_zip = st.file_uploader("Upload Watershed ZIP", type="zip")
    basin_name = "Default_Basin"
    if up_zip:
        basin_name = up_zip.name.replace(".zip", "")
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(up_zip, 'r') as z: z.extractall(td)
            shps = list(Path(td).rglob("*.shp"))
            if shps:
                st.session_state.active_gdf = gpd.read_file(shps[0]).to_crs("EPSG:4326")
                b = st.session_state.active_gdf.total_bounds
                st.session_state.map_view = pdk.ViewState(
                    latitude=(b[1] + b[3]) / 2, longitude=(b[0] + b[2]) / 2, zoom=11
                )

    if st.button("PROCESS RADAR DATA", use_container_width=True):
        if st.session_state.active_gdf is not None:
            s_dt = datetime.combine(s_date, datetime.strptime(s_time, "%H:%M").time())
            e_dt = datetime.combine(e_date, datetime.strptime(e_time, "%H:%M").time())
            tr = pd.date_range(s_dt, e_dt, freq='15min')
            cache, stats = {}, []
            pb = st.progress(0)
            for i, ts in enumerate(tr):
                ts_utc = ts if tz_mode == "UTC" else ts + timedelta(hours=5)
                path, val, bnds = get_radar_image(ts_utc)
                if path:
                    cache[ts.strftime("%H:%M")] = {"path": path, "bounds": bnds}
                    stats.append({"time": ts, "rain_in": val})
                pb.progress((i + 1) / len(tr))
            
            st.session_state.radar_cache = cache
            st.session_state.time_list = list(cache.keys())
            st.session_state.current_time_index = 0
            # Save this specific run to the vault
            st.session_state.basin_vault[basin_name] = pd.DataFrame(stats)

    # --- MULTI-FILE DOWNLOADER ---
    if st.session_state.basin_vault:
        st.write("---")
        st.subheader("📁 Processed Basins")
        
        # Select which CSV file to interact with
        target_basin = st.selectbox("Select CSV to Download", options=list(st.session_state.basin_vault.keys()))
        df_target = st.session_state.basin_vault[target_basin]
        
        if st.button(f"📊 PLOT {target_basin}", use_container_width=True):
            import plotly.express as px
            @st.dialog(f"Stats: {target_basin}", width="large")
            def modal():
                st.plotly_chart(px.bar(df_target, x='time', y='rain_in', template="plotly_dark"), use_container_width=True)
            modal()
        
        csv_data = df_target.to_csv(index=False).encode('utf-8')
        st.download_button(f"DOWNLOAD {target_basin}.CSV", data=csv_data, file_name=f"{target_basin}.csv", use_container_width=True)

# -----------------------------
# 6. ANIMATION LOGIC
# -----------------------------
if st.session_state.time_list and st.session_state.is_playing:
    st.session_state.current_time_index = (st.session_state.current_time_index + 1) % len(st.session_state.time_list)
    time.sleep(0.5)
    st.rerun()

# -----------------------------
# 7. MAP RENDER
# -----------------------------
layers = []
if st.session_state.time_list:
    current_time_str = st.session_state.time_list[st.session_state.current_time_index]
    curr = st.session_state.radar_cache[current_time_str]
    layers.append(pdk.Layer("BitmapLayer", image=curr["path"], bounds=curr["bounds"], opacity=0.7))

if st.session_state.active_gdf is not None:
    layers.append(pdk.Layer("GeoJsonLayer", st.session_state.active_gdf.__geo_interface__, 
                            stroked=True, filled=False, get_line_color=[255, 255, 255], line_width_min_pixels=3))

# Create deck with explicit height to prevent Streamlit clamping
deck = pdk.Deck(
    layers=layers,
    initial_view_state=st.session_state.map_view,
    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
)

# Render with explicit height - CSS will make it 100vh but this prevents initial clamp
st.pydeck_chart(deck, use_container_width=True, height=1000)

# -----------------------------
# 8. CONTROLS (positioned at bottom by CSS)
# -----------------------------
# -----------------------------
# 8. CONTROLS (floating via anchor CSS)
# -----------------------------
if st.session_state.time_list:
    with st.container():
        # anchor must be inside the SAME container as the widgets
        st.markdown('<div id="controls-anchor"></div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 10, 2])

        with col1:
            if st.session_state.is_playing:
                if st.button("⏸", key="pause_btn"):
                    st.session_state.is_playing = False
                    st.rerun()
            else:
                if st.button("▶", key="play_btn"):
                    st.session_state.is_playing = True
                    st.rerun()

        with col2:
            selected_index = st.select_slider(
                "",
                options=range(len(st.session_state.time_list)),
                value=st.session_state.current_time_index,
                format_func=lambda x: st.session_state.time_list[x],
                label_visibility="collapsed",
            )
            if selected_index != st.session_state.current_time_index:
                st.session_state.current_time_index = selected_index
                st.session_state.is_playing = False
                st.rerun()

        with col3:
            st.markdown(f"**{st.session_state.time_list[st.session_state.current_time_index]}**")



