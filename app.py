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
# 2. THE CSS FIX
# -----------------------------
st.markdown("""
<style>
    /* KILL THE TOP OFFSET: Remove Streamlit's header space */
    [data-testid="stHeader"] {
        display: none !important;
    }

    /* FULL HEIGHT ROOT */
    [data-testid="stAppViewContainer"] {
        height: 100vh !important;
        width: 100vw !important;
        overflow: hidden !important;
    }

    /* THE MAP: Force it to fill the vertical block */
    [data-testid="stVerticalBlock"] {
        gap: 0 !important;
    }
    
    .stPydeckChart {
        height: 100vh !important;
        width: 100vw !important;
        min-height: 100vh !important;
    }

    /* SIDEBAR LOCKDOWN: Width + No Hide + No Resize */
    [data-testid="stSidebar"] {
        min-width: 400px !important;
        max-width: 400px !important;
        width: 400px !important;
    }
    [data-testid="stSidebarResizer"], [data-testid="collapsedControl"], button[title="Collapse sidebar"] {
        display: none !important;
    }

    /* FLOATING BOTTOM CONTROLS */
    /* Only target the last vertical block for the slider */
    div[data-testid="stVerticalBlock"] > div:last-child {
        position: fixed !important;
        bottom: 20px !important;
        left: 420px !important;
        right: 20px !important;
        z-index: 999 !important;
        background: rgba(15, 15, 15, 0.9) !important;
        padding: 10px 25px !important;
        border-radius: 50px !important;
        border: 1px solid #444;
    }

    /* BUTTON STYLING: Only make the PLAY/PAUSE buttons round */
    /* We use a specific selector for buttons inside the floating controller */
    div[data-testid="column"] .stButton button {
        border-radius: 50% !important;
        width: 45px !important;
        height: 45px !important;
        padding: 0 !important;
    }

    /* LEAVE SIDEBAR BUTTONS ALONE */
    [data-testid="stSidebar"] .stButton button {
        border-radius: 4px !important;
        width: 100% !important;
        height: auto !important;
    }

    footer { visibility: hidden !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 3. STATE & ENGINE
# -----------------------------
if 'radar_cache' not in st.session_state: st.session_state.radar_cache = {}
if 'time_list' not in st.session_state: st.session_state.time_list = []
if 'active_gdf' not in st.session_state: st.session_state.active_gdf = None
if 'basin_vault' not in st.session_state: st.session_state.basin_vault = {} 
if 'map_view' not in st.session_state:
    st.session_state.map_view = pdk.ViewState(latitude=40.7, longitude=-74.0, zoom=9)
if "img_dir" not in st.session_state:
    st.session_state.img_dir = tempfile.mkdtemp(prefix="radar_png_")
if 'is_playing' not in st.session_state: st.session_state.is_playing = False
if 'current_time_index' not in st.session_state: st.session_state.current_time_index = 0

RADAR_COLORS = ['#76fffe', '#01a0fe', '#0001ef', '#01ef01', '#019001', '#ffff01', '#e7c001', '#ff9000', '#ff0101']
RADAR_CMAP = ListedColormap(RADAR_COLORS)

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
            img_path = os.path.join(st.session_state.img_dir, f"radar_{dt_utc.strftime('%H%M')}.png")
            plt.imsave(img_path, data, cmap=RADAR_CMAP, vmin=0.1, vmax=15.0)
            bounds = [float(subset.longitude.min()), float(subset.latitude.min()), 
                      float(subset.longitude.max()), float(subset.latitude.max())]
            return img_path, max(0.0, site_mean / 25.4), bounds
    except: return None, 0, None
    finally:
        if os.path.exists(tmp_grib): os.remove(tmp_grib)

# -----------------------------
# 4. SIDEBAR
# -----------------------------
with st.sidebar:
    st.title("CNR GIS Portal")
    tz_mode = st.radio("Timezone", ["Local (EST/EDT)", "UTC"])
    s_date = st.date_input("Start Date", value=datetime.now().date())
    e_date = st.date_input("End Date", value=datetime.now().date())

    c1, c2 = st.columns(2)
    s_time = c1.selectbox("Start", [f"{h:02d}:00" for h in range(24)], index=19)
    e_time = c2.selectbox("End", [f"{h:02d}:00" for h in range(24)], index=21)

    up_zip = st.file_uploader("Upload Watershed ZIP", type="zip")
    if up_zip:
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
            st.session_state.radar_cache, st.session_state.time_list = cache, list(cache.keys())
            st.session_state.current_time_index = 0
            st.session_state.basin_vault[up_zip.name] = pd.DataFrame(stats)

    if st.session_state.basin_vault:
        st.write("---")
        st.subheader("📁 Processed Basins")
        target_file = st.selectbox("Select CSV", options=list(st.session_state.basin_vault.keys()))
        df_target = st.session_state.basin_vault[target_file]
        
        if st.button(f"📊 SHOW PLOT", use_container_width=True):
            import plotly.express as px
            @st.dialog(f"Stats: {target_file}", width="large")
            def modal():
                st.plotly_chart(px.bar(df_target, x='time', y='rain_in', template="plotly_dark"), use_container_width=True)
            modal()
        
        st.download_button(f"DOWNLOAD CSV", data=df_target.to_csv(index=False).encode('utf-8'), 
                           file_name=f"{target_file}.csv", use_container_width=True)

# -----------------------------
# 5. ANIMATION & MAP
# -----------------------------
if st.session_state.time_list and st.session_state.is_playing:
    st.session_state.current_time_index = (st.session_state.current_time_index + 1) % len(st.session_state.time_list)
    time.sleep(0.4)
    st.rerun()

layers = []
if st.session_state.time_list:
    curr = st.session_state.radar_cache[st.session_state.time_list[st.session_state.current_time_index]]
    layers.append(pdk.Layer("BitmapLayer", image=curr["path"], bounds=curr["bounds"], opacity=0.7))

if st.session_state.active_gdf is not None:
    layers.append(pdk.Layer("GeoJsonLayer", st.session_state.active_gdf.__geo_interface__, 
                            stroked=True, filled=False, get_line_color=[255, 255, 255], line_width_min_pixels=3))

# Render map
st.pydeck_chart(pdk.Deck(
    layers=layers,
    initial_view_state=st.session_state.map_view,
    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
), use_container_width=True)

# -----------------------------
# 6. FLOATING CONTROLS
# -----------------------------
if st.session_state.time_list:
    c1, c2, c3 = st.columns([1, 10, 2])
    with c1:
        if st.button("⏸" if st.session_state.is_playing else "▶", key="playback_toggle"):
            st.session_state.is_playing = not st.session_state.is_playing
            st.rerun()
    with c2:
        new_idx = st.select_slider("", options=range(len(st.session_state.time_list)), 
                                   value=st.session_state.current_time_index,
                                   format_func=lambda x: st.session_state.time_list[x], label_visibility="collapsed")
        if new_idx != st.session_state.current_time_index:
            st.session_state.current_time_index = new_idx
            st.session_state.is_playing = False
            st.rerun()
    with c3:
        st.write(f"**{st.session_state.time_list[st.session_state.current_time_index]}**")
