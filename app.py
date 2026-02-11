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

# --- 1. THE REFINED "TOTAL TRANSPARENCY" CSS ---
st.set_page_config(layout="wide", page_title="CNR Radar Portal")

st.markdown("""
    <style>
        /* 1. Force every possible Streamlit background to be transparent */
        [data-testid="stAppViewContainer"], 
        [data-testid="stMain"], 
        [data-testid="stMainContainer"],
        .main,
        .stApp {
            background-color: transparent !important;
            background: transparent !important;
        }

        /* 2. Remove the 'belt' and padding */
        .block-container {
            padding: 0 !important;
            margin: 0 !important;
            max-width: 100vw !important;
        }

        /* 3. The Map: Full Screen Fixed background */
        iframe[title="pydeck.io"] {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            z-index: 0 !important; /* Base Layer */
            border: none !important;
        }

        /* 4. The Sidebar: Floating Glass */
        [data-testid="stSidebar"] {
            background-color: rgba(20, 20, 20, 0.75) !important;
            backdrop-filter: blur(12px) !important;
            z-index: 100 !important;
        }

        /* 5. The Slider: Floating UI */
        .stSlider {
            position: fixed !important;
            bottom: 40px !important;
            left: 360px !important; 
            right: 40px !important;
            z-index: 1000 !important;
            background: rgba(10, 10, 10, 0.9) !important;
            padding: 10px 40px !important;
            border-radius: 50px !important;
            border: 1px solid #444 !important;
        }

        /* 6. Sidebar Button Styling */
        div.stButton > button {
            width: 100% !important;
            height: 60px !important;
            font-weight: bold !important;
            text-transform: uppercase;
        }

        /* Hide UI clutter */
        header, footer, [data-testid="stHeader"] { visibility: hidden !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA ENGINE ---
RADAR_COLORS = ['#76fffe', '#01a0fe', '#0001ef', '#01ef01', '#019001', '#ffff01', '#e7c001', '#ff9000', '#ff0101']
RADAR_CMAP = ListedColormap(RADAR_COLORS)

if 'processed_df' not in st.session_state: st.session_state.processed_df = None
if 'radar_cache' not in st.session_state: st.session_state.radar_cache = {}
if 'time_list' not in st.session_state: st.session_state.time_list = []
if 'active_gdf' not in st.session_state: st.session_state.active_gdf = None
if 'map_view' not in st.session_state: 
    st.session_state.map_view = pdk.ViewState(latitude=40.7, longitude=-74.0, zoom=9)

def get_radar_image(dt_utc):
    ts_str = dt_utc.strftime("%Y%m%d-%H%M00") 
    url = f"https://noaa-mrms-pds.s3.amazonaws.com/CONUS/RadarOnly_QPE_15M_00.00/{dt_utc.strftime('%Y%m%d')}/MRMS_RadarOnly_QPE_15M_00.00_{ts_str}.grib2.gz"
    tmp_grib = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False).name
    try:
        r = requests.get(url, stream=True, timeout=10)
        if r.status_code != 200: return None, 0, None
        with gzip.GzipFile(fileobj=r.raw) as gz, open(tmp_grib, "wb") as f:
            shutil.copyfileobj(gz, f)
        with xr.open_dataset(tmp_grib, engine="cfgrib") as ds:
            da = ds[list(ds.data_vars)[0]].load()
            da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180).sortby("longitude")
            subset = da.sel(latitude=slice(42.5, 38.5), longitude=slice(-76.5, -72.5))
            site_mean = 0.0
            if st.session_state.active_gdf is not None:
                clipped = subset.rio.write_crs("EPSG:4326").rio.clip(st.session_state.active_gdf.geometry)
                site_mean = float(clipped.mean()) if not clipped.isnull().all() else 0.0
            data = subset.values
            data[data < 0.1] = np.nan 
            img_filename = f"radar_{dt_utc.strftime('%H%M')}.png"
            plt.imsave(img_filename, data, cmap=RADAR_CMAP, vmin=0.1, vmax=15.0)
            bounds = [float(subset.longitude.min()), float(subset.latitude.min()), 
                      float(subset.longitude.max()), float(subset.latitude.max())]
            return img_filename, max(0.0, site_mean/25.4), bounds
    except: return None, 0, None
    finally:
        if os.path.exists(tmp_grib): os.remove(tmp_grib)

# --- 3. UI ---
with st.sidebar:
    st.title("CNR GIS Portal")
    tz_mode = st.radio("Timezone", ["Local (EST/EDT)", "UTC"])
    s_date = st.date_input("Start Date", value=datetime.now().date())
    c1, c2 = st.columns(2)
    hours = [f"{h:02d}:00" for h in range(24)]
    s_time = c1.selectbox("Start", hours, index=19)
    e_time = c2.selectbox("End", hours, index=21)
    up_zip = st.file_uploader("Upload Watershed", type="zip")
    if up_zip:
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(up_zip, 'r') as z: z.extractall(td)
            shps = list(Path(td).rglob("*.shp"))
            if shps:
                st.session_state.active_gdf = gpd.read_file(shps[0]).to_crs("EPSG:4326")
                b = st.session_state.active_gdf.total_bounds
                st.session_state.map_view = pdk.ViewState(latitude=(b[1]+b[3])/2, longitude=(b[0]+b[2])/2, zoom=11)

    if st.button("PROCESS DATA"):
        if st.session_state.active_gdf is not None:
            s_dt = datetime.combine(s_date, datetime.strptime(s_time, "%H:%M").time())
            tr = pd.date_range(s_dt, s_dt + timedelta(hours=2), freq='15min')
            cache, stats = {}, []
            for ts in tr:
                path, val, bnds = get_radar_image(ts + timedelta(hours=5))
                if path:
                    cache[ts.strftime("%H:%M")] = {"path": path, "bounds": bnds}
                    stats.append({"time": ts, "rain_in": val})
            st.session_state.radar_cache, st.session_state.time_list, st.session_state.processed_df = cache, list(cache.keys()), pd.DataFrame(stats)

    if st.session_state.processed_df is not None:
        if st.button("SHOW PLOT"):
            import plotly.express as px
            @st.dialog("Rainfall Statistics", width="large")
            def modal():
                fig = px.bar(st.session_state.processed_df, x='time', y='rain_in', template="plotly_dark")
                fig.update_layout(bargap=0)
                st.plotly_chart(fig, use_container_width=True)
            modal()

# --- 4. THE BACKGROUND MAP ---
if st.session_state.time_list:
    t_idx = st.select_slider("Time", options=range(len(st.session_state.time_list)), format_func=lambda x: st.session_state.time_list[x], label_visibility="collapsed")
    curr = st.session_state.radar_cache[st.session_state.time_list[t_idx]]
    layers = [pdk.Layer("BitmapLayer", image=curr["path"], bounds=curr["bounds"], opacity=0.7)]
else:
    layers = []

st.pydeck_chart(pdk.Deck(
    layers=layers, 
    initial_view_state=st.session_state.map_view, 
    map_style="mapbox://styles/mapbox/dark-v10",
    height=2000 # Force a large height for the iframe
), key="radar_map")
