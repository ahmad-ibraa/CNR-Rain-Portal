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

# --- 1. THE "GHOST UI" CSS (Solves the Blackout) ---
st.set_page_config(layout="wide", page_title="CNR Radar Portal")

st.markdown("""
    <style>
        /* 1. Force the app background to be totally invisible */
        [data-testid="stAppViewContainer"], 
        [data-testid="stMain"], 
        .stApp,
        .main {
            background: transparent !important;
            background-color: transparent !important;
        }

        /* 2. Make the Map fill the entire window and sit at Z-index 0 */
        iframe[title="pydeck.io"] {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            z-index: 0 !important;
            border: none !important;
        }

        /* 3. The 'Ghost' Trick: Allow clicks to pass through the UI to the map */
        [data-testid="stMain"] {
            pointer-events: none !important;
        }
        
        /* 4. RE-ENABLE clicks for the slider and sidebar specifically */
        [data-testid="stSidebar"], .stSlider, .stDialog, .stButton {
            pointer-events: auto !important;
            z-index: 1000 !important;
        }

        /* 5. Glass Sidebar */
        [data-testid="stSidebar"] {
            background-color: rgba(15, 15, 15, 0.8) !important;
            backdrop-filter: blur(15px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* 6. Floating Timeline Slider */
        .stSlider {
            position: fixed !important;
            bottom: 40px !important;
            left: 360px !important; 
            right: 40px !important;
            background: rgba(10, 10, 10, 0.9) !important;
            padding: 10px 40px !important;
            border-radius: 50px !important;
            border: 1px solid #444 !important;
        }

        /* 7. Hide all standard Streamlit UI junk */
        header, footer, [data-testid="stHeader"] { visibility: hidden !important; height: 0; }
        .block-container { padding: 0 !important; margin: 0 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA ENGINE & STATE ---
RADAR_COLORS = ['#76fffe', '#01a0fe', '#0001ef', '#01ef01', '#019001', '#ffff01', '#e7c001', '#ff9000', '#ff0101']
RADAR_CMAP = ListedColormap(RADAR_COLORS)

if 'radar_cache' not in st.session_state: st.session_state.radar_cache = {}
if 'time_list' not in st.session_state: st.session_state.time_list = []
if 'processed_df' not in st.session_state: st.session_state.processed_df = None
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

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("CNR GIS Dashboard")
    s_date = st.date_input("Select Date", value=datetime.now().date())
    up_zip = st.file_uploader("Upload Watershed (ZIP)", type="zip")
    
    if up_zip:
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(up_zip, 'r') as z: z.extractall(td)
            shps = list(Path(td).rglob("*.shp"))
            if shps:
                st.session_state.active_gdf = gpd.read_file(shps[0]).to_crs("EPSG:4326")
                b = st.session_state.active_gdf.total_bounds
                st.session_state.map_view = pdk.ViewState(latitude=(b[1]+b[3])/2, longitude=(b[0]+b[2])/2, zoom=10)

    if st.button("PROCESS RADAR DATA", use_container_width=True):
        if st.session_state.active_gdf is not None:
            s_dt = datetime.combine(s_date, datetime.min.time()) + timedelta(hours=12) # Midday start
            tr = pd.date_range(s_dt, s_dt + timedelta(hours=6), freq='15min')
            cache, stats = {}, []
            for ts in tr:
                path, val, bnd = get_radar_image(ts + timedelta(hours=5))
                if path:
                    cache[ts.strftime("%H:%M")] = {"path": path, "bounds": bnd}
                    stats.append({"time": ts, "rain": val})
            st.session_state.radar_cache, st.session_state.time_list, st.session_state.processed_df = cache, list(cache.keys()), pd.DataFrame(stats)

    if st.session_state.processed_df is not None:
        if st.button("SHOW RAINFALL PLOT", use_container_width=True):
            import plotly.express as px
            @st.dialog("Basin Rain Statistics", width="large")
            def modal():
                st.plotly_chart(px.bar(st.session_state.processed_df, x='time', y='rain', template="plotly_dark"), use_container_width=True)
            modal()

# --- 4. MAP RENDER ---
if st.session_state.time_list:
    t_idx = st.select_slider("Timeline", options=range(len(st.session_state.time_list)), 
                             format_func=lambda x: st.session_state.time_list[x], label_visibility="collapsed")
    curr = st.session_state.radar_cache[st.session_state.time_list[t_idx]]
    layers = [
        pdk.Layer("BitmapLayer", image=curr["path"], bounds=curr["bounds"], opacity=0.7),
        pdk.Layer("GeoJsonLayer", st.session_state.active_gdf.__geo_interface__, stroked=True, filled=False, get_line_color=[255,255,255], line_width_min_pixels=2)
    ]
else:
    layers = []
    if st.session_state.active_gdf is not None:
        layers.append(pdk.Layer("GeoJsonLayer", st.session_state.active_gdf.__geo_interface__, stroked=True, filled=False, get_line_color=[255,255,255], line_width_min_pixels=2))

st.pydeck_chart(pdk.Deck(
    layers=layers, 
    initial_view_state=st.session_state.map_view, 
    map_style="mapbox://styles/mapbox/dark-v11"
), key="full_screen_map")
