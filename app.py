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

# --- 1. ROBUST CSS (No Ghosting) ---
st.set_page_config(layout="wide", page_title="CNR Radar Portal")

st.markdown("""
    <style>
        /* Force the map container to be as large as possible */
        .stPydeckChart {
            height: 85vh !important;
            width: 100% !important;
        }
        
        /* Reduce top padding so map is higher up */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 0rem !important;
        }

        /* Sidebar Styling - Solid & Reliable */
        [data-testid="stSidebar"] {
            background-color: #111 !important;
            border-right: 1px solid #333;
        }

        /* Large Action Button */
        div.stButton > button {
            width: 100% !important;
            height: 50px !important;
            background-color: #FF4B4B !important;
            color: white !important;
        }

        /* Hide standard header/footer */
        header, footer { visibility: hidden !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SETUP & STATE ---
RADAR_COLORS = ['#76fffe', '#01a0fe', '#0001ef', '#01ef01', '#019001', '#ffff01', '#e7c001', '#ff9000', '#ff0101']
RADAR_CMAP = ListedColormap(RADAR_COLORS)

if 'radar_cache' not in st.session_state: st.session_state.radar_cache = {}
if 'time_list' not in st.session_state: st.session_state.time_list = []
if 'active_gdf' not in st.session_state: st.session_state.active_gdf = None
if 'map_view' not in st.session_state: 
    st.session_state.map_view = pdk.ViewState(latitude=40.7, longitude=-74.0, zoom=9)

# --- 3. DATA ENGINE ---
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
            data = subset.values
            data[data < 0.1] = np.nan 
            img_filename = f"radar_{dt_utc.strftime('%H%M')}.png"
            plt.imsave(img_filename, data, cmap=RADAR_CMAP, vmin=0.1, vmax=15.0)
            bounds = [float(subset.longitude.min()), float(subset.latitude.min()), 
                      float(subset.longitude.max()), float(subset.latitude.max())]
            return img_filename, bounds
    except: return None, None
    finally:
        if os.path.exists(tmp_grib): os.remove(tmp_grib)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("CNR Portal")
    s_date = st.date_input("Date", value=datetime.now().date())
    up_zip = st.file_uploader("Upload Watershed", type="zip")
    
    if up_zip:
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(up_zip, 'r') as z: z.extractall(td)
            shps = list(Path(td).rglob("*.shp"))
            if shps:
                st.session_state.active_gdf = gpd.read_file(shps[0]).to_crs("EPSG:4326")
                b = st.session_state.active_gdf.total_bounds
                st.session_state.map_view = pdk.ViewState(latitude=(b[1]+b[3])/2, longitude=(b[0]+b[2])/2, zoom=10)

    if st.button("PROCESS RADAR"):
        # Just getting the last 1 hour of data for speed/testing
        s_dt = datetime.combine(s_date, datetime.min.time()) + timedelta(hours=12)
        tr = pd.date_range(s_dt, s_dt + timedelta(hours=1), freq='15min')
        cache = {}
        for ts in tr:
            path, bnd = get_radar_image(ts + timedelta(hours=5))
            if path: cache[ts.strftime("%H:%M")] = {"path": path, "bounds": bnd}
        st.session_state.radar_cache, st.session_state.time_list = cache, list(cache.keys())

# --- 5. MAIN CONTENT ---
if st.session_state.time_list:
    t_idx = st.select_slider("Select Time", options=range(len(st.session_state.time_list)), 
                             format_func=lambda x: st.session_state.time_list[x])
    curr = st.session_state.radar_cache[st.session_state.time_list[t_idx]]
    layers = [pdk.Layer("BitmapLayer", image=curr["path"], bounds=curr["bounds"], opacity=0.7)]
else:
    layers = []

if st.session_state.active_gdf is not None:
    layers.append(pdk.Layer("GeoJsonLayer", st.session_state.active_gdf.__geo_interface__, 
                            stroked=True, filled=False, get_line_color=[255, 255, 255], line_width_min_pixels=2))

# Using a standard Deck call without the "Nuclear" fixed position
st.pydeck_chart(pdk.Deck(
    layers=layers, 
    initial_view_state=st.session_state.map_view, 
    map_style="mapbox://styles/mapbox/dark-v10"
))
