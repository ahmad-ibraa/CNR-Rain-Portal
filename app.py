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

# --- 1. CLEAN & STABLE CSS ---
st.set_page_config(layout="wide", page_title="CNR Radar Portal")

st.markdown("""
    <style>
        /* 1. Force the main block to take up the full width without huge margins */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 0rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            max-width: 98% !important;
        }

        /* 2. Style the Sidebar so it stays visible and organized */
        [data-testid="stSidebar"] {
            background-color: #0e1117 !important;
            border-right: 1px solid #333 !important;
            min-width: 350px !important;
        }

        /* 3. Force the Map container to a massive height so it can't be 'small' */
        .stPydeckChart {
            height: 80vh !important;
            border: 1px solid #333 !important;
            border-radius: 10px;
        }

        /* 4. Improve button visibility */
        .stButton>button {
            width: 100% !important;
            border-radius: 5px;
            height: 3em;
            background-color: #FF4B4B;
            color: white;
            font-weight: bold;
        }

        header, footer { visibility: hidden !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SESSION STATE INITIALIZATION ---
if 'radar_cache' not in st.session_state: st.session_state.radar_cache = {}
if 'time_list' not in st.session_state: st.session_state.time_list = []
if 'processed_df' not in st.session_state: st.session_state.processed_df = None
if 'active_gdf' not in st.session_state: st.session_state.active_gdf = None
if 'map_view' not in st.session_state: 
    st.session_state.map_view = pdk.ViewState(latitude=40.7, longitude=-74.0, zoom=9)

# --- 3. DATA ENGINE ---
RADAR_COLORS = ['#76fffe', '#01a0fe', '#0001ef', '#01ef01', '#019001', '#ffff01', '#e7c001', '#ff9000', '#ff0101']
RADAR_CMAP = ListedColormap(RADAR_COLORS)

def get_radar_image(dt_utc):
    ts_str = dt_utc.strftime("%Y%m%d-%H%M00") 
    url = f"https://noaa-mrms-pds.s3.amazonaws.com/CONUS/RadarOnly_QPE_15M_00.00/{dt_utc.strftime('%Y%m%d')}/MRMS_RadarOnly_QPE_15M_00.00_{ts_str}.grib2.gz"
    tmp_grib = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False).name
    try:
        r = requests.get(url, stream=True, timeout=10)
        if r.status_code != 200: return None, None
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
    except Exception as e:
        return None, None
    finally:
        if os.path.exists(tmp_grib): os.remove(tmp_grib)

# --- 4. SIDEBAR (Full Content) ---
with st.sidebar:
    st.title("CNR Radar Portal")
    st.write("---")
    
    # Date and Time Selection
    s_date = st.date_input("Analysis Date", value=datetime.now().date())
    
    c1, c2 = st.columns(2)
    s_hour = c1.number_input("Start Hour (UTC)", 0, 23, 12)
    e_hour = c2.number_input("End Hour (UTC)", 0, 23, 14)
    
    # Watershed Upload
    up_zip = st.file_uploader("Upload Basin (ZIP with SHP)", type="zip")
    if up_zip:
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(up_zip, 'r') as z: z.extractall(td)
            shps = list(Path(td).rglob("*.shp"))
            if shps:
                st.session_state.active_gdf = gpd.read_file(shps[0]).to_crs("EPSG:4326")
                b = st.session_state.active_gdf.total_bounds
                st.session_state.map_view = pdk.ViewState(
                    latitude=(b[1]+b[3])/2, longitude=(b[0]+b[2])/2, zoom=10
                )
                st.success("Basin Loaded")

    # Action Buttons
    if st.button("RUN RADAR ANALYSIS"):
        with st.spinner("Fetching MRMS Data..."):
            s_dt = datetime.combine(s_date, datetime.min.time()) + timedelta(hours=s_hour)
            tr = pd.date_range(s_dt, s_dt + timedelta(hours=(e_hour-s_hour)), freq='15min')
            cache = {}
            for ts in tr:
                path, bnd = get_radar_image(ts)
                if path:
                    cache[ts.strftime("%H:%M")] = {"path": path, "bounds": bnd}
            st.session_state.radar_cache = cache
            st.session_state.time_list = list(cache.keys())

    if st.session_state.time_list:
        st.write("---")
        st.info(f"Loaded {len(st.session_state.time_list)} radar frames.")

# --- 5. MAIN CONTENT AREA ---
# We force the map to render in a container with a defined height
map_container = st.container()

with map_container:
    if st.session_state.time_list:
        # Putting the slider right above the map
        t_str = st.select_slider("Select Radar Timestamp", options=st.session_state.time_list)
        curr = st.session_state.radar_cache[t_str]
        
        layers = [
            pdk.Layer("BitmapLayer", image=curr["path"], bounds=curr["bounds"], opacity=0.7)
        ]
    else:
        layers = []
        st.warning("No radar data loaded. Use the sidebar to process a date.")

    if st.session_state.active_gdf is not None:
        layers.append(pdk.Layer(
            "GeoJsonLayer", 
            st.session_state.active_gdf.__geo_interface__, 
            stroked=True, filled=False, get_line_color=[255, 255, 255], line_width_min_pixels=2
        ))

    # This is the map render
    st.pydeck_chart(pdk.Deck(
        layers=layers, 
        initial_view_state=st.session_state.map_view, 
        map_style="mapbox://styles/mapbox/dark-v10",
        tooltip=True
    ))
