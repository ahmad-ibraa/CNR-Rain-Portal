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
from pathlib import Path
from datetime import datetime, timedelta
import leafmap.foliumap as leafmap
import plotly.express as px

# --- 1. PAGE CONFIG & UI OVERRIDES ---
st.set_page_config(layout="wide", page_title="CNR Radar Portal", page_icon="🌧️")

# Custom CSS to fix the sidebar toggle, remove margins, and size the map
st.markdown("""
    <style>
        /* 1. Remove padding around the main app container */
        .block-container {
            padding-top: 0.5rem !important;
            padding-bottom: 0rem !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
            max-width: 100%;
        }
        /* 2. Style the Header to be transparent but KEEP it in the DOM */
        [data-testid="stHeader"] {
            background-color: rgba(0,0,0,0);
            height: 3rem;
            visibility: visible;
        }
        /* 3. Style the Sidebar Toggle button (the arrow) so it is ALWAYS visible */
        [data-testid="stSidebarCollapsedControl"] {
            background-color: #262730;
            color: white;
            border-radius: 0 5px 5px 0;
            top: 10px;
            left: 0px;
            visibility: visible;
            display: flex !important;
        }
        /* 4. Force the Map Iframe to be long (75% of viewport height) */
        iframe {
            height: 75vh !important;
            width: 100% !important;
        }
        /* 5. Ensure the slider at the bottom is styled cleanly */
        .stSlider {
            padding: 10px 20px;
        }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SESSION STATE ---
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'raster_cache' not in st.session_state:
    st.session_state.raster_cache = {}
if 'time_list' not in st.session_state:
    st.session_state.time_list = []

# --- 3. HELPER FUNCTIONS ---
def get_tz_offset(dt):
    edt_start = datetime(dt.year, 3, 8 + (6 - datetime(dt.year, 3, 1).weekday()) % 7)
    edt_end = datetime(dt.year, 11, 1 + (6 - datetime(dt.year, 11, 1).weekday()) % 7)
    return 4 if edt_start <= dt < edt_end else 5

def download_s3_grib(file_type, dt_local):
    offset = get_tz_offset(dt_local)
    dt_utc = dt_local + timedelta(hours=offset)
    s3_folder = dt_utc.strftime("%Y%m%d")
    ts = dt_utc.strftime("%Y%m%d-%H%M00")
    
    base_url = "https://noaa-mrms-pds.s3.amazonaws.com/CONUS"
    if file_type == "RO":
        url = f"{base_url}/RadarOnly_QPE_15M_00.00/{s3_folder}/MRMS_RadarOnly_QPE_15M_00.00_{ts}.grib2.gz"
    else:
        url = f"{base_url}/MultiSensor_QPE_01H_Pass2_00.00/{s3_folder}/MRMS_MultiSensor_QPE_01H_Pass2_00.00_{ts}.grib2.gz"

    tmp_path = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False).name
    try:
        r = requests.get(url, stream=True, timeout=10)
        if r.status_code == 200:
            with gzip.GzipFile(fileobj=r.raw) as gz, open(tmp_path, "wb") as f:
                shutil.copyfileobj(gz, f)
            with xr.open_dataset(tmp_path, engine="cfgrib", backend_kwargs={"indexpath": ""}) as ds:
                da = ds[list(ds.data_vars)[0]].load()
                da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180).sortby("longitude")
                return da.rio.write_crs("EPSG:4326")
    except: return None
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)
    return None

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("🛰️ CNR Rain Portal")
    
    st.subheader("1. Datetime Range")
    c1, c2 = st.columns(2)
    start_dt = datetime.combine(c1.date_input("Start Date", datetime.now() - timedelta(days=2)), 
                                c2.time_input("Start Time", value=datetime.strptime("00:00", "%H:%M").time()))
    
    c3, c4 = st.columns(2)
    end_dt = datetime.combine(c3.date_input("End Date", datetime.now() - timedelta(days=1)), 
                              c4.time_input("End Time", value=datetime.strptime("23:45", "%H:%M").time()))

    st.subheader("2. Geometry")
    uploaded_zip = st.file_uploader("Upload ZIP Shapefile", type="zip")
    
    active_gdf = None
    if uploaded_zip:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(uploaded_zip, 'r') as z:
                z.extractall(tmp_dir)
            shps = list(Path(tmp_dir).rglob("*.shp"))
            if shps:
                active_gdf = gpd.read_file(shps[0]).to_crs("EPSG:4326")
                st.success("Geometry Loaded")

    if st.button("🚀 Process & Export", use_container_width=True):
        if active_gdf is not None:
            with st.spinner("Processing radar data..."):
                time_range = pd.date_range(start_dt, end_dt, freq='15min')
                temp_rasters, series_list = {}, []
                
                prog = st.progress(0)
                for i, ts in enumerate(time_range):
                    da = download_s3_grib("RO", ts)
                    if da is not None:
                        t_tif = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
                        da.rio.to_raster(t_tif)
                        time_str = ts.strftime("%Y-%m-%d %H:%M")
                        temp_rasters[time_str] = t_tif
                        clipped = da.rio.clip(active_gdf.geometry, active_gdf.crs, all_touched=True)
                        series_list.append({"time": ts, "rain_in": clipped.mean().item() / 25.4})
                    prog.progress((i + 1) / len(time_range))
                
                st.session_state.processed_df = pd.DataFrame(series_list).set_index("time")
                st.session_state.raster_cache = temp_rasters
                st.session_state.time_list = list(temp_rasters.keys())
        else:
            st.warning("Please upload a ZIP shapefile.")

# --- 5. MAIN PAGE LAYOUT ---

# Define the map container
m = leafmap.Map(center=[40.1, -74.5], zoom=8)

# Add Shapefile
if active_gdf is not None:
    m.add_gdf(active_gdf, layer_name="Boundaries")
    if not st.session_state.time_list:
        m.zoom_to_gdf(active_gdf)

# Display the Map (High visibility, takes 75% of screen length)
map_placeholder = st.empty()

# Add Time Slider BELOW the map
view_time = None
if st.session_state.time_list:
    st.markdown("### 🕰️ Radar Time Selection")
    view_time = st.select_slider("Slide to change radar view time:", options=st.session_state.time_list)

# Add Raster to map if slider is moved
if view_time and view_time in st.session_state.raster_cache:
    m.add_raster(st.session_state.raster_cache[view_time], layer_name="Radar", colormap="jet", opacity=0.5)

# Render the Map
with map_placeholder:
    m.to_streamlit(responsive=True)

# --- 6. RESULTS SECTION ---
if st.session_state.processed_df is not None:
    st.write("---")
    st.write("### 📊 Extracted Rainfall Statistics")
    col_a, col_b = st.columns([3, 1])
    with col_a:
        fig = px.bar(st.session_state.processed_df.reset_index(), x="time", y="rain_in", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        st.dataframe(st.session_state.processed_df.describe(), use_container_width=True)
        csv = st.session_state.processed_df.to_csv().encode('utf-8')
        st.download_button("📥 Download Results", csv, "rainfall.csv", "text/csv", use_container_width=True)
