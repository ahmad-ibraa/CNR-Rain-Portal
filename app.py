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

# --- 1. PAGE CONFIG & FULL-SCREEN CSS ---
st.set_page_config(layout="wide", page_title="CNR Radar Portal", page_icon="🌧️")

# This CSS removes the padding at the top and sides, and hides the scrollbar 
# to make the map feel like a native desktop GIS application.
st.markdown("""
    <style>
        /* Remove margins from the main content area */
        .block-container {
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
            padding-left: 0rem !important;
            padding-right: 0rem !important;
        }
        /* Hide the Streamlit header */
        header {visibility: hidden;}
        /* Make the map container fill the available height */
        iframe {
            width: 100% !important;
            height: 85vh !important;
        }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SESSION STATE INITIALIZATION ---
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'raster_cache' not in st.session_state:
    st.session_state.raster_cache = {}
if 'time_list' not in st.session_state:
    st.session_state.time_list = []

# --- 3. CORE PROCESSING FUNCTIONS ---
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

# --- 4. SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("🌧️ CNR Portal")
    st.header("1. Time Window")
    
    c1, c2 = st.columns(2)
    start_dt = datetime.combine(c1.date_input("Start", datetime.now() - timedelta(days=2)), 
                                c2.time_input("T1", value=datetime.strptime("00:00", "%H:%M").time()))
    
    c3, c4 = st.columns(2)
    end_dt = datetime.combine(c3.date_input("End", datetime.now() - timedelta(days=1)), 
                              c4.time_input("T2", value=datetime.strptime("23:45", "%H:%M").time()))

    st.header("2. Boundary")
    uploaded_zip = st.file_uploader("Upload ZIP Shapefile", type="zip")
    
    active_gdf = None
    if uploaded_zip:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(uploaded_zip, 'r') as z:
                z.extractall(tmp_dir)
            shps = list(Path(tmp_dir).rglob("*.shp"))
            if shps:
                active_gdf = gpd.read_file(shps[0]).to_crs("EPSG:4326")
                st.success("Boundary Ready")

    if st.button("🚀 Process Data", use_container_width=True):
        if active_gdf is not None:
            with st.spinner("Extracting..."):
                time_range = pd.date_range(start_dt, end_dt, freq='15min')
                temp_rasters, series_list = {}, []
                
                prog = st.progress(0)
                for i, ts in enumerate(time_range):
                    da = download_s3_grib("RO", ts)
                    if da is not None:
                        t_tif = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
                        da.rio.to_raster(t_tif)
                        temp_rasters[ts.strftime("%Y-%m-%d %H:%M")] = t_tif
                        clipped = da.rio.clip(active_gdf.geometry, active_gdf.crs, all_touched=True)
                        series_list.append({"time": ts, "rain_in": clipped.mean().item() / 25.4})
                    prog.progress((i + 1) / len(time_range))
                
                st.session_state.processed_df = pd.DataFrame(series_list).set_index("time")
                st.session_state.raster_cache = temp_rasters
                st.session_state.time_list = list(temp_rasters.keys())
        else:
            st.warning("Upload a ZIP first")

# --- 5. MAIN PAGE: FULL-WIDTH MAP ---

# Time Slider (only appears if data is ready)
view_time = None
if st.session_state.time_list:
    view_time = st.select_slider("Select Radar Timestamp", options=st.session_state.time_list)

m = leafmap.Map(center=[40.1, -74.5], zoom=8)

if view_time and view_time in st.session_state.raster_cache:
    m.add_raster(st.session_state.raster_cache[view_time], layer_name="Radar", colormap="terrain", opacity=0.6)

if active_gdf is not None:
    m.add_gdf(active_gdf, layer_name="User Boundary")
    # Auto-zoom on first load
    if not st.session_state.time_list:
        m.zoom_to_gdf(active_gdf)

# responsive=True + the CSS above makes this full-screen
m.to_streamlit(responsive=True)

# --- 6. RESULTS SECTION (BELOW MAP) ---
if st.session_state.processed_df is not None:
    st.markdown("### 📊 Rain Analysis")
    col_plot, col_stats = st.columns([3, 1])
    with col_plot:
        fig = px.bar(st.session_state.processed_df.reset_index(), x="time", y="rain_in", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    with col_stats:
        st.download_button("📥 Export CSV", st.session_state.processed_df.to_csv().encode('utf-8'), "radar_rain.csv", use_container_width=True)
        st.dataframe(st.session_state.processed_df.describe(), use_container_width=True)
