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

# --- APP CONFIG ---
st.set_page_config(layout="wide", page_title="CNR Radar Portal", page_icon="🌧️")

# Initialize Session State
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'raster_cache' not in st.session_state:
    st.session_state.raster_cache = {}
if 'time_list' not in st.session_state:
    st.session_state.time_list = []

# --- CORE FUNCTIONS ---

def get_tz_offset(dt):
    # Standard NJ logic (EDT=4, EST=5)
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

# --- SIDEBAR: INPUTS ---
with st.sidebar:
    st.header("1. Selection")
    
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start Date", datetime.now() - timedelta(days=2))
    start_time = col2.time_input("Start Time", value=datetime.strptime("00:00", "%H:%M").time())
    
    col3, col4 = st.columns(2)
    end_date = col3.date_input("End Date", datetime.now() - timedelta(days=1))
    end_time = col4.time_input("End Time", value=datetime.strptime("23:45", "%H:%M").time())
    
    start_dt = datetime.combine(start_date, start_time)
    end_dt = datetime.combine(end_date, end_time)

    st.divider()
    st.header("2. Boundary")
    uploaded_zip = st.file_uploader("Import Boundaries (ZIP)", type="zip")
    
    active_gdf = None
    if uploaded_zip:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(uploaded_zip, 'r') as z:
                z.extractall(tmp_dir)
            shps = list(Path(tmp_dir).rglob("*.shp"))
            if shps:
                active_gdf = gpd.read_file(shps[0]).to_crs("EPSG:4326")
                st.success(f"Loaded {len(active_gdf)} polygons")

    st.divider()
    if st.button("🚀 Process & Export", use_container_width=True):
        if active_gdf is None:
            st.error("Missing Shapefile!")
        else:
            with st.spinner("Downloading Radar Data..."):
                time_range = pd.date_range(start_dt, end_dt, freq='15min')
                temp_rasters = {}
                series_list = []
                
                prog = st.progress(0)
                for i, ts in enumerate(time_range):
                    da = download_s3_grib("RO", ts)
                    if da is not None:
                        # Save temp TIFF for map
                        t_tif = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
                        da.rio.to_raster(t_tif)
                        time_str = ts.strftime("%Y-%m-%d %H:%M")
                        temp_rasters[time_str] = t_tif
                        
                        # Clip and calculate mean
                        clipped = da.rio.clip(active_gdf.geometry, active_gdf.crs, all_touched=True)
                        series_list.append({"time": ts, "rain_in": clipped.mean().item() / 25.4})
                    
                    prog.progress((i + 1) / len(time_range))
                
                st.session_state.processed_df = pd.DataFrame(series_list).set_index("time")
                st.session_state.raster_cache = temp_rasters
                st.session_state.time_list = list(temp_rasters.keys())

# --- MAIN PAGE ---

# 1. Visualization Controls (Only show after processing)
view_time = None
if st.session_state.time_list:
    view_time = st.select_slider("🕰️ Visualizer Slider:", options=st.session_state.time_list)

# 2. The Map
m = leafmap.Map(center=[40.1, -74.5], zoom=8)

if view_time and view_time in st.session_state.raster_cache:
    m.add_raster(st.session_state.raster_cache[view_time], layer_name="Radar Layer", colormap="terrain", opacity=0.6)

if active_gdf is not None:
    m.add_gdf(active_gdf, layer_name="Boundaries")
    # If it's a new upload, center the map
    if not st.session_state.time_list:
        m.zoom_to_gdf(active_gdf)

m.to_streamlit(height=600)

# 3. Results Section
if st.session_state.processed_df is not None:
    st.divider()
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        fig = px.bar(st.session_state.processed_df.reset_index(), x="time", y="rain_in", title="Rainfall depth (inches)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col_b:
        st.subheader("Stats & Export")
        st.dataframe(st.session_state.processed_df.describe())
        csv_data = st.session_state.processed_df.to_csv().encode('utf-8')
        st.download_button("📥 Download CSV", csv_data, "rainfall_results.csv", "text/csv")
