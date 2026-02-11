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

# --- 1. PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="CNR Radar Portal")

# --- 2. DATA CLEANING & PROJECTION FIX ---
def process_mrms_to_points(dt_utc):
    """Downloads and converts radar to a lightweight dataframe."""
    ts_str = dt_utc.strftime("%Y%m%d-%H1500") 
    url = f"https://noaa-mrms-pds.s3.amazonaws.com/CONUS/RadarOnly_QPE_15M_00.00/{dt_utc.strftime('%Y%m%d')}/MRMS_RadarOnly_QPE_15M_00.00_{ts_str}.grib2.gz"
    
    tmp_path = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False).name
    try:
        r = requests.get(url, stream=True, timeout=10)
        if r.status_code != 200: return None
        
        with gzip.GzipFile(fileobj=r.raw) as gz, open(tmp_path, "wb") as f:
            shutil.copyfileobj(gz, f)
            
        with xr.open_dataset(tmp_path, engine="cfgrib") as ds:
            # 1. Fix Longitude Projection (CRITICAL for GIS)
            da = ds[list(ds.data_vars)[0]].load()
            da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180).sortby("longitude")
            
            # 2. Convert to lightweight points only where it's raining (RAM Optimization)
            df = da.to_dataframe(name='val').reset_index()
            df = df[df['val'] > 0.5] # Only keep actual rain pixels
            
            # 3. Clean up the large xarray object immediately
            return df[['latitude', 'longitude', 'val']]
    except: return None
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("🛰️ GIS Radar Portal")
    start_d = st.date_input("Date", value=datetime.now() - timedelta(days=1))
    up_zip = st.file_uploader("Upload Watershed ZIP", type="zip")
    
    if st.button("🚀 Process GIS Layers"):
        if up_zip:
            # Logic to extract and save boundary to state...
            st.session_state.processed_data = {}
            for h in range(24):
                dt = datetime.combine(start_d, datetime.min.time()) + timedelta(hours=h)
                # Convert to UTC for NOAA
                df_points = process_mrms_to_points(dt + timedelta(hours=5)) 
                if df_points is not None:
                    st.session_state.processed_data[h] = df_points
            st.success("Data Cached!")

# --- 4. THE PYDECK MAP (The "Pro" GIS Option) ---
st.subheader("Interactive Radar Map")

if 'processed_data' in st.session_state:
    # 1. Timeline Slider
    hour_idx = st.select_slider("Hour of Day", options=list(st.session_state.processed_data.keys()))
    current_df = st.session_state.processed_data[hour_idx]

    # 2. Radar Layer (Hexagons or Points)
    radar_layer = pdk.Layer(
        "ScatterplotLayer",
        current_df,
        get_position=["longitude", "latitude"],
        get_fill_color="[255 - (val * 10), 150, val * 50, 140]", # Dynamic coloring
        get_radius=2000,
        pickable=True,
    )

    # 3. Render GPU-Accelerated Map
    st.pydeck_chart(pdk.Deck(
        layers=[radar_layer],
        initial_view_state=pdk.ViewState(
            latitude=40.1, longitude=-74.5, zoom=8, pitch=45
        ),
        map_style="mapbox://styles/mapbox/dark-v10",
    ))
