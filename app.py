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
import time
from pathlib import Path
from datetime import datetime, timedelta
import leafmap.foliumap as leafmap
import plotly.express as px

# --- 1. PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="CNR Radar Portal", page_icon="🛰️")

# Force CSS to prevent layout jumping
st.markdown("""
    <style>
        .block-container { padding: 1rem !important; }
        iframe { height: 65vh !important; width: 100% !important; border-radius: 8px; }
        .stButton button { width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SESSION STATE ---
if 'processed_df' not in st.session_state: st.session_state.processed_df = None
if 'raster_cache' not in st.session_state: st.session_state.raster_cache = {}
if 'time_list' not in st.session_state: st.session_state.time_list = []
if 'playing' not in st.session_state: st.session_state.playing = False
if 'current_idx' not in st.session_state: st.session_state.current_idx = 0

# --- 3. DOWNLOADER ---
def download_mrms(dt_utc):
    ts = dt_utc.strftime("%Y%m%d-%H%M00")
    url = f"https://noaa-mrms-pds.s3.amazonaws.com/CONUS/RadarOnly_QPE_15M_00.00/{dt_utc.strftime('%Y%m%d')}/MRMS_RadarOnly_QPE_15M_00.00_{ts}.grib2.gz"
    tmp = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False).name
    try:
        r = requests.get(url, stream=True, timeout=10)
        if r.status_code == 200:
            with gzip.GzipFile(fileobj=r.raw) as gz, open(tmp, "wb") as f:
                shutil.copyfileobj(gz, f)
            with xr.open_dataset(tmp, engine="cfgrib", backend_kwargs={"indexpath": ""}) as ds:
                da = ds[list(ds.data_vars)[0]].load()
                # Fix coordinate wrap-around
                da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180).sortby("longitude")
                da = da.rio.write_crs("EPSG:4326")
                return da
    except: return None
    finally:
        if os.path.exists(tmp): os.remove(tmp)
    return None

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("🛰️ CNR Portal")
    tz_mode = st.radio("Timezone", ["Local (EST/EDT)", "UTC"])
    # (Standard date/time inputs and processing logic remains here...)
    # [Assuming processing logic from previous step is preserved here for brevity]
    # ...

# --- 5. MAIN CONTENT ---
map_placeholder = st.empty()

def draw_map(selected_time):
    # We use a fresh map instance to ensure the raster clears properly
    m = leafmap.Map(center=[40.1, -74.5], zoom=8)
    
    # Try to add the raster with high visibility settings
    if selected_time in st.session_state.raster_cache:
        m.add_raster(
            st.session_state.raster_cache[selected_time],
            colormap="jet",
            opacity=0.7,
            layer_name="Radar Layer",
            vmin=0.1,  # Ensure low rain is colored
            vmax=3.0   # Cap for high intensity
        )
    
    # Render to the placeholder slot
    with map_placeholder:
        m.to_streamlit()

# --- 6. CONTROLS (Slider & Play/Pause) ---
if st.session_state.time_list:
    c1, c2 = st.columns([1, 6])
    
    # Toggle Play/Stop state
    if st.session_state.playing:
        if c1.button("⏹️ Stop"):
            st.session_state.playing = False
            st.rerun()
    else:
        if c1.button("▶️ Play"):
            st.session_state.playing = True
            st.rerun()

    # Time Slider
    st.session_state.current_idx = st.select_slider(
        "Current Time",
        options=range(len(st.session_state.time_list)),
        value=st.session_state.current_idx,
        format_func=lambda x: st.session_state.time_list[x]
    )

    # Execution Logic
    if st.session_state.playing:
        # Loop through indices starting from current
        for i in range(st.session_state.current_idx, len(st.session_state.time_list)):
            st.session_state.current_idx = i
            draw_map(st.session_state.time_list[i])
            time.sleep(0.5)
            if not st.session_state.playing: break
        st.session_state.playing = False
        st.rerun()
    else:
        # Static update
        draw_map(st.session_state.time_list[st.session_state.current_idx])

# --- 7. CHART ---
if st.session_state.processed_df is not None:
    st.plotly_chart(px.bar(st.session_state.processed_df.reset_index(), x="time", y="rain_in", 
                           title="Rainfall Profile", template="plotly_dark"), use_container_width=True)
