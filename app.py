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

# --- 1. PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="CNR Radar Portal", page_icon="🛰️")

# --- 2. THE CSS FIX (No more squishing) ---
st.markdown("""
    <style>
        /* Remove the 'paper' margins around the app */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 0rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        
        /* Ensure the sidebar toggle is always visible and floating */
        [data-testid="stSidebarCollapsedControl"] {
            background-color: #262730;
            color: white;
            border-radius: 0 5px 5px 0;
            top: 10px;
            display: flex !important;
        }

        /* Target the map specifically to take up a massive amount of height */
        /* Using 85vh here prevents the 'squishing' while filling the screen */
        iframe {
            height: 85vh !important;
            width: 100% !important;
        }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SESSION STATE ---
if 'processed_df' not in st.session_state: st.session_state.processed_df = None
if 'raster_cache' not in st.session_state: st.session_state.raster_cache = {}
if 'time_list' not in st.session_state: st.session_state.time_list = []

# --- 4. HELPERS ---
def get_tz_offset(dt):
    edt_s = datetime(dt.year, 3, 8 + (6 - datetime(dt.year, 3, 1).weekday()) % 7)
    edt_e = datetime(dt.year, 11, 1 + (6 - datetime(dt.year, 11, 1).weekday()) % 7)
    return 4 if edt_s <= dt < edt_e else 5

def download_s3_grib(file_type, dt_local):
    offset = get_tz_offset(dt_local)
    dt_utc = dt_local + timedelta(hours=offset)
    s3_path = dt_utc.strftime("%Y%m%d")
    ts = dt_utc.strftime("%Y%m%d-%H%M00")
    base = "https://noaa-mrms-pds.s3.amazonaws.com/CONUS"
    f_prefix = "RadarOnly_QPE_15M" if file_type == "RO" else "MultiSensor_QPE_01H_Pass2"
    url = f"{base}/{f_prefix}_00.00/{s3_path}/MRMS_{f_prefix}_00.00_{ts}.grib2.gz"

    tmp = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False).name
    try:
        r = requests.get(url, stream=True, timeout=10)
        if r.status_code == 200:
            with gzip.GzipFile(fileobj=r.raw) as gz, open(tmp, "wb") as f:
                shutil.copyfileobj(gz, f)
            with xr.open_dataset(tmp, engine="cfgrib", backend_kwargs={"indexpath": ""}) as ds:
                da = ds[list(ds.data_vars)[0]].load()
                da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180).sortby("longitude")
                return da.rio.write_crs("EPSG:4326")
    except: return None
    finally:
        if os.path.exists(tmp): os.remove(tmp)
    return None

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("🛰️ CNR Rain Portal")
    
    c1, c2 = st.columns(2)
    s_dt = datetime.combine(c1.date_input("Start", datetime.now()-timedelta(2)), c2.time_input("T1", datetime.min.time()))
    c3, c4 = st.columns(2)
    e_dt = datetime.combine(c3.date_input("End", datetime.now()-timedelta(1)), c4.time_input("T2", datetime.max.time()))
    
    up_zip = st.file_uploader("Upload ZIP Shapefile", type="zip")
    active_gdf = None
    if up_zip:
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(up_zip, 'r') as z: z.extractall(td)
            shps = list(Path(td).rglob("*.shp"))
            if shps:
                active_gdf = gpd.read_file(shps[0]).to_crs("EPSG:4326")
                st.success("Geometry Ready")

    if st.button("🚀 Process & Export", use_container_width=True):
        if active_gdf is not None:
            with st.spinner("Processing..."):
                tr = pd.date_range(s_dt, e_dt, freq='15min')
                rc, sl = {}, []
                pb = st.progress(0)
                for i, ts in enumerate(tr):
                    da = download_s3_grib("RO", ts)
                    if da is not None:
                        tf = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
                        da.rio.to_raster(tf)
                        rc[ts.strftime("%Y-%m-%d %H:%M")] = tf
                        sl.append({"time": ts, "rain_in": da.rio.clip(active_gdf.geometry, active_gdf.crs).mean().item()/25.4})
                    pb.progress((i+1)/len(tr))
                st.session_state.processed_df, st.session_state.raster_cache, st.session_state.time_list = pd.DataFrame(sl).set_index("time"), rc, list(rc.keys())

    if st.session_state.processed_df is not None:
        st.divider()
        st.download_button("📥 Download CSV Output", st.session_state.processed_df.to_csv().encode('utf-8'), "rainfall.csv", use_container_width=True)

# --- 6. MAIN CONTENT ---

# Display the Time Slider at the VERY TOP of the main page
view_t = None
if st.session_state.time_list:
    view_t = st.select_slider("🕰️ Select Radar Time", options=st.session_state.time_list)

# The Map
m = leafmap.Map(center=[40.1, -74.5], zoom=8)
if active_gdf is not None:
    m.add_gdf(active_gdf, layer_name="Boundary")
    if not st.session_state.time_list: m.zoom_to_gdf(active_gdf)

if view_t and view_t in st.session_state.raster_cache:
    m.add_raster(st.session_state.raster_cache[view_t], layer_name="Radar", colormap="jet", opacity=0.5)

# Render Map
m.to_streamlit(responsive=True)

# Statistics bar chart (if data exists)
if st.session_state.processed_df is not None:
    st.plotly_chart(px.bar(st.session_state.processed_df.reset_index(), x="time", y="rain_in", template="plotly_dark"), use_container_width=True)
