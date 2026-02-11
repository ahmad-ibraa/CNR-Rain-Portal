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
st.set_page_config(layout="wide", page_title="CNR Radar Portal", page_icon="🛰️")

# CSS to fix the blue gutter, center the map, and stabilize the sidebar
st.markdown("""
    <style>
        .stApp {
            background-color: #0e1117;
        }
        /* Tightens the main area to prevent the blue background from bleeding through */
        .block-container {
            padding: 0rem !important;
            max-width: 100% !important;
            height: 100vh !important;
            display: flex;
            flex-direction: column;
        }
        /* Header and Sidebar Toggle Fix */
        [data-testid="stHeader"] {
            background-color: rgba(0,0,0,0);
            height: 0px;
        }
        [data-testid="stSidebarCollapsedControl"] {
            background-color: #262730;
            color: white;
            top: 15px;
            display: flex !important;
            z-index: 9999;
        }
        /* Map Height Adjustment */
        .map-frame iframe {
            height: 85vh !important; 
            width: 100% !important;
            border: none;
        }
        /* Slider styling at the bottom */
        .bottom-ui {
            padding: 10px 25px;
            background-color: #0e1117;
            border-top: 1px solid #31333f;
        }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SESSION STATE ---
if 'processed_df' not in st.session_state: st.session_state.processed_df = None
if 'raster_cache' not in st.session_state: st.session_state.raster_cache = {}
if 'time_list' not in st.session_state: st.session_state.time_list = []

# --- 3. HELPER FUNCTIONS ---
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

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("🛰️ CNR Rain Portal")
    
    # 1. Datetime Range
    c1, c2 = st.columns(2)
    s_dt = datetime.combine(c1.date_input("Start", datetime.now()-timedelta(2)), c2.time_input("T1", datetime.min.time()))
    c3, c4 = st.columns(2)
    e_dt = datetime.combine(c3.date_input("End", datetime.now()-timedelta(1)), c4.time_input("T2", datetime.max.time()))
    
    # 2. Boundary ZIP
    up_zip = st.file_uploader("Upload ZIP Shapefile", type="zip")
    active_gdf = None
    if up_zip:
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(up_zip, 'r') as z: z.extractall(td)
            shps = list(Path(td).rglob("*.shp"))
            if shps:
                active_gdf = gpd.read_file(shps[0]).to_crs("EPSG:4326")
                st.success("Geometry Ready")

    # 3. Process Button
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
                
                st.session_state.processed_df = pd.DataFrame(sl).set_index("time")
                st.session_state.raster_cache = rc
                st.session_state.time_list = list(rc.keys())
        else:
            st.warning("Upload a ZIP first")

    # 4. DOWNLOAD CSV (Placed directly under the button)
    if st.session_state.processed_df is not None:
        st.divider()
        st.success("Data Ready!")
        csv_data = st.session_state.processed_df.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Download CSV Output",
            data=csv_data,
            file_name=f"Rainfall_{s_dt.strftime('%Y%m%d')}_to_{e_dt.strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# --- 5. MAIN CONTENT ---
m = leafmap.Map(center=[40.1, -74.5], zoom=8)
if active_gdf is not None:
    m.add_gdf(active_gdf, layer_name="Boundary")
    if not st.session_state.time_list: m.zoom_to_gdf(active_gdf)

# Display Map
st.markdown('<div class="map-frame">', unsafe_allow_html=True)
map_sub = st.empty()
st.markdown('</div>', unsafe_allow_html=True)

# Time Selection Slider at bottom
view_t = None
if st.session_state.time_list:
    st.markdown('<div class="bottom-ui">', unsafe_allow_html=True)
    view_t = st.select_slider("🕰️ Select Radar Time:", options=st.session_state.time_list)
    st.markdown('</div>', unsafe_allow_html=True)

if view_t and view_t in st.session_state.raster_cache:
    m.add_raster(st.session_state.raster_cache[view_t], layer_name="Radar", colormap="jet", opacity=0.5)

with map_sub:
    m.to_streamlit(responsive=True)

# Graph below the map
if st.session_state.processed_df is not None:
    st.write("---")
    fig = px.bar(st.session_state.processed_df.reset_index(), x="time", y="rain_in", title="Mean Rainfall (Inches)", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
