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
import gc # Garbage Collector to free RAM
from pathlib import Path
from datetime import datetime, timedelta
import leafmap.foliumap as leafmap
import plotly.express as px

# --- 1. PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="CNR Radar Portal", page_icon="🛰️")

# --- 2. CACHING & MEMORY MGMT ---
# This prevents the app from re-downloading/re-processing the same hour
@st.cache_data(show_spinner=False, max_entries=24)
def get_mrms_data(ts_utc):
    ts_str = ts_utc.strftime("%Y%m%d-%H1500") 
    url = f"https://noaa-mrms-pds.s3.amazonaws.com/CONUS/RadarOnly_QPE_15M_00.00/{ts_utc.strftime('%Y%m%d')}/MRMS_RadarOnly_QPE_15M_00.00_{ts_str}.grib2.gz"
    
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp_file:
        tmp_path = tmp_file.name
    try:
        r = requests.get(url, stream=True, timeout=10)
        if r.status_code == 200:
            with gzip.GzipFile(fileobj=r.raw) as gz, open(tmp_path, "wb") as f:
                shutil.copyfileobj(gz, f)
            with xr.open_dataset(tmp_path, engine="cfgrib") as ds:
                da = ds[list(ds.data_vars)[0]].load()
                # Correct coordinates
                da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180).sortby("longitude")
                da = da.rio.write_crs("EPSG:4326")
                return da
    except Exception as e:
        return None
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)
    return None

if 'raster_cache' not in st.session_state: st.session_state.raster_cache = {}
if 'time_list' not in st.session_state: st.session_state.time_list = []
if 'playing' not in st.session_state: st.session_state.playing = False
if 'current_idx' not in st.session_state: st.session_state.current_idx = 0

# --- 3. HELPERS ---
def get_tz_offset(dt):
    edt_s = datetime(dt.year, 3, 8 + (6 - datetime(dt.year, 3, 1).weekday()) % 7)
    edt_e = datetime(dt.year, 11, 1 + (6 - datetime(dt.year, 11, 1).weekday()) % 7)
    return 4 if edt_s <= dt < edt_e else 5

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("🛰️ CNR Portal")
    tz_mode = st.radio("Timezone", ["Local (EST/EDT)", "UTC"])
    
    yesterday = datetime.now() - timedelta(days=1)
    start_d = st.date_input("Start Date", value=yesterday.date())
    end_d = st.date_input("End Date", value=yesterday.date())
    
    hours = [f"{h:02d}:00" for h in range(24)]
    c1, c2 = st.columns(2)
    start_t = c1.selectbox("Start Hour", hours, index=0)
    end_t = c2.selectbox("End Hour", hours, index=23)
    
    up_zip = st.file_uploader("Upload Watershed ZIP", type="zip")
    if up_zip:
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(up_zip, 'r') as z: z.extractall(td)
            shps = list(Path(td).rglob("*.shp"))
            if shps:
                st.session_state.active_gdf = gpd.read_file(shps[0]).to_crs("EPSG:4326")

    if st.button("🚀 Process Data", use_container_width=True):
        if 'active_gdf' in st.session_state:
            # Clear old cache to save RAM
            for f in st.session_state.raster_cache.values():
                if os.path.exists(f): os.remove(f)
            st.session_state.raster_cache = {}
            
            s_dt = datetime.combine(start_d, datetime.strptime(start_t, "%H:%M").time())
            e_dt = datetime.combine(end_d, datetime.strptime(end_t, "%H:%M").time())
            tr = pd.date_range(s_dt, e_dt, freq='1H')
            
            sl = []
            pb = st.progress(0)
            for i, ts in enumerate(tr):
                ts_utc = ts if tz_mode == "UTC" else ts + timedelta(hours=get_tz_offset(ts))
                da = get_mrms_data(ts_utc)
                if da is not None:
                    # Save small GeoTIFF for Map
                    map_da = da.where(da > 0.1, np.nan)
                    tf = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
                    map_da.rio.to_raster(tf)
                    st.session_state.raster_cache[ts.strftime("%Y-%m-%d %H:%M")] = tf
                    
                    # Calculate Stats
                    clipped = da.rio.clip(st.session_state.active_gdf.geometry, "EPSG:4326")
                    mean_val = float(clipped.mean()) if not clipped.isnull().all() else 0.0
                    sl.append({"time": ts, "rain_in": max(0.0, mean_val/25.4)})
                else:
                    sl.append({"time": ts, "rain_in": 0.0})
                pb.progress((i+1)/len(tr))
            
            st.session_state.processed_df = pd.DataFrame(sl).set_index("time")
            st.session_state.time_list = list(st.session_state.raster_cache.keys())
            gc.collect() # Force clear RAM

# --- 5. MAP RENDERING (THE "FIX") ---
map_placeholder = st.empty()

def render_map_frame(idx):
    # Centering logic
    center, zoom = [40.1, -74.5], 8
    if 'active_gdf' in st.session_state:
        b = st.session_state.active_gdf.total_bounds
        center = [(b[1]+b[3])/2, (b[0]+b[2])/2]
        zoom = 11

    m = leafmap.Map(center=center, zoom=zoom)
    if 'active_gdf' in st.session_state:
        m.add_gdf(st.session_state.active_gdf, layer_name="Site")
    
    if idx < len(st.session_state.time_list):
        t_key = st.session_state.time_list[idx]
        tif_path = st.session_state.raster_cache.get(t_key)
        if tif_path and os.path.exists(tif_path):
            m.add_raster(tif_path, colormap="jet", opacity=0.7, vmin=0.1, vmax=3.0)
    
    with map_placeholder:
        m.to_streamlit(key=f"map_{idx}") # Key prevents map duplication in RAM

# --- 6. DISPLAY ---
if st.session_state.time_list:
    c_p, c_s = st.columns([1, 6])
    if c_p.button("⏹️ Stop" if st.session_state.playing else "▶️ Play"):
        st.session_state.playing = not st.session_state.playing
        st.rerun()

    st.session_state.current_idx = c_s.select_slider(
        "Timeline", options=range(len(st.session_state.time_list)),
        value=st.session_state.current_idx,
        format_func=lambda x: st.session_state.time_list[x]
    )

    if st.session_state.playing:
        for i in range(st.session_state.current_idx, len(st.session_state.time_list)):
            st.session_state.current_idx = i
            render_map_frame(i)
            time.sleep(0.3)
            if not st.session_state.playing: break
        st.session_state.playing = False
        st.rerun()
    else:
        render_map_frame(st.session_state.current_idx)
else:
    # Show empty map if nothing processed
    render_map_frame(0)

if 'processed_df' in st.session_state:
    st.plotly_chart(px.bar(st.session_state.processed_df.reset_index(), x='time', y='rain_in', template="plotly_dark"), use_container_width=True)
