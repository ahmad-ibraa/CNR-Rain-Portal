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
import gc
from pathlib import Path
from datetime import datetime, timedelta
import leafmap.foliumap as leafmap
import plotly.express as px

# --- 1. PAGE CONFIG & FULL-HEIGHT CSS ---
st.set_page_config(layout="wide", page_title="CNR Radar Portal", page_icon="🛰️")

# This CSS forces the map container to stay large and prevents shrinking
st.markdown("""
    <style>
        .block-container { padding: 1rem !important; }
        /* Target the streamlit iframe specifically to maintain height */
        iframe { 
            height: 80vh !important; 
            min-height: 600px !important; 
            width: 100% !important; 
            border-radius: 10px;
        }
        /* Style the play button to be prominent */
        .stButton button { height: 3em; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SESSION STATE ---
if 'processed_df' not in st.session_state: st.session_state.processed_df = None
if 'raster_cache' not in st.session_state: st.session_state.raster_cache = {}
if 'time_list' not in st.session_state: st.session_state.time_list = []
if 'playing' not in st.session_state: st.session_state.playing = False
if 'current_idx' not in st.session_state: st.session_state.current_idx = 0
if 'active_gdf' not in st.session_state: st.session_state.active_gdf = None

# --- 3. HELPERS & DATA CACHING ---
@st.cache_data(show_spinner=False, max_entries=24)
def get_mrms_data(ts_utc):
    ts_str = ts_utc.strftime("%Y%m%d-%H1500") 
    url = f"https://noaa-mrms-pds.s3.amazonaws.com/CONUS/RadarOnly_QPE_15M_00.00/{ts_utc.strftime('%Y%m%d')}/MRMS_RadarOnly_QPE_15M_00.00_{ts_str}.grib2.gz"
    
    tmp_path = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False).name
    try:
        r = requests.get(url, stream=True, timeout=10)
        if r.status_code == 200:
            with gzip.GzipFile(fileobj=r.raw) as gz, open(tmp_path, "wb") as f:
                shutil.copyfileobj(gz, f)
            with xr.open_dataset(tmp_path, engine="cfgrib") as ds:
                da = ds[list(ds.data_vars)[0]].load()
                da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180).sortby("longitude")
                da = da.rio.write_crs("EPSG:4326")
                return da
    except: return None
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)
    return None

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
        if st.session_state.active_gdf is not None:
            # Cleanup previous files
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
                    # Save Raster for Map
                    map_da = da.where(da > 0.1, np.nan)
                    tf = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
                    map_da.rio.to_raster(tf)
                    st.session_state.raster_cache[ts.strftime("%Y-%m-%d %H:%M")] = tf
                    
                    # Stats calculation
                    clipped = da.rio.clip(st.session_state.active_gdf.geometry, "EPSG:4326")
                    mean_val = float(clipped.mean()) if not clipped.isnull().all() else 0.0
                    sl.append({"time": ts, "rain_in": max(0.0, mean_val/25.4)})
                else:
                    sl.append({"time": ts, "rain_in": 0.0})
                pb.progress((i+1)/len(tr))
            
            st.session_state.processed_df = pd.DataFrame(sl).set_index("time")
            st.session_state.time_list = list(st.session_state.raster_cache.keys())
            st.session_state.current_idx = 0
            gc.collect()

# --- 5. MAIN CONTENT (MAP) ---
map_placeholder = st.empty()

def render_map_frame(idx):
    # Determine center/zoom
    center, zoom = [40.1, -74.5], 8
    if st.session_state.active_gdf is not None:
        b = st.session_state.active_gdf.total_bounds
        center = [(b[1]+b[3])/2, (b[0]+b[2])/2]
        zoom = 12 # Tighter zoom for local sites

    m = leafmap.Map(center=center, zoom=zoom, draw_control=False, measure_control=False)
    
    if st.session_state.active_gdf is not None:
        m.add_gdf(st.session_state.active_gdf, layer_name="Watershed Boundary")
    
    if st.session_state.time_list and idx < len(st.session_state.time_list):
        t_key = st.session_state.time_list[idx]
        tif_path = st.session_state.raster_cache.get(t_key)
        if tif_path and os.path.exists(tif_path):
            # High visibility radar settings
            m.add_raster(tif_path, colormap="jet", opacity=0.75, vmin=0.1, vmax=3.0)
    
    with map_placeholder:
        # Key forces Streamlit to respect the frame and height
        m.to_streamlit(key=f"map_frame_{idx}", height=700)

# --- 6. TIMELINE CONTROLS ---
if st.session_state.time_list:
    c_p, c_s = st.columns([1, 6])
    
    btn_label = "⏹️ Stop" if st.session_state.playing else "▶️ Play Animation"
    if c_p.button(btn_label):
        st.session_state.playing = not st.session_state.playing
        st.rerun()

    st.session_state.current_idx = c_s.select_slider(
        "Timeline Slider", 
        options=range(len(st.session_state.time_list)),
        value=st.session_state.current_idx,
        format_func=lambda x: st.session_state.time_list[x],
        label_visibility="collapsed"
    )

    if st.session_state.playing:
        for i in range(st.session_state.current_idx, len(st.session_state.time_list)):
            st.session_state.current_idx = i
            render_map_frame(i)
            time.sleep(0.4)
            if not st.session_state.playing: break
        st.session_state.playing = False
        st.rerun()
    else:
        render_map_frame(st.session_state.current_idx)
else:
    # Always render a map even if no data is processed
    render_map_frame(0)

# --- 7. CHART ---
if st.session_state.processed_df is not None:
    st.plotly_chart(px.bar(st.session_state.processed_df.reset_index(), 
                           x='time', y='rain_in', 
                           title="Watershed Average Rainfall (Inches)",
                           template="plotly_dark",
                           color_discrete_sequence=['#00d4ff']), use_container_width=True)
