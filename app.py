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
if 'shp_name' not in st.session_state: st.session_state.shp_name = "output"
if 'active_gdf' not in st.session_state: st.session_state.active_gdf = None

# --- 3. HELPERS ---
def get_tz_offset(dt):
    edt_s = datetime(dt.year, 3, 8 + (6 - datetime(dt.year, 3, 1).weekday()) % 7)
    edt_e = datetime(dt.year, 11, 1 + (6 - datetime(dt.year, 11, 1).weekday()) % 7)
    return 4 if edt_s <= dt < edt_e else 5

# Set Global Constants
now = datetime.now()
yesterday = now - timedelta(days=1)
max_allowed_utc = datetime.utcnow() - timedelta(hours=1)

def download_mrms(dt_utc):
    # Automatically adjust to the 15-minute mark of the selected hour
    ts = dt_utc.strftime("%Y%m%d-%H1500") 
    url = f"https://noaa-mrms-pds.s3.amazonaws.com/CONUS/RadarOnly_QPE_15M_00.00/{dt_utc.strftime('%Y%m%d')}/MRMS_RadarOnly_QPE_15M_00.00_{ts}.grib2.gz"
    
    tmp = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False).name
    try:
        r = requests.get(url, stream=True, timeout=10)
        if r.status_code == 200:
            with gzip.GzipFile(fileobj=r.raw) as gz, open(tmp, "wb") as f:
                shutil.copyfileobj(gz, f)
            with xr.open_dataset(tmp, engine="cfgrib", backend_kwargs={"indexpath": ""}) as ds:
                da = ds[list(ds.data_vars)[0]].load()
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
    tz_mode = st.radio("Timezone", ["Local (EST/EDT)", "UTC"], index=0)
    
    # Defaulting both dates to yesterday
    start_d = st.date_input("Start Date", value=yesterday.date(), max_value=yesterday.date())
    end_d = st.date_input("End Date", value=yesterday.date(), max_value=yesterday.date())
    
    # Hourly selection only
    hours = [f"{h:02d}:00" for h in range(24)]
    
    c1, c2 = st.columns(2)
    start_t = c1.selectbox("Start Hour", hours, index=0)
    end_t = c2.selectbox("End Hour", hours, index=23)
    
    up_zip = st.file_uploader("Upload ZIP Shapefile", type="zip")
    if up_zip:
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(up_zip, 'r') as z: z.extractall(td)
            shps = list(Path(td).rglob("*.shp"))
            if shps:
                st.session_state.shp_name = shps[0].stem
                st.session_state.active_gdf = gpd.read_file(shps[0]).to_crs("EPSG:4326")

    if st.button("🚀 Process & Export", use_container_width=True):
        if st.session_state.active_gdf is not None:
            with st.spinner("Processing Hourly Samples..."):
                s_dt = datetime.combine(start_d, datetime.strptime(start_t, "%H:%M").time())
                e_dt = datetime.combine(end_d, datetime.strptime(end_t, "%H:%M").time())
                tr = pd.date_range(s_dt, e_dt, freq='1H')
                
                rc, sl = {}, []
                pb = st.progress(0)
                for i, ts in enumerate(tr):
                    ts_utc = ts if tz_mode == "UTC" else ts + timedelta(hours=get_tz_offset(ts))
                    da = download_mrms(ts_utc)
                    if da is not None:
                        # Visualization
                        map_da = da.where(da > 0.1, np.nan)
                        tf = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
                        map_da.rio.to_raster(tf)
                        rc[ts.strftime("%Y-%m-%d %H:%M")] = tf
                        
                        # Stats
                        clipped = da.rio.clip(st.session_state.active_gdf.geometry, st.session_state.active_gdf.crs, all_touched=True)
                        mean_val = float(clipped.mean()) if not clipped.isnull().all() else 0.0
                        sl.append({"time": ts, "rain_in": max(0.0, mean_val/25.4)})
                    else:
                        sl.append({"time": ts, "rain_in": 0.0})
                    pb.progress((i+1)/len(tr))
                
                st.session_state.processed_df = pd.DataFrame(sl).fillna(0.0).set_index("time")
                st.session_state.raster_cache = rc
                st.session_state.time_list = list(rc.keys())
        else: st.warning("Please upload a ZIP shapefile first.")

# --- 5. MAIN CONTENT ---
map_placeholder = st.empty()

def render_map(idx=None):
    m = leafmap.Map(center=[40.1, -74.5], zoom=8)
    if st.session_state.active_gdf is not None:
        m.add_gdf(st.session_state.active_gdf, layer_name="Site Boundary")
    if idx is not None and st.session_state.time_list:
        ts = st.session_state.time_list[idx]
        if ts in st.session_state.raster_cache:
            m.add_raster(st.session_state.raster_cache[ts], colormap="jet", opacity=0.7, vmin=0.1, vmax=2.5)
    with map_placeholder:
        m.to_streamlit()

if not st.session_state.time_list:
    render_map()
else:
    c_p, c_s = st.columns([1, 6])
    if st.session_state.playing:
        if c_p.button("⏹️ Stop"):
            st.session_state.playing = False
            st.rerun()
    else:
        if c_p.button("▶️ Play"):
            st.session_state.playing = True
            st.rerun()

    st.session_state.current_idx = c_s.select_slider(
        "Time Selection",
        options=range(len(st.session_state.time_list)),
        value=st.session_state.current_idx,
        format_func=lambda x: st.session_state.time_list[x],
        label_visibility="collapsed"
    )

    if st.session_state.playing:
        for i in range(st.session_state.current_idx, len(st.session_state.time_list)):
            st.session_state.current_idx = i
            render_map(i)
            time.sleep(0.5)
            if not st.session_state.playing: break
        st.session_state.playing = False
        st.rerun()
    else:
        render_map(st.session_state.current_idx)

# --- 6. CHART ---
if st.session_state.processed_df is not None:
    st.plotly_chart(px.bar(st.session_state.processed_df.reset_index(), x="time", y="rain_in", 
                           title=f"Rainfall Profile: {st.session_state.shp_name}", 
                           template="plotly_dark"), use_container_width=True)
