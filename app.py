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
import time
from datetime import datetime, timedelta
from pathlib import Path

# --- 1. PAGE CONFIG & RESPONSIVE CSS ---
st.set_page_config(layout="wide", page_title="CNR Radar Portal", page_icon="🛰️")

# Force the map to fill the vertical space of the browser
st.markdown("""
    <style>
        /* Remove extra padding around the app */
        .block-container { padding: 1rem !important; }
        
        /* Force Pydeck to occupy 75% of the screen height */
        [data-testid="stPydeckChart"] {
            height: 75vh !important;
            min-height: 500px;
        }
        
        /* Make the sidebar a bit cleaner */
        section[data-testid="stSidebar"] { width: 350px !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SESSION STATE ---
if 'processed_df' not in st.session_state: st.session_state.processed_df = None
if 'map_data_cache' not in st.session_state: st.session_state.map_data_cache = {}
if 'time_list' not in st.session_state: st.session_state.time_list = []
if 'active_gdf' not in st.session_state: st.session_state.active_gdf = None
if 'current_idx' not in st.session_state: st.session_state.current_idx = 0
if 'playing' not in st.session_state: st.session_state.playing = False

# --- 3. DATA PROCESSING ---
def get_mrms_points(dt_utc):
    ts_str = dt_utc.strftime("%Y%m%d-%H1500") 
    url = f"https://noaa-mrms-pds.s3.amazonaws.com/CONUS/RadarOnly_QPE_15M_00.00/{dt_utc.strftime('%Y%m%d')}/MRMS_RadarOnly_QPE_15M_00.00_{ts_str}.grib2.gz"
    
    tmp_path = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False).name
    try:
        r = requests.get(url, stream=True, timeout=10)
        if r.status_code != 200: return None, 0.0
        
        with gzip.GzipFile(fileobj=r.raw) as gz, open(tmp_path, "wb") as f:
            shutil.copyfileobj(gz, f)
            
        with xr.open_dataset(tmp_path, engine="cfgrib") as ds:
            da = ds[list(ds.data_vars)[0]].load()
            # Projection: 0-360 to -180-180
            da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180).sortby("longitude")
            da = da.rio.write_crs("EPSG:4326")

            site_mean = 0.0
            if st.session_state.active_gdf is not None:
                clipped = da.rio.clip(st.session_state.active_gdf.geometry, "EPSG:4326")
                site_mean = float(clipped.mean()) if not clipped.isnull().all() else 0.0

            # Local focus subset
            subset = da.sel(latitude=slice(43, 38), longitude=slice(-77, -71))
            df = subset.to_dataframe(name='val').reset_index()
            df = df[df['val'] > 0.1] 
            return df[['latitude', 'longitude', 'val']], max(0.0, site_mean/25.4)
    except: return None, 0.0
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("🛰️ CNR GIS Portal")
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
            s_dt = datetime.combine(start_d, datetime.strptime(start_t, "%H:%M").time())
            e_dt = datetime.combine(end_d, datetime.strptime(end_t, "%H:%M").time())
            tr = pd.date_range(s_dt, e_dt, freq='1H')
            map_cache, stats_list = {}, []
            pb = st.progress(0)
            for i, ts in enumerate(tr):
                ts_utc = ts if tz_mode == "UTC" else ts + timedelta(hours=5) 
                pts_df, rain_val = get_mrms_points(ts_utc)
                t_str = ts.strftime("%Y-%m-%d %H:%M")
                if pts_df is not None: map_cache[t_str] = pts_df
                stats_list.append({"time": ts, "rain_in": rain_val})
                pb.progress((i+1)/len(tr))
            st.session_state.processed_df = pd.DataFrame(stats_list)
            st.session_state.map_data_cache = map_cache
            st.session_state.time_list = list(map_cache.keys())
        else: st.warning("Upload Shapefile First")

# --- 5. MAIN CONTENT (MAP) ---
map_spot = st.empty()

def render_deck(idx=0):
    layers = []
    lat, lon, zoom = 40.7, -74.0, 8
    
    if st.session_state.active_gdf is not None:
        b = st.session_state.active_gdf.total_bounds
        lat, lon, zoom = (b[1]+b[3])/2, (b[0]+b[2])/2, 11
        layers.append(pdk.Layer("GeoJsonLayer", st.session_state.active_gdf.__geo_interface__, 
                                stroked=True, filled=True, get_fill_color=[255, 255, 255, 40],
                                get_line_color=[255, 255, 255], line_width_min_pixels=2))

    if st.session_state.time_list and idx < len(st.session_state.time_list):
        df = st.session_state.map_data_cache[st.session_state.time_list[idx]]
        if not df.empty:
            layers.append(pdk.Layer("ScreenGridLayer", df, get_position=["longitude", "latitude"], 
                                    get_weight="val", cell_size_pixels=12, 
                                    color_range=[[0,255,255,100],[0,0,255,180],[255,0,0,230]]))

    with map_spot:
        # Note: height=None here allows the CSS (vh) to control the display
        st.pydeck_chart(pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom),
            map_style="light" 
        ))

# Controls
if st.session_state.time_list:
    cp, cs = st.columns([1, 5])
    if cp.button("⏹️ Stop" if st.session_state.playing else "▶️ Play"):
        st.session_state.playing = not st.session_state.playing
        st.rerun()
    
    st.session_state.current_idx = cs.select_slider("Time Selection", 
                                                   options=range(len(st.session_state.time_list)), 
                                                   format_func=lambda x: st.session_state.time_list[x])
    
    if st.session_state.playing:
        for i in range(st.session_state.current_idx, len(st.session_state.time_list)):
            st.session_state.current_idx = i
            render_deck(i)
            time.sleep(0.4)
            if not st.session_state.playing: break
        st.session_state.playing = False
        st.rerun()
    else:
        render_deck(st.session_state.current_idx)
else:
    render_deck()

# --- 6. CHART ---
if st.session_state.processed_df is not None and not st.session_state.processed_df.empty:
    import plotly.express as px
    st.plotly_chart(px.bar(st.session_state.processed_df, x='time', y='rain_in', 
                           template="plotly_dark", color_discrete_sequence=['#00CCFF']), 
                           use_container_width=True)
