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
from pathlib import Path

# --- 1. PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="CNR Radar Portal", page_icon="🛰️")

# --- 2. SESSION STATE ---
if 'processed_df' not in st.session_state: st.session_state.processed_df = None
if 'map_data_cache' not in st.session_state: st.session_state.map_data_cache = {}
if 'time_list' not in st.session_state: st.session_state.time_list = []
if 'active_gdf' not in st.session_state: st.session_state.active_gdf = None
if 'current_idx' not in st.session_state: st.session_state.current_idx = 0

# --- 3. CORE DOWNLOADER (Lightweight Points) ---
def get_mrms_points(dt_utc):
    """Downloads MRMS and converts to lightweight GPS points for the map."""
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
            # PROJECTION FIX: Force -180 to 180 longitude
            da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180).sortby("longitude")
            da = da.rio.write_crs("EPSG:4326")

            # 1. Stats Calculation (Clipped to Site)
            site_mean = 0.0
            if st.session_state.active_gdf is not None:
                clipped = da.rio.clip(st.session_state.active_gdf.geometry, "EPSG:4326")
                site_mean = float(clipped.mean()) if not clipped.isnull().all() else 0.0

            # 2. Map Optimization: Only keep points in the local region to save RAM
            # (Roughly Northeast US: Lat 38-42, Lon -76 to -72)
            subset = da.sel(latitude=slice(42, 38), longitude=slice(-76, -72))
            df = subset.to_dataframe(name='val').reset_index()
            df = df[df['val'] > 0.1] # Only keep rain points
            
            return df[['latitude', 'longitude', 'val']], max(0.0, site_mean/25.4)
    except: return None, 0.0
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

# --- 4. SIDEBAR (Restored Full Logic) ---
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
                # Offset for UTC download
                ts_utc = ts if tz_mode == "UTC" else ts + timedelta(hours=5) 
                pts_df, rain_val = get_mrms_points(ts_utc)
                
                t_str = ts.strftime("%Y-%m-%d %H:%M")
                if pts_df is not None:
                    map_cache[t_str] = pts_df
                stats_list.append({"time": ts, "rain_in": rain_val})
                pb.progress((i+1)/len(tr))
            
            st.session_state.processed_df = pd.DataFrame(stats_list).set_index("time")
            st.session_state.map_data_cache = map_cache
            st.session_state.time_list = list(map_cache.keys())
        else: st.warning("Please upload a ZIP shapefile first.")

# --- 5. MAIN CONTENT (Pydeck Map) ---
st.subheader("Interactive Radar & Site Monitoring")

if st.session_state.time_list:
    # Time Selector
    st.session_state.current_idx = st.select_slider(
        "Timeline Slider", 
        options=range(len(st.session_state.time_list)),
        format_func=lambda x: st.session_state.time_list[x]
    )
    
    current_time = st.session_state.time_list[st.session_state.current_idx]
    current_map_df = st.session_state.map_data_cache[current_time]

    # Boundary GeoJSON for Pydeck
    boundary_data = st.session_state.active_gdf.__geo_interface__

    # Layers
    radar_layer = pdk.Layer(
        "ScreenGridLayer", # Excellent for "filled" look with massive RAM efficiency
        current_map_df,
        get_position=["longitude", "latitude"],
        get_weight="val",
        cell_size_pixels=10,
        color_range=[[0,255,255,100], [0,0,255,150], [255,0,0,200]],
        pickable=True,
    )

    site_layer = pdk.Layer(
        "GeoJsonLayer",
        boundary_data,
        stroked=True,
        filled=False,
        get_line_color=[255, 255, 255],
        line_width_min_pixels=2,
    )

    # View State
    b = st.session_state.active_gdf.total_bounds
    view_state = pdk.ViewState(
        latitude=(b[1]+b[3])/2, longitude=(b[0]+b[2])/2, 
        zoom=10, pitch=0
    )

    st.pydeck_chart(pdk.Deck(
        layers=[radar_layer, site_layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/satellite-v9", # Better GIS context
    ))

# --- 6. CHART ---
if st.session_state.processed_df is not None:
    st.plotly_chart(px.bar(st.session_state.processed_df.reset_index(), 
                           x='time', y='rain_in', 
                           title="Hourly Rainfall (Inches)",
                           template="plotly_dark"), use_container_width=True)
