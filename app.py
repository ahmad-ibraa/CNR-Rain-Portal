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

# --- 1. PAGE CONFIG & FULL-HEIGHT CSS ---
st.set_page_config(layout="wide", page_title="CNR Radar Portal", page_icon="🛰️")

st.markdown("""
    <style>
        .block-container { padding: 1rem 2rem 0rem 2rem !important; }
        div[data-testid="stPydeckChart"], iframe {
            height: 80vh !important;
            min-height: 700px !important;
        }
        header { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SESSION STATE ---
if 'processed_df' not in st.session_state: st.session_state.processed_df = None
if 'map_data_cache' not in st.session_state: st.session_state.map_data_cache = {}
if 'time_list' not in st.session_state: st.session_state.time_list = []
if 'active_gdf' not in st.session_state: st.session_state.active_gdf = None
if 'view_state' not in st.session_state: st.session_state.view_state = pdk.ViewState(latitude=40.7, longitude=-74.0, zoom=8)

# --- 3. NEXRAD COLOR PALETTE ---
# Standard rainfall colors: Light Blue -> Dark Blue -> Green -> Yellow -> Red
RAIN_COLORS = [
    [0, 191, 255, 140],   # DeepSkyBlue
    [0, 0, 255, 160],     # Blue
    [0, 255, 0, 180],     # Green
    [255, 255, 0, 200],   # Yellow
    [255, 165, 0, 220],   # Orange
    [255, 0, 0, 240]      # Red
]

# --- 4. DATA PROCESSING ---
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
            da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180).sortby("longitude")
            da = da.rio.write_crs("EPSG:4326")

            site_mean = 0.0
            if st.session_state.active_gdf is not None:
                clipped = da.rio.clip(st.session_state.active_gdf.geometry, "EPSG:4326")
                site_mean = float(clipped.mean()) if not clipped.isnull().all() else 0.0

            subset = da.sel(latitude=slice(43, 38), longitude=slice(-77, -71))
            df = subset.to_dataframe(name='val').reset_index()
            df = df[df['val'] > 0.1] 
            return df[['latitude', 'longitude', 'val']], max(0.0, site_mean/25.4)
    except: return None, 0.0
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

# --- 5. SIDEBAR ---
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
                b = st.session_state.active_gdf.total_bounds
                # Set the initial view once when file is uploaded
                st.session_state.view_state = pdk.ViewState(
                    latitude=(b[1]+b[3])/2, longitude=(b[0]+b[2])/2, zoom=11
                )

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
                if pts_df is not None: map_cache[ts.strftime("%Y-%m-%d %H:%M")] = pts_df
                stats_list.append({"time": ts, "rain_in": rain_val})
                pb.progress((i+1)/len(tr))
            st.session_state.processed_df = pd.DataFrame(stats_list)
            st.session_state.map_data_cache = map_cache
            st.session_state.time_list = list(map_cache.keys())

# --- 6. MAIN CONTENT ---
st.subheader("Radar GIS Viewer")

if st.session_state.time_list:
    current_idx = st.select_slider("Timeline", options=range(len(st.session_state.time_list)), 
                                   format_func=lambda x: st.session_state.time_list[x])
    
    df = st.session_state.map_data_cache[st.session_state.time_list[current_idx]]
    
    layers = [
        # GridLayer uses real-world meters, so it scales with zoom
        pdk.Layer(
            "GridLayer",
            df,
            get_position=["longitude", "latitude"],
            cell_size=1000,  # 1km resolution matches MRMS
            extruded=False,
            elevation_scale=0,
            get_fill_color="val", # We let the color_range handle the scale
            color_range=RAIN_COLORS,
            pickable=True,
        )
    ]
    
    if st.session_state.active_gdf is not None:
        layers.append(pdk.Layer("GeoJsonLayer", st.session_state.active_gdf.__geo_interface__, 
                                stroked=True, filled=True, get_fill_color=[255, 255, 255, 20],
                                get_line_color=[255, 255, 255], line_width_min_pixels=3))

    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=st.session_state.view_state,
        map_style="light"
    ))

# --- 7. CHART ---
if st.session_state.processed_df is not None and not st.session_state.processed_df.empty:
    import plotly.express as px
    st.plotly_chart(px.bar(st.session_state.processed_df, x='time', y='rain_in', 
                           template="plotly_dark", color_discrete_sequence=['#00CCFF']), 
                           use_container_width=True)
