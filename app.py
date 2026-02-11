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
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# --- 1. PAGE CONFIG & FULL-SCREEN HEIGHT ---
st.set_page_config(layout="wide", page_title="CNR Radar Portal")

st.markdown("""
    <style>
        .block-container { padding: 1rem 2rem 0rem 2rem !important; }
        /* This selector targets the pydeck container and forces vertical expansion */
        .stPydeckChart, div[data-testid="stPydeckChart"], iframe {
            height: 85vh !important;
            width: 100% !important;
        }
        header { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. NEXRAD COLOR PALETTE (The Professional Standard) ---
# Precise HEX codes for weather radar
COLORS = ['#76fffe', '#01a0fe', '#0001ef', '#01ef01', '#019001', '#ffff01', '#e7c001', '#ff9000', '#ff0101']
LEVELS = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 50.0]
CMAP = ListedColormap(COLORS)
NORM = BoundaryNorm(LEVELS, ncolors=len(COLORS))

# --- 3. SESSION STATE ---
if 'processed_df' not in st.session_state: st.session_state.processed_df = None
if 'radar_images' not in st.session_state: st.session_state.radar_images = {}
if 'time_list' not in st.session_state: st.session_state.time_list = []
if 'active_gdf' not in st.session_state: st.session_state.active_gdf = None
if 'view_state' not in st.session_state: 
    st.session_state.view_state = pdk.ViewState(latitude=40.7, longitude=-74.0, zoom=9)

# --- 4. DATA PROCESSING (GRIB -> PNG BITMAP) ---
def get_radar_bitmap(dt_utc):
    ts_str = dt_utc.strftime("%Y%m%d-%H1500") 
    url = f"https://noaa-mrms-pds.s3.amazonaws.com/CONUS/RadarOnly_QPE_15M_00.00/{dt_utc.strftime('%Y%m%d')}/MRMS_RadarOnly_QPE_15M_00.00_{ts_str}.grib2.gz"
    
    with tempfile.NamedTemporaryFile(suffix=".grib2") as tmp:
        r = requests.get(url, stream=True, timeout=10)
        if r.status_code != 200: return None, 0, None
        with gzip.GzipFile(fileobj=r.raw) as gz:
            shutil.copyfileobj(gz, tmp)
            
        with xr.open_dataset(tmp.name, engine="cfgrib") as ds:
            da = ds[list(ds.data_vars)[0]].load()
            da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180).sortby("longitude")
            
            # Regional crop to keep the image sharp
            subset = da.sel(latitude=slice(42.5, 38.5), longitude=slice(-76.5, -72.5))
            
            # Stats for chart
            site_mean = 0.0
            if st.session_state.active_gdf is not None:
                clipped = subset.rio.write_crs("EPSG:4326").rio.clip(st.session_state.active_gdf.geometry)
                site_mean = float(clipped.mean()) if not clipped.isnull().all() else 0.0

            # Generate Bitmap Image
            img_path = f"radar_{dt_utc.timestamp()}.png"
            data = subset.values
            data[data < 0.1] = np.nan # Transparency for no rain
            
            plt.imsave(img_path, data, cmap=CMAP, vmin=0.1, vmax=50.0)
            
            # Return image, rain value, and the precise bounding box for the map
            bounds = [float(subset.longitude.min()), float(subset.latitude.min()), 
                      float(subset.longitude.max()), float(subset.latitude.max())]
            return img_path, max(0.0, site_mean/25.4), bounds

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("🛰️ CNR GIS Portal")
    up_zip = st.file_uploader("Upload Watershed ZIP", type="zip")
    if up_zip:
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(up_zip, 'r') as z: z.extractall(td)
            shps = list(Path(td).rglob("*.shp"))
            if shps:
                st.session_state.active_gdf = gpd.read_file(shps[0]).to_crs("EPSG:4326")
                b = st.session_state.active_gdf.total_bounds
                st.session_state.view_state = pdk.ViewState(latitude=(b[1]+b[3])/2, longitude=(b[0]+b[2])/2, zoom=11)

    if st.button("🚀 Process Radar Data", use_container_width=True):
        if st.session_state.active_gdf is not None:
            # Simple 3-hour window around current time for example
            tr = pd.date_range(datetime.now()-timedelta(hours=4), datetime.now(), freq='1H')
            imgs, stats = {}, []
            pb = st.progress(0)
            for i, ts in enumerate(tr):
                img, val, bounds = get_radar_bitmap(ts + timedelta(hours=5))
                if img:
                    imgs[ts.strftime("%H:%M")] = {"path": img, "bounds": bounds}
                stats.append({"time": ts, "rain_in": val})
                pb.progress((i+1)/len(tr))
            st.session_state.radar_images = imgs
            st.session_state.time_list = list(imgs.keys())
            st.session_state.processed_df = pd.DataFrame(stats)

# --- 6. MAIN CONTENT ---
if st.session_state.time_list:
    t_key = st.select_slider("Select Time", options=st.session_state.time_list)
    radar_info = st.session_state.radar_images[t_key]

    layers = [
        pdk.Layer(
            "BitmapLayer",
            image=radar_info["path"],
            bounds=radar_info["bounds"],
            opacity=0.7
        )
    ]
    
    if st.session_state.active_gdf is not None:
        layers.append(pdk.Layer("GeoJsonLayer", st.session_state.active_gdf.__geo_interface__, 
                                stroked=True, filled=False, get_line_color=[255, 255, 255], line_width_min_pixels=3))

    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=st.session_state.view_state, map_style="dark"))
else:
    st.info("Upload a shapefile and process data to see the radar.")

# --- 7. RAINFALL CHART ---
if st.session_state.processed_df is not None:
    import plotly.express as px
    st.plotly_chart(px.bar(st.session_state.processed_df, x='time', y='rain_in', template="plotly_dark"))
