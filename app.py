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
from matplotlib.colors import ListedColormap

# --- 1. PAGE CONFIG & UI FIXES ---
st.set_page_config(layout="wide", page_title="CNR Radar Portal", page_icon="🛰️")

st.markdown("""
    <style>
        .block-container { padding: 1rem 2rem 0rem 2rem !important; }
        iframe, div[data-testid="stPydeckChart"] {
            height: 80vh !important;
            min-height: 750px !important;
        }
        section[data-testid="stSidebar"] { width: 400px !important; }
        header { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. NEXRAD COLORS ---
RADAR_COLORS = ['#76fffe', '#01a0fe', '#0001ef', '#01ef01', '#019001', '#ffff01', '#e7c001', '#ff9000', '#ff0101']
RADAR_CMAP = ListedColormap(RADAR_COLORS)

# --- 3. SESSION STATE ---
if 'processed_df' not in st.session_state: st.session_state.processed_df = None
if 'radar_cache' not in st.session_state: st.session_state.radar_cache = {}
if 'time_list' not in st.session_state: st.session_state.time_list = []
if 'active_gdf' not in st.session_state: st.session_state.active_gdf = None
if 'view_state' not in st.session_state: 
    st.session_state.view_state = pdk.ViewState(latitude=40.7, longitude=-74.0, zoom=9)

# --- 4. DATA ENGINE ---
def get_radar_image(dt_utc):
    ts_str = dt_utc.strftime("%Y%m%d-%H1500") 
    url = f"https://noaa-mrms-pds.s3.amazonaws.com/CONUS/RadarOnly_QPE_15M_00.00/{dt_utc.strftime('%Y%m%d')}/MRMS_RadarOnly_QPE_15M_00.00_{ts_str}.grib2.gz"
    
    tmp_grib = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False).name
    try:
        r = requests.get(url, stream=True, timeout=10)
        if r.status_code != 200: return None, 0, None
        with gzip.GzipFile(fileobj=r.raw) as gz, open(tmp_grib, "wb") as f:
            shutil.copyfileobj(gz, f)
            
        with xr.open_dataset(tmp_grib, engine="cfgrib") as ds:
            da = ds[list(ds.data_vars)[0]].load()
            da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180).sortby("longitude")
            subset = da.sel(latitude=slice(42.5, 38.5), longitude=slice( -76.5, -72.5))
            
            site_mean = 0.0
            if st.session_state.active_gdf is not None:
                clipped = subset.rio.write_crs("EPSG:4326").rio.clip(st.session_state.active_gdf.geometry)
                site_mean = float(clipped.mean()) if not clipped.isnull().all() else 0.0

            data = subset.values
            data[data < 0.1] = np.nan 
            
            # Use a static name to avoid filling up disk space
            img_filename = f"radar_{dt_utc.strftime('%H%M')}.png"
            plt.imsave(img_filename, data, cmap=RADAR_CMAP, vmin=0.1, vmax=25.0)
            
            bounds = [float(subset.longitude.min()), float(subset.latitude.min()), 
                      float(subset.longitude.max()), float(subset.latitude.max())]
            return img_filename, max(0.0, site_mean/25.4), bounds
    except: return None, 0, None
    finally:
        if os.path.exists(tmp_grib): os.remove(tmp_grib)

# --- 5. SIDEBAR (STAYS LOCKED) ---
with st.sidebar:
    st.title("🛰️ CNR GIS Portal")
    tz_mode = st.radio("Timezone", ["Local (EST/EDT)", "UTC"])
    yesterday = datetime.now() - timedelta(days=1)
    s_date = st.date_input("Start Date", value=yesterday.date())
    e_date = st.date_input("End Date", value=yesterday.date())
    
    hours = [f"{h:02d}:00" for h in range(24)]
    c1, c2 = st.columns(2)
    s_time = c1.selectbox("Start Hour", hours, index=18)
    e_time = c2.selectbox("End Hour", hours, index=22)
    
    up_zip = st.file_uploader("Upload Watershed ZIP", type="zip")
    if up_zip:
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(up_zip, 'r') as z: z.extractall(td)
            shps = list(Path(td).rglob("*.shp"))
            if shps:
                st.session_state.active_gdf = gpd.read_file(shps[0]).to_crs("EPSG:4326")
                b = st.session_state.active_gdf.total_bounds
                st.session_state.view_state = pdk.ViewState(latitude=(b[1]+b[3])/2, longitude=(b[0]+b[2])/2, zoom=11)

    if st.button("🚀 Process Data", use_container_width=True):
        if st.session_state.active_gdf is not None:
            s_dt = datetime.combine(s_date, datetime.strptime(s_time, "%H:%M").time())
            e_dt = datetime.combine(e_date, datetime.strptime(e_time, "%H:%M").time())
            tr = pd.date_range(s_dt, e_dt, freq='1H')
            
            cache, stats = {}, []
            pb = st.progress(0)
            for i, ts in enumerate(tr):
                ts_utc = ts if tz_mode == "UTC" else ts + timedelta(hours=5)
                path, val, bnds = get_radar_image(ts_utc)
                if path:
                    cache[ts.strftime("%Y-%m-%d %H:%M")] = {"path": path, "bounds": bnds}
                    stats.append({"time": ts, "rain_in": val})
                pb.progress((i+1)/len(tr))
            
            st.session_state.radar_cache = cache
            st.session_state.time_list = list(cache.keys())
            st.session_state.processed_df = pd.DataFrame(stats) if stats else None
        else:
            st.error("Upload ZIP first.")

# --- 6. MAP ---
st.subheader("Radar GIS Viewer")
if st.session_state.time_list:
    t_idx = st.select_slider("Time Selection", options=range(len(st.session_state.time_list)),
                             format_func=lambda x: st.session_state.time_list[x])
    
    current_data = st.session_state.radar_cache[st.session_state.time_list[t_idx]]
    layers = [
        pdk.Layer("BitmapLayer", image=current_data["path"], bounds=current_data["bounds"], opacity=0.75),
        pdk.Layer("GeoJsonLayer", st.session_state.active_gdf.__geo_interface__, 
                  stroked=True, filled=False, get_line_color=[255, 255, 255], line_width_min_pixels=3)
    ]
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=st.session_state.view_state, map_style="dark"))
else:
    # Show an empty map with the watershed boundary so the UI doesn't look broken
    layers = []
    if st.session_state.active_gdf is not None:
        layers.append(pdk.Layer("GeoJsonLayer", st.session_state.active_gdf.__geo_interface__, 
                                stroked=True, filled=False, get_line_color=[255, 255, 255], line_width_min_pixels=3))
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=st.session_state.view_state, map_style="dark"))

# --- 7. CHART (THE FIX) ---
if st.session_state.processed_df is not None and not st.session_state.processed_df.empty:
    import plotly.express as px
    # Ensure columns exist before plotting
    if 'time' in st.session_state.processed_df.columns and 'rain_in' in st.session_state.processed_df.columns:
        st.plotly_chart(px.bar(st.session_state.processed_df, x='time', y='rain_in', 
                               template="plotly_dark", color_discrete_sequence=['#00CCFF']), 
                               use_container_width=True)
else:
    st.info("Awaiting radar data... Use the sidebar to process a time range.")
