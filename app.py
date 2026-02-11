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

# --- 1. PAGE CONFIG & AGGRESSIVE FULL-SCREEN CSS ---
st.set_page_config(layout="wide", page_title="CNR Radar Portal")

st.markdown("""
    <style>
        /* Force the app to use 100% of the browser width */
        .block-container { 
            padding: 0rem !important; 
            max-width: 100vw !important; 
        }
        
        /* Hide UI clutter */
        header, footer { display: none !important; }
        
        /* Force the Map to fill the screen */
        div[data-testid="stPydeckChart"], iframe {
            height: 100vh !important;
            width: 100vw !important;
            position: fixed !important;
            top: 0;
            left: 0;
        }

        /* Float the slider at the bottom so it doesn't push the map up */
        .stSlider {
            position: fixed;
            bottom: 30px;
            left: 320px; /* Offset for sidebar */
            right: 50px;
            z-index: 1000;
            background: rgba(25, 25, 25, 0.8);
            padding: 10px 25px;
            border-radius: 15px;
            border: 1px solid #444;
        }

        /* Style the Sidebar Button to be a large, wide rectangle */
        div.stButton > button {
            width: 100% !important;
            height: 60px !important;
            font-size: 18px !important;
            font-weight: bold !important;
            margin-top: 10px;
        }

        /* Make the Dialog (Graph) much larger */
        div[data-testid="stDialog"] div[role="dialog"] {
            width: 80vw !important;
            max-width: 1200px !important;
        }
    </style>
    """, unsafe_allow_html=True)

# --- 2. COLORS & PERSISTENT STATE ---
RADAR_COLORS = ['#76fffe', '#01a0fe', '#0001ef', '#01ef01', '#019001', '#ffff01', '#e7c001', '#ff9000', '#ff0101']
RADAR_CMAP = ListedColormap(RADAR_COLORS)

if 'processed_df' not in st.session_state: st.session_state.processed_df = None
if 'radar_cache' not in st.session_state: st.session_state.radar_cache = {}
if 'time_list' not in st.session_state: st.session_state.time_list = []
if 'active_gdf' not in st.session_state: st.session_state.active_gdf = None
if 'map_view' not in st.session_state: 
    st.session_state.map_view = pdk.ViewState(latitude=40.7, longitude=-74.0, zoom=9)

# --- 3. DATA ENGINE (15-min) ---
def get_radar_image(dt_utc):
    ts_str = dt_utc.strftime("%Y%m%d-%H%M00") 
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
            subset = da.sel(latitude=slice(42.5, 38.5), longitude=slice(-76.5, -72.5))
            
            site_mean = 0.0
            if st.session_state.active_gdf is not None:
                clipped = subset.rio.write_crs("EPSG:4326").rio.clip(st.session_state.active_gdf.geometry)
                site_mean = float(clipped.mean()) if not clipped.isnull().all() else 0.0

            data = subset.values
            data[data < 0.1] = np.nan 
            
            img_filename = f"radar_{dt_utc.strftime('%H%M')}.png"
            plt.imsave(img_filename, data, cmap=RADAR_CMAP, vmin=0.1, vmax=15.0)
            
            bounds = [float(subset.longitude.min()), float(subset.latitude.min()), 
                      float(subset.longitude.max()), float(subset.latitude.max())]
            return img_filename, max(0.0, site_mean/25.4), bounds
    except: return None, 0, None
    finally:
        if os.path.exists(tmp_grib): os.remove(tmp_grib)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("CNR GIS Portal")
    tz_mode = st.radio("Timezone", ["Local (EST/EDT)", "UTC"])
    s_date = st.date_input("Start Date", value=datetime.now().date())
    e_date = st.date_input("End Date", value=datetime.now().date())
    
    hours = [f"{h:02d}:00" for h in range(24)]
    c1, c2 = st.columns(2)
    s_time = c1.selectbox("Start Hour", hours, index=19)
    e_time = c2.selectbox("End Hour", hours, index=21)
    
    up_zip = st.file_uploader("Upload Watershed ZIP", type="zip")
    if up_zip:
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(up_zip, 'r') as z: z.extractall(td)
            shps = list(Path(td).rglob("*.shp"))
            if shps:
                st.session_state.active_gdf = gpd.read_file(shps[0]).to_crs("EPSG:4326")
                b = st.session_state.active_gdf.total_bounds
                st.session_state.map_view = pdk.ViewState(
                    latitude=(b[1]+b[3])/2, longitude=(b[0]+b[2])/2, zoom=11
                )

    if st.button("PROCESS RADAR DATA"):
        if st.session_state.active_gdf is not None:
            s_dt = datetime.combine(s_date, datetime.strptime(s_time, "%H:%M").time())
            e_dt = datetime.combine(e_date, datetime.strptime(e_time, "%H:%M").time())
            tr = pd.date_range(s_dt, e_dt, freq='15min')
            
            cache, stats = {}, []
            pb = st.progress(0)
            for i, ts in enumerate(tr):
                ts_utc = ts if tz_mode == "UTC" else ts + timedelta(hours=5)
                path, val, bnds = get_radar_image(ts_utc)
                if path:
                    cache[ts.strftime("%H:%M")] = {"path": path, "bounds": bnds}
                    stats.append({"time": ts, "rain_in": val})
                pb.progress((i+1)/len(tr))
            
            st.session_state.radar_cache = cache
            st.session_state.time_list = list(cache.keys())
            st.session_state.processed_df = pd.DataFrame(stats)

    if st.session_state.processed_df is not None:
        st.write("---")
        if st.button("SHOW PLOT", type="primary"):
            import plotly.express as px
            @st.dialog("Rainfall Statistics", width="large")
            def modal():
                fig = px.bar(st.session_state.processed_df, x='time', y='rain_in', template="plotly_dark")
                fig.update_layout(bargap=0, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
            modal()

# --- 5. MAIN MAP ---
if st.session_state.time_list:
    # Floating Slider (Positioned via CSS)
    t_idx = st.select_slider("Select Time", options=range(len(st.session_state.time_list)),
                             format_func=lambda x: st.session_state.time_list[x], label_visibility="collapsed")
    
    current_data = st.session_state.radar_cache[st.session_state.time_list[t_idx]]
    layers = [
        pdk.Layer("BitmapLayer", image=current_data["path"], bounds=current_data["bounds"], opacity=0.7),
        pdk.Layer("GeoJsonLayer", st.session_state.active_gdf.__geo_interface__, 
                  stroked=True, filled=False, get_line_color=[255, 255, 255], line_width_min_pixels=3)
    ]
else:
    layers = []
    if st.session_state.active_gdf is not None:
        layers.append(pdk.Layer("GeoJsonLayer", st.session_state.active_gdf.__geo_interface__, 
                                stroked=True, filled=False, get_line_color=[255, 255, 255], line_width_min_pixels=3))

# Using a constant 'key' in st.pydeck_chart is the secret to stopping the flickering/resetting
st.pydeck_chart(pdk.Deck(
    layers=layers, 
    initial_view_state=st.session_state.map_view, 
    map_style="dark",
), key="radar_map")
