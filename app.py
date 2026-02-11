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
import rioxarray 

# -----------------------------
# 1. PAGE CONFIG & FULLSCREEN CSS
# -----------------------------
st.set_page_config(layout="wide", page_title="CNR Radar Portal")

st.markdown("""
<style>
    /* 1. Force the App to be a true fullscreen canvas */
    [data-testid="stAppViewContainer"] {
        background-color: #0e1117;
    }

    /* 2. Fix the Map to the background - This kills the 'half screen' bug */
    .stPydeckChart {
        position: fixed !important;
        top: 0;
        left: 0;
        width: 100vw !important;
        height: 100vh !important;
        z-index: 0;
    }

    /* 3. Make the sidebar solid so it doesn't clash with map colors */
    [data-testid="stSidebar"] {
        background-color: #111 !important;
        z-index: 100;
        border-right: 1px solid #333;
    }

    /* 4. Style the Timeline Slider at the bottom */
    .stSlider {
        position: fixed !important;
        bottom: 30px !important;
        left: 380px !important;
        right: 40px !important;
        z-index: 1000 !important;
        background: rgba(10, 10, 10, 0.85) !important;
        padding: 10px 30px !important;
        border-radius: 50px !important;
        border: 1px solid #444;
    }

    /* Hide standard Streamlit headers */
    header, footer { visibility: hidden !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 2. STATE & DIRECTORY
# -----------------------------
RADAR_COLORS = ['#76fffe', '#01a0fe', '#0001ef', '#01ef01', '#019001', '#ffff01', '#e7c001', '#ff9000', '#ff0101']
RADAR_CMAP = ListedColormap(RADAR_COLORS)

if 'processed_df' not in st.session_state: st.session_state.processed_df = None
if 'radar_cache' not in st.session_state: st.session_state.radar_cache = {}
if 'time_list' not in st.session_state: st.session_state.time_list = []
if 'active_gdf' not in st.session_state: st.session_state.active_gdf = None
if 'map_view' not in st.session_state:
    st.session_state.map_view = pdk.ViewState(latitude=40.7, longitude=-74.0, zoom=9)

if "img_dir" not in st.session_state:
    st.session_state.img_dir = tempfile.mkdtemp(prefix="radar_png_")

# -----------------------------
# 3. RADAR ENGINE
# -----------------------------
def get_radar_image(dt_utc):
    ts_str = dt_utc.strftime("%Y%m%d-%H%M00")
    url = f"https://noaa-mrms-pds.s3.amazonaws.com/CONUS/RadarOnly_QPE_15M_00.00/{dt_utc.strftime('%Y%m%d')}/MRMS_RadarOnly_QPE_15M_00.00_{ts_str}.grib2.gz"
    tmp_grib = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False).name
    try:
        r = requests.get(url, stream=True, timeout=20)
        if r.status_code != 200: return None, 0, None
        with gzip.GzipFile(fileobj=r.raw) as gz, open(tmp_grib, "wb") as f:
            shutil.copyfileobj(gz, f)
        with xr.open_dataset(tmp_grib, engine="cfgrib") as ds:
            da = ds[list(ds.data_vars)[0]].load()
            da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180).sortby("longitude")
            subset = da.sel(latitude=slice(42.5, 38.5), longitude=slice(-76.5, -72.5))
            
            site_mean = 0.0
            if st.session_state.active_gdf is not None:
                sub = subset.rio.write_crs("EPSG:4326", inplace=False)
                clipped = sub.rio.clip(st.session_state.active_gdf.geometry, st.session_state.active_gdf.crs, drop=False)
                if not clipped.isnull().all():
                    site_mean = float(clipped.mean().values)

            data = subset.values.astype("float32")
            data[data < 0.1] = np.nan
            img_path = os.path.join(st.session_state.img_dir, f"radar_{dt_utc.strftime('%H%M')}.png")
            plt.imsave(img_path, data, cmap=RADAR_CMAP, vmin=0.1, vmax=15.0)
            bounds = [float(subset.longitude.min()), float(subset.latitude.min()), 
                      float(subset.longitude.max()), float(subset.latitude.max())]
            return img_path, max(0.0, site_mean / 25.4), bounds
    except: return None, 0, None
    finally:
        if os.path.exists(tmp_grib): os.remove(tmp_grib)

# -----------------------------
# 4. SIDEBAR
# -----------------------------
with st.sidebar:
    st.title("CNR GIS Portal")
    tz_mode = st.radio("Timezone", ["Local (EST/EDT)", "UTC"])
    s_date = st.date_input("Start Date", value=datetime.now().date())
    e_date = st.date_input("End Date", value=datetime.now().date())

    c1, c2 = st.columns(2)
    hours = [f"{h:02d}:00" for h in range(24)]
    s_time = c1.selectbox("Start", hours, index=19)
    e_time = c2.selectbox("End", hours, index=21)

    up_zip = st.file_uploader("Upload Watershed ZIP", type="zip")
    if up_zip:
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(up_zip, 'r') as z: z.extractall(td)
            shps = list(Path(td).rglob("*.shp"))
            if shps:
                st.session_state.active_gdf = gpd.read_file(shps[0]).to_crs("EPSG:4326")
                b = st.session_state.active_gdf.total_bounds
                st.session_state.map_view = pdk.ViewState(
                    latitude=(b[1] + b[3]) / 2, longitude=(b[0] + b[2]) / 2, zoom=11
                )

    if st.button("PROCESS RADAR DATA", use_container_width=True):
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
                pb.progress((i + 1) / len(tr))
            st.session_state.radar_cache, st.session_state.time_list = cache, list(cache.keys())
            st.session_state.processed_df = pd.DataFrame(stats)

    # --- RESTORED & IMPROVED DATA SECTION ---
    if st.session_state.processed_df is not None:
        st.write("---")
        if st.button("📊 SHOW ANALYSIS PLOT", use_container_width=True):
            import plotly.express as px
            @st.dialog("Rainfall Statistics", width="large")
            def modal():
                fig = px.bar(st.session_state.processed_df, x='time', y='rain_in', 
                             title="Basin Average Rainfall (inches)", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            modal()
        
        st.subheader("📁 Data Downloads")
        # File selector to choose which timestamp to download
        selected_file_time = st.selectbox("Select Timestamp to View/Download", st.session_state.time_list)
        
        # Display the specific value for that timestamp
        val = st.session_state.processed_df[st.session_state.processed_df['time'].dt.strftime("%H:%M") == selected_file_time]['rain_in'].values[0]
        st.metric(label=f"Rainfall at {selected_file_time}", value=f"{val:.4f} in")
        
        # Download the full CSV
        full_csv = st.session_state.processed_df.to_csv(index=False).encode('utf-8')
        st.download_button("DOWNLOAD FULL CSV", data=full_csv, file_name="radar_full_stats.csv", mime='text/csv', use_container_width=True)

# -----------------------------
# 5. MAP RENDER
# -----------------------------
layers = []
if st.session_state.time_list:
    # Use select_slider with a custom style (handled in CSS above)
    t_str = st.select_slider("", options=st.session_state.time_list, label_visibility="collapsed")
    curr = st.session_state.radar_cache[t_str]
    layers.append(pdk.Layer("BitmapLayer", image=curr["path"], bounds=curr["bounds"], opacity=0.7))

if st.session_state.active_gdf is not None:
    layers.append(pdk.Layer("GeoJsonLayer", st.session_state.active_gdf.__geo_interface__, 
                            stroked=True, filled=False, get_line_color=[255, 255, 255], line_width_min_pixels=3))

st.pydeck_chart(pdk.Deck(
    layers=layers,
    initial_view_state=st.session_state.map_view,
    map_style="mapbox://styles/mapbox/dark-v11"
))
