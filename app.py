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
# 1. PAGE CONFIG & PERMANENT LAYOUT
# -----------------------------
st.set_page_config(layout="wide", page_title="CNR Radar Portal", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* 1. MAP: Force to true 100% background covering everything */
    .stPydeckChart {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        z-index: -1 !important;
    }

    /* 2. SIDEBAR: Lock width, disable resize handle, and hide collapse button */
    [data-testid="stSidebar"] {
        min-width: 400px !important;
        max-width: 400px !important;
        width: 400px !important;
        background-color: #111 !important;
    }
    
    /* Disable the resize 'drag' bar */
    [data-testid="stSidebarResizer"] {
        display: none !important;
    }

    /* Hide the 'X' button that allows hiding the sidebar */
    button[title="Collapse sidebar"] {
        display: none !important;
    }

    /* 3. MAIN CONTAINER: Make it transparent so the map shows through */
    [data-testid="stAppViewContainer"] {
        background-color: transparent !important;
    }
    
    .main .block-container {
        padding: 0 !important;
        margin: 0 !important;
    }

    /* 4. SLIDER: Floating at bottom */
    .stSlider {
        position: fixed !important;
        bottom: 20px !important;
        left: 430px !important;
        right: 30px !important;
        z-index: 1000 !important;
        background: rgba(20, 20, 20, 0.9) !important;
        padding: 10px 30px !important;
        border-radius: 50px !important;
        border: 1px solid #333;
    }

    header, footer { visibility: hidden !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 2. STATE & ENGINE
# -----------------------------
if 'radar_cache' not in st.session_state: st.session_state.radar_cache = {}
if 'time_list' not in st.session_state: st.session_state.time_list = []
if 'active_gdf' not in st.session_state: st.session_state.active_gdf = None
if 'basin_vault' not in st.session_state: st.session_state.basin_vault = {} 
if 'map_view' not in st.session_state:
    st.session_state.map_view = pdk.ViewState(latitude=40.7, longitude=-74.0, zoom=9)
if "img_dir" not in st.session_state:
    st.session_state.img_dir = tempfile.mkdtemp(prefix="radar_png_")

RADAR_COLORS = ['#76fffe', '#01a0fe', '#0001ef', '#01ef01', '#019001', '#ffff01', '#e7c001', '#ff9000', '#ff0101']
RADAR_CMAP = ListedColormap(RADAR_COLORS)

def get_radar_image(dt_utc):
    ts_str = dt_utc.strftime("%Y%m%d-%H%M00")
    url = f"https://noaa-mrms-pds.s3.amazonaws.com/CONUS/RadarOnly_QPE_15M_00.00/{dt_utc.strftime('%Y%m%d')}/MRMS_RadarOnly_QPE_15M_00.00_{ts_str}.grib2.gz"
    tmp_grib = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False).name
    try:
        r = requests.get(url, stream=True, timeout=15)
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
# 3. SIDEBAR (PERMANENT)
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
    basin_name = "Default_Basin"
    if up_zip:
        basin_name = up_zip.name.replace(".zip", "")
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
            st.session_state.basin_vault[basin_name] = pd.DataFrame(stats)

    if st.session_state.basin_vault:
        st.write("---")
        st.subheader("📁 Processed Basins")
        target_basin = st.selectbox("Select CSV to Download", options=list(st.session_state.basin_vault.keys()))
        df_target = st.session_state.basin_vault[target_basin]
        
        if st.button(f"📊 PLOT {target_basin}", use_container_width=True):
            import plotly.express as px
            @st.dialog(f"Stats: {target_basin}", width="large")
            def modal():
                st.plotly_chart(px.bar(df_target, x='time', y='rain_in', template="plotly_dark"), use_container_width=True)
            modal()
        
        csv_data = df_target.to_csv(index=False).encode('utf-8')
        st.download_button(f"DOWNLOAD {target_basin}.CSV", data=csv_data, file_name=f"{target_basin}.csv", use_container_width=True)

# -----------------------------
# 4. MAP (LOCKED FULLSCREEN)
# -----------------------------
layers = []
if st.session_state.time_list:
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
