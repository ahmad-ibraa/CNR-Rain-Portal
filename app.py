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
import time
import base64
from zoneinfo import ZoneInfo

NY_TZ  = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

# =============================
# 1) PAGE CONFIG & CSS
# =============================
st.set_page_config(layout="wide", page_title="CNR Radar Portal", initial_sidebar_state="expanded")

st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"] { height: 100vh; overflow: hidden; background: #000; }
    .main .block-container { padding: 0 !important; max-width: 100vw !important; }
    [data-testid="stSidebar"] { min-width: 400px; max-width: 400px; background: rgba(17,17,17,0.95); z-index: 5000; }
    header, footer, [data-testid="stToolbar"] { display: none; }
    .control-bar {
        position: fixed; left: 420px; right: 20px; bottom: 20px; z-index: 10000;
        background: rgba(15,15,15,0.9); padding: 15px; border-radius: 50px;
        border: 1px solid #444; backdrop-filter: blur(10px);
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.5rem !important; }
</style>
""", unsafe_allow_html=True)

# =============================
# 2) HELPERS
# =============================
def clean_da(da):
    da = da.squeeze()
    keep = {"time", "latitude", "longitude"}
    to_drop = [c for c in da.coords if c not in keep]
    da = da.drop_vars(to_drop, errors="ignore")
    da = da.assign_coords(
        latitude=da.latitude.round(4),
        longitude=da.longitude.round(4)
    )
    return da.sortby("latitude", ascending=False).sortby("longitude")

def load_precip(file_type, dt_local):
    dt_utc = dt_local.replace(tzinfo=NY_TZ).astimezone(UTC_TZ).replace(tzinfo=None)
    ts, ymd = dt_utc.strftime("%Y%m%d-%H%M00"), dt_utc.strftime("%Y%m%d")
    
    if file_type == "RO":
        url = f"https://noaa-mrms-pds.s3.amazonaws.com/CONUS/RadarOnly_QPE_15M_00.00/{ymd}/MRMS_RadarOnly_QPE_15M_00.00_{ts}.grib2.gz"
    else:
        url = f"https://noaa-mrms-pds.s3.amazonaws.com/CONUS/MultiSensor_QPE_01H_Pass2_00.00/{ymd}/MRMS_MultiSensor_QPE_01H_Pass2_00.00_{ts}.grib2.gz"
    
    tmp = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False).name
    try:
        r = requests.get(url, stream=True, timeout=15)
        if r.status_code != 200: return None
        with gzip.GzipFile(fileobj=r.raw) as gz, open(tmp, "wb") as f:
            shutil.copyfileobj(gz, f)
        with xr.open_dataset(tmp, engine="cfgrib") as ds:
            da = ds[list(ds.data_vars)[0]].load()
            da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180)
            return clean_da(da).rio.write_crs("EPSG:4326")
    except: return None
    finally:
        if os.path.exists(tmp): os.remove(tmp)

# =============================
# 3) STATE & SIDEBAR
# =============================
for key in ["radar_cache", "time_list", "basin_vault"]:
    if key not in st.session_state: st.session_state[key] = {}
if "current_time_index" not in st.session_state: st.session_state.current_time_index = 0
if "is_playing" not in st.session_state: st.session_state.is_playing = False

with st.sidebar:
    st.title("CNR Radar Portal")
    
    # Restored Date/Time Inputs
    s_date = st.date_input("Start Date", value=datetime.now().date())
    e_date = st.date_input("End Date", value=datetime.now().date())
    
    t_cols = st.columns(2)
    hours = [f"{h:02d}:00" for h in range(24)]
    s_time = t_cols[0].selectbox("Start Time", hours, index=19)
    e_time = t_cols[1].selectbox("End Time", hours, index=22)
    
    up_zip = st.file_uploader("Watershed ZIP", type="zip")
    if up_zip:
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(up_zip, "r") as z: z.extractall(td)
            shps = list(Path(td).rglob("*.shp"))
            if shps:
                st.session_state.active_gdf = gpd.read_file(shps[0]).to_crs("EPSG:4326")
                b = st.session_state.active_gdf.total_bounds
                st.session_state.map_view = pdk.ViewState(latitude=(b[1]+b[3])/2, longitude=(b[0]+b[2])/2, zoom=10)

    if st.button("PROCESS DATA", use_container_width=True):
        if st.session_state.get("active_gdf") is None:
            st.error("Upload a ZIP first")
        else:
            # Combine Date and Time
            start_dt = datetime.combine(s_date, datetime.strptime(s_time, "%H:%M").time())
            end_dt = datetime.combine(e_date, datetime.strptime(e_time, "%H:%M").time())
            
            ro_times = pd.date_range(start_dt, end_dt, freq="15min")
            mrms_times = pd.date_range(start_dt + timedelta(hours=1), end_dt, freq="1H")

            # 1. Download
            ro_list, mrms_list = [], []
            msg = st.empty()
            
            for t in ro_times:
                msg.info(f"Downloading RO: {t:%H:%M}")
                da = load_precip("RO", t.to_pydatetime())
                if da is not None: ro_list.append(da.assign_coords(time=t))
            
            for t in mrms_times:
                msg.info(f"Downloading MRMS: {t:%H:%M}")
                da = load_precip("MRMS", t.to_pydatetime())
                if da is not None: mrms_list.append(da.assign_coords(time=t))

            if not ro_list or not mrms_list:
                st.error("Missing data for selected range.")
            else:
                # 2. Alignment Logic
                msg.info("Aligning Grids...")
                ro = xr.concat(ro_list, dim="time")
                mrms = xr.concat(mrms_list, dim="time")
                ro = ro.reindex_like(mrms, method="nearest")
                
                # 3. Scaling
                msg.info("Calculating Scaled Precip...")
                final_frames = []
                for t_mrms in mrms.time:
                    t_end = pd.to_datetime(t_mrms.values)
                    t_start = t_end - timedelta(minutes=45)
                    ro_hour_sum = ro.sel(time=slice(t_start, t_end)).sum(dim="time")
                    mrms_val = mrms.sel(time=t_mrms)
                    scale = xr.where(ro_hour_sum > 0.1, mrms_val / ro_hour_sum, 1.0).clip(0, 10)
                    
                    for t_sub in pd.date_range(t_start, t_end, freq="15min"):
                        if t_sub in ro.time:
                            final_frames.append((t_sub, ro.sel(time=t_sub) * scale))

                # 4. Render
                RADAR_CMAP = ListedColormap(["#76fffe", "#01a0fe", "#0001ef", "#01ef01", "#019001", "#ffff01", "#e7c001", "#ff9000", "#ff0101"])
                img_dir = tempfile.mkdtemp()
                cache = {}
                for ts, da in final_frames:
                    path = os.path.join(img_dir, f"r_{ts:%H%M}.png")
                    arr = da.values
                    arr[arr < 0.1] = np.nan
                    plt.imsave(path, arr, cmap=RADAR_CMAP, vmin=0.1, vmax=15.0)
                    bounds = [float(da.longitude.min()), float(da.latitude.min()), float(da.longitude.max()), float(da.latitude.max())]
                    cache[f"{ts:%Y-%m-%d %H:%M}"] = {"path": path, "bounds": bounds}
                
                st.session_state.radar_cache = cache
                st.session_state.time_list = sorted(list(cache.keys()))
                msg.success(f"Processed {len(cache)} frames.")

# =============================
# 4) RENDER MAP & PLAYER
# =============================
if st.session_state.time_list and st.session_state.is_playing:
    st.session_state.current_time_index = (st.session_state.current_time_index + 1) % len(st.session_state.time_list)
    time.sleep(0.5)
    st.rerun()

if st.session_state.time_list:
    curr_key = st.session_state.time_list[st.session_state.current_time_index]
    curr = st.session_state.radar_cache[curr_key]
    
    layers = [pdk.Layer("BitmapLayer", image=curr["path"], bounds=curr["bounds"], opacity=0.7)]
    if st.session_state.get("active_gdf") is not None:
        layers.append(pdk.Layer("GeoJsonLayer", st.session_state.active_gdf.__geo_interface__, stroked=True, filled=False, get_line_color=[255,255,255], line_width_min_pixels=2))
    
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=st.session_state.get("map_view"), map_style="mapbox://styles/mapbox/dark-v10"), height=1000)

    # Controls
    st.markdown('<div class="control-bar">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 8, 2])
    with col1:
        if st.button("⏸" if st.session_state.is_playing else "▶"):
            st.session_state.is_playing = not st.session_state.is_playing
            st.rerun()
    with col2:
        new_idx = st.select_slider(" ", options=range(len(st.session_state.time_list)), value=st.session_state.current_time_index, format_func=lambda x: st.session_state.time_list[x], label_visibility="collapsed")
        if new_idx != st.session_state.current_time_index:
            st.session_state.current_time_index = new_idx
            st.session_state.is_playing = False
            st.rerun()
    with col3: st.write(f"**{curr_key}**")
    st.markdown('</div>', unsafe_allow_html=True)
