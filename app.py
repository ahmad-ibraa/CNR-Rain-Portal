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
from pathlib import Path
from datetime import datetime, timedelta
import leafmap.foliumap as leafmap
import plotly.express as px

# --- 1. PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="CNR Radar Portal", page_icon="🛰️")

st.markdown("""
    <style>
        .block-container { padding: 1rem !important; }
        iframe { height: 75vh !important; width: 100% !important; border-radius: 8px; }
        [data-testid="stSidebarCollapsedControl"] { display: flex !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. TIME HELPERS ---
def get_tz_offset(dt):
    edt_s = datetime(dt.year, 3, 8 + (6 - datetime(dt.year, 3, 1).weekday()) % 7)
    edt_e = datetime(dt.year, 11, 1 + (6 - datetime(dt.year, 11, 1).weekday()) % 7)
    return 4 if edt_s <= dt < edt_e else 5

now_utc = datetime.utcnow()
max_allowed_utc = now_utc - timedelta(hours=1)

if 'processed_df' not in st.session_state: st.session_state.processed_df = None
if 'raster_cache' not in st.session_state: st.session_state.raster_cache = {}
if 'time_list' not in st.session_state: st.session_state.time_list = []
if 'shp_name' not in st.session_state: st.session_state.shp_name = "output"

# --- 3. DOWNLOADER ---
def download_mrms(dt_utc):
    ts = dt_utc.strftime("%Y%m%d-%H%M00")
    url = f"https://noaa-mrms-pds.s3.amazonaws.com/CONUS/RadarOnly_QPE_15M_00.00/{dt_utc.strftime('%Y%m%d')}/MRMS_RadarOnly_QPE_15M_00.00_{ts}.grib2.gz"
    
    tmp = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False).name
    try:
        r = requests.get(url, stream=True, timeout=10)
        if r.status_code == 200:
            with gzip.GzipFile(fileobj=r.raw) as gz, open(tmp, "wb") as f:
                shutil.copyfileobj(gz, f)
            with xr.open_dataset(tmp, engine="cfgrib", backend_kwargs={"indexpath": ""}) as ds:
                var_name = list(ds.data_vars)[0]
                da = ds[var_name].load()
                # Correct projection and coordinates
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
    
    max_view_dt = max_allowed_utc if tz_mode == "UTC" else (max_allowed_utc - timedelta(hours=get_tz_offset(max_allowed_utc)))
    
    start_d = st.date_input("Start Date", value=max_view_dt - timedelta(days=1), max_value=max_view_dt.date())
    end_d = st.date_input("End Date", value=max_view_dt.date(), max_value=max_view_dt.date())
    
    hours = [f"{h:02d}:00" for h in range(24)]
    if end_d == max_view_dt.date():
        hours = [f"{h:02d}:00" for h in range(max_view_dt.hour + 1)]

    c1, c2 = st.columns(2)
    start_t = c1.selectbox("Start Time", hours, index=0)
    end_t = c2.selectbox("End Time", hours, index=len(hours)-1)
    
    up_zip = st.file_uploader("Upload ZIP Shapefile", type="zip")
    active_gdf = None
    if up_zip:
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(up_zip, 'r') as z: z.extractall(td)
            shps = list(Path(td).rglob("*.shp"))
            if shps:
                st.session_state.shp_name = shps[0].stem
                active_gdf = gpd.read_file(shps[0]).to_crs("EPSG:4326")

    if st.button("🚀 Process & Export", use_container_width=True):
        if active_gdf is not None:
            with st.spinner("Crunching Radar Layers..."):
                s_dt = datetime.combine(start_d, datetime.strptime(start_t, "%H:%M").time())
                e_dt = datetime.combine(end_d, datetime.strptime(end_t, "%H:%M").time())
                tr = pd.date_range(s_dt, e_dt, freq='1H')
                
                rc, sl = {}, []
                pb = st.progress(0)
                for i, ts in enumerate(tr):
                    ts_utc = ts if tz_mode == "UTC" else ts + timedelta(hours=get_tz_offset(ts))
                    da = download_mrms(ts_utc)
                    if da is not None:
                        # Save GeoTIFF for Map (CONUS)
                        # We mask values <= 0 for map display only so it's transparent where dry
                        map_da = da.where(da > 0.1, np.nan)
                        tf = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
                        map_da.rio.to_raster(tf)
                        rc[ts.strftime("%Y-%m-%d %H:%M")] = tf
                        
                        # Stats Calculation (Filled with 0 if empty)
                        clipped = da.rio.clip(active_gdf.geometry, active_gdf.crs, all_touched=True)
                        mean_val = float(clipped.mean()) if not clipped.isnull().all() else 0.0
                        # Check for 0 specifically
                        if np.isnan(mean_val) or mean_val < 0: mean_val = 0.0
                        sl.append({"time": ts, "rain_in": mean_val/25.4})
                    else:
                        # If file is missing, still add a 0 entry to keep time steps continuous
                        sl.append({"time": ts, "rain_in": 0.0})
                    pb.progress((i+1)/len(tr))
                
                st.session_state.processed_df = pd.DataFrame(sl).set_index("time")
                st.session_state.raster_cache = rc
                st.session_state.time_list = list(rc.keys())
        else: st.warning("Upload ZIP first")

    if st.session_state.processed_df is not None:
        st.divider()
        csv_name = f"rainfall_{st.session_state.shp_name}.csv"
        st.download_button(label=f"📥 {csv_name}", data=st.session_state.processed_df.to_csv().encode('utf-8'), file_name=csv_name)

# --- 5. MAIN CONTENT ---
m = leafmap.Map(center=[40.1, -74.5], zoom=7)
if active_gdf is not None:
    m.add_gdf(active_gdf, layer_name="Site Boundary")

if st.session_state.time_list:
    # This adds the playback bar with slider and play button at the bottom
    m.add_time_slider(
        st.session_state.raster_cache,
        labels=st.session_state.time_list,
        time_interval=500,
        position="bottomright"
    )

m.to_streamlit()

# --- 6. CHART ---
if st.session_state.processed_df is not None:
    st.plotly_chart(px.bar(st.session_state.processed_df.reset_index(), x="time", y="rain_in", 
                           title=f"Precipitation Trend ({tz_mode})", template="plotly_dark"), use_container_width=True)
