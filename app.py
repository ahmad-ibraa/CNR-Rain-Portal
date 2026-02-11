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
        .block-container { padding: 1.5rem !important; }
        iframe { height: 70vh !important; width: 100% !important; border-radius: 8px; border: 1px solid #31333f; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. TIME CONSTRAINTS ---
now = datetime.now()
max_allowed_dt = now - timedelta(hours=1)

if 'processed_df' not in st.session_state: st.session_state.processed_df = None
if 'raster_cache' not in st.session_state: st.session_state.raster_cache = {}
if 'time_list' not in st.session_state: st.session_state.time_list = []
if 'shp_name' not in st.session_state: st.session_state.shp_name = "output"

# --- 3. DOWNLOADER ---
def download_s3_grib(dt_local):
    # Determine UTC offset (EDT vs EST)
    edt_s = datetime(dt_local.year, 3, 8 + (6 - datetime(dt_local.year, 3, 1).weekday()) % 7)
    edt_e = datetime(dt_local.year, 11, 1 + (6 - datetime(dt_local.year, 11, 1).weekday()) % 7)
    offset = 4 if edt_s <= dt_local < edt_e else 5
    dt_utc = dt_local + timedelta(hours=offset)
    
    ts = dt_utc.strftime("%Y%m%d-%H%M00")
    url = f"https://noaa-mrms-pds.s3.amazonaws.com/CONUS/RadarOnly_QPE_15M_00.00/{dt_utc.strftime('%Y%m%d')}/MRMS_RadarOnly_QPE_15M_00.00_{ts}.grib2.gz"

    tmp = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False).name
    try:
        r = requests.get(url, stream=True, timeout=10)
        if r.status_code == 200:
            with gzip.GzipFile(fileobj=r.raw) as gz, open(tmp, "wb") as f:
                shutil.copyfileobj(gz, f)
            with xr.open_dataset(tmp, engine="cfgrib", backend_kwargs={"indexpath": ""}) as ds:
                da = ds[list(ds.data_vars)[0]].load()
                da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180).sortby("longitude")
                # Filter noise for map clarity
                da = da.where(da > 0.1) 
                return da.rio.write_crs("EPSG:4326")
    except: return None
    finally:
        if os.path.exists(tmp): os.remove(tmp)
    return None

# --- 4. SIDEBAR SELECTION ---
with st.sidebar:
    st.title("🛰️ CNR Portal")
    
    # Date Pickers restricted to max_allowed_dt
    start_d = st.date_input("Start Date", value=max_allowed_dt - timedelta(days=1), max_value=max_allowed_dt.date())
    end_d = st.date_input("End Date", value=max_allowed_dt.date(), max_value=max_allowed_dt.date())
    
    # Logic to filter hours for "Today"
    full_hours = [f"{h:02d}:00" for h in range(24)]
    if end_d == max_allowed_dt.date():
        selectable_hours = [f"{h:02d}:00" for h in range(max_allowed_dt.hour + 1)]
    else:
        selectable_hours = full_hours

    col1, col2 = st.columns(2)
    start_t = col1.selectbox("Start Time", full_hours, index=0)
    end_t = col2.selectbox("End Time", selectable_hours, index=len(selectable_hours)-1)
    
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
            with st.spinner("Processing MRMS Data..."):
                s_dt = datetime.combine(start_d, datetime.strptime(start_t, "%H:%M").time())
                e_dt = datetime.combine(end_d, datetime.strptime(end_t, "%H:%M").time())
                
                tr = pd.date_range(s_dt, e_dt, freq='1H')
                rc, sl = {}, []
                pb = st.progress(0)
                for i, ts in enumerate(tr):
                    da = download_s3_grib(ts)
                    if da is not None:
                        tf = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
                        da.rio.to_raster(tf)
                        rc[ts.strftime("%Y-%m-%d %H:%M")] = tf
                        clipped = da.rio.clip(active_gdf.geometry, active_gdf.crs, all_touched=True)
                        sl.append({"time": ts, "rain_in": float(clipped.mean())/25.4})
                    pb.progress((i+1)/len(tr))
                st.session_state.processed_df = pd.DataFrame(sl).set_index("time")
                st.session_state.raster_cache = rc
                st.session_state.time_list = list(rc.keys())
        else: st.warning("Please upload a shapefile first.")

    if st.session_state.processed_df is not None:
        st.divider()
        st.subheader("📁 Output Files")
        csv_name = f"rainfall_{st.session_state.shp_name}.csv"
        st.download_button(label=f"📥 {csv_name}", 
                           data=st.session_state.processed_df.to_csv().encode('utf-8'), 
                           file_name=csv_name)

# --- 5. MAIN CONTENT ---

# Use a standard Selectbox for Time Selection instead of a slider
selected_view_time = None
if st.session_state.time_list:
    st.info("### 🔍 Radar View Selection")
    selected_view_time = st.selectbox(
        "Select specific timestamp to display on map:",
        options=st.session_state.time_list,
        index=len(st.session_state.time_list)-1
    )

# Initialize Map
m = leafmap.Map(center=[40.1, -74.5], zoom=7)

if active_gdf is not None:
    m.add_gdf(active_gdf, layer_name="Site Boundary")

# Render radar for chosen timestamp
if selected_view_time and selected_view_time in st.session_state.raster_cache:
    m.add_raster(st.session_state.raster_cache[selected_view_time], 
                 layer_name="CONUS Radar", colormap="jet", opacity=0.5)

# Render Map
m.to_streamlit()

# --- 6. CHART ---
if st.session_state.processed_df is not None:
    st.plotly_chart(px.bar(st.session_state.processed_df.reset_index(), x="time", y="rain_in", 
                           title=f"Mean Rainfall Trend: {st.session_state.shp_name} (Inches)",
                           template="plotly_dark"), use_container_width=True)
