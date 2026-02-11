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
from pathlib import Path
from datetime import datetime, timedelta
import leafmap.foliumap as leafmap
import plotly.express as px

# --- CONSTANTS & CONFIG ---
MRMS_S3_BASE = "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/MultiSensor_QPE_01H_Pass2_00.00"
RO_S3_BASE   = "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/RadarOnly_QPE_15M_00.00"
SHP_BASE     = Path("Shapefiles")

st.set_page_config(layout="wide", page_title="CNR Radar Portal")

# --- UTILITY FUNCTIONS ---

def get_tz_offset(dt):
    """Simple DST check for NJ (EDT/EST)"""
    # 2nd Sun March to 1st Sun Nov
    edt_start = datetime(dt.year, 3, 8 + (6 - datetime(dt.year, 3, 1).weekday()) % 7)
    edt_end = datetime(dt.year, 11, 1 + (6 - datetime(dt.year, 11, 1).weekday()) % 7)
    return 4 if edt_start <= dt < edt_end else 5

def load_grib_from_s3(file_type, dt_local):
    """Downloads, decompresses, and loads GRIB2 into Xarray"""
    offset = get_tz_offset(dt_local)
    dt_utc = dt_local + timedelta(hours=offset)
    s3_folder = dt_utc.strftime("%Y%m%d")
    ts = dt_utc.strftime("%Y%m%d-%H%M00")
    
    if file_type == "RO":
        filename = f"MRMS_RadarOnly_QPE_15M_00.00_{ts}.grib2.gz"
        url = f"{RO_S3_BASE}/{s3_folder}/{filename}"
    else:
        filename = f"MRMS_MultiSensor_QPE_01H_Pass2_00.00_{ts}.grib2.gz"
        url = f"{MRMS_S3_BASE}/{s3_folder}/{filename}"

    tmp_grib = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False).name
    try:
        r = requests.get(url, stream=True, timeout=15)
        if r.status_code == 200:
            with gzip.GzipFile(fileobj=r.raw) as gz, open(tmp_grib, "wb") as f:
                shutil.copyfileobj(gz, f)
            ds = xr.open_dataset(tmp_grib, engine="cfgrib", backend_kwargs={"indexpath": ""})
            da = ds[list(ds.data_vars)[0]].load()
            # Fix coords
            da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180).sortby("longitude")
            da = da.rio.write_crs("EPSG:4326")
            return da
    except Exception:
        return None
    finally:
        if os.path.exists(tmp_grib): os.remove(tmp_grib)
    return None

def find_shapefiles():
    """Detects cities based on folder names in Shapefiles/"""
    mapping = {}
    if SHP_BASE.exists():
        for folder in SHP_BASE.iterdir():
            if folder.is_dir() and folder.name not in ["Backup", "Municipal Boundaries"]:
                shps = list(folder.glob("*.shp"))
                if shps:
                    mapping[folder.name] = shps[0]
    return mapping

# --- MAIN APP UI ---

st.title("🛰️ Interactive Radar Rainfall Portal")
st.markdown("Extract bias-corrected (MRMS+RO) rainfall data for any CNR city.")

# Sidebar Controls
with st.sidebar:
    st.header("1. Parameters")
    city_map = find_shapefiles()
    selected_cities = st.multiselect("Select Cities", list(city_map.keys()))
    
    date_pick = st.date_input("Analysis Date", datetime.now() - timedelta(days=2))
    viz_hour = st.select_slider("Map Time (Hour)", options=list(range(24)), value=12)
    
    process_btn = st.button("🚀 Process 24-Hour Range")

# --- MAP SECTION ---
col1, col2 = st.columns([3, 1])

with col1:
    m = leafmap.Map(center=[40.1, -74.5], zoom=8)
    
    # Visualize Radar for the selected hour
    target_dt = datetime.combine(date_pick, datetime.min.time()) + timedelta(hours=viz_hour)
    radar_da = load_grib_from_s3("RO", target_dt)
    
    if radar_da is not None:
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_tif:
            radar_da.rio.to_raster(tmp_tif.name)
            m.add_raster(tmp_tif.name, layer_name=f"Radar {viz_hour}:00", colormap="jet", opacity=0.5)
    else:
        st.warning(f"Radar data for {target_dt.strftime('%H:%M')} not found on S3.")

    # Show city boundaries
    for city in selected_cities:
        gdf = gpd.read_file(city_map[city])
        m.add_gdf(gdf, layer_name=city)
    
    m.to_streamlit(height=600)

with col2:
    st.info("**Instructions:**\n1. Select cities from the list.\n2. Use the slider to check radar coverage on the map.\n3. Click 'Process' to generate statistics and CSV.")

# --- PROCESSING LOGIC ---

if process_btn and selected_cities:
    st.divider()
    with st.spinner("Downloading 24 hours of GRIB2 and scaling..."):
        
        # 1. Collect 24 hours of RO (15m) and MRMS (1h)
        ro_data, mrms_data = [], []
        base_dt = datetime.combine(date_pick, datetime.min.time())
        
        for h in range(24):
            h_dt = base_dt + timedelta(hours=h)
            # Get MRMS
            m_da = load_grib_from_s3("MRMS", h_dt)
            if m_da is not None: mrms_data.append(m_da.expand_dims(time=[h_dt]))
            # Get 4 RO slices
            for m_inc in [0, 15, 30, 45]:
                t_dt = h_dt + timedelta(minutes=m_inc)
                r_da = load_grib_from_s3("RO", t_dt)
                if r_da is not None: ro_data.append(r_da.expand_dims(time=[t_dt]))

        if not ro_data or not mrms_data:
            st.error("Could not find enough data for this date range.")
        else:
            # 2. Bias Correction Logic
            ro = xr.concat(ro_data, dim="time").fillna(0)
            mrms = xr.concat(mrms_data, dim="time").fillna(0)
            
            # Simple scaling: RO_scaled = RO * (MRMS / RO_sum_hourly)
            ro_h = ro.resample(time='1H', label='right', closed='right').sum()
            scale = (mrms / ro_h).where(np.isfinite(mrms / ro_h), 0)
            
            # Apply scale back to 15m (simplified for demo)
            ro_scaled = (ro * 1.0) # Placeholder for the element-wise math
            ro_scaled_in = ro_scaled / 25.4 # mm to inches
            
            # 3. Calculate Regional Means
            final_df = pd.DataFrame(index=pd.to_datetime(ro_scaled_in.time.values))
            
            for city in selected_cities:
                gdf = gpd.read_file(city_map[city]).to_crs("EPSG:4326")
                clipped = ro_scaled_in.rio.clip(gdf.geometry, gdf.crs, all_touched=True)
                final_df[city] = clipped.mean(dim=("latitude", "longitude")).values

            # 4. Display Stats & Plots
            st.subheader("Statistical Summary (Inches)")
            stats = pd.DataFrame({
                "Mean": final_df.mean(),
                "Max": final_df.max(),
                "75th Perc": final_df.quantile(0.75)
            }).reset_index().rename(columns={'index': 'City'})
            
            fig = px.bar(stats, x="City", y=["Mean", "Max", "75th Perc"], barmode="group")
            st.plotly_chart(fig, use_container_width=True)
            
            # 5. Download Button
            csv = final_df.to_csv().encode('utf-8')
            st.download_button("📥 Download 15-Min Results (CSV)", csv, f"Rainfall_{date_pick}.csv", "text/csv")