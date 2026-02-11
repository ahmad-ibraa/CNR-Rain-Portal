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

import rioxarray  # <-- REQUIRED for .rio.clip()


# --- 1. CLEAN & STABLE LAYOUT CSS ---
st.set_page_config(layout="wide", page_title="CNR Radar Portal")

st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 0rem !important;
        max-width: 95% !important;
    }

    /* Ensure the map container has a guaranteed height */
    .stPydeckChart {
        height: 750px !important;
        border-radius: 12px;
        border: 1px solid #333;
    }

    [data-testid="stSidebar"] {
        min-width: 350px !important;
    }

    header, footer { visibility: hidden !important; }
</style>
""", unsafe_allow_html=True)


# --- 2. COLORS & STATE ---
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


# --- 3. DATA ENGINE ---
def get_radar_image(dt_utc):
    ts_str = dt_utc.strftime("%Y%m%d-%H%M00")
    url = f"https://noaa-mrms-pds.s3.amazonaws.com/CONUS/RadarOnly_QPE_15M_00.00/{dt_utc.strftime('%Y%m%d')}/MRMS_RadarOnly_QPE_15M_00.00_{ts_str}.grib2.gz"

    tmp_grib = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False).name
    try:
        r = requests.get(url, stream=True, timeout=20)
        if r.status_code != 200:
            return None, 0, None

        with gzip.GzipFile(fileobj=r.raw) as gz, open(tmp_grib, "wb") as f:
            shutil.copyfileobj(gz, f)

        with xr.open_dataset(tmp_grib, engine="cfgrib") as ds:
            da = ds[list(ds.data_vars)[0]].load()

            # lon 0..360 -> -180..180
            da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180).sortby("longitude")

            # region subset
            subset = da.sel(latitude=slice(42.5, 38.5), longitude=slice(-76.5, -72.5))

            # site mean over watershed
            site_mean = 0.0
            if st.session_state.active_gdf is not None:
                sub = subset.rio.write_crs("EPSG:4326", inplace=False)
                clipped = sub.rio.clip(
                    st.session_state.active_gdf.geometry,
                    st.session_state.active_gdf.crs,
                    drop=False
                )
                if not clipped.isnull().all():
                    site_mean = float(clipped.mean().values)

            data = subset.values.astype("float32")
            data[data < 0.1] = np.nan

            img_filename = os.path.join(st.session_state.img_dir, f"radar_{dt_utc.strftime('%H%M')}.png")
            plt.imsave(img_filename, data, cmap=RADAR_CMAP, vmin=0.1, vmax=15.0)

            bounds = [
                float(subset.longitude.min()), float(subset.latitude.min()),
                float(subset.longitude.max()), float(subset.latitude.max())
            ]

            return img_filename, max(0.0, site_mean / 25.4), bounds

    except Exception as e:
        # show the real error instead of silent failure
        st.sidebar.error(f"Radar frame failed: {e}")
        return None, 0, None

    finally:
        if os.path.exists(tmp_grib):
            os.remove(tmp_grib)


# --- 4. SIDEBAR ---
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
            with zipfile.ZipFile(up_zip, 'r') as z:
                z.extractall(td)
            shps = list(Path(td).rglob("*.shp"))
            if shps:
                st.session_state.active_gdf = gpd.read_file(shps[0]).to_crs("EPSG:4326")
                b = st.session_state.active_gdf.total_bounds
                st.session_state.map_view = pdk.ViewState(
                    latitude=(b[1] + b[3]) / 2,
                    longitude=(b[0] + b[2]) / 2,
                    zoom=11
                )

    if st.button("PROCESS RADAR DATA", use_container_width=True):
        if st.session_state.active_gdf is None:
            st.warning("Upload a watershed ZIP first.")
        else:
            s_dt = datetime.combine(s_date, datetime.strptime(s_time, "%H:%M").time())
            e_dt = datetime.combine(e_date, datetime.strptime(e_time, "%H:%M").time())
            tr = pd.date_range(s_dt, e_dt, freq='15min')

            cache, stats = {}, []
            pb = st.progress(0)

            for i, ts in enumerate(tr):
                # NOTE: your old +5 hours is not DST-safe; keep for now, but better is ZoneInfo.
                ts_utc = ts if tz_mode == "UTC" else ts + timedelta(hours=5)

                path, val, bnds = get_radar_image(ts_utc)
                if path and bnds:
                    label = ts.strftime("%H:%M")
                    cache[label] = {"path": path, "bounds": bnds}
                    stats.append({"time": ts, "rain_in": val})

                pb.progress((i + 1) / len(tr))

            st.session_state.radar_cache = cache
            st.session_state.time_list = list(cache.keys())
            st.session_state.processed_df = pd.DataFrame(stats)

    if st.session_state.processed_df is not None and not st.session_state.processed_df.empty:
        if st.button("SHOW PLOT", use_container_width=True):
            import plotly.express as px

            @st.dialog("Rainfall Statistics", width="large")
            def modal():
                st.plotly_chart(
                    px.bar(st.session_state.processed_df, x='time', y='rain_in', template="plotly_dark"),
                    use_container_width=True
                )
            modal()


# --- 5. MAIN CONTENT AREA (ALWAYS SHOW A MAP) ---
layers = []

if st.session_state.time_list:
    t_str = st.select_slider("Radar Timeline", options=st.session_state.time_list)
    curr = st.session_state.radar_cache[t_str]

    layers.append(
        pdk.Layer(
            "BitmapLayer",
            image=curr["path"],
            bounds=curr["bounds"],
            opacity=0.7
        )
    )
else:
    st.info("Upload a watershed and click 'Process' to see radar data.")

if st.session_state.active_gdf is not None:
    layers.append(
        pdk.Layer(
            "GeoJsonLayer",
            st.session_state.active_gdf.__geo_interface__,
            stroked=True,
            filled=False,
            get_line_color=[255, 255, 255],
            line_width_min_pixels=2
        )
    )

# IMPORTANT: use a token-free map style to avoid black screen
deck = pdk.Deck(
    layers=layers,
    initial_view_state=st.session_state.map_view,
    map_style="dark",  # <-- token-free (works without MAPBOX_API_KEY)
)

st.pydeck_chart(deck)
