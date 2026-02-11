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
import time
import pydeck as pdk
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import rioxarray  # enables .rio accessor


# -----------------------------
# 1) PAGE CONFIG
# -----------------------------
st.set_page_config(layout="wide", page_title="CNR Radar Portal")


# -----------------------------
# 2) CONSTANTS / STATE
# -----------------------------
RADAR_COLORS = [
    "#76fffe", "#01a0fe", "#0001ef",
    "#01ef01", "#019001",
    "#ffff01", "#e7c001",
    "#ff9000", "#ff0101"
]
RADAR_CMAP = ListedColormap(RADAR_COLORS)

NY_TZ = ZoneInfo("America/New_York")

if "processed_df" not in st.session_state:
    st.session_state.processed_df = None

if "radar_cache" not in st.session_state:
    st.session_state.radar_cache = {}  # key: "HH:MM" -> dict(path,bounds)

if "time_list" not in st.session_state:
    st.session_state.time_list = []

if "active_gdf" not in st.session_state:
    st.session_state.active_gdf = None

if "map_view" not in st.session_state:
    st.session_state.map_view = pdk.ViewState(latitude=40.7, longitude=-74.0, zoom=9)

if "img_dir" not in st.session_state:
    st.session_state.img_dir = tempfile.mkdtemp(prefix="radar_png_")

if "sidebar_collapsed" not in st.session_state:
    st.session_state.sidebar_collapsed = False

if "playing" not in st.session_state:
    st.session_state.playing = False

if "t_idx" not in st.session_state:
    st.session_state.t_idx = 0

if "show_loading" not in st.session_state:
    st.session_state.show_loading = False


# -----------------------------
# 3) FULLSCREEN MAP + UI OVERLAYS (CSS)
# -----------------------------
slider_left = "60px" if st.session_state.sidebar_collapsed else "380px"
sidebar_transform = "translateX(-340px)" if st.session_state.sidebar_collapsed else "translateX(0px)"
hamburger_left = "12px" if st.session_state.sidebar_collapsed else "330px"

loading_css = """
.loading-overlay{
  position: fixed; inset: 0;
  background: rgba(0,0,0,0.55);
  z-index: 2000;
  display:flex; align-items:center; justify-content:center;
}
.spinner{
  width: 58px; height: 58px;
  border: 6px solid rgba(255,255,255,0.25);
  border-top: 6px solid rgba(255,75,75,0.95);
  border-radius: 50%;
  animation: spin 0.9s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.loading-text{
  margin-top: 14px;
  color: rgba(255,255,255,0.9);
  font-weight: 700;
  letter-spacing: 0.4px;
  text-align:center;
}
"""

st.markdown(
    f"""
<style>
/* ----- true fullscreen, no scroll ----- */
html, body, [data-testid="stAppViewContainer"] {{
    margin: 0 !important;
    padding: 0 !important;
    height: 100vh !important;
    width: 100vw !important;
    overflow: hidden !important;
}}
.main .block-container {{
    padding: 0 !important;
    margin: 0 !important;
    max-width: 100vw !important;
}}

/* Hide Streamlit chrome */
header, footer {{ visibility: hidden !important; }}

/* Make pydeck iframe full-screen background */
iframe[title="pydeck.io"] {{
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    height: 100vh !important;
    width: 100vw !important;
    z-index: 0 !important;
}}

/* Sidebar glass overlay + collapsible */
[data-testid="stSidebar"] {{
    background-color: rgba(20, 20, 20, 0.75) !important;
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
    z-index: 1000 !important;
    transform: {sidebar_transform};
    transition: transform 180ms ease-in-out;
}}
/* Prevent the "gap" Streamlit reserves */
[data-testid="stSidebar"] + div {{
    margin-left: 0 !important;
}}

/* Floating timeline slider */
.stSlider {{
    position: fixed;
    bottom: 36px;
    left: {slider_left};
    right: 40px;
    z-index: 1200;
    background: rgba(15, 15, 15, 0.86);
    padding: 12px 28px;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.08);
}}

/* Clean sidebar buttons */
div.stButton > button {{
    width: 100% !important;
    height: 52px !important;
    background-color: #ff4b4b !important;
    color: white !important;
    border: none !important;
    font-weight: 800 !important;
    text-transform: uppercase;
}}

/* Floating hamburger */
.floating-hamburger {{
    position: fixed;
    top: 14px;
    left: {hamburger_left};
    z-index: 1400;
}}

/* Loading overlay */
{loading_css}
</style>
""",
    unsafe_allow_html=True,
)

# Optional loading overlay HTML
if st.session_state.show_loading:
    st.markdown(
        """
        <div class="loading-overlay">
          <div>
            <div class="spinner"></div>
            <div class="loading-text">Processing radar frames…</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# 4) HELPERS (polished)
# -----------------------------
def local_dt_to_utc(dt_naive_local: datetime) -> datetime:
    """Interpret a naive datetime as America/New_York and convert to UTC aware."""
    dt_local = dt_naive_local.replace(tzinfo=NY_TZ)
    return dt_local.astimezone(ZoneInfo("UTC"))

def utc_dt_to_local_str(dt_utc: datetime) -> str:
    """UTC aware -> local display string HH:MM."""
    return dt_utc.astimezone(NY_TZ).strftime("%H:%M")

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_mrms_grib_gz(url: str) -> bytes | None:
    """Download MRMS grib2.gz content. Cached to reduce flicker & repeat downloads."""
    try:
        r = requests.get(url, stream=True, timeout=20)
        if r.status_code != 200:
            return None
        return r.content
    except Exception:
        return None

def get_radar_frame(dt_utc: datetime, bbox=None):
    """
    Returns: (png_path, site_mean_in, bounds)
    - Saves PNG into a session temp folder so files persist across reruns.
    """
    ts_str = dt_utc.strftime("%Y%m%d-%H%M00")
    day_str = dt_utc.strftime("%Y%m%d")
    url = (
        "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/"
        f"RadarOnly_QPE_15M_00.00/{day_str}/"
        f"MRMS_RadarOnly_QPE_15M_00.00_{ts_str}.grib2.gz"
    )

    gz_bytes = fetch_mrms_grib_gz(url)
    if gz_bytes is None:
        return None, 0.0, None

    tmp_grib = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False).name
    try:
        # decompress -> tmp grib
        with gzip.GzipFile(fileobj=tempfile.SpooledTemporaryFile()) as _:
            pass

        with gzip.GzipFile(fileobj=tempfile.SpooledTemporaryFile()) as _:
            pass

        # write gz_bytes then decompress
        tmp_gz = tempfile.NamedTemporaryFile(suffix=".gz", delete=False).name
        with open(tmp_gz, "wb") as f:
            f.write(gz_bytes)

        with gzip.open(tmp_gz, "rb") as gz, open(tmp_grib, "wb") as out:
            shutil.copyfileobj(gz, out)

        # open grib
        with xr.open_dataset(tmp_grib, engine="cfgrib") as ds:
            var = list(ds.data_vars)[0]
            da = ds[var].load()

            # normalize lon to [-180,180]
            da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180).sortby("longitude")

            # If watershed uploaded, zoom to its bounds with a small buffer
            if bbox is not None:
                minx, miny, maxx, maxy = bbox
                pad_x = (maxx - minx) * 0.10 + 0.05
                pad_y = (maxy - miny) * 0.10 + 0.05
                lon0, lon1 = (minx - pad_x), (maxx + pad_x)
                lat0, lat1 = (miny - pad_y), (maxy + pad_y)
                subset = da.sel(latitude=slice(lat1, lat0), longitude=slice(lon0, lon1))
            else:
                # fallback box
                subset = da.sel(latitude=slice(42.5, 38.5), longitude=slice(-76.5, -72.5))

            # clip mean over watershed if present
            site_mean_in = 0.0
            if st.session_state.active_gdf is not None and subset.size > 0:
                try:
                    sub = subset.rio.write_crs("EPSG:4326", inplace=False)
                    clipped = sub.rio.clip(
                        st.session_state.active_gdf.geometry,
                        st.session_state.active_gdf.crs,
                        drop=False
                    )
                    if not clipped.isnull().all():
                        # MRMS QPE is mm over the interval; convert mean mm -> inches
                        site_mean_in = float(clipped.mean().values) / 25.4
                except Exception:
                    site_mean_in = 0.0

            data = subset.values.astype("float32")
            data[data < 0.1] = np.nan

            # Save image to persistent per-session temp dir
            out_name = f"radar_{dt_utc.strftime('%Y%m%d_%H%M')}.png"
            out_path = os.path.join(st.session_state.img_dir, out_name)
            if not os.path.exists(out_path):
                plt.imsave(out_path, data, cmap=RADAR_CMAP, vmin=0.1, vmax=15.0)

            bounds = [
                float(subset.longitude.min()),
                float(subset.latitude.min()),
                float(subset.longitude.max()),
                float(subset.latitude.max()),
            ]
            return out_path, max(0.0, site_mean_in), bounds

    except Exception:
        return None, 0.0, None
    finally:
        for p in [tmp_grib]:
            if p and os.path.exists(p):
                os.remove(p)
        # tmp_gz cleanup
        try:
            if "tmp_gz" in locals() and os.path.exists(tmp_gz):
                os.remove(tmp_gz)
        except Exception:
            pass


def build_timerange(start_date, end_date, start_hhmm, end_hhmm, tz_mode):
    """Return a list of UTC datetimes at 15-min increments + local label list."""
    s_local = datetime.combine(start_date, datetime.strptime(start_hhmm, "%H:%M").time())
    e_local = datetime.combine(end_date, datetime.strptime(end_hhmm, "%H:%M").time())

    # Ensure forward range
    if e_local < s_local:
        e_local = s_local

    # Build local naive range
    local_times = pd.date_range(s_local, e_local, freq="15min").to_pydatetime().tolist()

    if tz_mode == "UTC":
        # Interpret chosen times as UTC (rare, but keep option)
        utc_times = [dt.replace(tzinfo=ZoneInfo("UTC")) for dt in local_times]
        labels = [dt.strftime("%H:%M") for dt in local_times]
        return utc_times, labels

    # Interpret chosen times as America/New_York and convert to UTC
    utc_times = [local_dt_to_utc(dt) for dt in local_times]
    labels = [dt.strftime("%H:%M") for dt in local_times]
    return utc_times, labels


# -----------------------------
# 5) FLOATING HAMBURGER (sidebar toggle)
# -----------------------------
with st.container():
    st.markdown('<div class="floating-hamburger">', unsafe_allow_html=True)
    if st.button("☰", key="toggle_sidebar"):
        st.session_state.sidebar_collapsed = not st.session_state.sidebar_collapsed
        st.experimental_rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# 6) SIDEBAR
# -----------------------------
with st.sidebar:
    st.title("CNR GIS Portal")

    tz_mode = st.radio("Timezone", ["Local (EST/EDT)", "UTC"], index=0)

    # Dates
    s_date = st.date_input("Start Date", value=datetime.now().date())
    e_date = st.date_input("End Date", value=datetime.now().date())

    # Hours
    c1, c2 = st.columns(2)
    hours = [f"{h:02d}:00" for h in range(24)]
    s_time = c1.selectbox("Start", hours, index=19)
    e_time = c2.selectbox("End", hours, index=21)

    st.write("")

    # Upload watershed zip (shapefile)
    up_zip = st.file_uploader("Upload Watershed ZIP", type="zip")

    if up_zip:
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(up_zip, "r") as z:
                z.extractall(td)
            shps = list(Path(td).rglob("*.shp"))
            if shps:
                gdf = gpd.read_file(shps[0])
                gdf = gdf.to_crs("EPSG:4326")
                st.session_state.active_gdf = gdf

                b = gdf.total_bounds  # (minx, miny, maxx, maxy)
                st.session_state.map_view = pdk.ViewState(
                    latitude=(b[1] + b[3]) / 2,
                    longitude=(b[0] + b[2]) / 2,
                    zoom=11
                )

    st.write("")
    st.write("")

    # Process button
    if st.button("PROCESS RADAR DATA"):
        if st.session_state.active_gdf is None:
            st.warning("Upload a watershed ZIP first.")
        else:
            st.session_state.show_loading = True
            st.experimental_rerun()

    # If we re-ran with loading enabled, do the heavy work now
    if st.session_state.show_loading:
        try:
            bbox = st.session_state.active_gdf.total_bounds if st.session_state.active_gdf is not None else None

            utc_times, labels = build_timerange(
                s_date, e_date, s_time, e_time,
                "UTC" if tz_mode == "UTC" else "LOCAL"
            )

            cache = {}
            stats = []
            pb = st.progress(0)

            for i, (dt_utc, label) in enumerate(zip(utc_times, labels)):
                path, val_in, bnds = get_radar_frame(dt_utc, bbox=bbox)
                if path and bnds:
                    cache[label] = {"path": path, "bounds": bnds}
                    stats.append({"time": label, "rain_in": val_in})

                pb.progress((i + 1) / max(1, len(utc_times)))

            st.session_state.radar_cache = cache
            st.session_state.time_list = list(cache.keys())
            st.session_state.processed_df = pd.DataFrame(stats)

            # reset timeline index
            st.session_state.t_idx = 0
            st.session_state.playing = False

        finally:
            st.session_state.show_loading = False
            st.experimental_rerun()

    # Polished controls (playback + plot)
    if st.session_state.time_list:
        st.write("---")
        cc1, cc2 = st.columns([1, 1])
        if cc1.button("▶ PLAY" if not st.session_state.playing else "⏸ PAUSE"):
            st.session_state.playing = not st.session_state.playing
            st.experimental_rerun()
        if cc2.button("⟲ RESET"):
            st.session_state.t_idx = 0
            st.session_state.playing = False
            st.experimental_rerun()

    if st.session_state.processed_df is not None and not st.session_state.processed_df.empty:
        st.write("---")
        if st.button("SHOW PLOT"):
            import plotly.express as px

            @st.dialog("Rainfall Statistics", width="large")
            def modal():
                df = st.session_state.processed_df.copy()
                fig = px.bar(df, x="time", y="rain_in", template="plotly_dark")
                fig.update_layout(bargap=0, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

            modal()


# -----------------------------
# 7) TIMELINE / AUTOPLAY (polish)
# -----------------------------
if st.session_state.time_list:
    n = len(st.session_state.time_list)
    st.session_state.t_idx = int(np.clip(st.session_state.t_idx, 0, max(0, n - 1)))

    # If playing, advance and rerun (simple autoplay)
    if st.session_state.playing:
        st.session_state.t_idx = (st.session_state.t_idx + 1) % n
        time.sleep(0.35)  # smooth-ish animation; adjust as desired
        st.experimental_rerun()

    # Floating slider (keeps your "full-screen map" feel)
    t_idx = st.select_slider(
        "Timeline",
        options=list(range(n)),
        value=st.session_state.t_idx,
        format_func=lambda i: st.session_state.time_list[i],
        label_visibility="collapsed",
        key="timeline_slider",
    )
    st.session_state.t_idx = t_idx


# -----------------------------
# 8) BACKGROUND MAP LAYERS
# -----------------------------
layers = []

if st.session_state.time_list:
    key = st.session_state.time_list[st.session_state.t_idx]
    current = st.session_state.radar_cache.get(key)

    if current:
        layers.append(
            pdk.Layer(
                "BitmapLayer",
                image=current["path"],
                bounds=current["bounds"],
                opacity=0.70,
            )
        )

if st.session_state.active_gdf is not None:
    layers.append(
        pdk.Layer(
            "GeoJsonLayer",
            st.session_state.active_gdf.__geo_interface__,
            stroked=True,
            filled=False,
            get_line_color=[255, 255, 255],
            line_width_min_pixels=3,
        )
    )

deck = pdk.Deck(
    layers=layers,
    initial_view_state=st.session_state.map_view,
    map_style="dark",
)

st.pydeck_chart(deck, key="radar_map")
