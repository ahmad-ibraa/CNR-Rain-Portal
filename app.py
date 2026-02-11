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

# -----------------------------
# 1) PAGE CONFIG
# -----------------------------
st.set_page_config(layout="wide", page_title="CNR Radar Portal", initial_sidebar_state="expanded")

# -----------------------------
# 2) CSS (fullscreen map + locked sidebar + floating controls)
# -----------------------------
st.markdown("""
<style>
/* ---- GLOBAL: no scrollbars ---- */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"]{
  height:100vh !important; width:100vw !important;
  margin:0 !important; padding:0 !important;
  overflow:hidden !important;
}
body { background:#000 !important; }

/* remove main padding */
.main .block-container{
  padding:0 !important; margin:0 !important;
  max-width:100vw !important;
}

/* hide streamlit chrome */
header, footer, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"]{
  display:none !important; height:0 !important; visibility:hidden !important;
}

/* ---- SIDEBAR locked ---- */
[data-testid="stSidebar"]{
  position:fixed !important;
  left:0 !important; top:0 !important;
  height:100vh !important;

  min-width:400px !important;
  max-width:400px !important;
  width:400px !important;

  background:rgba(17,17,17,0.95) !important;
  backdrop-filter: blur(10px);
  border-right:1px solid rgba(255,255,255,0.08);
  z-index:4000 !important;
}
[data-testid="stSidebarResizer"],
[data-testid="collapsedControl"],
button[title="Collapse sidebar"]{
  display:none !important;
}

/* tighten sidebar spacing */
[data-testid="stSidebarContent"]{
  height:100vh !important;
  overflow:auto !important;
  padding:10px 12px !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{ gap:0.30rem !important; }
[data-testid="stSidebar"] .block-container{ padding:0 !important; margin:0 !important; }

/* ---- DECK.GL fullscreen (this is what fixed your offsets) ---- */
#deckgl-wrapper{
  position:fixed !important;
  inset:0 !important;
  width:100vw !important;
  height:100vh !important;
  margin:0 !important;
  padding:0 !important;
  transform:none !important;
  z-index:0 !important;
}
#view-default-view,
#view-default-view > div,
#view-default-view .mapboxgl-map,
#view-default-view .mapboxgl-canvas-container,
#view-default-view .overlays{
  position:absolute !important;
  inset:0 !important;
  width:100% !important;
  height:100% !important;
}
canvas#deckgl-overlay,
canvas.mapboxgl-canvas{
  position:absolute !important;
  inset:0 !important;
  width:100% !important;
  height:100% !important;
}

/* ---- FLOATING CONTROLS ---- */
/* Float the Streamlit element container that contains our anchor */
div[data-testid="element-container"]:has(#controls-anchor){
  position:fixed !important;
  left:420px !important;
  right:18px !important;
  bottom:18px !important;

  z-index:9999 !important;
  background:rgba(15,15,15,0.92) !important;
  padding:12px 18px !important;
  border-radius:999px !important;
  border:1px solid rgba(255,255,255,0.12);
  backdrop-filter: blur(10px);

  display:block !important;
  pointer-events:auto !important;
}
div[data-testid="element-container"]:has(#controls-anchor) *{
  visibility:visible !important;
  pointer-events:auto !important;
}
div[data-testid="element-container"]:has(#controls-anchor) .stButton button{
  border-radius:999px !important;
  width:44px !important;
  height:44px !important;
  padding:0 !important;
  font-size:18px !important;
}

/* nicer primary buttons in sidebar */
[data-testid="stSidebar"] .stButton button{
  width:100% !important;
  border-radius:10px !important;
  height:44px !important;
  font-weight:600 !important;
}

/* clickable output links */
.output-link a{
  text-decoration:none !important;
  font-weight:600 !important;
}
.output-link a:hover{
  text-decoration:underline !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 3) STATE
# -----------------------------
if "radar_cache" not in st.session_state: st.session_state.radar_cache = {}
if "time_list" not in st.session_state: st.session_state.time_list = []
if "active_gdf" not in st.session_state: st.session_state.active_gdf = None
if "basin_vault" not in st.session_state: st.session_state.basin_vault = {}   # basin_name -> df
if "map_view" not in st.session_state: st.session_state.map_view = pdk.ViewState(latitude=40.7, longitude=-74.0, zoom=9)
if "img_dir" not in st.session_state: st.session_state.img_dir = tempfile.mkdtemp(prefix="radar_png_")
if "is_playing" not in st.session_state: st.session_state.is_playing = False
if "current_time_index" not in st.session_state: st.session_state.current_time_index = 0
if "processing_msg" not in st.session_state: st.session_state.processing_msg = ""

RADAR_COLORS = ['#76fffe', '#01a0fe', '#0001ef', '#01ef01', '#019001', '#ffff01', '#e7c001', '#ff9000', '#ff0101']
RADAR_CMAP = ListedColormap(RADAR_COLORS)

def csv_download_link(df: pd.DataFrame, filename: str, label: str):
    """Create a 'hyperlink' that downloads a CSV via a data URL."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(csv_bytes).decode()
    href = f'data:text/csv;base64,{b64}'
    st.markdown(
        f'<div class="output-link">📄 <a download="{filename}" href="{href}">{label}</a></div>',
        unsafe_allow_html=True
    )

# -----------------------------
# 4) RADAR ENGINE
# -----------------------------
def get_radar_image(dt_utc):
    ts_str = dt_utc.strftime("%Y%m%d-%H%M00")
    url = (
        "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/"
        f"RadarOnly_QPE_15M_00.00/{dt_utc.strftime('%Y%m%d')}/"
        f"MRMS_RadarOnly_QPE_15M_00.00_{ts_str}.grib2.gz"
    )
    tmp_grib = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False).name

    try:
        r = requests.get(url, stream=True, timeout=15)
        if r.status_code != 200:
            return None, 0.0, None

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

            # IMPORTANT: include date to avoid overwriting when running multiple days
            img_path = os.path.join(st.session_state.img_dir, f"radar_{dt_utc.strftime('%Y%m%d_%H%M')}.png")
            plt.imsave(img_path, data, cmap=RADAR_CMAP, vmin=0.1, vmax=15.0)

            bounds = [
                float(subset.longitude.min()), float(subset.latitude.min()),
                float(subset.longitude.max()), float(subset.latitude.max())
            ]
            return img_path, max(0.0, site_mean / 25.4), bounds

    except Exception:
        return None, 0.0, None
    finally:
        if os.path.exists(tmp_grib):
            os.remove(tmp_grib)

# -----------------------------
# 5) SIDEBAR UI
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

    up_zip = st.file_uploader("Watershed Boundary (ZIP)", type="zip")
    basin_name = "Default_Basin"

    if up_zip:
        basin_name = up_zip.name.replace(".zip", "")
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(up_zip, "r") as z:
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

    # status label
    if st.session_state.processing_msg:
        st.caption(st.session_state.processing_msg)

    if st.button("Process Radar Data", use_container_width=True):
        if st.session_state.active_gdf is None:
            st.session_state.processing_msg = "Upload a watershed ZIP first."
            st.rerun()

        s_dt = datetime.combine(s_date, datetime.strptime(s_time, "%H:%M").time())
        e_dt = datetime.combine(e_date, datetime.strptime(e_time, "%H:%M").time())
        tr = pd.date_range(s_dt, e_dt, freq="15min")

        cache, stats = {}, []
        pb = st.progress(0.0)
        msg = st.empty()

        for i, ts in enumerate(tr):
            msg.info(f"Downloading & rendering MRMS frame {i+1}/{len(tr)} ({ts.strftime('%Y-%m-%d %H:%M')}) …")
            ts_utc = ts if tz_mode == "UTC" else ts + timedelta(hours=5)

            path, val, bnds = get_radar_image(ts_utc)
            if path and bnds:
                cache[ts.strftime("%H:%M")] = {"path": path, "bounds": bnds}
                stats.append({"time": ts, "rain_in": val})

            pb.progress((i + 1) / len(tr))

        msg.success(f"Complete: {len(cache)} frames processed for {basin_name}.")
        st.session_state.processing_msg = f"Last run: {basin_name} ({len(cache)} frames)"
        st.session_state.radar_cache = cache
        st.session_state.time_list = list(cache.keys())
        st.session_state.current_time_index = 0
        st.session_state.basin_vault[basin_name] = pd.DataFrame(stats)

        st.rerun()

    # outputs list (click-to-download)
    if st.session_state.basin_vault:
        st.divider()
        st.subheader("Outputs")

        # show all outputs as clickable downloads
        for name, df in st.session_state.basin_vault.items():
            csv_download_link(df, filename=f"{name}.csv", label=f"{name}.csv")

        # optional plot viewer for selected output
        st.divider()
        pick = st.selectbox("Rainfall Summary", options=list(st.session_state.basin_vault.keys()))
        if st.button("View Rainfall Summary", use_container_width=True):
            import plotly.express as px
            df_target = st.session_state.basin_vault[pick]

            @st.dialog(f"Rainfall Summary — {pick}", width="large")
            def modal():
                fig = px.bar(df_target, x="time", y="rain_in", template="plotly_dark")
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

            modal()

# -----------------------------
# 6) ANIMATION LOGIC
# -----------------------------
if st.session_state.time_list and st.session_state.is_playing:
    st.session_state.current_time_index = (st.session_state.current_time_index + 1) % len(st.session_state.time_list)
    time.sleep(0.5)
    st.rerun()

# -----------------------------
# 7) MAP RENDER
# -----------------------------
layers = []
if st.session_state.time_list:
    current_time_str = st.session_state.time_list[st.session_state.current_time_index]
    curr = st.session_state.radar_cache[current_time_str]
    layers.append(
        pdk.Layer("BitmapLayer", image=curr["path"], bounds=curr["bounds"], opacity=0.70)
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
    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
)

st.pydeck_chart(deck, use_container_width=True, height=1000)

# -----------------------------
# 8) CONTROLS (floating via anchor)
# -----------------------------
if st.session_state.time_list:
    with st.container():
        st.markdown('<div id="controls-anchor"></div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 10, 2])

        with col1:
            if st.session_state.is_playing:
                if st.button("⏸", key="pause_btn"):
                    st.session_state.is_playing = False
                    st.rerun()
            else:
                if st.button("▶", key="play_btn"):
                    st.session_state.is_playing = True
                    st.rerun()

        with col2:
            selected_index = st.select_slider(
                "",
                options=range(len(st.session_state.time_list)),
                value=st.session_state.current_time_index,
                format_func=lambda x: st.session_state.time_list[x],
                label_visibility="collapsed",
            )
            if selected_index != st.session_state.current_time_index:
                st.session_state.current_time_index = selected_index
                st.session_state.is_playing = False
                st.rerun()

        with col3:
            st.markdown(f"**{st.session_state.time_list[st.session_state.current_time_index]}**")
