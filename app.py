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
import calendar

# =============================
# 1) PAGE CONFIG
# =============================
st.set_page_config(layout="wide", page_title="CNR Radar Portal", initial_sidebar_state="expanded")

# =============================
# 2) CSS (fix deckgl stacking + locked sidebar + floating controls)
# =============================
st.markdown(
    """
<style>
/* ---- GLOBAL ---- */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"]{
  height:100vh !important; width:100vw !important;
  margin:0 !important; padding:0 !important;
  overflow:hidden !important;
}
body { background:#000 !important; }

/* remove Streamlit main padding */
.main .block-container{
  padding:0 !important; margin:0 !important;
  max-width:100vw !important;
}

/* hide streamlit chrome */
header, footer, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"]{
  display:none !important; height:0 !important; visibility:hidden !important;
}

/* ---- SIDEBAR (locked) ---- */
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
  z-index:5000 !important;
}
[data-testid="stSidebarResizer"],
[data-testid="collapsedControl"],
button[title="Collapse sidebar"]{
  display:none !important;
}
[data-testid="stSidebarContent"]{
  height:100vh !important;
  overflow:auto !important;
  padding:10px 12px !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{ gap:0.28rem !important; }
[data-testid="stSidebar"] .block-container{ padding:0 !important; margin:0 !important; }

/* ---- KEY FIX: Put Streamlit main layer above DeckGL ---- */
section.main{
  position: relative !important;
  z-index: 3000 !important;
}

/* ---- DeckGL fullscreen offsets fix, but keep it UNDER controls ---- */
#deckgl-wrapper{
  position:fixed !important;
  inset:0 !important;
  width:100vw !important;
  height:100vh !important;
  margin:0 !important;
  padding:0 !important;
  transform:none !important;
  z-index: 1 !important;         /* <<< LOWER than controls */
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

/* ---- Floating controls: target ONLY our main controls ---- */
/* (We keep ALL other buttons in the sidebar.) */
section.main div[data-testid="stButton"]{
  position:fixed !important;
  left:420px !important;
  bottom:18px !important;
  z-index: 12000 !important;

  background: rgba(15,15,15,0.92) !important;
  padding: 10px 12px !important;
  border-radius: 999px !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  backdrop-filter: blur(10px);
}
section.main div[data-testid="stButton"] button{
  border-radius:999px !important;
  width:44px !important;
  height:44px !important;
  padding:0 !important;
  font-size:18px !important;
}

/* slider container */
section.main div[data-testid="stSlider"]{
  position:fixed !important;
  left:480px !important;    /* sidebar 400 + spacing */
  right:140px !important;   /* room for time label */
  bottom:18px !important;
  z-index: 12000 !important;

  background: rgba(15,15,15,0.92) !important;
  padding: 10px 16px !important;
  border-radius: 999px !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  backdrop-filter: blur(10px);
}

/* floating time label */
#time-float{
  position:fixed !important;
  right:22px !important;
  bottom:28px !important;
  z-index: 12001 !important;

  color:#fff !important;
  font-weight:600 !important;
  background:rgba(15,15,15,0.65) !important;
  padding:6px 10px !important;
  border-radius:10px !important;
  border:1px solid rgba(255,255,255,0.10) !important;
  backdrop-filter: blur(10px);
}

/* sidebar buttons nicer */
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
""",
    unsafe_allow_html=True,
)

# =============================
# 3) STATE
# =============================
if "radar_cache" not in st.session_state: st.session_state.radar_cache = {}
if "time_list" not in st.session_state: st.session_state.time_list = []
if "active_gdf" not in st.session_state: st.session_state.active_gdf = None
if "basin_vault" not in st.session_state: st.session_state.basin_vault = {}  # basin_name -> df
if "map_view" not in st.session_state: st.session_state.map_view = pdk.ViewState(latitude=40.7, longitude=-74.0, zoom=9)
if "img_dir" not in st.session_state: st.session_state.img_dir = tempfile.mkdtemp(prefix="radar_png_")
if "is_playing" not in st.session_state: st.session_state.is_playing = False
if "current_time_index" not in st.session_state: st.session_state.current_time_index = 0
if "processing_msg" not in st.session_state: st.session_state.processing_msg = ""

RADAR_COLORS = ["#76fffe", "#01a0fe", "#0001ef", "#01ef01", "#019001", "#ffff01", "#e7c001", "#ff9000", "#ff0101"]
RADAR_CMAP = ListedColormap(RADAR_COLORS)

MRMS_S3_BASE = "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/MultiSensor_QPE_01H_Pass2_00.00"
RO_S3_BASE   = "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/RadarOnly_QPE_15M_00.00"


def csv_download_link(df: pd.DataFrame, filename: str, label: str):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(csv_bytes).decode()
    href = f"data:text/csv;base64,{b64}"
    st.markdown(
        f'<div class="output-link">📄 <a download="{filename}" href="{href}">{label}</a></div>',
        unsafe_allow_html=True,
    )



def ny_is_dst(dt_local: datetime) -> bool:
    """Return True if dt_local is in EDT (DST), else EST, using US rules."""
    year = dt_local.year
    c = calendar.Calendar(firstweekday=6)  # Sunday=0 in monthdayscalendar weeks

    # DST starts: 2nd Sunday in March at 2:00
    march_weeks = c.monthdayscalendar(year, 3)
    march_sundays = [w[0] for w in march_weeks if w[0] != 0]
    dst_start = datetime(year, 3, march_sundays[1], 2, 0)

    # DST ends: 1st Sunday in November at 2:00
    nov_weeks = c.monthdayscalendar(year, 11)
    nov_sundays = [w[0] for w in nov_weeks if w[0] != 0]
    dst_end = datetime(year, 11, nov_sundays[0], 2, 0)

    return dst_start <= dt_local < dst_end

def local_to_utc(dt_local: datetime) -> datetime:
    """Convert NY local naive datetime -> UTC naive datetime."""
    offset = 4 if ny_is_dst(dt_local) else 5
    return dt_local + timedelta(hours=offset)


def ceil_to_hour(dt: datetime) -> datetime:
    """Ceil a naive datetime to the next hour (if not already on the hour)."""
    dt0 = dt.replace(minute=0, second=0, microsecond=0)
    return dt0 if dt == dt0 else (dt0 + timedelta(hours=1))

def open_grib_from_s3(base_url: str, filename: str) -> xr.DataArray:
    """Download gz GRIB2 to temp, open with cfgrib, return DataArray loaded in RAM."""
    full_url = f"{base_url}/{filename[:8]}/{filename}"  # filename starts with YYYYMMDD-...
    tmp_path = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False).name

    r = requests.get(full_url, stream=True, timeout=25)
    if r.status_code == 404:
        return None
    r.raise_for_status()

    try:
        with gzip.GzipFile(fileobj=r.raw) as gz, open(tmp_path, "wb") as f:
            shutil.copyfileobj(gz, f)

        with xr.open_dataset(tmp_path, engine="cfgrib", backend_kwargs={"indexpath": ""}) as ds:
            var = list(ds.data_vars)[0]
            da = ds[var].clip(min=0).load()

        # lon wrap to -180..180
        da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180).sortby("longitude")
        return da
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def load_precip(file_type: str, dt_local: datetime) -> xr.DataArray:
    """
    file_type: "RO" or "MRMS"
    dt_local: naive local (NY) time representing end time (RO timestamp is exact 15-min time, MRMS timestamp is hourly)
    S3 uses UTC-based folder & timestamp; we convert local->UTC for the filename.
    """
    dt_utc = local_to_utc(dt_local)
    ts = dt_utc.strftime("%Y%m%d-%H%M00")
    ymd = dt_utc.strftime("%Y%m%d")

    if file_type == "RO":
        filename = f"MRMS_RadarOnly_QPE_15M_00.00_{ts}.grib2.gz"
        base = RO_S3_BASE
    elif file_type == "MRMS":
        filename = f"MRMS_MultiSensor_QPE_01H_Pass2_00.00_{ts}.grib2.gz"
        base = MRMS_S3_BASE
    else:
        raise ValueError("file_type must be 'RO' or 'MRMS'")

    # Our helper expects path base/YYYYMMDD/filename; build that URL directly:
    full_url = f"{base}/{ymd}/{filename}"

    tmp_path = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False).name
    r = requests.get(full_url, stream=True, timeout=25)
    if r.status_code == 404:
        return None
    r.raise_for_status()

    try:
        with gzip.GzipFile(fileobj=r.raw) as gz, open(tmp_path, "wb") as f:
            shutil.copyfileobj(gz, f)

        with xr.open_dataset(tmp_path, engine="cfgrib", backend_kwargs={"indexpath": ""}) as ds:
            var = list(ds.data_vars)[0]
            da = ds[var].clip(min=0).load()

        da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180).sortby("longitude")
        da = da.rio.write_crs("EPSG:4326")
        return da
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def save_frame_png(da: xr.DataArray, dt_local: datetime) -> tuple[str, list]:
    """Save raster to PNG and return (path, bounds)."""
    data = da.values.astype("float32")
    data[data < 0.1] = np.nan

    img_path = os.path.join(st.session_state.img_dir, f"radar_{dt_local.strftime('%Y%m%d_%H%M')}.png")
    plt.imsave(img_path, data, cmap=RADAR_CMAP, vmin=0.1, vmax=15.0)

    bounds = [
        float(da.longitude.min()),
        float(da.latitude.min()),
        float(da.longitude.max()),
        float(da.latitude.max()),
    ]
    return img_path, bounds

# =============================
# 4) SIDEBAR
# =============================
with st.sidebar:
    st.title("CNR GIS Portal")

    s_date = st.date_input("Start Date", value=datetime.now().date())
    e_date = st.date_input("End Date", value=datetime.now().date())

    c1, c2 = st.columns(2)
    hours = [f"{h:02d}:00" for h in range(24)]
    s_time = c1.selectbox("Start Time", hours, index=19)
    e_time = c2.selectbox("End Time", hours, index=21)

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
                    zoom=11,
                )

    if st.session_state.processing_msg:
        st.caption(st.session_state.processing_msg)

    if st.button("Process Radar Data", use_container_width=True):
        if st.session_state.active_gdf is None:
            st.session_state.processing_msg = "Please upload a watershed boundary ZIP first."
            st.rerun()

        # user input local start/end (naive)
        start_dt = datetime.combine(s_date, datetime.strptime(s_time, "%H:%M").time())
        end_dt   = datetime.combine(e_date, datetime.strptime(e_time, "%H:%M").time())

        # ---------
        # ALIGNMENT (matches your notebook intent)
        # RO: starts at start + 15 minutes
        # MRMS: starts at first full hour >= start + 45 minutes
        # ---------
        ro_start   = start_dt + timedelta(minutes=15)
        ro_end     = end_dt
        mrms_start = ceil_to_hour(start_dt + timedelta(minutes=45))
        mrms_end   = end_dt.replace(minute=0, second=0, microsecond=0)

        # build time lists
        ro_times = list(pd.date_range(ro_start, ro_end, freq="15min"))
        mrms_times = list(pd.date_range(mrms_start, mrms_end, freq="1H"))

        cache = {}
        stats = []

        pb = st.progress(0.0)
        msg = st.empty()

        # 1) download RO 15-min frames
        ro_list = []
        kept_ro_times = []
        for i, t in enumerate(ro_times):
            msg.info(f"Downloading RadarOnly 15-min ({i+1}/{len(ro_times)}) — {t.strftime('%Y-%m-%d %H:%M')}")
            da = load_precip("RO", t)
            if da is None:
                pb.progress((i + 1) / (len(ro_times) + len(mrms_times)))
                continue

            # clip to watershed
            sub = da.rio.write_crs("EPSG:4326", inplace=False)
            clipped = sub.rio.clip(st.session_state.active_gdf.geometry, st.session_state.active_gdf.crs, drop=False)
            clipped = clipped.fillna(0)

            ro_list.append(clipped)
            kept_ro_times.append(pd.Timestamp(t).to_pydatetime())
            pb.progress((i + 1) / (len(ro_times) + len(mrms_times)))

        if len(ro_list) < 4:
            msg.error("Not enough RadarOnly frames found for the selected period.")
            st.session_state.processing_msg = "No valid RadarOnly frames."
            st.rerun()

        ro = xr.concat(ro_list, dim="time").assign_coords(time=kept_ro_times)

        # 2) download MRMS hourly frames
        mrms_list = []
        kept_mrms_times = []
        for j, t in enumerate(mrms_times):
            msg.info(f"Downloading MRMS 1-hr ({j+1}/{len(mrms_times)}) — {t.strftime('%Y-%m-%d %H:%M')}")
            da = load_precip("MRMS", t)
            if da is None:
                pb.progress((len(ro_times) + j + 1) / (len(ro_times) + len(mrms_times)))
                continue

            sub = da.rio.write_crs("EPSG:4326", inplace=False)
            clipped = sub.rio.clip(st.session_state.active_gdf.geometry, st.session_state.active_gdf.crs, drop=False)
            clipped = clipped.fillna(0)

            mrms_list.append(clipped)
            kept_mrms_times.append(pd.Timestamp(t).to_pydatetime())
            pb.progress((len(ro_times) + j + 1) / (len(ro_times) + len(mrms_times)))

        if len(mrms_list) == 0:
            msg.error("No MRMS hourly frames found for the aligned hourly window.")
            st.session_state.processing_msg = "No valid MRMS frames."
            st.rerun()

        mrms = xr.concat(mrms_list, dim="time").assign_coords(time=kept_mrms_times)

        # 3) Build RO hourly sums aligned to MRMS times
        # For each MRMS time T: sum RO over [T-45min, T] => 4 frames
        ro_hourly = []
        for T in mrms.time.values:
            Tdt = pd.to_datetime(str(T)).to_pydatetime()
            block = ro.sel(time=slice(Tdt - timedelta(minutes=45), Tdt)).sum(dim="time")
            ro_hourly.append(block)
        ro_hourly_da = xr.concat(ro_hourly, dim="time").assign_coords(time=mrms.time.values)

        # 4) Cell-wise scaling rasters per hour: MRMS / RO_hourly
        ratio_list = []
        for i in range(len(mrms.time)):
            mrms_slice = mrms.isel(time=i)
            ro_slice   = ro_hourly_da.isel(time=i)

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = mrms_slice / ro_slice
                ratio = ratio.where(np.isfinite(ratio), 0)

            ratio_list.append(ratio)

        scaling_da = xr.concat(ratio_list, dim="time").assign_coords(time=mrms.time.values)

        # 5) Build scaled RO 15-min using hourly scaling raster
        # We assign each RO frame to an hourly bin based on which MRMS time window it belongs to:
        # If MRMS time is T, then RO frames in (T-45, T] get scaling raster of that hour.
        ro_scaled_list = []
        ro_scaled_times = []

        for t in ro.time.values:
            tdt = pd.to_datetime(str(t)).to_pydatetime()

            # find the MRMS hour whose window ends at T such that tdt in [T-45, T]
            # easiest: choose the smallest MRMS time >= tdt, but ensure within 45 min window
            candidates = [pd.to_datetime(str(x)).to_pydatetime() for x in mrms.time.values]
            # first hour end >= tdt
            hour_end = next((T for T in candidates if T >= tdt), None)
            if hour_end is None:
                continue
            if tdt < hour_end - timedelta(minutes=45):
                continue  # not within that hour's 4-frame block

            hour_idx = candidates.index(hour_end)
            scale_raster = scaling_da.isel(time=hour_idx)

            ro_slice_15 = ro.sel(time=tdt)
            scaled = ro_slice_15 * scale_raster
            ro_scaled_list.append(scaled)
            ro_scaled_times.append(tdt)

        if len(ro_scaled_list) == 0:
            msg.error("Scaling produced no valid RO frames (check time alignment / data availability).")
            st.session_state.processing_msg = "Scaling failed."
            st.rerun()

        ro_scaled_da = xr.concat(ro_scaled_list, dim="time").assign_coords(time=ro_scaled_times)

        # 6) Render PNG frames + build cache/time_list + mean series
        cache = {}
        stats = []

        msg.info("Rendering frames for map playback…")
        for i, tdt in enumerate(ro_scaled_da.time.values):
            dt_local = pd.to_datetime(str(tdt)).to_pydatetime()

            da = ro_scaled_da.sel(time=dt_local)
            # NOTE: da is already clipped to watershed earlier in pipeline
            site_mean_in = float(da.mean().values) / 25.4  # mm -> inches (mean depth)

            img_path, bounds = save_frame_png(da, dt_local)

            label = dt_local.strftime("%Y-%m-%d %H:%M")
            cache[label] = {"path": img_path, "bounds": bounds}
            stats.append({"time": dt_local, "rain_in": site_mean_in})

        st.session_state.radar_cache = cache
        st.session_state.time_list = list(cache.keys())
        st.session_state.current_time_index = 0
        st.session_state.basin_vault[basin_name] = pd.DataFrame(stats)

        msg.success(f"Complete: {len(cache)} scaled RO frames • MRMS hours used: {len(mrms.time)}")
        st.session_state.processing_msg = f"Last run: {basin_name} • {len(cache)} frames (RO→MRMS scaled)"
        st.rerun()

    # outputs
    if st.session_state.basin_vault:
        st.divider()
        st.subheader("Outputs")
        for name, df in st.session_state.basin_vault.items():
            csv_download_link(df, filename=f"{name}.csv", label=f"{name}.csv")

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

# =============================
# 5) ANIMATION LOGIC
# =============================
if st.session_state.time_list and st.session_state.is_playing:
    st.session_state.current_time_index = (st.session_state.current_time_index + 1) % len(st.session_state.time_list)
    time.sleep(0.5)
    st.rerun()

# =============================
# 6) MAP RENDER
# =============================
layers = []

if st.session_state.time_list:
    current_key = st.session_state.time_list[st.session_state.current_time_index]
    curr = st.session_state.radar_cache[current_key]
    layers.append(pdk.Layer("BitmapLayer", image=curr["path"], bounds=curr["bounds"], opacity=0.70))

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

# =============================
# 7) CONTROLS (this is your “radar view” slider)
# =============================
# IMPORTANT: Keep ONLY these widgets in main. Everything else stays in sidebar.
if st.session_state.time_list:
    # Play/pause
    if st.session_state.is_playing:
        if st.button("⏸", key="pause_btn"):
            st.session_state.is_playing = False
            st.rerun()
    else:
        if st.button("▶", key="play_btn"):
            st.session_state.is_playing = True
            st.rerun()

    # Slider: index-based so it stays stable
    selected_index = st.select_slider(
        " ",
        options=range(len(st.session_state.time_list)),
        value=st.session_state.current_time_index,
        format_func=lambda i: st.session_state.time_list[i],
        label_visibility="collapsed",
        key="timeline_slider",
    )

    if selected_index != st.session_state.current_time_index:
        st.session_state.current_time_index = selected_index
        st.session_state.is_playing = False
        st.rerun()

    st.markdown(
        f'<div id="time-float">{st.session_state.time_list[st.session_state.current_time_index]}</div>',
        unsafe_allow_html=True,
    )

