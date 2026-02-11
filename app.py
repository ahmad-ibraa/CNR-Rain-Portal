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

# ---- robust timezone handling (no manual DST math) ----
from zoneinfo import ZoneInfo

NY_TZ  = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

# =============================
# 1) PAGE CONFIG
# =============================
st.set_page_config(layout="wide", page_title="CNR Radar Portal", initial_sidebar_state="expanded")

# =============================
# 2) CSS
#   - keep deckgl fullscreen
#   - ensure controls sit ABOVE deckgl
#   - DO NOT use :has()
# =============================
st.markdown(
    """
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"]{
  height:100vh !important; width:100vw !important;
  margin:0 !important; padding:0 !important;
  overflow:hidden !important;
}
body { background:#000 !important; }

.main .block-container{
  padding:0 !important; margin:0 !important;
  max-width:100vw !important;
}

header, footer, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"]{
  display:none !important; height:0 !important; visibility:hidden !important;
}

/* Sidebar locked */
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

/* DeckGL fullscreen, UNDER controls */
#deckgl-wrapper{
  position:fixed !important;
  inset:0 !important;
  width:100vw !important;
  height:100vh !important;
  margin:0 !important;
  padding:0 !important;
  transform:none !important;
  z-index: 1 !important;
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

/* --- Floating control bar --- */
.control-bar{
  position: fixed !important;
  left: 420px !important;   /* sidebar 400 + 20 */
  right: 18px !important;
  bottom: 18px !important;
  z-index: 12000 !important;

  background: rgba(15,15,15,0.92) !important;
  padding: 12px 16px !important;
  border-radius: 999px !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  backdrop-filter: blur(10px);
}
.control-bar *{
  pointer-events: auto !important;
}
.control-bar .stButton button{
  border-radius:999px !important;
  width:44px !important;
  height:44px !important;
  padding:0 !important;
  font-size:18px !important;
}

/* Output links */
.output-link a{
  text-decoration:none !important;
  font-weight:600 !important;
}
.output-link a:hover{ text-decoration:underline !important; }

/* Sidebar buttons */
[data-testid="stSidebar"] .stButton button{
  width:100% !important;
  border-radius:10px !important;
  height:44px !important;
  font-weight:650 !important;
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
if "ws_bounds" not in st.session_state: st.session_state.ws_bounds = None

RADAR_COLORS = ["#76fffe", "#01a0fe", "#0001ef", "#01ef01", "#019001", "#ffff01", "#e7c001", "#ff9000", "#ff0101"]
RADAR_CMAP = ListedColormap(RADAR_COLORS)

MRMS_S3_BASE = "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/MultiSensor_QPE_01H_Pass2_00.00"
RO_S3_BASE   = "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/RadarOnly_QPE_15M_00.00"

# =============================
# 4) HELPERS
# =============================
def csv_download_link(df: pd.DataFrame, filename: str, label: str):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(csv_bytes).decode()
    href = f"data:text/csv;base64,{b64}"
    st.markdown(
        f'<div class="output-link">📄 <a download="{filename}" href="{href}">{label}</a></div>',
        unsafe_allow_html=True,
    )

def normalize_grid(da: xr.DataArray) -> xr.DataArray:
    keep = {"latitude", "longitude"}
    drop = [c for c in da.coords if c not in keep]
    da = da.drop_vars(drop, errors="ignore")

    for d in list(da.dims):
        if d not in ("latitude", "longitude"):
            da = da.squeeze(d, drop=True)

    # consistent ordering
    da = da.sortby("latitude", ascending=False).sortby("longitude")
    return da

def local_naive_to_utc(dt_local_naive: datetime) -> datetime:
    """NY local naive -> UTC naive, correct DST via zoneinfo."""
    aware_local = dt_local_naive.replace(tzinfo=NY_TZ)
    aware_utc = aware_local.astimezone(UTC_TZ)
    return aware_utc.replace(tzinfo=None)

def ceil_to_hour(dt: datetime) -> datetime:
    dt0 = dt.replace(minute=0, second=0, microsecond=0)
    return dt0 if dt == dt0 else (dt0 + timedelta(hours=1))

def load_precip(file_type: str, dt_local_naive: datetime) -> xr.DataArray | None:
    """
    dt_local_naive: local NY time (naive) representing the product timestamp/end time.
    NOAA MRMS S3 naming is UTC.
    """
    dt_utc = local_naive_to_utc(dt_local_naive)
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

    url = f"{base}/{ymd}/{filename}"
    tmp_path = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False).name

    r = requests.get(url, stream=True, timeout=25)
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
        da = normalize_grid(da)
        return da
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def drop_time_coord(da: xr.DataArray) -> xr.DataArray:
    return da.drop_vars("time", errors="ignore")

def save_frame_png(da: xr.DataArray, dt_local_naive: datetime) -> tuple[str, list]:
    data = da.values.astype("float32")
    data[data < 0.1] = np.nan
    img_path = os.path.join(st.session_state.img_dir, f"radar_{dt_local_naive.strftime('%Y%m%d_%H%M')}.png")
    plt.imsave(img_path, data, cmap=RADAR_CMAP, vmin=0.1, vmax=15.0)

    bounds = [
        float(da.longitude.min()),
        float(da.latitude.min()),
        float(da.longitude.max()),
        float(da.latitude.max()),
    ]
    return img_path, bounds

def watershed_mean_inch(da_full: xr.DataArray, ws_gdf: gpd.GeoDataFrame) -> float:
    """Compute mean over watershed, but do NOT affect displayed raster."""
    if ws_gdf is None:
        return float(da_full.mean().values) / 25.4

    sub = da_full.rio.write_crs("EPSG:4326", inplace=False)
    try:
        clipped = sub.rio.clip(ws_gdf.geometry, ws_gdf.crs, drop=True)
        if clipped.size == 0:
            return float(da_full.mean().values) / 25.4
        return float(clipped.mean().values) / 25.4
    except Exception:
        return float(da_full.mean().values) / 25.4

# =============================
# 5) SIDEBAR UI
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
                st.session_state.ws_bounds = b  # only for zoom framing if you want
                st.session_state.map_view = pdk.ViewState(
                    latitude=(b[1] + b[3]) / 2,
                    longitude=(b[0] + b[2]) / 2,
                    zoom=11,
                )

    if st.session_state.processing_msg:
        st.caption(st.session_state.processing_msg)

    if st.button("Run Processing", use_container_width=True):
        try:
            if st.session_state.active_gdf is None:
                st.session_state.processing_msg = "Upload a watershed boundary ZIP first."
                st.rerun()

            start_dt = datetime.combine(s_date, datetime.strptime(s_time, "%H:%M").time())
            end_dt   = datetime.combine(e_date, datetime.strptime(e_time, "%H:%M").time())

            # alignment (your rule)
            ro_start   = start_dt + timedelta(minutes=15)
            ro_end     = end_dt
            mrms_start = ceil_to_hour(start_dt + timedelta(minutes=45))
            mrms_end   = end_dt.replace(minute=0, second=0, microsecond=0)

            ro_times   = list(pd.date_range(ro_start, ro_end, freq="15min"))
            mrms_times = list(pd.date_range(mrms_start, mrms_end, freq="1H"))

            pb = st.progress(0.0)
            msg = st.empty()

            total_steps = max(1, len(ro_times) + len(mrms_times) + 30)
            step_i = 0

            # ---------- 1) Download RO (NO CLIP) ----------
            ro_list, ro_kept = [], []
            for i, t in enumerate(ro_times):
                msg.info(f"RadarOnly 15-min → {i+1}/{len(ro_times)} • {t:%Y-%m-%d %H:%M} (NY local)")
                da = load_precip("RO", t)
                step_i += 1; pb.progress(min(1.0, step_i/total_steps))
                if da is None:
                    continue
                ro_list.append(da.astype("float32"))
                ro_kept.append(pd.Timestamp(t).to_pydatetime())

            if len(ro_list) < 4:
                raise RuntimeError("Not enough RadarOnly frames found in the selected period.")

            ro = xr.concat(ro_list, dim="time").assign_coords(time=ro_kept)

            # ---------- 2) Download MRMS (NO CLIP) ----------
            mrms_list, mrms_kept = [], []
            for j, t in enumerate(mrms_times):
                msg.info(f"MRMS 1-hr → {j+1}/{len(mrms_times)} • {t:%Y-%m-%d %H:%M} (NY local)")
                da = load_precip("MRMS", t)
                step_i += 1; pb.progress(min(1.0, step_i/total_steps))
                if da is None:
                    continue
                mrms_list.append(da.astype("float32"))
                mrms_kept.append(pd.Timestamp(t).to_pydatetime())

            if len(mrms_list) == 0:
                raise RuntimeError("No MRMS hourly frames found for the aligned window.")

            mrms = xr.concat(mrms_list, dim="time").assign_coords(time=mrms_kept)

            # ---------- 3) Force RO grid to match MRMS grid (spatial only) ----------
            msg.info("Aligning RO grid to MRMS grid…")
            ro = ro.interp_like(mrms.isel(time=0), method="nearest")
            ro = ro.astype("float32")
            mrms = mrms.astype("float32")
            step_i += 3; pb.progress(min(1.0, step_i/total_steps))

            # ---------- 4) RO hourly sums aligned to MRMS (require 4 frames) ----------
            msg.info("Computing RO hourly sums aligned to MRMS…")
            ro_hourly, valid_times = [], []
            for T in mrms.time.values:
                Tdt = pd.to_datetime(str(T)).to_pydatetime()
                block = ro.sel(time=slice(Tdt - timedelta(minutes=45), Tdt))
                if block.sizes.get("time", 0) != 4:
                    continue
                ro_hourly.append(block.sum(dim="time").astype("float32"))
                valid_times.append(Tdt)

            if not valid_times:
                raise RuntimeError("No MRMS hours had a complete 4-frame RO block to scale.")

            mrms = mrms.sel(time=pd.to_datetime(valid_times))
            ro_hourly_da = xr.concat(ro_hourly, dim="time").assign_coords(time=pd.to_datetime(valid_times))
            step_i += 5; pb.progress(min(1.0, step_i/total_steps))

            # ---------- 5) Scaling rasters (MRMS / RO_hourly) ----------
            msg.info("Computing hourly scaling rasters (MRMS / RO_hourly)…")
            eps = 0.01  # mm
            ratio_list = []
            for i in range(len(mrms.time)):
                mrms_slice = mrms.isel(time=i)
                ro_slice   = ro_hourly_da.isel(time=i).interp_like(mrms_slice, method="nearest")

                ratio = xr.where(ro_slice > eps, mrms_slice / ro_slice, 0.0)
                ratio = ratio.where(np.isfinite(ratio), 0.0)
                ratio = ratio.clip(min=0.0, max=50.0).astype("float32")
                ratio_list.append(ratio)

            scaling_da = xr.concat(ratio_list, dim="time").assign_coords(time=mrms.time.values)
            step_i += 8; pb.progress(min(1.0, step_i/total_steps))

            # ---------- 6) Scale each RO 15-min frame using the hour-end bin ----------
            msg.info("Applying hourly scaling to 15-min RO frames…")
            mrms_ends = [pd.to_datetime(str(x)).to_pydatetime() for x in mrms.time.values]

            ro_scaled_list, ro_scaled_times = [], []
            for t in ro.time.values:
                tdt = pd.to_datetime(str(t)).to_pydatetime()
                hour_end = next((T for T in mrms_ends if T >= tdt), None)
                if hour_end is None:
                    continue
                if tdt < hour_end - timedelta(minutes=45):
                    continue

                hour_idx = mrms_ends.index(hour_end)

                ro_slice_15  = drop_time_coord(ro.sel(time=tdt))
                scale_raster = drop_time_coord(scaling_da.isel(time=hour_idx))

                scaled = (ro_slice_15 * scale_raster).astype("float32")
                scaled = normalize_grid(scaled)

                ro_scaled_list.append(scaled)
                ro_scaled_times.append(tdt)

            if len(ro_scaled_list) == 0:
                raise RuntimeError("Scaling produced zero frames (check overlap/time alignment).")

            ro_scaled_da = xr.concat(ro_scaled_list, dim="time").assign_coords(time=ro_scaled_times)
            step_i += 8; pb.progress(min(1.0, step_i/total_steps))

            # ---------- 7) Render frames for playback (NO CLIP) ----------
            msg.info("Rendering map frames for playback…")
            cache, stats = {}, []

            for i, tdt in enumerate(ro_scaled_da.time.values):
                dt_local = pd.to_datetime(str(tdt)).to_pydatetime()
                da = ro_scaled_da.sel(time=dt_local)

                # Stats can use watershed clip (optional), but display is full raster
                mean_in = watershed_mean_inch(da, st.session_state.active_gdf)

                img_path, bounds = save_frame_png(da, dt_local)

                label = dt_local.strftime("%Y-%m-%d %H:%M")  # NY local label
                cache[label] = {"path": img_path, "bounds": bounds}
                stats.append({"time": dt_local, "rain_in": mean_in})

                step_i += 1; pb.progress(min(1.0, step_i/total_steps))

            st.session_state.radar_cache = cache
            st.session_state.time_list = list(cache.keys())
            st.session_state.current_time_index = 0
            st.session_state.is_playing = False
            st.session_state.basin_vault[basin_name] = pd.DataFrame(stats)

            msg.success(f"Complete: {len(cache)} frames • MRMS hours used: {len(mrms.time)}")
            st.session_state.processing_msg = f"Last run: {basin_name} • {len(cache)} frames (RO scaled to MRMS hourly)"
            st.rerun()

        except Exception as e:
            st.session_state.processing_msg = f"Error: {type(e).__name__}: {e}"
            st.error(st.session_state.processing_msg)
            st.exception(e)
            st.stop()

    # outputs list
    if st.session_state.basin_vault:
        st.divider()
        st.subheader("Outputs")
        for name, df in st.session_state.basin_vault.items():
            csv_download_link(df, filename=f"{name}.csv", label=f"{name}.csv")

# =============================
# 6) ANIMATION LOOP
# =============================
if st.session_state.time_list and st.session_state.is_playing:
    st.session_state.current_time_index = (st.session_state.current_time_index + 1) % len(st.session_state.time_list)
    time.sleep(0.5)
    st.rerun()

# =============================
# 7) MAP RENDER
# =============================
layers = []
if st.session_state.time_list:
    key = st.session_state.time_list[st.session_state.current_time_index]
    curr = st.session_state.radar_cache[key]
    layers.append(pdk.Layer("BitmapLayer", image=curr["path"], bounds=curr["bounds"], opacity=0.70))

# Watershed outline only (no clipping)
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
# 8) CONTROLS (ALWAYS ABOVE MAP)
#   - we wrap in a div with class control-bar that CSS can target
# =============================
if st.session_state.time_list:
    st.markdown('<div class="control-bar">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 10, 2])

    with c1:
        if st.session_state.is_playing:
            if st.button("⏸", key="pause_btn"):
                st.session_state.is_playing = False
                st.rerun()
        else:
            if st.button("▶", key="play_btn"):
                st.session_state.is_playing = True
                st.rerun()

    with c2:
        idx = st.select_slider(
            " ",
            options=list(range(len(st.session_state.time_list))),
            value=int(st.session_state.current_time_index),
            format_func=lambda i: st.session_state.time_list[i],
            label_visibility="collapsed",
            key="timeline_slider",
        )
        if idx != st.session_state.current_time_index:
            st.session_state.current_time_index = idx
            st.session_state.is_playing = False
            st.rerun()

    with c3:
        st.markdown(f"**{st.session_state.time_list[st.session_state.current_time_index]}**")

    st.markdown("</div>", unsafe_allow_html=True)
