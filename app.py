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
import gc  # Added for memory management
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import rioxarray
import time
import base64
from zoneinfo import ZoneInfo
import matplotlib.dates as mdates
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
UTC_TZ = ZoneInfo("UTC")

# user-selectable timezones (add/remove as you like)
TZ_OPTIONS = {
    "UTC": "UTC",
    "New York (ET)": "America/New_York",
    "Chicago (CT)": "America/Chicago",
    "Denver (MT)": "America/Denver",
    "Los Angeles (PT)": "America/Los_Angeles",
}

# --- minimum allowed timestamp (absolute truth in UTC) ---
MIN_UTC = datetime(2020, 10, 15, 0, 0, 0)  # naive UTC baseline


# =============================
# 1) PAGE CONFIG
# =============================
st.set_page_config(layout="wide", page_title="CNR Radar Portal", initial_sidebar_state="expanded")

# =============================
# 2) CSS (FULL RESTORATION + SLIDER FIX)
# =============================
st.markdown(
    """
<style>
/* --- 1. GLOBAL LAYOUT --- */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"]{
  height:100vh !important; width:100vw !important;
  margin:0 !important; padding:0 !important;
  overflow:hidden !important;
  background:#000 !important;
}

.main .block-container{
  padding:0 !important; margin:0 !important;
  max-width:100vw !important;
}

header, footer, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"]{
  display:none !important; height:0 !important; visibility:hidden !important;
}

/* --- 2. SIDEBAR (Locked & Styled) --- */
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
[data-testid="stSidebarResizer"], [data-testid="collapsedControl"], button[title="Collapse sidebar"]{
  display:none !important;
}
[data-testid="stSidebarContent"]{
  height:100vh !important;
  overflow:auto !important;
  padding:10px 12px !important;
}

/* Sidebar Button Styling */
[data-testid="stSidebar"] .stButton button{
  width:100% !important;
  border-radius:10px !important;
  height:44px !important;
  font-weight:650 !important;
  background: rgba(226,36,44,0.92) !important;
}

/* --- 3. MAP OVERLAY --- */
#deckgl-wrapper{
  position:fixed !important;
  inset:0 !important;
  z-index: 0 !important;
}

/* --- 4. FLOATING CONTROL BAR (The Slider Fix) --- */
/* The main container must allow clicks to pass through to the map, 
   but we RE-ENABLE them for the controls. */
   #deckgl-wrapper, #deckgl-wrapper canvas {
  pointer-events: auto !important;
}

[data-testid="stSidebar"], .control-bar-wrapper, .stButton, .stSlider, .stSelectSlider {
    pointer-events: auto !important;
}

.control-bar-wrapper {
  position: fixed !important;
  left: 420px !important;
  right: 18px !important;
  bottom: 18px !important;
  z-index: 1000000 !important; /* Extremely high to stay on top */
  background: rgba(15,15,15,0.92) !important;
  padding: 12px 16px !important;
  border-radius: 999px !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  backdrop-filter: blur(10px);
}

/* Round Play/Pause Button */
.control-bar-wrapper .stButton button {
  border-radius: 999px !important;
  width: 44px !important;
  height: 44px !important;
  padding: 0 !important;
  font-size: 18px !important;
}

/* --- 5. MISC --- */
.output-link a{ text-decoration:none !important; font-weight:600 !important; }
.output-link a:hover{ text-decoration:underline !important; }

</style>
""",
    unsafe_allow_html=True,
)
st.markdown("""
<style>
/* map still interactive */
#deckgl-wrapper, #deckgl-wrapper canvas { pointer-events:auto !important; }

/* floating controls container (JS adds this class) */
.floating-controls{
  position: fixed !important;
  left: 420px !important;
  right: 18px !important;
  bottom: 18px !important;
  z-index: 1000000 !important;
  background: rgba(15,15,15,0.92) !important;
  padding: 20px 20px !important;
  border-radius: 999px !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  backdrop-filter: blur(10px);
  pointer-events: auto !important;
  height: 100px !important;
  box-sizing: border-box !important;
  max-width: calc(100vw - 420px - 200px) !important;
  overflow: hidden !important;
}

.floating-controls *{ pointer-events:auto !important; }

/* allow Streamlit columns to shrink instead of overflow */
.floating-controls [data-testid="column"]{ min-width: 0 !important; }

/* timestamp truncation */
.floating-controls .timestamp{
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
/* ---------- Floating bar: center contents ---------- */
.floating-controls{
  height: 100px !important;                  /* give it a stable height */
  display: flex !important;
  justify-content: center !important;   /* horizontal centering */
  align-items: center !important;           /* vertical centering */
}
/* Make the immediate child take full width so centering is consistent */
.floating-controls > div{
  width: 100% !important;
  display: flex !important;
  justify-content: center !important;
  align-items: center !important;
}


/* Streamlit columns row inside the floating bar */
.floating-controls [data-testid="stHorizontalBlock"]{
  align-items: center !important;           /* vertical centering */
  gap: 14px !important;
}

/* Reduce extra vertical padding Streamlit adds around widgets */
.floating-controls [data-testid="stSlider"]{
  margin-top: -6px !important;
  margin-bottom: -6px !important;
}

/* ---------- Slider restyle (BaseWeb slider) ---------- */
.floating-controls [data-baseweb="slider"]{
  padding: 0 10px !important;
}

/* Track (unfilled + filled) - BaseWeb usually uses 2 bars */
.floating-controls [data-baseweb="slider"] > div{
  align-items: center !important;
}

/* These selectors are intentionally broad because Streamlit hashes classnames */
.floating-controls [data-baseweb="slider"] div[role="progressbar"]{
  height: 8px !important;
  border-radius: 999px !important;
  background: rgba(255,255,255,0.14) !important;  /* unfilled track */
}

/* Filled portion sits inside the progressbar in many Streamlit builds */
.floating-controls [data-baseweb="slider"] div[role="progressbar"] > div{
  height: 8px !important;
  border-radius: 999px !important;
  background: rgba(1,160,254,0.95) !important;    /* filled track */
  box-shadow: 0 0 10px rgba(1,160,254,0.25);
}

/* Thumb */
.floating-controls [data-baseweb="slider"] div[role="slider"]{
  width: 18px !important;
  height: 18px !important;
  border-radius: 999px !important;
  background: #01a0fe !important;
  border: 2px solid rgba(255,255,255,0.85) !important;
  box-shadow: 0 6px 16px rgba(0,0,0,0.45), 0 0 14px rgba(1,160,254,0.25) !important;
}

/* Optional: make the timestamp look more “HUD-like” */
.floating-controls .timestamp{
  opacity: 0.95;
  letter-spacing: 0.2px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* --- Fix BaseWeb slider geometry inside floating-controls --- */

/* 1) Remove Streamlit's extra padding around the widget block */
.floating-controls [data-testid="stSelectSlider"],
.floating-controls [data-testid="stSlider"]{
  padding: 0 !important;
  margin: 0 !important;
}

/* 2) Force the widget container to not “overhang” */
.floating-controls [data-testid="stSelectSlider"] > div,
.floating-controls [data-testid="stSlider"] > div{
  width: 100% !important;
  min-width: 0 !important;
}

/* 3) BaseWeb slider root: remove left/right padding that makes thumb start “too far left” */
.floating-controls [data-baseweb="slider"]{
  padding-left: 0 !important;
  padding-right: 0 !important;
  margin: 0 !important;
  width: 100% !important;
  box-sizing: border-box !important;
}

/* 4) The rail/track wrapper sometimes has negative margin or extra width */
.floating-controls [data-baseweb="slider"] > div{
  margin: 0 !important;
  width: 100% !important;
  box-sizing: border-box !important;
}

/* 5) Progressbar (track) should not extend past the container */
.floating-controls [data-baseweb="slider"] div[role="progressbar"]{
  width: 100% !important;
  margin: 0 !important;
  box-sizing: border-box !important;
}

/* 6) Some builds render the filled bar as a child with its own margin */
.floating-controls [data-baseweb="slider"] div[role="progressbar"] > div{
  margin: 0 !important;
  box-sizing: border-box !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* -------- Clean slider layout -------- */
.floating-controls{
  background: rgba(0,0,0,0.85) !important;   /* darker */
  border: 1px solid rgba(255,255,255,0.08) !important;
  padding: 20px 20px !important;            /* keep your height look */
  border-radius: 999px !important;
  backdrop-filter: blur(10px);
}


/* Make slider only 75% width and centered */
.floating-controls [data-testid="stHorizontalBlock"] {
    width: 75% !important;
}

/* Remove default extra padding */
.floating-controls [data-testid="stSlider"] {
    margin: 0 !important;
}

/* -------- Track styling -------- */
.floating-controls [data-baseweb="slider"] div[role="progressbar"] {
    height: 30px !important;                 /* thicker track */
    border_radius: 4px !important;
    background: rgba(255,255,255,0.2) !important;
}

/* Filled portion */
.floating-controls [data-baseweb="slider"] div[role="progressbar"] > div {
    height: 10px !important;
    border_radius: 4px !important;
    background: #01a0fe !important;
}

/* -------- Vertical line thumb -------- */
.floating-controls [data-baseweb="slider"] div[role="slider"] {
    width: 4px !important;                   /* thin vertical line */
    height: 28px !important;                 /* tall */
    border_radius: 2px !important;
    background: #01a0fe !important;
    border: none !important;
    box_shadow: 0 0 10px rgba(1,160,254,0.6) !important;
    margin-top: -9px !important;             /* vertically center it */
}

/* Timestamp clean look */
.floating-controls .timestamp {
    color: #01a0fe !important;
    font-family: monospace !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

</style>
""", unsafe_allow_html=True)



# =============================
# 3) STATE (must be before any st.session_state access)
# =============================
defaults = {
    "selected_boundaries": {},  # { "Name": GeoDataFrame }
    "radar_cache": {},
    "time_list": [],
    "active_gdf": None,
    "basin_vault": {},
    "map_view": pdk.ViewState(latitude=40.7, longitude=-74.0, zoom=9),
    "img_dir": tempfile.mkdtemp(prefix="radar_png_"),
    "is_playing": False,
    "current_time_index": 0,
    "processing_msg": "",
    "tz_name": "America/New_York",
    "radar_footprint": None,
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
if "current_time_label" not in st.session_state:
    st.session_state.current_time_label = None
LOCAL_TZ = ZoneInfo(st.session_state.get("tz_name", "America/New_York"))
CFGRIB_LOCK = threading.Lock()

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
    if da is None: return None
    return da.assign_coords(latitude=da.latitude.round(4), longitude=da.longitude.round(4))

def normalize_to_latlon(da: xr.DataArray) -> xr.DataArray:
    if da is None: return None
    da = normalize_grid(da)
    drop_coords = [c for c in da.coords if c not in ("latitude", "longitude")]
    da = da.drop_vars(drop_coords, errors="ignore")
    allowed = {"latitude", "longitude"}
    for d in list(da.dims):
        if d in allowed: continue
        da = da.isel({d: 0})
    da = da.sortby("latitude", ascending=False).sortby("longitude")
    return da

def normalize_to_tlatlon(da: xr.DataArray) -> xr.DataArray:
    if da is None: return None
    da = normalize_grid(da)
    drop_coords = [c for c in da.coords if c not in ("time", "latitude", "longitude")]
    da = da.drop_vars(drop_coords, errors="ignore")
    allowed = {"time", "latitude", "longitude"}
    for d in list(da.dims):
        if d in allowed: continue
        da = da.isel({d: 0})
    da = da.sortby("latitude", ascending=False).sortby("longitude")
    return da

def nearest_index(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    src = np.asarray(src).astype("float64")
    tgt = np.asarray(tgt).astype("float64")
    descending = src[0] > src[-1]
    src_work = src[::-1] if descending else src
    pos = np.searchsorted(src_work, tgt)
    pos = np.clip(pos, 1, len(src_work) - 1)
    left = src_work[pos - 1]
    right = src_work[pos]
    choose_right = (np.abs(tgt - right) < np.abs(tgt - left))
    idx = pos - 1 + choose_right.astype(int)
    if descending: idx = (len(src) - 1) - idx
    return idx

def align_ro_to_mrms_grid_nearest(ro: xr.DataArray, mrms0: xr.DataArray) -> xr.DataArray:
    ro = normalize_to_tlatlon(ro)
    mrms0 = normalize_to_latlon(mrms0)
    ro_lat, ro_lon = np.asarray(ro.latitude.values, dtype="float64"), np.asarray(ro.longitude.values, dtype="float64")
    mr_lat, mr_lon = np.asarray(mrms0.latitude.values, dtype="float64"), np.asarray(mrms0.longitude.values, dtype="float64")
    lat_idx = nearest_index(ro_lat, mr_lat)
    lon_idx = nearest_index(ro_lon, mr_lon)
    data = ro.values.astype("float32")
    aligned = data[:, lat_idx[:, None], lon_idx[None, :]]
    return xr.DataArray(
        aligned, dims=("time", "latitude", "longitude"),
        coords={"time": ro.time.values, "latitude": mr_lat, "longitude": mr_lon}
    )

def local_naive_to_utc(dt_local_naive: datetime) -> datetime:
    aware_local = dt_local_naive.replace(tzinfo=LOCAL_TZ)
    aware_utc = aware_local.astimezone(UTC_TZ)
    return aware_utc.replace(tzinfo=None)

def utc_naive_to_local_naive(dt_utc_naive: datetime) -> datetime:
    aware_utc = dt_utc_naive.replace(tzinfo=UTC_TZ)
    aware_local = aware_utc.astimezone(LOCAL_TZ)
    return aware_local.replace(tzinfo=None)

def ceil_to_hour(dt: datetime) -> datetime:
    dt0 = dt.replace(minute=0, second=0, microsecond=0)
    return dt0 if dt == dt0 else (dt0 + timedelta(hours=1))
    
def fetch_precip_safe(file_type: str, t: datetime):
    try:
        da = load_precip(file_type, t)
        return (t, da)
    except Exception:
        return (t, None)

def load_precip(file_type: str, dt_local_naive: datetime) -> xr.DataArray | None:
    dt_utc = local_naive_to_utc(dt_local_naive)
    ts, ymd = dt_utc.strftime("%Y%m%d-%H%M00"), dt_utc.strftime("%Y%m%d")
    base = RO_S3_BASE if file_type == "RO" else MRMS_S3_BASE
    filename = f"MRMS_{'RadarOnly_QPE_15M' if file_type=='RO' else 'MultiSensor_QPE_01H_Pass2'}_00.00_{ts}.grib2.gz"
    url = f"{base}/{ymd}/{filename}"
    tmp_path = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False).name
    try:
        r = requests.get(url, stream=True, timeout=25)
        if r.status_code == 404: return None
        r.raise_for_status()
        with gzip.GzipFile(fileobj=r.raw) as gz, open(tmp_path, "wb") as f:
            shutil.copyfileobj(gz, f)
        with CFGRIB_LOCK:
            with xr.open_dataset(tmp_path, engine="cfgrib", backend_kwargs={"indexpath": ""}) as ds:
                var = list(ds.data_vars)[0]
                da = ds[var].clip(min=0).load()
        da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180).sortby("longitude")
        da = da.rio.write_crs("EPSG:4326")
        return normalize_to_latlon(da)
    except: return None
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

def drop_time_coord(da: xr.DataArray) -> xr.DataArray:
    return da.drop_vars("time", errors="ignore")

def save_frame_png(da: xr.DataArray, dt_local_naive: datetime) -> tuple[str, list]:
    data = da.values.astype("float32")
    data[data < 0.1] = np.nan
    img_path = os.path.join(st.session_state.img_dir, f"radar_{dt_local_naive.strftime('%Y%m%d_%H%M')}.png")
    plt.imsave(img_path, data, cmap=RADAR_CMAP, vmin=0.1, vmax=15.0)
    bounds = [float(da.longitude.min()), float(da.latitude.min()), float(da.longitude.max()), float(da.latitude.max())]
    return img_path, bounds

def watershed_mean_inch(da_full: xr.DataArray, ws_gdf: gpd.GeoDataFrame) -> float:
    if ws_gdf is None: return float(da_full.mean().values) / 25.4
    try:
        sub = da_full.rio.write_crs("EPSG:4326", inplace=False)
        clipped = sub.rio.clip(ws_gdf.geometry, ws_gdf.crs, drop=True)
        return float(clipped.mean().values) / 25.4 if clipped.size > 0 else float(da_full.mean().values) / 25.4
    except: return float(da_full.mean().values) / 25.4

def _edges_from_centers(c):
    """Given 1D centers, return 1D edges with len = len(c)+1 (extrapolated at ends)."""
    c = np.asarray(c, dtype="float64")
    mid = (c[:-1] + c[1:]) / 2.0
    e = np.empty(len(c) + 1, dtype="float64")
    e[1:-1] = mid
    e[0] = c[0] - (mid[0] - c[0])
    e[-1] = c[-1] + (c[-1] - mid[-1])
    return e

def _add_seg(segset, a, b):
    """Add an undirected segment (a->b) using canonical ordering for dedupe."""
    if a <= b:
        segset.add((a, b))
    else:
        segset.add((b, a))

def _stitch_segments_to_rings(segments):
    """
    Stitch undirected segments into one or more closed rings.
    segments: set of ((x1,y1),(x2,y2)) in float tuples
    returns list of rings, each ring is list of (x,y) (closed: last == first)
    """
    # adjacency: point -> list of neighbors
    adj = {}
    for a, b in segments:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    rings = []
    used = set()

    def take_edge(u, v):
        used.add((u, v)); used.add((v, u))

    for start in list(adj.keys()):
        # find an unused edge from start
        nbrs = adj.get(start, [])
        nxt = None
        for v in nbrs:
            if (start, v) not in used:
                nxt = v
                break
        if nxt is None:
            continue

        ring = [start]
        u, v = start, nxt
        take_edge(u, v)
        ring.append(v)

        # walk until we return to start
        while True:
            nbrs = adj.get(v, [])
            # choose next neighbor that isn't the immediate back edge and isn't used
            cand = None
            for w in nbrs:
                if w == u:
                    continue
                if (v, w) not in used:
                    cand = w
                    break
            if cand is None:
                # either dead-end or only back edge remains; try closing if possible
                if ring[-1] == start:
                    break
                # attempt to close directly
                if start in nbrs and (v, start) not in used:
                    take_edge(v, start)
                    ring.append(start)
                break

            u, v = v, cand
            take_edge(u, v)
            ring.append(v)

            if v == start:
                break

        # keep only valid closed rings (>= 4 points including closure)
        if len(ring) >= 4 and ring[0] == ring[-1]:
            rings.append(ring)

    return rings

def _ring_area_xy(ring):
    """Shoelace area for ring [(x,y),...,(x,y)] closed."""
    x = np.array([p[0] for p in ring], dtype="float64")
    y = np.array([p[1] for p in ring], dtype="float64")
    return 0.5 * np.abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))

def grid_footprint_geojson_from_mask(lat_centers, lon_centers, valid_mask_2d):
    """
    Build an irregular footprint polygon from a boolean mask over (lat, lon) grid.
    valid_mask_2d shape: (nlat, nlon) True where cell is valid (kept/extracted).
    Returns GeoJSON FeatureCollection with Polygon or MultiPolygon.
    """
    latc = np.asarray(lat_centers, dtype="float64")
    lonc = np.asarray(lon_centers, dtype="float64")
    mask = np.asarray(valid_mask_2d, dtype=bool)

    # edges (cell boundaries) in degrees
    late = _edges_from_centers(latc)  # len nlat+1
    lone = _edges_from_centers(lonc)  # len nlon+1

    nlat, nlon = mask.shape
    segs = set()

    # For each valid cell, add boundary edges where neighbor is invalid/outside.
    # We build segments in (lon, lat) space (GeoJSON convention).
    for i in range(nlat):
        for j in range(nlon):
            if not mask[i, j]:
                continue

            # corners of this cell
            # lat edges: i..i+1, lon edges: j..j+1
            # (lon,lat) corners:
            bl = (float(lone[j]),   float(late[i]))     # bottom-left
            br = (float(lone[j+1]), float(late[i]))     # bottom-right
            tr = (float(lone[j+1]), float(late[i+1]))   # top-right
            tl = (float(lone[j]),   float(late[i+1]))   # top-left

            # neighbors (N,S,E,W) in index space (i-1 is "north" if lat is descending—doesn't matter for topology)
            # North edge: between tl-tr, neighbor i-1
            if i == 0 or not mask[i-1, j]:
                _add_seg(segs, tl, tr)
            # South edge: between bl-br, neighbor i+1
            if i == nlat-1 or not mask[i+1, j]:
                _add_seg(segs, bl, br)
            # West edge: between bl-tl, neighbor j-1
            if j == 0 or not mask[i, j-1]:
                _add_seg(segs, bl, tl)
            # East edge: between br-tr, neighbor j+1
            if j == nlon-1 or not mask[i, j+1]:
                _add_seg(segs, br, tr)

    rings = _stitch_segments_to_rings(segs)
    if not rings:
        return None

    # If multiple rings, return MultiPolygon (or choose largest only)
    # Here: keep all, but you can keep only largest by area if you want.
    rings_sorted = sorted(rings, key=_ring_area_xy, reverse=True)

    # Build GeoJSON coords. GeoJSON wants [ [ [lon,lat], ... ] ] for Polygon
    if len(rings_sorted) == 1:
        coords = [[list(p) for p in rings_sorted[0]]]
        geom = {"type": "Polygon", "coordinates": coords}
    else:
        polys = []
        for r in rings_sorted:
            polys.append([[list(p) for p in r]])
        geom = {"type": "MultiPolygon", "coordinates": polys}

    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"name": "Radar Grid Footprint"},
            "geometry": geom
        }]
    }

# -----------------------------
# Modal / popup plot helper
# -----------------------------
try:
    _HAS_DIALOG = hasattr(st, "dialog")
except Exception:
    _HAS_DIALOG = False

def show_plot_popup(title: str, df: pd.DataFrame):
    """Open a popup modal with the rainfall bar chart."""
    # compute width (days)
    if len(df) > 1:
        delta_seconds = (df['time'].iloc[1] - df['time'].iloc[0]).total_seconds()
        width = delta_seconds / 86400
    else:
        delta_seconds = 3600
        width = 0.01

    def _render():
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(df['time'], df['rain_in'], width=width, align='edge',
               edgecolor='white', linewidth=0.5)

        ax.set_xlim(df['time'].min(), df['time'].max() + timedelta(seconds=delta_seconds))

        locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax.set_title(f"{title} Rainfall (in)", color='white', fontsize=8)
        ax.tick_params(axis='both', colors='white', labelsize=6)
        ax.set_facecolor('#111')
        fig.patch.set_facecolor('#111')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')

        st.pyplot(fig)
        plt.close(fig)

    # Prefer st.dialog (real popup). If not available, fallback is inline expander render.
    if _HAS_DIALOG:
        @st.dialog(title)
        def _dlg():
            _render()
        _dlg()
    else:
        # Fallback: still gated behind button, but not a true popup
        st.info("Your Streamlit version doesn’t support dialogs. Showing plot inline.")
        _render()

# =============================
# 5) SIDEBAR UI
# =============================
with st.sidebar:
    st.title("CNR GIS Portal")
    
    # --- 1. INITIALIZE VARIABLES ---
    show_muni_map = False
    muni_file = "nj_munis.geojson"
    muni_gdf = None
    
    if "selected_areas" not in st.session_state:
        st.session_state.selected_areas = {}
    
    if os.path.exists(muni_file):
        muni_gdf = gpd.read_file(muni_file).to_crs("EPSG:4326")

    # --- 2. TIMEZONE & DATE PICKERS ---
    tz_label = st.selectbox(
        "Time Zone",
        list(TZ_OPTIONS.keys()),
        index=list(TZ_OPTIONS.values()).index(st.session_state.tz_name) if st.session_state.tz_name in TZ_OPTIONS.values() else 1
    )
    st.session_state.tz_name = TZ_OPTIONS[tz_label]
    LOCAL_TZ = ZoneInfo(st.session_state.tz_name)

    min_local_dt = utc_naive_to_local_naive(MIN_UTC)
    s_date = st.date_input("Start Date", value=max(datetime.now().date(), min_local_dt.date()), min_value=min_local_dt.date())
    e_date = st.date_input("End Date", value=max(datetime.now().date(), min_local_dt.date()), min_value=min_local_dt.date())

    c1, c2 = st.columns(2)
    s_time = c1.selectbox("Start Time", hours := [f"{h:02d}:00" for h in range(24)], index=0)
    e_time = c2.selectbox("End Time", hours, index=0)

    st.divider()
    st.subheader("Area Selection")

    # [CHANGE]: Default value set to False for performance
    if muni_gdf is not None:
        show_muni_map = st.checkbox("Show municipalities on map", value=False)

    # --- 3. SEARCH & ADD LOGIC ---
    if muni_gdf is not None:
        muni_names = sorted(muni_gdf["GNIS_NAME"].dropna().unique().tolist())
        col_search, col_add = st.columns([0.8, 0.2])

        with col_search:
            drop_selection = st.selectbox(
                "Search Municipality", 
                ["Type to start..."] + muni_names,
                key="muni_picker_search"
            )
        
        # ZOOM Logic
        if drop_selection != "Type to start...":
            target = muni_gdf[muni_gdf["GNIS_NAME"] == drop_selection].copy()
            b = target.total_bounds
            new_view = pdk.ViewState(latitude=(b[1]+b[3])/2, longitude=(b[0]+b[2])/2, zoom=12, pitch=0, bearing=0)
            

        with col_add:
            st.write("##")
            if st.button("✚", help="Add to Active Selections"):
                if drop_selection != "Type to start...":
                    st.session_state.selected_areas[drop_selection] = muni_gdf[muni_gdf["GNIS_NAME"] == drop_selection].copy()
                    st.rerun()
                else:
                    st.toast("Select a name first!", icon="⚠️")

    # --- 4. UPLOAD LOGIC ---
    up_zip = st.file_uploader("Add Watershed Boundary (ZIP)", type="zip")
    if up_zip:
        if up_zip.name not in st.session_state.selected_areas:
            with tempfile.TemporaryDirectory() as td:
                with zipfile.ZipFile(up_zip, "r") as z: z.extractall(td)
                if shps := list(Path(td).rglob("*.shp")):
                    new_gdf = gpd.read_file(shps[0]).to_crs("EPSG:4326")
                    st.session_state.selected_areas[up_zip.name] = new_gdf
                    
                    b = new_gdf.total_bounds
                    st.session_state.map_view = pdk.ViewState(latitude=(b[1]+b[3])/2, longitude=(b[0]+b[2])/2, zoom=12)
                    st.rerun()

    # --- 5. ACTIVE SELECTIONS LIST ---
    if st.session_state.selected_areas:
        st.write("---")
        st.caption("Active Selections:")
        for name in list(st.session_state.selected_areas.keys()):
            cols = st.columns([0.8, 0.2])
            cols[0].markdown(f"📍 **{name}**")
            if cols[1].button("✕", key=f"del_{name}"):
                del st.session_state.selected_areas[name]
                st.rerun()

    # --- 6. PROCESSING ENGINE (UPDATED) ---
    if st.button("Run Processing", use_container_width=True):
        if not st.session_state.selected_areas:
            st.error("Select at least one area first.")
        else:
            try:
                # 1. Combine all areas to determine the TOTAL bounding box for data fetching
                combined_gdf = pd.concat(st.session_state.selected_areas.values())
                b = combined_gdf.total_bounds
                
                # Expand bounds slightly to ensure full coverage
                BUFFER_DEG = 0.35
                lon_min, lat_min, lon_max, lat_max = b[0]-BUFFER_DEG, b[1]-BUFFER_DEG, b[2]+BUFFER_DEG, b[3]+BUFFER_DEG
                
                start_dt = datetime.combine(s_date, datetime.strptime(s_time, "%H:%M").time())
                end_dt   = datetime.combine(e_date, datetime.strptime(e_time, "%H:%M").time())
                if start_dt < min_local_dt: start_dt = min_local_dt

                ro_times = list(pd.date_range(start_dt + timedelta(minutes=15), end_dt, freq="15min"))
                mrms_times = list(pd.date_range(ceil_to_hour(start_dt + timedelta(minutes=45)), end_dt.replace(minute=0, second=0, microsecond=0), freq="1h"))

                pb, msg = st.progress(0.0), st.empty()
                
                # --- FETCH RO ---
                ro_list, ro_kept = [], []
                max_workers = min(6, (os.cpu_count() or 4))
                
                futs = []
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    for t in ro_times:
                        futs.append(ex.submit(fetch_precip_safe, "RO", t.to_pydatetime()))
                
                    total = len(futs)
                    done = 0
                    for fut in as_completed(futs):
                        tdt, da = fut.result()
                        done += 1
                        msg.info(f"RO → {done}/{total}")
                        pb.progress(min(1.0, done / max(1, (len(ro_times) + len(mrms_times)))))
                
                        if da is not None:
                            da = da.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
                            ro_list.append(da.astype("float32"))
                            ro_kept.append(tdt)
                
                # keep chronological order (important for concat + animation)
                if ro_kept:
                    ro_kept, ro_list = zip(*sorted(zip(ro_kept, ro_list), key=lambda x: x[0]))
                    ro_kept, ro_list = list(ro_kept), list(ro_list)
                else:
                    ro_kept, ro_list = [], []
                
                if len(ro_list) < 4: raise RuntimeError("Insufficient RO data.")
                ro = xr.concat(ro_list, dim="time").assign_coords(time=ro_kept)
                del ro_list; gc.collect()

                # --- FETCH MRMS ---
                mrms_list, mrms_kept = [], []

                futs = []
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    for t in mrms_times:
                        futs.append(ex.submit(fetch_precip_safe, "MRMS", t.to_pydatetime()))
                
                    total = len(futs)
                    done = 0
                    for fut in as_completed(futs):
                        tdt, da = fut.result()
                        done += 1
                        msg.info(f"MRMS → {done}/{total}")
                        pb.progress(min(1.0, (len(ro_kept) + done) / max(1, (len(ro_times) + len(mrms_times)))))
                
                        if da is not None:
                            da = da.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
                            mrms_list.append(da.astype("float32"))
                            mrms_kept.append(tdt)
                
                if mrms_kept:
                    mrms_kept, mrms_list = zip(*sorted(zip(mrms_kept, mrms_list), key=lambda x: x[0]))
                    mrms_kept, mrms_list = list(mrms_kept), list(mrms_list)
                else:
                    mrms_kept, mrms_list = [], []
                
                if not mrms_list: raise RuntimeError("No MRMS data found.")
                mrms = xr.concat(mrms_list, dim="time").assign_coords(time=mrms_kept)
                del mrms_list; gc.collect()
                m0 = mrms.isel(time=0).values
                valid_mask = np.isfinite(m0)
                
                st.session_state.radar_footprint = grid_footprint_geojson_from_mask(
                    lat_centers=mrms.latitude.values,
                    lon_centers=mrms.longitude.values,
                    valid_mask_2d=valid_mask
                )

                
                msg.info("Aligning grids...")
                ro = align_ro_to_mrms_grid_nearest(ro, mrms.isel(time=0))
                gc.collect()

                
                msg.info("Calculating Bias Scaling...")
                ro_hourly, v_times = [], []
                for T in mrms.time.values:
                    Tdt = pd.to_datetime(str(T)).to_pydatetime()
                    block = ro.sel(time=slice(Tdt - timedelta(minutes=45) - timedelta(seconds=1), Tdt))
                    if block.sizes.get("time", 0) == 4:
                        ro_hourly.append(block.sum(dim="time").astype("float32"))
                        v_times.append(Tdt)
                
                if not v_times: raise RuntimeError("No matching RO/MRMS windows found.")
                
                mrms_subset = mrms.sel(time=pd.to_datetime(v_times))
                ro_hr_da = xr.concat(ro_hourly, dim="time").assign_coords(time=pd.to_datetime(v_times))
                scaling = xr.where(ro_hr_da > 0.01, mrms_subset / ro_hr_da, 0.0).clip(0, 50).astype("float32")
                del ro_hr_da, ro_hourly, mrms_subset; gc.collect()

                msg.info("Rendering frames & Calculating Stats...")
                
                # [CHANGE] Prepare structure to hold stats for EACH city separately
                city_stats_container = {name: [] for name in st.session_state.selected_areas.keys()}
                cache = {}
                
                m_ends = [pd.to_datetime(str(x)).to_pydatetime() for x in mrms.time.values]
                
                for t in ro.time.values:
                    tdt = pd.to_datetime(str(t)).to_pydatetime()
                    h_end = next((T for T in m_ends if T >= tdt), None)
                    
                    if h_end and tdt >= h_end - timedelta(minutes=45):
                        h_idx = m_ends.index(h_end)
                        # Create the visual frame (covers entire area)
                        s_frame = (drop_time_coord(ro.sel(time=tdt)) * drop_time_coord(scaling.isel(time=h_idx))).astype("float32")
                        
                        # Save visual for map
                        img, bnds = save_frame_png(s_frame, tdt)
                        lbl = tdt.strftime("%Y-%m-%d %H:%M")
                        cache[lbl] = {"path": img, "bounds": bnds}
                        
                        # [CHANGE] Loop through EVERY selected city to calculate its specific rainfall
                        for area_name, area_gdf in st.session_state.selected_areas.items():
                            val = watershed_mean_inch(s_frame, area_gdf)
                            city_stats_container[area_name].append({"time": tdt, "rain_in": val})

                # Update Global State
                st.session_state.radar_cache = cache
                st.session_state.time_list = sorted(cache.keys())
                st.session_state.current_time_label = st.session_state.time_list[0] if st.session_state.time_list else None
                
                # [CHANGE] Save DataFrame for every city into the vault
                st.session_state.basin_vault = {name: pd.DataFrame(stats) for name, stats in city_stats_container.items()}
                
                msg.success("Complete."); st.rerun()
            except Exception as e:
                st.exception(e)
                st.stop()

    # --- 7. DOWNLOADS & PLOTS (UPDATED to loop all results) ---
    if st.session_state.basin_vault:
        st.divider()
        st.subheader("Results & Exports")
        
        # [CHANGE] Loop through basin_vault items to display multiple graphs/downloads
        for name, df in st.session_state.basin_vault.items():
            with st.expander(f"📊 {name}", expanded=False):
                # Download Link
                csv_download_link(df, f"{name}_rain_data.csv", f"Download CSV")

                # 1. Calculate the correct bar width (Matplotlib uses 'days' as the unit)
                if len(df) > 1:
                    # Get time difference between first two points in seconds
                    delta_seconds = (df['time'].iloc[1] - df['time'].iloc[0]).total_seconds()
                    # Convert to days (seconds / 86400)
                    width = delta_seconds / 86400
                else:
                    width = 0.01  # Fallback width if only one point exists
    
                plot_key = f"plot_{name}"
                if st.button("Plot", key=plot_key):
                    show_plot_popup(f"Plot {name}", df)

# =============================
# 6) ANIMATION & MAIN DISPLAY
# =============================
if st.session_state.is_playing:
    n = len(st.session_state.time_list)
    if n > 0:
        st.session_state.current_time_index = (st.session_state.current_time_index + 1) % n
        time.sleep(0.5)
        st.rerun()

layers = []

# Layer 1: Municipalities
if show_muni_map and muni_gdf is not None:
    layers.append(pdk.Layer(
        "GeoJsonLayer",
        muni_gdf,
        pickable=True,
        stroked=True,
        filled=True,
        get_fill_color=[255, 255, 255, 10],
        get_line_color=[255, 255, 255, 60],
        line_width_min_pixels=1,
    ))

if st.session_state.get("radar_footprint"):
    layers.append(pdk.Layer(
        "GeoJsonLayer",
        st.session_state.radar_footprint,
        pickable=False,
        stroked=True,
        filled=False,
        get_line_color=[180, 180, 180],
        line_width_min_pixels=2,
    ))

# Layer 2: Radar
if st.session_state.time_list and st.session_state.current_time_label:
    # Validate timestamp exists in current list
    if st.session_state.current_time_label not in st.session_state.time_list:
        st.session_state.current_time_label = st.session_state.time_list[0]
        
    curr = st.session_state.radar_cache[st.session_state.current_time_label]
    layers.append(pdk.Layer(
        "BitmapLayer",
        image=curr["path"],
        bounds=curr["bounds"],
        opacity=0.70
    ))

# Layer 3: Selected Highlights (Blue outlines)
if st.session_state.selected_areas:
    # We overlay ALL selected areas
    active_overlay = pd.concat(st.session_state.selected_areas.values())
    layers.append(pdk.Layer(
        "GeoJsonLayer",
        active_overlay.__geo_interface__,
        stroked=True,
        filled=False,
        get_line_color=[1, 160, 254],
        line_width_min_pixels=3
    ))




deck = pdk.Deck(
    layers=layers,
    initial_view_state=st.session_state.map_view,
    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    tooltip={"text": "{GNIS_NAME}"} if show_muni_map else None
)

selection_hash = "_".join(st.session_state.selected_areas.keys())
deck_key = f"map_{selection_hash}"
map_event = st.pydeck_chart(deck, width="stretch", height=1000, key=deck_key)

# 7. INTERACTIVITY: Catch Map Clicks
if map_event is not None:
    try:
        last_clicked = map_event["last_clicked"]
        if last_clicked:
            picked_obj = last_clicked.get("object")
            if picked_obj and "GNIS_NAME" in picked_obj:
                name = picked_obj["GNIS_NAME"]
                # Add to selection if not present
                if name not in st.session_state.selected_areas:
                    target = muni_gdf[muni_gdf["GNIS_NAME"] == name].copy()
                    st.session_state.selected_areas[name] = target
                    
                    # Update ViewState
                    b = target.total_bounds
                    st.session_state.map_view = pdk.ViewState(
                        latitude=(b[1]+b[3])/2, 
                        longitude=(b[0]+b[2])/2, 
                        zoom=12,
                        pitch=0,
                        bearing=0
                    )
                    st.rerun()
    except (KeyError, TypeError):
        pass
# =============================
# 7) FLOATING CONTROLS
# =============================
import streamlit.components.v1 as components

if st.session_state.time_list:
    with st.container():
        st.markdown('<div id="control_bar_anchor"></div>', unsafe_allow_html=True)

        col_play, col_slider, col_txt = st.columns([1.2, 12.8, 4])

        with col_play:
            icon = "⏸" if st.session_state.is_playing else "▶"
            if st.button(icon, key="timeline_play_pause", help="Play/Pause"):
                st.session_state.is_playing = not st.session_state.is_playing
                st.rerun()
        
        with col_slider:
            chosen = st.select_slider(
                "Timeline",
                options=st.session_state.time_list,
                value=st.session_state.current_time_label or st.session_state.time_list[0],
                label_visibility="collapsed",
                key="timeline_slider"
            )
            if chosen != st.session_state.current_time_label:
                st.session_state.current_time_label = chosen
                # if user manually scrubs, pause playback (optional; remove if you don’t want this)
                st.session_state.is_playing = False
                st.rerun()
        
        with col_txt:
            ts = st.session_state.current_time_label or ""
            st.markdown(
                f'<p class="timestamp" style="color:#01a0fe; margin:0; font-family:monospace; font-size:14px; font-weight:bold; line-height:44px;">{ts}</p>',
                unsafe_allow_html=True
            )
    # your existing JS that adds .floating-controls stays the same
    components.html("""
    <script>
    (function() {
      const doc = window.parent.document;
      const anchor = doc.querySelector('#control_bar_anchor');
      if (!anchor) return;
      const wrap =
        anchor.closest('[data-testid="stVerticalBlock"]') ||
        anchor.closest('.element-container') ||
        anchor.parentElement;
      if (!wrap) return;
      wrap.classList.add('floating-controls');
    })();
    </script>
    """, height=0)




