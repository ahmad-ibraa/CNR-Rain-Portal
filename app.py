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
import folium
from folium import FeatureGroup
from streamlit_folium import st_folium
from functools import lru_cache
from folium.features import GeoJsonTooltip
from folium import Map
from streamlit_folium import st_folium

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
  padding: 12px 16px !important;
  border-radius: 999px !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  backdrop-filter: blur(10px);
  pointer-events: auto !important;

  box-sizing: border-box !important;
  max-width: calc(100vw - 420px - 18px) !important;
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
  height: 70px !important;                 /* give it a stable height */
  display: flex !important;
  align-items: center !important;          /* vertical centering */
}

/* Streamlit columns row inside the floating bar */
.floating-controls [data-testid="stHorizontalBlock"]{
  align-items: center !important;          /* vertical centering */
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




# =============================
# 3) STATE (must be before any st.session_state access)
# =============================
defaults = {
    "radar_cache": {},
    "time_list": [],
    "active_gdf": None,
    "basin_vault": {},
    "map_view": pdk.ViewState(latitude=40.7, longitude=-74.0, zoom=9),
    "img_dir": tempfile.mkdtemp(prefix="radar_png_"),
    "is_playing": False,
    "current_time_index": 0,
    "processing_msg": "",
    "tz_name": "America/New_York",   # default selection
    "selected_munis": [],          # list of GNIS_NAME strings
    "show_munis": True,            # toggle default
    "search_query": "",            # city search text
    "mode": "select",   # "select" (folium) or "view" (pydeck)

}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
if "current_time_label" not in st.session_state:
    st.session_state.current_time_label = None



RADAR_COLORS = ["#76fffe", "#01a0fe", "#0001ef", "#01ef01", "#019001", "#ffff01", "#e7c001", "#ff9000", "#ff0101"]
RADAR_CMAP = ListedColormap(RADAR_COLORS)

MRMS_S3_BASE = "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/MultiSensor_QPE_01H_Pass2_00.00"
RO_S3_BASE   = "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/RadarOnly_QPE_15M_00.00"

# =============================
# 4) HELPERS
# =============================
MUNI_GEOJSON_PATH = "data/nj_municipalities.geojson"  # <-- change to your actual repo path
MUNI_NAME_FIELD = "GNIS_NAME"


def render_muni_picker_map(muni_geojson: dict, center=(40.1, -74.6), zoom=8):
    m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB dark_matter")

    folium.GeoJson(
        muni_geojson,
        name="NJ Municipalities",
        style_function=lambda feat: {
            "color": "#ffffff",
            "weight": 1,
            "fillColor": "#000000",
            "fillOpacity": 0.00,
        },
        highlight_function=lambda feat: {
            "weight": 3,
            "color": "#01a0fe",
            "fillOpacity": 0.10,
        },
        tooltip=GeoJsonTooltip(fields=[MUNI_NAME_FIELD], aliases=["Municipality:"]),
    ).add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)

    out = st_folium(m, width=None, height=950, returned_objects=["last_clicked"])
    clicked = out.get("last_clicked")

    if clicked and isinstance(clicked, dict):
        props = clicked.get("properties", {}) or {}
        name = props.get(MUNI_NAME_FIELD)

        if name and name not in st.session_state.selected_munis:
            st.session_state.selected_munis.append(name)
            st.rerun()


def load_munis_geojson(path: str) -> dict:
    # Load raw GeoJSON as dict (folium uses it directly)
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
        
@st.cache_data(show_spinner=False, ttl=24*3600)
def geocode_place(query: str):
    # Nominatim requires a User-Agent
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": 1}
    headers = {"User-Agent": "cnr-rainfall-streamlit/1.0 (contact: you@example.com)"}
    r = requests.get(url, params=params, headers=headers, timeout=15)
    r.raise_for_status()
    js = r.json()
    if not js:
        return None
    return float(js[0]["lat"]), float(js[0]["lon"])

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

# =============================
# 5) SIDEBAR UI
# =============================
with st.sidebar:
    st.title("CNR GIS Portal")
    st.divider()
    st.subheader("Map Layers")

    st.session_state.show_munis = st.checkbox("Show NJ Municipalities", value=st.session_state.show_munis)

    st.caption("Click a municipality on the map to select it.")
    if st.session_state.selected_munis:
        st.write("Selected municipalities:")
        st.write(st.session_state.selected_munis)

        if st.button("Clear selected municipalities"):
            st.session_state.selected_munis = []
            st.rerun()

    st.divider()
    st.subheader("Search (Zoom to place)")
    q = st.text_input("City / place", value=st.session_state.search_query, placeholder="e.g., Newark, NJ")
    st.session_state.search_query = q

    if st.button("Zoom"):
        if q.strip():
            hit = geocode_place(q.strip())
            if hit is None:
                st.warning("No results found.")
            else:
                lat, lon = hit
                # store as your view center
                st.session_state.map_view = pdk.ViewState(latitude=lat, longitude=lon, zoom=11)
                st.rerun()
    tz_label = st.selectbox(
        "Time Zone",
        list(TZ_OPTIONS.keys()),
        index=list(TZ_OPTIONS.values()).index(st.session_state.tz_name)
              if st.session_state.tz_name in TZ_OPTIONS.values()
              else 1  # default to New York
    )
    
    st.session_state.tz_name = TZ_OPTIONS[tz_label]
    LOCAL_TZ = ZoneInfo(st.session_state.tz_name)

    min_local_dt = utc_naive_to_local_naive(MIN_UTC)
    min_local_date = min_local_dt.date()
    min_local_time = min_local_dt.time()
    
    s_date = st.date_input("Start Date", value=max(datetime.now().date(), min_local_date), min_value=min_local_date)
    e_date = st.date_input("End Date", value=max(datetime.now().date(), min_local_date), min_value=min_local_date)

    c1, c2 = st.columns(2)
    s_time = c1.selectbox("Start Time", hours := [f"{h:02d}:00" for h in range(24)], index=00)
    e_time = c2.selectbox("End Time", hours, index=00)
    up_zip = st.file_uploader("Watershed Boundary (ZIP)", type="zip")
    basin_name = up_zip.name.replace(".zip", "") if up_zip else "Default_Basin"

    if up_zip:
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(up_zip, "r") as z: z.extractall(td)
            if shps := list(Path(td).rglob("*.shp")):
                st.session_state.active_gdf = gpd.read_file(shps[0]).to_crs("EPSG:4326")
                b = st.session_state.active_gdf.total_bounds
                st.session_state.map_view = pdk.ViewState(latitude=(b[1]+b[3])/2, longitude=(b[0]+b[2])/2, zoom=11)

    if st.session_state.processing_msg: st.caption(st.session_state.processing_msg)

    if st.button("Run Processing", use_container_width=True):
        try:
            if st.session_state.active_gdf is None:
                st.session_state.processing_msg = "Upload a watershed boundary ZIP first."; st.rerun()

            # 1. Setup spatial bounds (CROP TO BASIN)
            b = st.session_state.active_gdf.total_bounds
            # Buffer by ~0.1 degrees to ensure no edge issues
            BUFFER_DEG = 0.35  # ~35-40 km-ish
            lon_min, lat_min, lon_max, lat_max = b[0]-BUFFER_DEG, b[1]-BUFFER_DEG, b[2]+BUFFER_DEG, b[3]+BUFFER_DEG


            start_dt = datetime.combine(s_date, datetime.strptime(s_time, "%H:%M").time())
            end_dt   = datetime.combine(e_date, datetime.strptime(e_time, "%H:%M").time())
            
            # clamp to MIN_UTC expressed in local time
            if start_dt < min_local_dt:
                start_dt = min_local_dt
                st.warning(f"Start time clamped to minimum: {min_local_dt.strftime('%Y-%m-%d %H:%M')} ({tz_label})")

            ro_times = list(pd.date_range(start_dt + timedelta(minutes=15), end_dt, freq="15min"))
            mrms_times = list(pd.date_range(ceil_to_hour(start_dt + timedelta(minutes=45)), end_dt.replace(minute=0, second=0, microsecond=0), freq="1H"))

            pb, msg = st.progress(0.0), st.empty()
            
            # 2. Download and CROP RO Data
            ro_list, ro_kept = [], []
            for i, t in enumerate(ro_times):
                msg.info(f"RO → {i+1}/{len(ro_times)}")
                da = load_precip("RO", t.to_pydatetime())
                if da is not None:
                    # SUBSET IMMEDIATELY
                    da = da.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
                    ro_list.append(da.astype("float32"))
                    ro_kept.append(t.to_pydatetime())
            
            if len(ro_list) < 4: raise RuntimeError("Insufficient RO data.")
            ro = xr.concat(ro_list, dim="time").assign_coords(time=ro_kept)
            del ro_list; gc.collect()

            # 3. Download and CROP MRMS Data
            mrms_list, mrms_kept = [], []
            for j, t in enumerate(mrms_times):
                msg.info(f"MRMS → {j+1}/{len(mrms_times)}")
                da = load_precip("MRMS", t.to_pydatetime())
                if da is not None:
                    # SUBSET IMMEDIATELY
                    da = da.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
                    mrms_list.append(da.astype("float32"))
                    mrms_kept.append(t.to_pydatetime())
            
            if not mrms_list: raise RuntimeError("No MRMS data found.")
            mrms = xr.concat(mrms_list, dim="time").assign_coords(time=mrms_kept)
            del mrms_list; gc.collect()
            
            # 4. Alignment (Now much faster because arrays are tiny)
            msg.info("Aligning grids...")
            ro = align_ro_to_mrms_grid_nearest(ro, mrms.isel(time=0))
            gc.collect()

            # 5. Bias Scaling
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

            # 6. Final Scaling and PNG Generation
            msg.info("Rendering frames...")
            cache, stats = {}, []
            m_ends = [pd.to_datetime(str(x)).to_pydatetime() for x in mrms.time.values]
            
            for t in ro.time.values:
                tdt = pd.to_datetime(str(t)).to_pydatetime()
                h_end = next((T for T in m_ends if T >= tdt), None)
                if h_end and tdt >= h_end - timedelta(minutes=45):
                    h_idx = m_ends.index(h_end)
                    # Apply spatial scaling
                    s_frame = (drop_time_coord(ro.sel(time=tdt)) * drop_time_coord(scaling.isel(time=h_idx))).astype("float32")
                    
                    img, bnds = save_frame_png(s_frame, tdt)
                    lbl = tdt.strftime("%Y-%m-%d %H:%M")
                    cache[lbl] = {"path": img, "bounds": bnds}
                    stats.append({"time": tdt, "rain_in": watershed_mean_inch(s_frame, st.session_state.active_gdf)})

            st.session_state.is_playing = False
            st.session_state.current_time_index = 0
            st.session_state.radar_cache = cache
            st.session_state.time_list = sorted(cache.keys())
            st.session_state.current_time_label = st.session_state.time_list[0] if st.session_state.time_list else None
            st.session_state.current_time_index = 0
            st.session_state.is_playing = False
            st.session_state.basin_vault[basin_name] = pd.DataFrame(stats)
            st.session_state.mode = "view"
            msg.success("Complete."); st.rerun()
        except Exception as e:
            st.error(f"Error: {e}"); st.exception(e); st.stop()

# =============================
# 6) MAIN DISPLAY
# =============================

# ---- A) Folium municipality picker (BEFORE processing) ----
if st.session_state.mode == "select":
    st.markdown("### Municipality selection (click a polygon)")
    if st.session_state.show_munis:
        muni_geojson = load_munis_geojson(MUNI_GEOJSON_PATH)

        # pick a sensible center: use watershed if uploaded, else NJ-ish
        if st.session_state.active_gdf is not None:
            b = st.session_state.active_gdf.total_bounds
            center = ((b[1] + b[3]) / 2, (b[0] + b[2]) / 2)
            zoom = 10
        else:
            center = (40.1, -74.6)
            zoom = 8

        render_muni_picker_map(muni_geojson, center=center, zoom=zoom)
    else:
        st.info("Enable 'Show NJ Municipalities' to pick by click.")

# ---- B) PyDeck radar viewer (AFTER processing) ----
else:
    layers = []

    if st.session_state.time_list and st.session_state.current_time_label:
        if st.session_state.current_time_label not in st.session_state.time_list:
            st.session_state.current_time_label = st.session_state.time_list[0]

        curr = st.session_state.radar_cache[st.session_state.current_time_label]
        layers.append(pdk.Layer(
            "BitmapLayer",
            image=curr["path"],
            bounds=curr["bounds"],
            opacity=0.70
        ))

    if st.session_state.active_gdf is not None:
        layers.append(pdk.Layer(
            "GeoJsonLayer",
            st.session_state.active_gdf.__geo_interface__,
            stroked=True,
            filled=False,
            get_line_color=[255, 255, 255],
            line_width_min_pixels=3
        ))

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=st.session_state.map_view,
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
    )

    deck_key = f"map_{st.session_state.current_time_label}"
    st.pydeck_chart(deck, use_container_width=True, height=1000, key=deck_key)


# =============================
# 7) FLOATING CONTROLS
# =============================
import streamlit.components.v1 as components

if st.session_state.mode == "view" and st.session_state.time_list:
    with st.container():
        st.markdown('<div id="control_bar_anchor"></div>', unsafe_allow_html=True)

        col_slider, col_txt = st.columns([14, 4])

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

# =============================
# 8) OUTPUTS (STYLIZED BAR CHART)
# =============================
with st.sidebar:
    if st.session_state.basin_vault:
        st.divider()
        st.subheader("Rainfall Analysis")
        
        df = st.session_state.basin_vault[basin_name]
        
        fig, ax = plt.subplots(figsize=(5, 3.5))
        # Use a calculated width for bars (approx 10 mins in day-units)
        ax.bar(df["time"], df["rain_in"], color="#01a0fe", width=0.006, edgecolor="white", linewidth=0.3)
        
        # Highlight current time in the bar chart
        curr_time = pd.to_datetime(st.session_state.current_time_label)
        ax.axvline(curr_time, color="#ff0101", linestyle="--", alpha=0.8, lw=1)
        
        ax.set_ylabel("Inches", color="gray")
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        ax.tick_params(colors='gray', labelsize=8)
        
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        csv_download_link(df, f"{basin_name}_rain.csv", f"Export {basin_name} Data")



