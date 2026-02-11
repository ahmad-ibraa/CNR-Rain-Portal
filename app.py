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
import gc
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import rioxarray
import base64
from zoneinfo import ZoneInfo
import folium
from folium.features import GeoJsonTooltip
from streamlit_folium import st_folium
from shapely.geometry import Point
import json
import streamlit.components.v1 as components

# =========================================================
# 0) CONSTANTS
# =========================================================
UTC_TZ = ZoneInfo("UTC")

TZ_OPTIONS = {
    "UTC": "UTC",
    "New York (ET)": "America/New_York",
    "Chicago (CT)": "America/Chicago",
    "Denver (MT)": "America/Denver",
    "Los Angeles (PT)": "America/Los_Angeles",
}

MIN_UTC = datetime(2020, 10, 15, 0, 0, 0)  # naive UTC baseline

MRMS_S3_BASE = "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/MultiSensor_QPE_01H_Pass2_00.00"
RO_S3_BASE   = "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/RadarOnly_QPE_15M_00.00"

MUNI_GEOJSON_PATH = "nj_munis.geojson"
MUNI_NAME_FIELD   = "GNIS_NAME"

RADAR_COLORS = ["#76fffe", "#01a0fe", "#0001ef", "#01ef01", "#019001", "#ffff01", "#e7c001", "#ff9000", "#ff0101"]
RADAR_CMAP = ListedColormap(RADAR_COLORS)

# =========================================================
# 1) PAGE CONFIG
# =========================================================
st.set_page_config(layout="wide", page_title="CNR Radar Portal", initial_sidebar_state="expanded")

# =========================================================
# 2) CSS: FIX TOP OFFSET (keep your existing CSS; add this)
# =========================================================
st.markdown(
    """
<style>
/* Ensure nothing pushes the main content down */
.main, section.main { padding-top: 0rem !important; margin-top: 0rem !important; }
.main .block-container { padding-top: 0rem !important; margin-top: 0rem !important; }
div[data-testid="stAppViewContainer"] { padding-top: 0rem !important; margin-top: 0rem !important; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# 3) STATE
# =========================================================
def init_state():
    defaults = {
        # radar rendering cache
        "radar_cache": {},
        "time_list": [],
        "current_time_label": None,
        "is_playing": False,
        "current_time_index": 0,

        # boundaries
        "active_gdf": None,             # used for processing + pydeck outline (chosen boundary)
        "boundary_source": None,        # "upload" | "muni"
        "boundary_name": None,

        # multi-muni selection (folium overlays)
        "muni_layers": {},              # dict[name] -> geojson dict (single-feature FC)

        # view + outputs
        "basin_vault": {},
        "map_view": pdk.ViewState(latitude=40.1, longitude=-74.6, zoom=8),
        "img_dir": tempfile.mkdtemp(prefix="radar_png_"),
        "processing_msg": "",

        # UI
        "tz_name": "America/New_York",
        "show_munis": True,
        "search_query": "",

        # mode
        "mode": "select",               # "select" (folium) | "view" (pydeck)
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

LOCAL_TZ = ZoneInfo(st.session_state.tz_name)

# =========================================================
# 4) CACHED LOADERS
# =========================================================
@st.cache_data(show_spinner=False)
def load_munis_cached(path: str) -> tuple[gpd.GeoDataFrame, dict]:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    geojson_dict = json.loads(gdf.to_json())
    return gdf, geojson_dict

@st.cache_data(show_spinner=False, ttl=24 * 3600)
def geocode_place(query: str):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": 1}
    headers = {"User-Agent": "cnr-rainfall-streamlit/1.0 (contact: you@example.com)"}
    r = requests.get(url, params=params, headers=headers, timeout=15)
    r.raise_for_status()
    js = r.json()
    if not js:
        return None
    return float(js[0]["lat"]), float(js[0]["lon"])

# =========================================================
# 5) HELPERS
# =========================================================
def csv_download_link(df: pd.DataFrame, filename: str, label: str):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(csv_bytes).decode()
    href = f"data:text/csv;base64,{b64}"
    st.markdown(
        f'<div class="output-link">📄 <a download="{filename}" href="{href}">{label}</a></div>',
        unsafe_allow_html=True,
    )

def utc_naive_to_local_naive(dt_utc_naive: datetime) -> datetime:
    aware_utc = dt_utc_naive.replace(tzinfo=UTC_TZ)
    aware_local = aware_utc.astimezone(LOCAL_TZ)
    return aware_local.replace(tzinfo=None)

def local_naive_to_utc(dt_local_naive: datetime) -> datetime:
    aware_local = dt_local_naive.replace(tzinfo=LOCAL_TZ)
    aware_utc = aware_local.astimezone(UTC_TZ)
    return aware_utc.replace(tzinfo=None)

def ceil_to_hour(dt: datetime) -> datetime:
    dt0 = dt.replace(minute=0, second=0, microsecond=0)
    return dt0 if dt == dt0 else (dt0 + timedelta(hours=1))

def normalize_grid(da: xr.DataArray) -> xr.DataArray:
    return da.assign_coords(latitude=da.latitude.round(4), longitude=da.longitude.round(4))

def normalize_to_latlon(da: xr.DataArray) -> xr.DataArray:
    da = normalize_grid(da)
    drop_coords = [c for c in da.coords if c not in ("latitude", "longitude")]
    da = da.drop_vars(drop_coords, errors="ignore")
    for d in list(da.dims):
        if d not in ("latitude", "longitude"):
            da = da.isel({d: 0})
    da = da.sortby("latitude", ascending=False).sortby("longitude")
    return da

def normalize_to_tlatlon(da: xr.DataArray) -> xr.DataArray:
    da = normalize_grid(da)
    drop_coords = [c for c in da.coords if c not in ("time", "latitude", "longitude")]
    da = da.drop_vars(drop_coords, errors="ignore")
    for d in list(da.dims):
        if d not in ("time", "latitude", "longitude"):
            da = da.isel({d: 0})
    da = da.sortby("latitude", ascending=False).sortby("longitude")
    return da

def nearest_index(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    src = np.asarray(src, dtype="float64")
    tgt = np.asarray(tgt, dtype="float64")
    descending = src[0] > src[-1]
    src_work = src[::-1] if descending else src
    pos = np.searchsorted(src_work, tgt)
    pos = np.clip(pos, 1, len(src_work) - 1)
    left = src_work[pos - 1]
    right = src_work[pos]
    choose_right = (np.abs(tgt - right) < np.abs(tgt - left))
    idx = pos - 1 + choose_right.astype(int)
    if descending:
        idx = (len(src) - 1) - idx
    return idx

def align_ro_to_mrms_grid_nearest(ro: xr.DataArray, mrms0: xr.DataArray) -> xr.DataArray:
    ro = normalize_to_tlatlon(ro)
    mrms0 = normalize_to_latlon(mrms0)

    ro_lat = np.asarray(ro.latitude.values, dtype="float64")
    ro_lon = np.asarray(ro.longitude.values, dtype="float64")
    mr_lat = np.asarray(mrms0.latitude.values, dtype="float64")
    mr_lon = np.asarray(mrms0.longitude.values, dtype="float64")

    lat_idx = nearest_index(ro_lat, mr_lat)
    lon_idx = nearest_index(ro_lon, mr_lon)

    data = ro.values.astype("float32")
    aligned = data[:, lat_idx[:, None], lon_idx[None, :]]
    return xr.DataArray(
        aligned,
        dims=("time", "latitude", "longitude"),
        coords={"time": ro.time.values, "latitude": mr_lat, "longitude": mr_lon},
    )

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
    if ws_gdf is None:
        return float(da_full.mean().values) / 25.4
    try:
        sub = da_full.rio.write_crs("EPSG:4326", inplace=False)
        clipped = sub.rio.clip(ws_gdf.geometry, ws_gdf.crs, drop=True)
        if clipped.size == 0:
            return float(da_full.mean().values) / 25.4
        return float(clipped.mean().values) / 25.4
    except Exception:
        return float(da_full.mean().values) / 25.4

def load_precip(file_type: str, dt_local_naive: datetime) -> xr.DataArray | None:
    dt_utc = local_naive_to_utc(dt_local_naive)
    ts, ymd = dt_utc.strftime("%Y%m%d-%H%M00"), dt_utc.strftime("%Y%m%d")
    base = RO_S3_BASE if file_type == "RO" else MRMS_S3_BASE
    filename = f"MRMS_{'RadarOnly_QPE_15M' if file_type=='RO' else 'MultiSensor_QPE_01H_Pass2'}_00.00_{ts}.grib2.gz"
    url = f"{base}/{ymd}/{filename}"

    tmp_path = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False).name
    try:
        r = requests.get(url, stream=True, timeout=25)
        if r.status_code == 404:
            return None
        r.raise_for_status()

        with gzip.GzipFile(fileobj=r.raw) as gz, open(tmp_path, "wb") as f:
            shutil.copyfileobj(gz, f)

        with xr.open_dataset(tmp_path, engine="cfgrib", backend_kwargs={"indexpath": ""}) as ds:
            var = list(ds.data_vars)[0]
            da = ds[var].clip(min=0).load()

        da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180).sortby("longitude")
        da = da.rio.write_crs("EPSG:4326")
        return normalize_to_latlon(da)

    except Exception:
        return None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# =========================================================
# 6) FOLIUM: MULTI-MUNICIPALITY PICKER (NO PYDECK SWITCH)
# =========================================================
def render_muni_picker_map(muni_gdf: gpd.GeoDataFrame, muni_geojson: dict, center=(40.1, -74.6), zoom=8):
    m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB dark_matter", control_scale=False)

    # base muni layer (clickable)
    if st.session_state.show_munis:
        folium.GeoJson(
            muni_geojson,
            name="NJ Municipalities",
            style_function=lambda feat: {"color": "#ffffff", "weight": 1, "fillColor": "#000000", "fillOpacity": 0.00},
            highlight_function=lambda feat: {"weight": 3, "color": "#01a0fe", "fillOpacity": 0.10},
            tooltip=GeoJsonTooltip(fields=[MUNI_NAME_FIELD], aliases=["Municipality:"]),
        ).add_to(m)

    # overlays: selected muni boundaries (multi)
    for name, gj in st.session_state.muni_layers.items():
        folium.GeoJson(
            gj,
            name=name,
            style_function=lambda feat: {"color": "#01a0fe", "weight": 3, "fillOpacity": 0.0},
        ).add_to(m)

    out = st_folium(
        m,
        height=950,
        width=None,
        key="muni_map",
        returned_objects=["last_clicked"],
    )

    clicked = out.get("last_clicked")
    if not (st.session_state.show_munis and clicked and isinstance(clicked, dict) and "lat" in clicked and "lng" in clicked):
        return

    pt = Point(clicked["lng"], clicked["lat"])
    hit = muni_gdf[muni_gdf.geometry.contains(pt)]
    if hit.empty:
        hit = muni_gdf[muni_gdf.geometry.intersects(pt)]
    if hit.empty:
        return

    sel = hit.iloc[[0]].copy()
    name = str(sel.iloc[0][MUNI_NAME_FIELD]) if MUNI_NAME_FIELD in sel.columns else "Selected municipality"

    # If already selected, do nothing
    if name in st.session_state.muni_layers:
        return

    # store overlay (folium display)
    st.session_state.muni_layers[name] = json.loads(sel.to_json())

    # OPTIONAL: also set "active_gdf" to last clicked, so processing uses it by default
    st.session_state.active_gdf = sel
    st.session_state.boundary_source = "muni"
    st.session_state.boundary_name = name

    # update view center for later pydeck
    b = sel.total_bounds
    st.session_state.map_view = pdk.ViewState(
        latitude=(b[1] + b[3]) / 2,
        longitude=(b[0] + b[2]) / 2,
        zoom=11,
    )

    st.rerun()

# =========================================================
# 7) SIDEBAR (DE-CLUTTERED)
# =========================================================
with st.sidebar:
    # checkbox only (no header/description)
    st.session_state.show_munis = st.checkbox("Show NJ Municipalities", value=st.session_state.show_munis)

    # city search: input + icon button beside it
    c1, c2 = st.columns([12, 2])
    with c1:
        q = st.text_input(
            "City / place",
            value=st.session_state.search_query,
            placeholder="e.g., Newark, NJ",
            label_visibility="collapsed",
        )
        st.session_state.search_query = q
    with c2:
        do_zoom = st.button("🔍", use_container_width=True)

    if do_zoom and q.strip():
        hit = geocode_place(q.strip())
        if hit is None:
            st.warning("No results found.")
        else:
            lat, lon = hit
            st.session_state.map_view = pdk.ViewState(latitude=lat, longitude=lon, zoom=11)
            st.rerun()

    # minimal selected-muni list (only appears when you have selections)
    if st.session_state.muni_layers:
        st.caption("Selected municipalities")
        for name in list(st.session_state.muni_layers.keys()):
            r1, r2 = st.columns([10, 2])
            r1.write(name)
            if r2.button("✕", key=f"rm_{name}", use_container_width=True):
                del st.session_state.muni_layers[name]
                # if you removed the active muni, clear active_gdf if it matched
                if st.session_state.boundary_source == "muni" and st.session_state.boundary_name == name:
                    st.session_state.active_gdf = None
                    st.session_state.boundary_source = None
                    st.session_state.boundary_name = None
                st.rerun()

    # ---- Everything else stays as-is (timezone/date/upload/run) ----
    st.divider()

    tz_label = st.selectbox(
        "Time Zone",
        list(TZ_OPTIONS.keys()),
        index=list(TZ_OPTIONS.values()).index(st.session_state.tz_name)
        if st.session_state.tz_name in TZ_OPTIONS.values()
        else 1,
    )
    st.session_state.tz_name = TZ_OPTIONS[tz_label]
    LOCAL_TZ = ZoneInfo(st.session_state.tz_name)

    min_local_dt = utc_naive_to_local_naive(MIN_UTC)
    min_local_date = min_local_dt.date()

    s_date = st.date_input("Start Date", value=max(datetime.now().date(), min_local_date), min_value=min_local_date)
    e_date = st.date_input("End Date", value=max(datetime.now().date(), min_local_date), min_value=min_local_date)

    hours = [f"{h:02d}:00" for h in range(24)]
    c3, c4 = st.columns(2)
    s_time = c3.selectbox("Start Time", hours, index=0)
    e_time = c4.selectbox("End Time", hours, index=0)

    up_zip = st.file_uploader("Watershed Boundary (ZIP)", type="zip")
    basin_name = up_zip.name.replace(".zip", "") if up_zip else "Default_Basin"

    if up_zip:
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(up_zip, "r") as z:
                z.extractall(td)
            shps = list(Path(td).rglob("*.shp"))
            if shps:
                gdf = gpd.read_file(shps[0]).to_crs("EPSG:4326")
                st.session_state.active_gdf = gdf
                st.session_state.boundary_source = "upload"
                st.session_state.boundary_name = basin_name

                b = gdf.total_bounds
                st.session_state.map_view = pdk.ViewState(
                    latitude=(b[1] + b[3]) / 2,
                    longitude=(b[0] + b[2]) / 2,
                    zoom=11,
                )
                # stay in folium select until processing is run
                st.session_state.mode = "select"
                st.rerun()

    if st.session_state.processing_msg:
        st.caption(st.session_state.processing_msg)

    if st.button("Run Processing", use_container_width=True):
        try:
            if st.session_state.active_gdf is None:
                st.session_state.processing_msg = "Select a municipality or upload a watershed boundary ZIP first."
                st.rerun()

            b = st.session_state.active_gdf.total_bounds
            BUFFER_DEG = 0.35
            lon_min, lat_min, lon_max, lat_max = b[0]-BUFFER_DEG, b[1]-BUFFER_DEG, b[2]+BUFFER_DEG, b[3]+BUFFER_DEG

            start_dt = datetime.combine(s_date, datetime.strptime(s_time, "%H:%M").time())
            end_dt   = datetime.combine(e_date, datetime.strptime(e_time, "%H:%M").time())

            if start_dt < min_local_dt:
                start_dt = min_local_dt
                st.warning(f"Start time clamped to minimum: {min_local_dt.strftime('%Y-%m-%d %H:%M')} ({tz_label})")

            ro_times   = list(pd.date_range(start_dt + timedelta(minutes=15), end_dt, freq="15min"))
            mrms_times = list(pd.date_range(
                ceil_to_hour(start_dt + timedelta(minutes=45)),
                end_dt.replace(minute=0, second=0, microsecond=0),
                freq="1H"
            ))

            pb, msg = st.progress(0.0), st.empty()
            total_steps = max(1, len(ro_times) + len(mrms_times) + 6)
            step_i = 0

            # --- RO (crop early) ---
            ro_list, ro_kept = [], []
            for i, t in enumerate(ro_times):
                msg.info(f"RO → {i+1}/{len(ro_times)}")
                da = load_precip("RO", t.to_pydatetime())
                if da is not None:
                    da = da.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
                    ro_list.append(da.astype("float32"))
                    ro_kept.append(t.to_pydatetime())
                step_i += 1
                pb.progress(min(1.0, step_i/total_steps))

            if len(ro_list) < 4:
                raise RuntimeError("Insufficient RO data.")
            ro = xr.concat(ro_list, dim="time").assign_coords(time=ro_kept)
            del ro_list
            gc.collect()

            # --- MRMS (crop early) ---
            mrms_list, mrms_kept = [], []
            for j, t in enumerate(mrms_times):
                msg.info(f"MRMS → {j+1}/{len(mrms_times)}")
                da = load_precip("MRMS", t.to_pydatetime())
                if da is not None:
                    da = da.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
                    mrms_list.append(da.astype("float32"))
                    mrms_kept.append(t.to_pydatetime())
                step_i += 1
                pb.progress(min(1.0, step_i/total_steps))

            if not mrms_list:
                raise RuntimeError("No MRMS data found.")
            mrms = xr.concat(mrms_list, dim="time").assign_coords(time=mrms_kept)
            del mrms_list
            gc.collect()

            # --- Align ---
            msg.info("Aligning grids...")
            ro = align_ro_to_mrms_grid_nearest(ro, mrms.isel(time=0))
            step_i += 1
            pb.progress(min(1.0, step_i/total_steps))
            gc.collect()

            # --- Bias scaling ---
            msg.info("Calculating Bias Scaling...")
            ro_hourly, v_times = [], []
            for T in mrms.time.values:
                Tdt = pd.to_datetime(str(T)).to_pydatetime()
                block = ro.sel(time=slice(Tdt - timedelta(minutes=45) - timedelta(seconds=1), Tdt))
                if block.sizes.get("time", 0) == 4:
                    ro_hourly.append(block.sum(dim="time").astype("float32"))
                    v_times.append(Tdt)

            if not v_times:
                raise RuntimeError("No matching RO/MRMS windows found.")

            mrms_subset = mrms.sel(time=pd.to_datetime(v_times))
            ro_hr_da = xr.concat(ro_hourly, dim="time").assign_coords(time=pd.to_datetime(v_times))
            scaling = xr.where(ro_hr_da > 0.01, mrms_subset / ro_hr_da, 0.0).clip(0, 50).astype("float32")

            del ro_hr_da, ro_hourly, mrms_subset
            step_i += 1
            pb.progress(min(1.0, step_i/total_steps))
            gc.collect()

            # --- Final scaling + PNG frames ---
            msg.info("Rendering frames...")
            cache, stats = {}, []
            m_ends = [pd.to_datetime(str(x)).to_pydatetime() for x in mrms.time.values]

            for t in ro.time.values:
                tdt = pd.to_datetime(str(t)).to_pydatetime()
                h_end = next((T for T in m_ends if T >= tdt), None)
                if h_end and tdt >= h_end - timedelta(minutes=45):
                    h_idx = m_ends.index(h_end)
                    s_frame = (drop_time_coord(ro.sel(time=tdt)) * drop_time_coord(scaling.isel(time=h_idx))).astype("float32")

                    img, bnds = save_frame_png(s_frame, tdt)
                    lbl = tdt.strftime("%Y-%m-%d %H:%M")
                    cache[lbl] = {"path": img, "bounds": bnds}
                    stats.append({"time": tdt, "rain_in": watershed_mean_inch(s_frame, st.session_state.active_gdf)})

            st.session_state.radar_cache = cache
            st.session_state.time_list = sorted(cache.keys())
            st.session_state.current_time_label = st.session_state.time_list[0] if st.session_state.time_list else None
            st.session_state.is_playing = False
            st.session_state.current_time_index = 0
            st.session_state.basin_vault[basin_name] = pd.DataFrame(stats)

            st.session_state.mode = "view"
            msg.success("Complete.")
            st.rerun()

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)
            st.stop()

    # Rainfall chart + export (unchanged)
    if st.session_state.basin_vault and basin_name in st.session_state.basin_vault:
        st.divider()
        df = st.session_state.basin_vault[basin_name]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(df["time"], df["rain_in"], width=0.006, edgecolor="white", linewidth=0.3)
        if st.session_state.current_time_label:
            curr_time = pd.to_datetime(st.session_state.current_time_label)
            ax.axvline(curr_time, linestyle="--", alpha=0.8, lw=1)
        ax.set_ylabel("Inches", color="gray")
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")
        ax.tick_params(colors="gray", labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        csv_download_link(df, f"{basin_name}_rain.csv", f"Export {basin_name} Data")

# =========================================================
# 8) MAIN DISPLAY (IMPORTANT: render map FIRST to avoid offset)
# =========================================================
if st.session_state.mode == "select":
    muni_gdf, muni_geojson = load_munis_cached(MUNI_GEOJSON_PATH)

    # center/zoom: if you have an active boundary, center there
    if st.session_state.active_gdf is not None:
        b = st.session_state.active_gdf.total_bounds
        center = ((b[1] + b[3]) / 2, (b[0] + b[2]) / 2)
        zoom = 10
    else:
        center = (40.1, -74.6)
        zoom = 8

    # render folium map (multi-muni overlays; does NOT switch to pydeck)
    render_muni_picker_map(muni_gdf, muni_geojson, center=center, zoom=zoom)

else:
    layers = []

    # Radar bitmap
    if st.session_state.time_list and st.session_state.current_time_label:
        if st.session_state.current_time_label not in st.session_state.time_list:
            st.session_state.current_time_label = st.session_state.time_list[0]
        curr = st.session_state.radar_cache[st.session_state.current_time_label]
        layers.append(
            pdk.Layer("BitmapLayer", image=curr["path"], bounds=curr["bounds"], opacity=0.70)
        )

    # Boundary outline (uses active_gdf)
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
    st.pydeck_chart(deck, use_container_width=True, height=1000, key=f"map_{st.session_state.current_time_label}")

# =========================================================
# 9) FLOATING CONTROLS (unchanged)
# =========================================================
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
                key="timeline_slider",
            )
            if chosen != st.session_state.current_time_label:
                st.session_state.current_time_label = chosen
                st.rerun()

        with col_txt:
            ts = st.session_state.current_time_label or ""
            st.markdown(
                f'<p class="timestamp" style="margin:0; font-family:monospace; font-size:14px; font-weight:bold; line-height:44px;">{ts}</p>',
                unsafe_allow_html=True,
            )

    components.html(
        """
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
""",
        height=0,
    )
