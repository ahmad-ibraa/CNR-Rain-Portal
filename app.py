import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
from datetime import datetime
import os

# --- 1. THE "NO-ESCAPE" FULLSCREEN CSS ---
st.set_page_config(layout="wide", page_title="CNR Radar Portal")

st.markdown("""
    <style>
        /* KILL ALL PADDING EVERYWHERE */
        html, body, [data-testid="stAppViewContainer"], .main, .block-container {
            margin: 0 !important;
            padding: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            overflow: hidden !important;
        }

        /* FORCE PYDECK TO BE THE LITERAL BACKGROUND */
        iframe[title="pydeck.io"] {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            border: none !important;
            z-index: 0 !important;
        }

        /* FLOATING SIDEBAR (GLASS EFFECT) */
        [data-testid="stSidebar"] {
            background-color: rgba(20, 20, 20, 0.7) !important;
            backdrop-filter: blur(15px) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
            z-index: 100 !important;
        }

        /* FLOATING SLIDER AT THE BOTTOM */
        div[data-testid="stSlider"] {
            position: fixed !important;
            bottom: 40px !important;
            left: 360px !important; 
            right: 40px !important;
            z-index: 1000 !important;
            background: rgba(10, 10, 10, 0.8) !important;
            padding: 10px 40px !important;
            border-radius: 100px !important;
        }

        /* HIDE UI ELEMENTS */
        header, footer, [data-testid="stHeader"] { 
            visibility: hidden !important; 
            height: 0 !important;
        }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA / STATE (Simplified for testing) ---
if 'map_view' not in st.session_state:
    st.session_state.map_view = pdk.ViewState(latitude=40.7, longitude=-74.0, zoom=9)

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("CNR GIS Portal")
    st.date_input("Select Date", value=datetime.now().date())
    st.file_uploader("Upload Watershed", type="zip")
    if st.button("PROCESS DATA"):
        st.success("Ready!")

# --- 4. THE MAP ---
# Crucial: Use width="stretch" or leave default with full-page CSS
st.pydeck_chart(pdk.Deck(
    initial_view_state=st.session_state.map_view,
    map_style="mapbox://styles/mapbox/dark-v10", # Dark mode is better for overlays
    layers=[]
), key="radar_map")
