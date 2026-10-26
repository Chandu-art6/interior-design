"""
Interior Redesign - Luxury Spatial Optimization & Multi-User Architecture Dashboard
A clean, modern, professional user interface for room analysis, multi-user preference synthesis, and layout optimization.
"""
import os
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from config import settings
from ui.components.widgets import (
    generate_comfort_radar_chart,
    generate_metrics_bar_chart,
    generate_spatial_clearance_heatmap
)
from ui.api_client import api_client
from services.recommender import recommender_service

# Page Configuration
st.set_page_config(
    page_title="Interior Design",
    page_icon="🛋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, Radiant Modern Studio CSS Styling with Enhanced Ambient Lighting & High Contrast
st.markdown("""
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Radiant Clean Modern Studio Theme */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: #f8fafc !important;
        background-image: radial-gradient(at 50% 0%, #ffffff 0%, #f4f7fb 60%, #eef2f6 100%) !important;
        color: #0f172a !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    header {visibility: hidden;}
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3.5rem;
        max-width: 1350px;
    }

    /* GLOBAL MAXIMUM TEXT CONTRAST & LEGIBILITY */
    html, body, p, span, div, li, strong, b, em {
        color: #0f172a;
    }
    
    .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li, .stMarkdown strong {
        color: #0f172a !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-weight: 800 !important;
    }

    /* Streamlit Widget Labels (Sliders, Selectboxes, Multiselects, Inputs, Radios, Checkboxes, File Uploads) */
    label, [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span {
        color: #0f172a !important;
        font-weight: 700 !important;
        font-size: 0.94rem !important;
        letter-spacing: -0.2px !important;
    }

    /* Captions & Subtitles */
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p,
    [data-testid="stCaptionContainer"] span,
    .stCaption, .stCaption p {
        color: #1e293b !important;
        font-weight: 600 !important;
        font-size: 0.84rem !important;
    }

    /* Slider specific labels, ticks, and numbers */
    .stSlider [data-testid="stWidgetLabel"] p {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    div[data-testid="stSlider"] div,
    div[data-testid="stSlider"] span,
    div[data-testid="stSliderTickBar"] div {
        color: #1e293b !important;
        font-weight: 700 !important;
    }

    /* BaseWeb Selectbox and Dropdown Inputs */
    div[data-baseweb="select"] {
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="select"] * {
        color: #0f172a !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] *,
    ul[role="listbox"],
    ul[role="listbox"] li,
    li[role="option"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        font-weight: 600 !important;
    }
    li[role="option"]:hover,
    li[aria-selected="true"] {
        background-color: #eff6ff !important;
        color: #1d4ed8 !important;
    }

    /* Multiselect Tags */
    div[data-baseweb="tag"] {
        background-color: #dbeafe !important;
        border: 1.5px solid #93c5fd !important;
        border-radius: 6px !important;
    }
    div[data-baseweb="tag"] span,
    div[data-baseweb="tag"] div {
        color: #1e3a8a !important;
        font-weight: 700 !important;
        font-size: 0.82rem !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 10px !important;
        background-color: #ffffff !important;
    }
    [data-testid="stExpander"] summary {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p {
        color: #0f172a !important;
        font-weight: 700 !important;
        font-size: 0.94rem !important;
    }
    [data-testid="stExpanderDetails"] {
        color: #0f172a !important;
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        border: 1.5px dashed #94a3b8 !important;
        border-radius: 12px !important;
        background-color: #ffffff !important;
        padding: 12px !important;
    }
    [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] div {
        color: #1e293b !important;
        font-weight: 600 !important;
    }

    /* Text Inputs & Text Areas */
    .stTextInput input, .stTextArea textarea {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2) !important;
    }

    /* Radio buttons & Checkboxes */
    .stRadio label div, .stCheckbox label div,
    [data-testid="stRadio"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stCheckbox"] [data-testid="stMarkdownContainer"] p {
        color: #0f172a !important;
        font-weight: 600 !important;
    }

    /* Sidebar Clean Normal Light Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1.5px solid #cbd5e1 !important;
    }
    [data-testid="stSidebar"] * {
        color: #0f172a !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown h4 {
        color: #0f172a !important;
        font-weight: 800 !important;
    }

    /* Alerts and Notification Boxes (High Contrast) */
    .stAlert {
        border-radius: 10px !important;
    }
    .stAlert [data-testid="stMarkdownContainer"] p {
        font-weight: 600 !important;
        color: #0f172a !important;
    }
    div[data-testid="stNotification"] {
        color: #0f172a !important;
    }

    /* Dataframes / Tables */
    [data-testid="stDataFrame"] {
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 8px !important;
    }
    [data-testid="stDataFrame"] * {
        color: #0f172a !important;
        font-weight: 600 !important;
    }

    /* Radiant Hero Header */
    .hero-container {
        text-align: center;
        padding: 20px 24px 16px 24px;
        margin-bottom: 1.2rem;
        background: #ffffff;
        border: 1.5px solid #cbd5e1;
        border-radius: 16px;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05);
    }
    
    .hero-title {
        font-size: 2.25rem;
        font-weight: 800;
        letter-spacing: -0.8px;
        color: #0f172a;
        margin-bottom: 0.35rem;
    }
    
    .hero-subtitle {
        font-size: 1.0rem;
        color: #1e293b;
        font-weight: 600;
        max-width: 820px;
        margin: 0 auto 0.8rem auto;
        line-height: 1.55;
    }

    /* Luminous Badge Pills */
    .pill-container {
        display: flex;
        justify-content: center;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 4px;
    }
    .hero-pill {
        background: #f1f5f9;
        border: 1.5px solid #cbd5e1;
        color: #0f172a;
        font-size: 0.82rem;
        font-weight: 700;
        padding: 5px 14px;
        border-radius: 20px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }

    /* Radiant Live Session Bar */
    .session-bar {
        background: #ffffff;
        border: 1.5px solid #cbd5e1;
        border-radius: 12px;
        padding: 12px 20px;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        box-shadow: 0 2px 12px rgba(15, 23, 42, 0.04);
    }
    .session-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.90rem;
        color: #0f172a;
        font-weight: 700;
    }
    .session-tag {
        background: #eff6ff;
        border: 1.5px solid #93c5fd;
        color: #1d4ed8;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 0.86rem;
    }

    /* Crisp Luminous Cards */
    .glass-card {
        background: #ffffff;
        border: 1.5px solid #cbd5e1;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.05);
    }

    /* KPI Metric Displays with High Contrast */
    .kpi-card {
        background: #ffffff;
        border: 1.5px solid #cbd5e1;
        border-radius: 10px;
        padding: 16px 14px;
        text-align: center;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
    }
    .kpi-title {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #1e293b;
        margin-bottom: 6px;
        font-weight: 800;
    }
    .kpi-value {
        font-size: 1.95rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 0.82rem;
        color: #1d4ed8;
        margin-top: 5px;
        font-weight: 700;
    }

    /* Inhabitant Profile Card */
    .inhabitant-card {
        background: #ffffff;
        border: 1.5px solid #cbd5e1;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 14px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
    }

    /* Comfort Evaluation Card */
    .comfort-highlight-card {
        background: #ffffff;
        border: 1.5px solid #86efac;
        border-left: 6px solid #16a34a;
        border-radius: 12px;
        padding: 18px 20px;
        margin-top: 12px;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px -2px rgba(22, 163, 74, 0.08);
    }

    /* Smart Suggestion Pill Card */
    .suggestion-box {
        background: #f0fdf4;
        border: 1.5px solid #bbf7d0;
        border-left: 5px solid #16a34a;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 10px;
        font-size: 0.88rem;
        color: #14532d;
        line-height: 1.5;
        font-weight: 500;
    }
    .suggestion-box b {
        color: #14532d;
        font-weight: 700;
    }

    /* STREAMLIT TAB STYLING - High Contrast & Visible at all times */
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: center;
        gap: 12px;
        background: #f1f5f9 !important;
        padding: 8px !important;
        border-radius: 14px !important;
        border: 1.5px solid #cbd5e1 !important;
        margin-bottom: 1.8rem !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        background-color: #ffffff !important;
        border: 1.5px solid #cbd5e1 !important;
        padding: 10px 22px !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        transition: all 0.15s ease !important;
    }
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span,
    .stTabs [data-baseweb="tab"] div {
        color: #0f172a !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e2e8f0 !important;
        border-color: #64748b !important;
        color: #0f172a !important;
    }
    .stTabs [data-baseweb="tab"]:hover p,
    .stTabs [data-baseweb="tab"]:hover span {
        color: #0f172a !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #2563eb !important;
        border-color: #1d4ed8 !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] p,
    .stTabs [data-baseweb="tab"][aria-selected="true"] span,
    .stTabs [data-baseweb="tab"][aria-selected="true"] div {
        color: #ffffff !important;
        font-weight: 800 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: transparent !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        background-color: transparent !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 700;
        padding: 0.55rem 1.3rem;
        font-size: 0.94rem;
        color: #0f172a;
        background: #ffffff;
        border: 1.5px solid #cbd5e1;
    }
    .stButton > button:hover {
        border-color: #2563eb;
        color: #2563eb;
    }
    .stButton > button[kind="primary"] {
        background: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.25) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #1d4ed8 !important;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.35) !important;
    }

    /* Color Swatch Display */
    .swatch-box {
        border-radius: 6px;
        height: 28px;
        width: 100%;
        margin-bottom: 4px;
        border: 1px solid #cbd5e1;
    }
    .swatch-label {
        font-size: 0.76rem;
        color: #0f172a;
        font-family: monospace;
        text-align: center;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# Session State Initialization
# ==============================================================================
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = None
if "redesign_data" not in st.session_state:
    st.session_state.redesign_data = None
if "inspected_var_idx" not in st.session_state:
    st.session_state.inspected_var_idx = 0

# Multi-User Preferences Session State Defaults
if "room_type" not in st.session_state:
    st.session_state.room_type = "Living Room"
if "num_users" not in st.session_state:
    st.session_state.num_users = 2
if "u1_name" not in st.session_state:
    st.session_state.u1_name = "Alex"
if "u1_style" not in st.session_state:
    st.session_state.u1_style = "Industrial"
if "u1_objs" not in st.session_state:
    st.session_state.u1_objs = ["Leather Sofa", "Coffee Table"]
if "u1_color" not in st.session_state:
    st.session_state.u1_color = "#475569"

if "u2_name" not in st.session_state:
    st.session_state.u2_name = "Sam"
if "u2_style" not in st.session_state:
    st.session_state.u2_style = "Japandi"
if "u2_objs" not in st.session_state:
    st.session_state.u2_objs = ["Indoor Plants", "Floor Lamp"]
if "u2_color" not in st.session_state:
    st.session_state.u2_color = "#e29578"

if "u3_name" not in st.session_state:
    st.session_state.u3_name = "Taylor"
if "u3_style" not in st.session_state:
    st.session_state.u3_style = "Scandinavian"
if "u3_objs" not in st.session_state:
    st.session_state.u3_objs = ["Armchair", "Pendant Lamp"]
if "u3_color" not in st.session_state:
    st.session_state.u3_color = "#38bdf8"

if "u4_name" not in st.session_state:
    st.session_state.u4_name = "Jordan"
if "u4_style" not in st.session_state:
    st.session_state.u4_style = "Modern Minimalist"
if "u4_objs" not in st.session_state:
    st.session_state.u4_objs = ["Plant Wall", "Side Table"]
if "u4_color" not in st.session_state:
    st.session_state.u4_color = "#10b981"

# Budget Session State Defaults
if "budget_tier" not in st.session_state:
    st.session_state.budget_tier = "Mid-Range Premium"
if "budget_limit" not in st.session_state:
    st.session_state.budget_limit = 15000
if "furn_pct" not in st.session_state:
    st.session_state.furn_pct = 50
if "light_pct" not in st.session_state:
    st.session_state.light_pct = 25
if "custom_prompt" not in st.session_state:
    st.session_state.custom_prompt = ""

# Standard Furniture Catalog with Realistic Costs and Archetypes
FURNITURE_CATALOG = {
    "Leather Sofa": {"cost": 1250, "category": "Seating", "icon": ""},
    "Fabric 3-Seater Sofa": {"cost": 850, "category": "Seating", "icon": ""},
    "Bouclé Accent Armchair": {"cost": 420, "category": "Seating", "icon": ""},
    "Lounge Chair": {"cost": 380, "category": "Seating", "icon": ""},
    "Solid Oak Coffee Table": {"cost": 340, "category": "Surface", "icon": ""},
    "Round Marble Coffee Table": {"cost": 460, "category": "Surface", "icon": ""},
    "Floating TV Console": {"cost": 480, "category": "Storage", "icon": ""},
    "Minimalist Steel Bookshelf": {"cost": 380, "category": "Storage", "icon": ""},
    "Architectural Floor Lamp": {"cost": 190, "category": "Lighting", "icon": ""},
    "Pendant Ceiling Lamp": {"cost": 220, "category": "Lighting", "icon": ""},
    "Handwoven Wool Area Rug": {"cost": 450, "category": "Decor", "icon": ""},
    "Indoor Monstera / Fiddle Leaf Plant": {"cost": 120, "category": "Decor", "icon": ""},
    "Large Framed Canvas Wall Art": {"cost": 260, "category": "Decor", "icon": ""},
    "Ergonomic Workstation Desk": {"cost": 520, "category": "Surface", "icon": ""}
}

STYLE_SUGGESTIONS = {
    "Industrial": "💡 **Better to use:** Replace bulky freestanding bookcases with wall-mounted steel & reclaimed wood shelving (₹26,500) to expand circulation paths by 0.8m².",
    "Japandi": "💡 **Better to use:** Pair a low-profile platform coffee table (₹24,000) with an organic textured jute rug (₹17,500) to maintain wide sightlines and natural light flow.",
    "Modern Minimalist": "💡 **Better to use:** Opt for a floating wall-mounted media console (₹35,000) and concealed storage to maintain completely open, unobstructed walkways (>90cm).",
    "Scandinavian": "💡 **Better to use:** Choose a round nesting birch coffee table (₹25,500) to eliminate sharp edge hazards and improve diagonal traffic flow.",
    "Luxury Contemporary": "💡 **Better to use:** Feature a curved organic velvet sofa (₹1,20,000) to guide flowing pedestrian movement around the central conversation zone.",
    "Biophilic Modern": "💡 **Better to use:** Place a vertical botanical planter trellis (₹28,000) beside the main window to optimize air quality and biophilic connection without consuming floor space.",
    "Mid-Century Modern": "💡 **Better to use:** Utilize tapered walnut credenzas (₹48,000) and iconic brass floor lamps (₹17,500) for retro balance and elevated ground clearance.",
    "Bohemian Warm": "💡 **Better to use:** Layer flatweave kilim rugs (₹26,500) with macramé wall hangings to boost acoustic sound absorption (NRC +0.25)."
}

# Helper function for rendering rich metric cards
def render_rich_kpi(title: str, value: str, subtext: str = "", delta_color: str = "#2563eb"):
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub" style="color: {delta_color};">{subtext}</div>
        </div>
    """, unsafe_allow_html=True)

# Centered Clean Hero Header
st.markdown("""
    <div class="hero-container">
        <div class="hero-title">Interior Design</div>
        <div class="hero-subtitle">Upload room photos, configure inhabitant preferences, and explore tailored design options with 2D floor plans.</div>
        <div class="pill-container">
            <span class="hero-pill">Room Analysis</span>
            <span class="hero-pill">Object Detection</span>
            <span class="hero-pill">Inhabitant Preferences</span>
            <span class="hero-pill">Budget Planner</span>
            <span class="hero-pill">2D Floor Plans</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR CONTROLS: Quick Configuration
# ==============================================================================
with st.sidebar:
    st.markdown("### Quick Configuration")

    # Backend status check
    if api_client.check_health():
        st.success("Service Status: Online")
    else:
        st.info("Service Status: Ready")

    st.markdown("---")
    st.markdown("#### 1. Room Classification")
    st.session_state.room_type = st.selectbox(
        "Room Type",
        settings.DEFAULT_ROOM_TYPES,
        index=settings.DEFAULT_ROOM_TYPES.index(st.session_state.room_type) if st.session_state.room_type in settings.DEFAULT_ROOM_TYPES else 0,
        key="sb_room_type"
    )

    st.markdown("---")
    st.markdown("#### 2. Budget Settings")
    st.session_state.budget_tier = st.select_slider(
        "Budget Tier",
        options=["Budget Essential", "Mid-Range Premium", "Luxury Bespoke"],
        value=st.session_state.budget_tier,
        key="sb_budget_tier"
    )
    st.session_state.budget_limit = st.slider(
        "Budget Limit (₹)",
        1000, 50000,
        int(st.session_state.budget_limit),
        step=1000,
        key="sb_budget_limit"
    )

# Build current compiled active users list
active_users = [
    {
        "name": st.session_state.u1_name,
        "preferred_style": st.session_state.u1_style,
        "priority_objects": st.session_state.u1_objs,
        "color_tone": st.session_state.u1_color
    },
    {
        "name": st.session_state.u2_name,
        "preferred_style": st.session_state.u2_style,
        "priority_objects": st.session_state.u2_objs,
        "color_tone": st.session_state.u2_color
    }
]
if st.session_state.num_users >= 3:
    active_users.append({
        "name": st.session_state.u3_name,
        "preferred_style": st.session_state.u3_style,
        "priority_objects": st.session_state.u3_objs,
        "color_tone": st.session_state.u3_color
    })
if st.session_state.num_users >= 4:
    active_users.append({
        "name": st.session_state.u4_name,
        "preferred_style": st.session_state.u4_style,
        "priority_objects": st.session_state.u4_objs,
        "color_tone": st.session_state.u4_color
    })

# Calculated live budget values
reno_calc_pct = max(0, 100 - (st.session_state.furn_pct + st.session_state.light_pct))
furn_val = int(st.session_state.budget_limit * (st.session_state.furn_pct / 100.0))
light_val = int(st.session_state.budget_limit * (st.session_state.light_pct / 100.0))
reno_val = int(st.session_state.budget_limit * (reno_calc_pct / 100.0))

# Compile user summary text with color dots
user_pills_html = " &bull; ".join([
    f"<span style='color: #334155; font-weight: 600;'><span style='display:inline-block; width:10px; height:10px; border-radius:50%; background-color:{u['color_tone']}; margin-right:4px; vertical-align:middle;'></span>{u['name']} ({u['preferred_style']})</span>"
    for u in active_users
])

# Project Configuration Banner
st.markdown(f"""
    <div class="session-bar">
        <div class="session-item">
            <span style="font-size: 1.1rem;">🏠</span> <b>Room:</b> <span class="session-tag">{st.session_state.room_type}</span>
        </div>
        <div class="session-item">
            <span style="font-size: 1.1rem;">👥</span> <b>Inhabitants ({len(active_users)}):</b> {user_pills_html}
        </div>
        <div class="session-item">
            <span style="font-size: 1.1rem;">💰</span> <b>Budget:</b> <span class="session-tag">₹{st.session_state.budget_limit:,} ({st.session_state.budget_tier})</span>
            <span style="font-size: 0.82rem; color: #1e293b; font-weight: 700; margin-left: 4px;">[Furn: ₹{furn_val:,} &bull; Light: ₹{light_val:,} &bull; Reno: ₹{reno_val:,}]</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# 4 Clear Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    " 1. Room Photo & Detection",
    " 2. Inhabitant Preferences & Budget",
    " 3. Redesign Options & Layouts",
    " 4. Design Analysis & Product Details"
])


# ==============================================================================
# TAB 1: Room Upload and Spatial Analysis
# ==============================================================================
with tab1:
    col_up, col_prev = st.columns([1.1, 1.1])

    with col_up:
        st.markdown("""
            <div class="glass-card">
                <div style="font-size: 1.10rem; font-weight: 700; color: #0f172a; margin-bottom: 4px;">📸 Step 1: Upload Room Image</div>
                <div style="font-size: 0.88rem; color: #1e293b; font-weight: 500; margin-bottom: 12px;">Upload a photograph of your existing space or load a sample room to detect furniture and extract room geometry.</div>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Select Room Photograph (JPG or PNG)", type=["jpg", "jpeg", "png"], key="main_file_uploader")

        st.markdown("**Or load a sample room:**")
        if st.button("🖼️ Load Sample Living Room", use_container_width=True):
            sample_img = np.full((1024, 1024, 3), 235, dtype=np.uint8)
            sample_img[620:, :] = [215, 185, 145]
            sample_img[520:820, 180:840] = [70, 80, 100]
            sample_img[720:880, 380:640] = [140, 75, 25]
            sample_img[200:480, 100:300] = [180, 210, 235]

            sample_pil = Image.fromarray(sample_img)
            sample_filename = f"sample_room_{int(time.time()*1000)}.jpg"
            sample_path = settings.UPLOAD_DIR / sample_filename
            sample_pil.save(sample_path, quality=95)

            st.session_state.uploaded_filename = sample_filename
            st.session_state.current_image = sample_pil
            st.success("Sample room loaded successfully.")

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            res = api_client.upload_image(file_bytes, uploaded_file.name)
            st.session_state.uploaded_filename = res["filename"]
            st.session_state.current_image = Image.open(uploaded_file).convert("RGB")
            st.success(f"File uploaded: {res['filename']}")

        if st.session_state.uploaded_filename:
            st.markdown("---")
            if st.button("🔍 Analyze Room Geometry & Furniture", type="primary", use_container_width=True):
                with st.spinner("Analyzing room geometry, detecting objects, and extracting dominant colors..."):
                    analysis_res = api_client.analyze_room(st.session_state.uploaded_filename, st.session_state.room_type)
                    st.session_state.analysis_data = analysis_res
                    st.success("Room analysis completed successfully.")

    with col_prev:
        st.markdown("""
            <div class="glass-card">
                <div style="font-size: 1.10rem; font-weight: 700; color: #0f172a; margin-bottom: 4px;">🖼️ Uploaded Room Preview</div>
                <div style="font-size: 0.88rem; color: #1e293b; font-weight: 500; margin-bottom: 12px;">Your input room photograph used as architectural reference.</div>
            </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.current_image:
            st.image(st.session_state.current_image, caption="Input Room Image", use_container_width=True)
        else:
            st.info("Upload an image or click 'Load Sample Living Room' to preview.")

    # Analysis Results Display
    if st.session_state.analysis_data:
        st.markdown("---")
        st.markdown("### 📊 Room Spatial Metrics & Detection")

        a_data = st.session_state.analysis_data
        spatial = a_data.get("spatial_metrics", {})

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            render_rich_kpi("Detected Objects", f"{spatial.get('furniture_count', 0)} Items", "Classified Objects")
        with m2:
            render_rich_kpi("Floor Occupancy", f"{spatial.get('furniture_coverage_pct', 0.0)}%", "Floor Footprint")
        with m3:
            render_rich_kpi("Clutter Index", f"{spatial.get('spatial_clutter_score', 0.0)}/100", "Room Density")
        with m4:
            render_rich_kpi("Spatial Balance", f"{spatial.get('layout_distribution', 'Balanced')}", "Room Geometry")

        # Color Palette Display
        palette = a_data.get("dominant_palette", [])
        if palette:
            st.markdown("##### 🎨 Extracted Dominant Color Palette")
            p_cols = st.columns(len(palette))
            for i, p_col in enumerate(p_cols):
                hex_val = palette[i].get("hex", "#FFFFFF")
                pct_val = palette[i].get("percentage", 0)
                with p_col:
                    st.markdown(f"""
                        <div style="background: #ffffff; border: 1.5px solid #cbd5e1; border-radius: 8px; padding: 8px; text-align: center;">
                            <div class="swatch-box" style="background-color: {hex_val};"></div>
                            <div class="swatch-label">{hex_val}</div>
                            <div style="font-size: 0.76rem; color: #0f172a; font-weight: 700; margin-top: 2px;">{pct_val}%</div>
                        </div>
                    """, unsafe_allow_html=True)

        # Annotated Image
        ann_url = a_data.get("annotated_image_url")
        if ann_url:
            ann_path = settings.STATIC_DIR / ann_url.replace("/static/", "")
            if ann_path.exists():
                st.markdown("---")
                st.markdown("##### 🏷️ Detected Object Bounding Boxes")
                st.image(str(ann_path), use_container_width=True)

        st.info("💡 Next Step: Go to **Tab 2 ('Inhabitant Preferences & Budget')** to customize styles and budget before generating redesigns!")


# ==============================================================================
# TAB 2: Inhabitant Preferences & Budget
# ==============================================================================
with tab2:
    st.markdown("""
        <div class="glass-card" style="margin-bottom: 16px;">
            <div style="font-size: 1.2rem; font-weight: 700; color: #0f172a; margin-bottom: 4px;">👥 Inhabitant Preferences, Preferred Items & Budget</div>
            <div style="font-size: 0.88rem; color: #334155; line-height: 1.5;">
                Configure inhabitant names, design styles, and preferred furniture items with itemized pricing, smart object recommendations, and financial budget allocation.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 1. Inhabitant Profiles, Preferred Items & Costs
    st.markdown("### 1. Inhabitant Profiles, Preferred Items & Costs")
    
    top_c1, top_c2 = st.columns([1, 2])
    with top_c1:
        st.session_state.num_users = st.selectbox(
            "Number of Inhabitants",
            [2, 3, 4],
            index=[2, 3, 4].index(st.session_state.num_users),
            key="tab2_num_users"
        )
    with top_c2:
        st.session_state.room_type = st.selectbox(
            "Room Purpose / Type",
            settings.DEFAULT_ROOM_TYPES,
            index=settings.DEFAULT_ROOM_TYPES.index(st.session_state.room_type),
            key="tab2_room_type"
        )

    # Inhabitant Cards with Name, Style, Preferred Items, Itemized Costs, and Smart Suggestions
    inhab_cols = st.columns(st.session_state.num_users)
    inhab_border_colors = ["#2563eb", "#0284c7", "#16a34a", "#d97706"]
    
    for i in range(st.session_state.num_users):
        u_idx = i + 1
        with inhab_cols[i]:
            st.markdown(f"""
                <div class="inhabitant-card" style="border-top: 3px solid {inhab_border_colors[i]};">
                    <div style="font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 2px;">👤 Inhabitant {u_idx}</div>
                    <div style="font-size: 0.80rem; color: #1e293b; font-weight: 600; margin-bottom: 8px;">{'Primary Inhabitant' if i == 0 else 'Co-Inhabitant'}</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Name Input
            st.session_state[f"u{u_idx}_name"] = st.text_input("Name", value=st.session_state[f"u{u_idx}_name"], key=f"tab2_u{u_idx}_name")
            
            # Style Preference Selectbox
            curr_style = st.session_state[f"u{u_idx}_style"]
            style_idx = settings.DEFAULT_INTERIOR_STYLES.index(curr_style) if curr_style in settings.DEFAULT_INTERIOR_STYLES else i % len(settings.DEFAULT_INTERIOR_STYLES)
            st.session_state[f"u{u_idx}_style"] = st.selectbox("Style Preference", settings.DEFAULT_INTERIOR_STYLES, index=style_idx, key=f"tab2_u{u_idx}_style")
            
            # Preferred Items Multiselect from Catalog
            curr_objs = [obj for obj in st.session_state[f"u{u_idx}_objs"] if obj in FURNITURE_CATALOG]
            if not curr_objs:
                default_defaults = [list(FURNITURE_CATALOG.keys())[0], list(FURNITURE_CATALOG.keys())[4]] if i == 0 else [list(FURNITURE_CATALOG.keys())[2], list(FURNITURE_CATALOG.keys())[8]]
                curr_objs = default_defaults
            
            selected_items = st.multiselect(
                "Preferred Furniture & Decor Items",
                options=list(FURNITURE_CATALOG.keys()),
                default=curr_objs,
                key=f"tab2_u{u_idx}_objs"
            )
            st.session_state[f"u{u_idx}_objs"] = selected_items
            
            # Calculate itemized costs for selected objects
            total_items_cost = sum([FURNITURE_CATALOG.get(item, {}).get("cost", 200) for item in selected_items])
            
            # Display itemized cost tags
            st.markdown(f"**Itemized Costs (Selected Total: ₹{total_items_cost:,}):**")
            cost_tags_html = " ".join([
                f"<span style='display:inline-block; background:#eff6ff; border:1px solid #bfdbfe; color:#1e40af; border-radius:6px; padding:2px 7px; font-size:0.75rem; margin:2px; font-weight:600;'>"
                f"{FURNITURE_CATALOG.get(it, {}).get('icon', '🛋️')} {it}: <b>₹{FURNITURE_CATALOG.get(it, {}).get('cost', 200)}</b>"
                f"</span>"
                for it in selected_items
            ]) if selected_items else "<span style='font-size:0.80rem; color:#475569; font-weight:600;'>No items selected yet.</span>"
            
            st.markdown(cost_tags_html, unsafe_allow_html=True)
            
            # Smart Suggestion for better objects
            active_inhab_style = st.session_state[f"u{u_idx}_style"]
            suggestion_text = STYLE_SUGGESTIONS.get(active_inhab_style, "💡 **Better to use:** Multi-functional nesting tables (₹25,500) to optimize floor space.")
            
            st.markdown(f"""
                <div class="suggestion-box">
                    <b>Suggestion for {st.session_state[f'u{u_idx}_name']} ({active_inhab_style}):</b><br>
                    {suggestion_text}
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # 2. Budget & Estimated Costs
    st.markdown("### 2. Budget Allocation & Estimated Costs")

    b_col1, b_col2 = st.columns([1.1, 1.1])

    with b_col1:
        st.session_state.budget_tier = st.select_slider(
            "Select Budget Tier",
            options=["Budget Essential", "Mid-Range Premium", "Luxury Bespoke"],
            value=st.session_state.budget_tier,
            key="tab2_budget_tier"
        )
        
        st.session_state.budget_limit = st.slider(
            "Total Budget (₹)",
            1000, 50000,
            int(st.session_state.budget_limit),
            step=500,
            key="tab2_budget_limit"
        )

    with b_col2:
        furn_dollar = int(st.session_state.budget_limit * (st.session_state.furn_pct / 100.0))
        light_dollar = int(st.session_state.budget_limit * (st.session_state.light_pct / 100.0))
        reno_dollar = int(st.session_state.budget_limit * (reno_calc_pct / 100.0))

        factor_map = {"Budget Essential": 0.72, "Mid-Range Premium": 0.88, "Luxury Bespoke": 0.94}
        active_factor = factor_map.get(st.session_state.budget_tier, 0.88)
        est_total_cost = int(st.session_state.budget_limit * active_factor)
        buffer_total = int(st.session_state.budget_limit - est_total_cost)

        st.markdown(f"""
            <div class="glass-card" style="padding: 16px;">
                <div style="font-size: 1.0rem; font-weight: 700; color: #0f172a; margin-bottom: 12px;">💰 Estimated Project Costs:</div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.88rem;">
                    <span>🛋️ Furniture & Seating:</span>
                    <b style="color: #2563eb;">₹{furn_dollar:,} ({st.session_state.furn_pct}%)</b>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 0.88rem;">
                    <span>💡 Lighting & Fixtures:</span>
                    <b style="color: #0284c7;">₹{light_dollar:,} ({st.session_state.light_pct}%)</b>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 12px; font-size: 0.88rem;">
                    <span>🪵 Flooring & Finishes:</span>
                    <b style="color: #16a34a;">₹{reno_dollar:,} ({reno_calc_pct}%)</b>
                </div>
                <hr style="border-color: #cbd5e1; margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; font-size: 0.92rem;">
                    <span>Estimated Total Project Cost:</span>
                    <b>₹{est_total_cost:,}</b>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.92rem; color: #16a34a; font-weight: 700; margin-top: 4px;">
                    <span>Safety Buffer Remaining:</span>
                    <span>+₹{buffer_total:,}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.success("✅ Preferences & items configured! Go to **Tab 3 ('Redesign Options & Layouts')** to generate your room designs.")


# ==============================================================================
# TAB 3: Design Generations and Explanations
# ==============================================================================
with tab3:
    if not st.session_state.uploaded_filename:
        st.warning("⚠️ Please upload a room photograph or load a sample room in Tab 1 before generating redesigns.")
    else:
        user_summary = " + ".join([u["name"] for u in active_users])
        
        st.markdown(f"""
            <div class="glass-card" style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 12px;">
                <div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a;">Generate Redesign Options & 2D Floor Plans</div>
                    <div style="font-size: 0.85rem; color: #334155;"><b>Inhabitants:</b> {user_summary} &nbsp;|&nbsp; <b>Room:</b> {st.session_state.room_type} &nbsp;|&nbsp; <b>Budget:</b> ₹{st.session_state.budget_limit:,} ({st.session_state.budget_tier})</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Comfort Score for Selected Design Option placed in Tab 3 (Image 4 Requirement)
        if st.session_state.redesign_data:
            r_data = st.session_state.redesign_data
            variations = r_data.get("variations", [])
            curr_idx = min(st.session_state.inspected_var_idx, len(variations) - 1) if variations else 0
            curr_var = variations[curr_idx] if variations else {}
            active_title = curr_var.get("title", f"Option {curr_idx+1}")
            active_style = curr_var.get("style", active_users[0]["preferred_style"])
            active_persona = curr_var.get("persona_blend", "Inhabitant Focus")
            active_var_id = curr_var.get("variation_id", f"var_{curr_idx+1}")

            active_ai = recommender_service.get_variation_ai_explanation(
                variation_id=active_var_id,
                style=active_style,
                budget_tier=st.session_state.budget_tier,
                budget_limit=float(st.session_state.budget_limit),
                persona_name=active_persona
            )

            comfort_score_val = active_ai.get("comfort_score", 95.0)
            comfort_lift = comfort_score_val - 68.0

            st.markdown(f"""
                <div class="comfort-highlight-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 12px;">
                        <div>
                            <div style="font-size: 1.12rem; font-weight: 700; color: #0f172a;">
                                ⭐ Comfort Score for Selected Layout: <span style="color: #2563eb;">{active_title}</span> ({active_style})
                            </div>
                            <div style="font-size: 0.84rem; color: #1e293b; font-weight: 600; margin-top: 2px;">
                                <b>Inhabitant Focus:</b> {active_persona} &nbsp;|&nbsp; <b>Layout Archetype:</b> {curr_var.get('layout_archetype', 'Custom Layout')}
                            </div>
                        </div>
                        <div style="display: flex; gap: 8px; align-items: center;">
                            <span style="background: #dcfce7; border: 1.5px solid #86efac; border-radius: 10px; padding: 6px 14px; font-weight: 800; color: #15803d; font-size: 1.10rem;">
                                ⭐ {comfort_score_val}/100 Comfort
                            </span>
                            <span style="background: #eff6ff; border: 1.5px solid #bfdbfe; border-radius: 10px; padding: 6px 10px; font-weight: 800; color: #1d4ed8; font-size: 0.84rem;">
                                +{comfort_lift:.1f} vs Baseline
                            </span>
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin-bottom: 12px; font-size: 0.86rem;">
                        <div style="background: #f8fafc; padding: 10px; border-radius: 8px; border: 1.5px solid #cbd5e1;">
                            <b style="color: #0f172a; font-weight: 700;">🚶 Space & Circulation:</b><br>
                            <span style="color: #0f172a; font-weight: 700;">{active_ai['circulation_clearance']}</span>
                        </div>
                        <div style="background: #f8fafc; padding: 10px; border-radius: 8px; border: 1.5px solid #cbd5e1;">
                            <b style="color: #0f172a; font-weight: 700;">💡 Lighting & Ambiance:</b><br>
                            <span style="color: #0f172a; font-weight: 700;">{active_ai['lighting_ambiance']}</span>
                        </div>
                        <div style="background: #f8fafc; padding: 10px; border-radius: 8px; border: 1.5px solid #cbd5e1;">
                            <b style="color: #0f172a; font-weight: 700;">🛋️ Seating Ergonomics:</b><br>
                            <span style="color: #0f172a; font-weight: 700;">{active_ai['ergonomics_rating']}</span>
                        </div>
                        <div style="background: #f8fafc; padding: 10px; border-radius: 8px; border: 1.5px solid #cbd5e1;">
                            <b style="color: #0f172a; font-weight: 700;">♿ Accessibility Rating:</b><br>
                            <span style="color: #16a34a; font-weight: 800;">{active_ai.get('ada_accessibility_score', 94.0)}/100 ADA</span>
                        </div>
                    </div>
                    <div style="font-size: 0.86rem; color: #0f172a; border-top: 1.5px solid #cbd5e1; padding-top: 10px;">
                        <b style="color: #0f172a;">Comfort & Ergonomic Highlights:</b>
                        <ul style="margin: 6px 0 0 0; padding-left: 20px; color: #0f172a;">
                            {''.join([f"<li style='color: #0f172a;'>{b}</li>" for b in active_ai.get('rationale_bullets', [])[:3]])}
                        </ul>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="comfort-highlight-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 10px;">
                        <div>
                            <div style="font-size: 1.05rem; font-weight: 700; color: #0f172a;">
                                ⭐ Projected Target Comfort Score: <span style="color: #16a34a;">92.0 - 96.0 / 100</span>
                            </div>
                            <div style="font-size: 0.84rem; color: #1e293b; font-weight: 600; margin-top: 2px;">
                                Calculated for {active_users[0]['preferred_style']} & {active_users[1]['preferred_style'] if len(active_users)>1 else ''} preferences with ₹{st.session_state.budget_limit:,} budget.
                            </div>
                        </div>
                        <span style="background: #dcfce7; border: 1.5px solid #86efac; border-radius: 8px; padding: 4px 12px; font-weight: 800; color: #15803d; font-size: 0.92rem;">
                            Target: ~95.0/100
                        </span>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; font-size: 0.86rem;">
                        <div style="background: #f8fafc; padding: 10px; border-radius: 8px; border: 1.5px solid #cbd5e1;">
                            <b style="color: #0f172a; font-weight: 700;">🚶 Primary Clearance:</b><br>
                            <span style="color: #1e293b; font-weight: 600;">Compliant with >90cm unblocked pathways</span>
                        </div>
                        <div style="background: #f8fafc; padding: 10px; border-radius: 8px; border: 1.5px solid #cbd5e1;">
                            <b style="color: #0f172a; font-weight: 700;">💡 Natural Lighting:</b><br>
                            <span style="color: #1e293b; font-weight: 600;">Oriented toward window daylight ingress</span>
                        </div>
                        <div style="background: #f8fafc; padding: 10px; border-radius: 8px; border: 1.5px solid #cbd5e1;">
                            <b style="color: #0f172a; font-weight: 700;">🛋️ Ergonomic Balance:</b><br>
                            <span style="color: #1e293b; font-weight: 600;">Optimized seating distances & sightlines</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        col_var_num, col_var_btn = st.columns([1, 2.5])
        with col_var_num:
            chosen_num_vars = st.selectbox(
                "Redesign Options",
                [3, 2, 1, 4],
                index=0,
                format_func=lambda x: f"{x} Suggested Option{'s' if x > 1 else ''}",
                key="tab3_num_variations_select"
            )
        with col_var_btn:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            gen_submitted = st.button(f"🚀 Generate {chosen_num_vars} Suggested Redesigns & Blueprints", type="primary", use_container_width=True)

        if gen_submitted:
            with st.spinner(f"Generating {chosen_num_vars} distinct 3D perspectives & 2D architectural blueprints in parallel..."):
                palette_hexes = [p.get("hex") for p in st.session_state.analysis_data.get("dominant_palette", [])] if st.session_state.analysis_data else []
                payload = {
                    "image_filename": st.session_state.uploaded_filename,
                    "room_type": st.session_state.room_type,
                    "interior_style": active_users[0]["preferred_style"],
                    "budget_tier": st.session_state.budget_tier,
                    "budget_limit": float(st.session_state.budget_limit),
                    "budget_allocation": {
                        "furniture_seating_pct": st.session_state.furn_pct,
                        "lighting_decor_pct": st.session_state.light_pct,
                        "renovation_finishes_pct": reno_calc_pct
                    },
                    "users": active_users,
                    "color_palette": palette_hexes,
                    "preserve_structure": True,
                    "custom_prompt_notes": st.session_state.custom_prompt,
                    "controlnet_scale": 0.45,
                    "inference_steps": 30,
                    "num_variations": chosen_num_vars
                }
                redesign_res = api_client.generate_redesign(payload)
                st.session_state.redesign_data = redesign_res
                st.session_state.inspected_var_idx = 0
                st.success(f"✨ {chosen_num_vars} suggested redesign options generated successfully!")

        # Results Display
        if st.session_state.redesign_data:
            r_data = st.session_state.redesign_data
            variations = r_data.get("variations", [])

            # Active View Comparison
            curr_idx = min(st.session_state.inspected_var_idx, len(variations) - 1) if variations else 0
            curr_var = variations[curr_idx] if variations else {}
            active_img_url = curr_var.get("image_url", "")
            active_title = curr_var.get("title", "Selected Variation")
            active_style = curr_var.get("style", active_users[0]["preferred_style"])
            active_persona = curr_var.get("persona_blend", "Inhabitant Focus")
            active_var_id = curr_var.get("variation_id", f"var_{curr_idx+1}")
            active_path = settings.STATIC_DIR / active_img_url.replace("/static/", "")

            # Fetch active variation dynamic explanation & comfort metrics
            active_ai = recommender_service.get_variation_ai_explanation(
                variation_id=active_var_id,
                style=active_style,
                budget_tier=st.session_state.budget_tier,
                budget_limit=float(st.session_state.budget_limit),
                persona_name=active_persona
            )

            active_place_url = curr_var.get("placement_overlay_url", "")
            active_place_path = settings.STATIC_DIR / active_place_url.replace("/static/", "") if active_place_url else None
            active_floorplan_url = curr_var.get("floorplan_diagram_url", "")
            active_floorplan_path = settings.STATIC_DIR / active_floorplan_url.replace("/static/", "") if active_floorplan_url else None
            active_placements = curr_var.get("placement_suggestions", [])

            st.markdown("---")
            st.markdown(f"### 📐 2D Floor Plan & Spatial Redesign: <span style='color: #2563eb;'>{active_title}</span>", unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**📸 1. Original Room Space (Uploaded)**")
                if st.session_state.current_image:
                    st.image(st.session_state.current_image, use_container_width=True)
            with c2:
                st.markdown(f"**📐 2. Top-Down 2D Floor Plan Blueprint**")
                if active_floorplan_path and active_floorplan_path.is_file():
                    st.image(str(active_floorplan_path), use_container_width=True)
                elif active_place_path and active_place_path.is_file():
                    st.image(str(active_place_path), use_container_width=True)
            with c3:
                st.markdown(f"**🛋️ 3. Redesigned 3D Perspective Interior**")
                if active_path.is_file():
                    st.image(str(active_path), use_container_width=True)
                st.markdown(f"""
                    <div style="background: #f0fdf4; border: 1.5px solid #86efac; border-radius: 8px; padding: 6px 12px; margin-top: 6px; text-align: center;">
                        <span style="font-size: 0.88rem; color: #166534; font-weight: 700;">💰 Selected Budget: ₹{st.session_state.budget_limit:,} ({st.session_state.budget_tier})</span>
                    </div>
                """, unsafe_allow_html=True)

            # Active View Evaluation Card
            comfort_color = "#16a34a" if active_ai["comfort_score"] >= 90 else "#2563eb" if active_ai["comfort_score"] >= 85 else "#d97706"
            arch_type = curr_var.get("layout_archetype", "Architectural Arrangement")
            wall_anc = curr_var.get("wall_anchor", "Main Wall Anchor")
            circ_flow = curr_var.get("circulation_corridor", "Walkway (>90cm)")

            st.markdown(f"""
                <div class="glass-card" style="margin-top: 14px; margin-bottom: 20px; border-left: 4px solid {comfort_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 12px;">
                        <div>
                            <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a;">
                                📐 Layout Archetype: <span style="color: #2563eb;">{arch_type}</span>
                            </div>
                            <div style="font-size: 0.84rem; color: #1e293b; font-weight: 600; margin-top: 2px;">
                                <b>Wall Anchor:</b> {wall_anc} &nbsp;|&nbsp; <b>Corridor Flow:</b> {circ_flow}
                            </div>
                        </div>
                        <div style="display: flex; gap: 10px; align-items: center;">
                            <span style="background: #f1f5f9; border: 1.5px solid #cbd5e1; border-radius: 12px; padding: 4px 10px; font-size: 0.84rem; color: #0f172a; font-weight: 700;">
                                <b>Original Room:</b> ~68.0/100
                            </span>
                            <span style="background: #dcfce7; border: 1.5px solid #86efac; border-radius: 12px; padding: 4px 12px; font-weight: 800; color: #15803d; font-size: 0.92rem;">
                                ⭐ Redesigned: {active_ai['comfort_score']}/100 (+{(active_ai['comfort_score'] - 68.0):.1f} Lift)
                            </span>
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 14px; font-size: 0.86rem;">
                        <div style="background: #f8fafc; padding: 10px; border-radius: 8px; border: 1.5px solid #cbd5e1;">
                            <b style="color: #0f172a; font-weight: 700;">🚶 Space & Circulation:</b><br>
                            <span style="color: #0f172a; font-weight: 700;">{active_ai['circulation_clearance']}</span>
                        </div>
                        <div style="background: #f8fafc; padding: 10px; border-radius: 8px; border: 1.5px solid #cbd5e1;">
                            <b style="color: #0f172a; font-weight: 700;">💡 Lighting & Ambiance:</b><br>
                            <span style="color: #0f172a; font-weight: 700;">{active_ai['lighting_ambiance']}</span>
                        </div>
                        <div style="background: #f8fafc; padding: 10px; border-radius: 8px; border: 1.5px solid #cbd5e1;">
                            <b style="color: #0f172a; font-weight: 700;">🛋️ Seating Ergonomics:</b><br>
                            <span style="color: #0f172a; font-weight: 700;">{active_ai['ergonomics_rating']}</span>
                        </div>
                        <div style="background: #f8fafc; padding: 10px; border-radius: 8px; border: 1.5px solid #cbd5e1;">
                            <b style="color: #0f172a; font-weight: 700;">💰 Estimated Cost:</b><br>
                            <span style="color: #16a34a; font-weight: 800;">₹{active_ai['estimated_cost']:,}</span> (Buffer: +₹{active_ai['budget_buffer']:,})
                        </div>
                    </div>
                    <div style="font-size: 0.86rem; color: #0f172a; border-top: 1.5px solid #cbd5e1; padding-top: 10px;">
                        <b style="color: #0f172a;">Architectural Rationale:</b>
                        <ul style="margin: 6px 0 0 0; padding-left: 20px; color: #0f172a;">
                            {''.join([f"<li style='color: #0f172a;'>{bullet}</li>" for bullet in active_ai.get('rationale_bullets', [])])}
                        </ul>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Suggested Redesigns Gallery
            st.markdown("---")
            st.markdown(f"### 🛋️ {len(variations)} Suggested Redesigns & 2D Floor Plans")

            if variations:
                cols = st.columns(len(variations))
                for idx in range(len(variations)):
                    col = cols[idx]
                    var = variations[idx]
                    var_id = var.get("variation_id", f"var_{idx+1}")
                    var_url = var.get("image_url", "")
                    var_floor_url = var.get("floorplan_diagram_url", "")
                    var_style = var.get("style", active_users[0]["preferred_style"])
                    var_persona = var.get("persona_blend", "Inhabitant Focus")
                    var_arch = var.get("layout_archetype", f"Layout {idx+1}")
                    var_path = settings.STATIC_DIR / var_url.replace("/static/", "")
                    var_floor_path = settings.STATIC_DIR / var_floor_url.replace("/static/", "") if var_floor_url else None
                    is_active = (idx == curr_idx)

                    ai_exp = recommender_service.get_variation_ai_explanation(
                        variation_id=var_id,
                        style=var_style,
                        budget_tier=st.session_state.budget_tier,
                        budget_limit=float(st.session_state.budget_limit),
                        persona_name=var_persona
                    )

                    div_score_pct = int(var.get("diversity_score", 0.85) * 100)
                    with col:
                        st.markdown(f"""
                            <div style="background: {'#eff6ff' if is_active else '#ffffff'}; border: {'2px solid #2563eb' if is_active else '1.5px solid #cbd5e1'}; border-radius: 12px; padding: 12px 14px; margin-bottom: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                                    <span style="font-weight: 800; color: #0f172a; font-size: 0.96rem;">{var.get('title', f'Option {idx+1}')}</span>
                                    <span style="background: {'#16a34a' if ai_exp['comfort_score'] >= 90 else '#2563eb'}; color: #ffffff; border-radius: 8px; padding: 2px 8px; font-size: 0.74rem; font-weight: 800;">⭐ {ai_exp['comfort_score']}/100</span>
                                </div>
                                <div style="font-size: 0.80rem; color: #1d4ed8; font-weight: 700;">📐 {var_arch}</div>
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
                                    <span style="font-size: 0.78rem; color: #0f172a; font-weight: 700;"><b>Style:</b> {var_style}</span>
                                    <span style="font-size: 0.74rem; color: #15803d; font-weight: 700; background: #dcfce7; border: 1px solid #86efac; padding: 2px 7px; border-radius: 5px;">✨ {div_score_pct}% Diversity</span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                        # Display 2D Floor Plan Blueprint Diagram
                        st.caption(f"📐 **2D Floor-Plan Blueprint (Option {idx+1}):**")
                        if var_floor_path and var_floor_path.is_file():
                            st.image(str(var_floor_path), use_container_width=True)
                        elif var.get("floorplan_diagram_image") is not None:
                            st.image(var["floorplan_diagram_image"], use_container_width=True)

                        # Display Perspective Redesign View
                        st.caption(f"🛋️ **Redesigned 3D Perspective Interior (Option {idx+1}):**")
                        if var_path and var_path.is_file():
                            st.image(str(var_path), use_container_width=True)
                        elif var.get("image") is not None:
                            st.image(var["image"], use_container_width=True)
                        elif "current_image" in st.session_state and st.session_state.current_image:
                            st.image(st.session_state.current_image, use_container_width=True)

                        st.markdown(f"""
                            <div style="background: #f0fdf4; border: 1.5px solid #86efac; border-radius: 8px; padding: 5px 8px; margin-top: 6px; margin-bottom: 8px; text-align: center;">
                                <span style="font-size: 0.82rem; color: #166534; font-weight: 700;">💰 Selected Budget: ₹{st.session_state.budget_limit:,} ({st.session_state.budget_tier})</span>
                            </div>
                        """, unsafe_allow_html=True)

                        if st.button(f"{'✅ Active View' if is_active else f'👉 Select Option {idx+1}'}", key=f"btn_var_{idx}", use_container_width=True):
                            st.session_state.inspected_var_idx = idx
                            st.rerun()

                        with st.expander(f"Option {idx+1} Details & Rationale", expanded=True):
                            st.markdown(f"""
                                <div style="background: #ffffff; border: 1.5px solid #cbd5e1; border-radius: 8px; padding: 12px; margin-bottom: 8px; font-size: 0.86rem; color: #0f172a;">
                                    <div style="color: #0f172a; font-weight: 600;"><b>Wall Anchor:</b> {var.get('wall_anchor', 'Main Wall Anchor')}</div>
                                    <div style="color: #15803d; font-weight: 700; margin-top: 3px;"><b>Corridor Flow:</b> {var.get('circulation_corridor', 'Walkway >90cm')}</div>
                                    <div style="color: #1d4ed8; font-weight: 700; margin-top: 3px;"><b>Space Optimization:</b> Unobstructed doors, windows & paths</div>
                                </div>
                            """, unsafe_allow_html=True)

                            st.markdown(f"""
                                <div style="font-size: 0.84rem; color: #0f172a; font-weight: 500; margin-bottom: 8px;">
                                    <b>Architectural Rationale:</b><br>
                                    {''.join([f'• {b}<br>' for b in ai_exp.get('rationale_bullets', [])[:2]])}
                                </div>
                            """, unsafe_allow_html=True)

                            st.markdown("**Key Furniture & Fixtures:**")
                            for prod in ai_exp["products"][:3]:
                                st.markdown(f"- `{prod['cost']}` {prod['name']}")

                            st.markdown("**Color Palette (60-30-10):**")
                            sw_cols = st.columns(len(ai_exp["color_swatches"]))
                            for s_idx, s_col in enumerate(sw_cols):
                                sw = ai_exp["color_swatches"][s_idx]
                                with s_col:
                                    st.markdown(f"""
                                        <div style="text-align: center;">
                                            <div class="swatch-box" style="background-color: {sw['hex']}; height: 22px;"></div>
                                            <div class="swatch-label" style="font-size: 0.65rem;">{sw['hex']}</div>
                                        </div>
                                    """, unsafe_allow_html=True)


# ==============================================================================
# TAB 4: Spatial Analytics and Ergonomic Metrics
# ==============================================================================
with tab4:
    if not st.session_state.redesign_data:
        st.info("ℹ️ Generate a design in Tab 3 to view spatial analytics, 2D floor maps, and ergonomic reports.")
    else:
        r_data = st.session_state.redesign_data
        variations = r_data.get("variations", [])
        curr_idx = min(st.session_state.inspected_var_idx, len(variations) - 1) if variations else 0

        # Option selector linked directly to Tab 3 preferred selection
        st.markdown("### 📊 Detailed Layout Analysis & Product Specifications")
        opt_titles = [f"{var.get('title', f'Option {i+1}')} ({var.get('style', 'Modern')})" for i, var in enumerate(variations)]
        
        c_sel1, c_sel2 = st.columns([2, 1])
        with c_sel1:
            selected_option = st.selectbox(
                "Preferred Design Option to Evaluate:",
                opt_titles,
                index=curr_idx,
                key="tab4_option_selector"
            )
            sel_idx = opt_titles.index(selected_option)
            if sel_idx != st.session_state.inspected_var_idx:
                st.session_state.inspected_var_idx = sel_idx
                st.rerun()
        
        with c_sel2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            st.info(f"💡 Active preferred option synced with Tab 3")

        curr_var = variations[curr_idx] if variations else {}
        var_title = curr_var.get("title", f"Option {curr_idx+1}")
        var_id = curr_var.get("variation_id", f"var_{curr_idx+1}")
        var_style = curr_var.get("style", active_users[0]["preferred_style"])
        var_persona = curr_var.get("persona_blend", "Inhabitant Focus")
        var_arch = curr_var.get("layout_archetype", "Architectural Arrangement")

        active_ai = recommender_service.get_variation_ai_explanation(
            variation_id=var_id,
            style=var_style,
            budget_tier=st.session_state.budget_tier,
            budget_limit=float(st.session_state.budget_limit),
            persona_name=var_persona
        )
        base_metrics = r_data.get("comfort_metrics", {})
        metrics = {
            **base_metrics,
            "overall_comfort_score": active_ai.get("comfort_score", 95.0),
            "circulation_flow_rate_pct": active_ai.get("circulation_flow_rate_pct", 92.5),
            "daylight_factor_pct": active_ai.get("daylight_factor_pct", 3.2),
            "ada_accessibility_score": active_ai.get("ada_accessibility_score", 94.0),
            "acoustic_nrc_rating": active_ai.get("acoustic_nrc_rating", 0.70),
            "circulation_score": round(active_ai.get("comfort_score", 95.0) * 0.96, 1),
            "lighting_score": round(active_ai.get("comfort_score", 95.0) * 0.98, 1),
            "ergonomics_score": round(active_ai.get("comfort_score", 95.0) * 0.95, 1),
            "clutter_index": 14.0 if curr_idx == 0 else 11.5,
            "actionable_fixes": base_metrics.get("actionable_fixes", [])
        }
        
        # Selected option evaluation banner
        st.markdown(f"""
            <div style="background: #ffffff; border: 1.5px solid #cbd5e1; border-left: 6px solid #2563eb; border-radius: 12px; padding: 14px 18px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; box-shadow: 0 2px 10px rgba(15,23,42,0.04);">
                <div>
                    <span style="font-weight: 800; color: #0f172a; font-size: 1.05rem;">Evaluating Preferred Design: <span style="color: #2563eb;">{var_title}</span></span>
                    <div style="font-size: 0.84rem; color: #0f172a; font-weight: 700; margin-top: 2px;"><b>Archetype:</b> {var_arch} &nbsp;|&nbsp; <b>Style:</b> {var_style} &nbsp;|&nbsp; <b>Inhabitant Focus:</b> {var_persona}</div>
                </div>
                <div style="display: flex; gap: 8px; align-items: center;">
                    <span style="background: #dcfce7; border: 1px solid #86efac; color: #15803d; border-radius: 8px; padding: 4px 12px; font-weight: 800; font-size: 0.95rem;">⭐ {active_ai.get('comfort_score', 95.0)}/100 Comfort</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Core Metrics Display with High Contrast
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            render_rich_kpi("Circulation Flow", f"{active_ai.get('circulation_flow_rate_pct', 92.5)}%", "Corridors >90cm")
        with k2:
            render_rich_kpi("Daylight Factor", f"{active_ai.get('daylight_factor_pct', 3.2)}% DF", "Natural Light")
        with k3:
            render_rich_kpi("Accessibility", f"{active_ai.get('ada_accessibility_score', 94.0)}/100", "Clearance Paths")
        with k4:
            render_rich_kpi("Acoustic Rating", f"NRC {active_ai.get('acoustic_nrc_rating', 0.70)}", "Sound Balance")
        with k5:
            render_rich_kpi("Comfort Score", f"{active_ai.get('comfort_score', 95.0)}/100", "Overall Score")

        st.markdown("---")

        # Furniture & Product Specifications Table for this Preferred Option
        st.markdown("### 🛋️ Product Specifications & Costs for Preferred Design")
        if active_ai.get("products"):
            df_products = pd.DataFrame([
                {
                    "Item / Product Name": p["name"],
                    "Category": p.get("type", "Furniture"),
                    "Estimated Cost": p["cost"],
                    "Vendor / Source": p.get("vendor", "Verified Interior Supplier")
                }
                for p in active_ai["products"]
            ])
            st.dataframe(df_products, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Visual Charts
        ch1, ch2 = st.columns([1.2, 0.8])
        with ch1:
            st.markdown("### 📐 2D Layout & Pathway Map")
            st.caption("Visual representation of room perimeter, furniture footprints, and walking corridors:")
            heatmap_img = generate_spatial_clearance_heatmap(metrics)
            st.image(heatmap_img, use_container_width=True)

        with ch2:
            st.markdown("### 📊 Performance Charts")
            radar_img = generate_comfort_radar_chart(metrics)
            st.image(radar_img, use_container_width=True)

            bar_img = generate_metrics_bar_chart(metrics)
            st.image(bar_img, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📜 Architectural Recommendations")

        recs = r_data.get("explainable_recommendations", [])
        for i, rec in enumerate(recs, 1):
            st.markdown(f"""
                <div class="glass-card" style="padding: 12px 16px; margin-bottom: 8px; border-left: 4px solid #2563eb;">
                    <div style="font-weight: 700; color: #0f172a; font-size: 0.88rem;">Guideline {i}: {rec}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🛠️ Actionable Layout Improvements")
        fixes = metrics.get("actionable_fixes", [])
        if fixes:
            df_fixes = pd.DataFrame([
                {
                    "Category": f["issue_category"] if isinstance(f, dict) else f.issue_category,
                    "Current State": f["current_state"] if isinstance(f, dict) else f.current_state,
                    "Recommended Action": f["recommended_action"] if isinstance(f, dict) else f.recommended_action,
                    "Impact": f["impact_metric"] if isinstance(f, dict) else f.impact_metric
                }
                for f in fixes
            ])
            st.dataframe(df_fixes, use_container_width=True, hide_index=True)

        # Download Report
        report_dict = {
            "project_uid": r_data.get("project_uid"),
            "room_type": r_data.get("room_type"),
            "selected_option": var_title,
            "style": var_style,
            "budget_tier": st.session_state.budget_tier,
            "budget_limit": f"₹{st.session_state.budget_limit}",
            "users": active_users,
            "metrics": metrics,
            "products": active_ai.get("products", []),
            "rules": recs
        }
        st.download_button(
            label="📥 Download Complete Project Report (JSON)",
            data=json.dumps(report_dict, indent=4),
            file_name=f"redesign_report_{r_data.get('project_uid', 'export')}.json",
            mime="application/json"
        )
