from __future__ import annotations

import streamlit as st

# Color Palette Tokens
BG_COLOR = "#0B0D12"
CARD_BG = "#151821"
BORDER_COLOR = "#272C36"
TEXT_PRIMARY = "#F8FAFC"
TEXT_SECONDARY = "#94A3B8"
ACCENT_PRIMARY = "#FF4B4B"
ACCENT_MUTED = "rgba(255, 75, 75, 0.15)"
SUCCESS_COLOR = "#10B981"


def apply_theme():
    """
    Injects custom CSS design tokens and layout styling into Streamlit.
    """
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&display=swap');

        /* Main container layout */
        .main .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }}

        /* Buttons */
        .stButton > button {{
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease-in-out;
        }}
        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, #FF4B4B 0%, #D93838 100%);
            border: none;
            box-shadow: 0 4px 12px rgba(255, 75, 75, 0.25);
        }}
        .stButton > button[kind="primary"]:hover {{
            box-shadow: 0 6px 16px rgba(255, 75, 75, 0.4);
            transform: translateY(-1px);
        }}

        /* Studio Brief Card */
        .studio-card {{
            background-color: {CARD_BG};
            border: 1px solid {BORDER_COLOR};
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        }}

        /* Metric Pill Banner */
        .metric-banner {{
            display: flex;
            gap: 1rem;
            background-color: {CARD_BG};
            border: 1px solid {BORDER_COLOR};
            border-radius: 10px;
            padding: 1rem 1rem;
            margin-bottom: 1rem;
            justify-content: space-around;
        }}
        .metric-item {{
            text-align: left;
        }}
        .metric-value {{
            font-size: 1.5rem;
            font-weight: 700;
            color: {TEXT_PRIMARY};
            line-height: 1.2;
        }}
        .metric-label {{
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: {TEXT_SECONDARY};
            margin-top: 0.25rem;
        }}

        /* Article Reading View */
        .article-reader {{
            max-width: 850px;
            margin: 0 auto;
            font-family: 'Newsreader', serif;
            font-size: 1.15rem;
            line-height: 1.8;
            color: {TEXT_PRIMARY};
        }}
        .article-meta {{
            font-family: 'Inter', sans-serif;
            font-size: 0.9rem;
            color: {TEXT_SECONDARY};
            border-bottom: 1px solid {BORDER_COLOR};
            padding-bottom: 1rem;
            margin-bottom: 1rem;
        }}

        /* Images and Responsive Visuals */
        div[data-testid="stImage"] img, div[data-testid="stMarkdownContainer"] img {{
            max-width: 100% !important;
            width: auto !important;
            max-height: 500px !important;
            height: auto !important;
            object-fit: contain !important;
            border-radius: 8px !important;
            margin: 1rem auto !important;
            display: block !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4) !important;
        }}

        /* Status Badge */
        .status-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: #10B981;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
        }}

        /* Library Cards */
        .library-card {{
            background-color: {CARD_BG};
            border: 1px solid {BORDER_COLOR};
            border-radius: 10px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            transition: border-color 0.2s ease-in-out;
        }}
        .library-card:hover {{
            border-color: #3B82F6;
        }}

        /* Workflow Step Indicator */
        .workflow-bar {{
            background-color: {CARD_BG};
            border: 1px solid {BORDER_COLOR};
            border-radius: 10px;
            padding: 1rem 1rem;
            margin-bottom: 1rem;
        }}

        /* Sidebar Navbar - Remove top margin & padding */
        section[data-testid="stSidebar"] {{
            padding-top: 0 !important;
            margin-top: 0 !important;
        }}
        section[data-testid="stSidebar"] .block-container,
        div[data-testid="stSidebarUserContent"],
        div[data-testid="stSidebarContent"] {{
            padding-top: 0.5rem !important;
            margin-top: 0 !important;
        }}
        div[data-testid="stSidebarHeader"], .st-emotion-cache-10p9htt {{
            padding-top: 0 !important;
            margin-top: 0 !important;
            margin-bottom: 0 !important;
            height: auto !important;
        }}

        /* Sidebar - Dividers */
        section[data-testid="stSidebar"] hr,
        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] hr {{
            display: block !important;
            margin: 0.75rem 0 !important;
            border: none !important;
            border-top: 1px solid rgba(255, 255, 255, 0.12) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

