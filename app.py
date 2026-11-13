"""Etkileşimli Hesaplamalı Simülatör — entry point.

Run:  streamlit run app.py
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.settings import global_settings  # noqa: E402
from ui.style import apply_style  # noqa: E402

st.set_page_config(page_title="Classical & Quantum Imaging Simulator", page_icon="🔬", layout="wide",
                   initial_sidebar_state="expanded")
apply_style()

pages = [
    st.Page("app_pages/home.py", title="Ana Sayfa", icon=":material/home:", default=True),
    st.Page("app_pages/synthetic_cell.py", title="Hücre", icon=":material/biotech:"),
    st.Page("app_pages/channel_1.py", title="Kanal 1 · Koherent", icon=":material/flare:"),
    st.Page("app_pages/channel_2.py", title="Kanal 2 · Korelasyon", icon=":material/compare_arrows:"),
    st.Page("app_pages/channel_3.py", title="Kanal 3 · Dolanık", icon=":material/all_inclusive:"),
    st.Page("app_pages/comparison.py", title="Karşılaştırma", icon=":material/insights:"),
    st.Page("app_pages/validation_page.py", title="Doğrulama", icon=":material/fact_check:"),
]
nav = st.navigation(pages, position="top")
global_settings()
nav.run()
