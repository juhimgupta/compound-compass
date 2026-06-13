#!/usr/bin/env python3
"""Compound Compass — Streamlit entry point."""

from __future__ import annotations

import streamlit as st

from config import APP_TITLE

# ── Page config ──────────────────────────────────────────────────────────── #
st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="collapsed")

# ── CSS ──────────────────────────────────────────────────────────────────── #
st.markdown("""
<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background-color:#F4F4F4; }
[data-testid="stHeader"] { background-color:#F4F4F4; }
#MainMenu, footer { visibility:hidden; }
.notice {
    background:#FFFBF2; border-left:5px solid #FFA500;
    padding:12px 18px; border-radius:0 6px 6px 0;
    font-size:0.83rem; line-height:1.65; color:#444; margin-bottom:10px;
}
.card {
    background:#FFFFFF; border-radius:8px;
    padding:20px 20px 14px 20px; border:1px solid #E0E0E0;
}
.meta-label {
    font-size:0.70rem; font-weight:600; color:#999;
    text-transform:uppercase; letter-spacing:0.07em; margin:10px 0 2px 0;
}
.meta-value { font-size:0.97rem; font-weight:700; color:#222; }
.rule-orange { border:none; border-top:2px solid #FFA500; opacity:0.45; margin:20px 0 12px 0; }
.sig-box { border-radius:0 6px 6px 0; padding:11px 16px 13px 16px; margin-bottom:10px; }
.sig-box-title { font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:6px; }
.sig-box ul { margin:0; padding-left:18px; font-size:0.82rem; line-height:1.9; }
[data-baseweb="tab"] { font-size:0.88rem !important; font-weight:600 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────── #
for _k, _v in {
    "page": "input",
    "candidate_info": {},
    "comp_page": 0,
    "rec_idx": 0,
    "selected_faers_comparators": [],
    "selected_trial_comparators": [],
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Routing ──────────────────────────────────────────────────────────────── #
if st.session_state.page == "input":
    from page_input import render_input_page
    render_input_page()
elif st.session_state.page == "results":
    from page_results import render_results_page
    render_results_page()
