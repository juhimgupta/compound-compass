from __future__ import annotations

import streamlit as st

from utils import _paths_for, _csv
from ui_components import _render_header
from tab_candidate import render_candidate_tab
from tab_ae import render_ae_tab
from tab_target import render_target_tab
from tab_trials import render_trials_tab


def render_results_page() -> None:
    cand = st.session_state.candidate_info
    target, slug = cand["target"], cand["slug"]
    dirs, files, rel = _paths_for(slug)

    hdr_l, hdr_r = st.columns([5, 1])
    with hdr_l:
        _render_header(
            target=target,
            indication=cand.get("indication", "—"),
            comparator_label="FAERS and trial comparators selected in analysis tabs",
            centered=False,
        )
    with hdr_r:
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        if st.button("New analysis", use_container_width=True):
            st.session_state.page = "input"
            st.session_state.candidate_info = {}
            st.session_state.comp_page = 0
            st.session_state.rec_idx = 0
            st.session_state.selected_faers_comparators = []
            st.session_state.selected_trial_comparators = []
            st.rerun()

    st.markdown('<hr class="rule-orange"/>', unsafe_allow_html=True)

    tab_candidate, tab_ae, tab_target, tab_trials = st.tabs([
        "Candidate Analysis", "Adverse Event Analysis", "Target Context", "Clinical Trial Context"
    ])

    with tab_candidate:
        render_candidate_tab(files, cand)

    with tab_ae:
        df_sim = _csv(files["similarity"])
        render_ae_tab(files, rel, dirs, cand, df_sim)

    with tab_target:
        render_target_tab(files, target)

    with tab_trials:
        df_sim = _csv(files["similarity"])
        render_trials_tab(files, dirs, target, cand, df_sim)
