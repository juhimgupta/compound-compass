from __future__ import annotations

import sys

import streamlit as st

from config import TARGET_CONFIGS, TARGET_OPTIONS
from utils import _slugify_target, _paths_for, _run, _engine_script
from ui_components import _render_header


def render_input_page() -> None:
    st.markdown("""
    <style>
    .stTextInput label p, .stTextArea label p, .stSelectbox label p {
        font-size: 1.5rem !important;
        font-weight: 800 !important;
        color: #222 !important;
    }
    .stTextInput input { font-size: 1.3rem !important; padding: 0.55rem 0.75rem !important; }
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div { font-size: 1.3rem !important; }
    .stTextArea textarea { font-size: 1.3rem !important; }
    .stFormSubmitButton button { font-size: 1.25rem !important; padding: 0.7rem 2.5rem !important; }
    .notice { font-size: 1rem !important; line-height: 1.7 !important; }
    </style>
    """, unsafe_allow_html=True)

    _render_header(centered=True)
    st.markdown('<hr class="rule-orange"/>', unsafe_allow_html=True)
    st.markdown("""
    <div class="notice"><strong>Compound Compass is a triage and hypothesis-generation tool.</strong><br>
    All outputs require expert scientific, clinical, and regulatory review before any development decision.</div>
    """, unsafe_allow_html=True)

    with st.form("candidate_form"):
        f_name = st.text_input("Candidate name *", placeholder="e.g. Internal-TKI-001")
        target_choice = st.selectbox("Target *", options=TARGET_OPTIONS, index=0,
                                     help="Choose a configured target or enter a custom target symbol.")
        if target_choice == "Other / custom target":
            f_target_custom = st.text_input("Custom target symbol", placeholder="e.g. MET")
            selected_target = str(f_target_custom).strip().upper()
            default_indication = "Target-associated disease context"
            config = {
                "slug": _slugify_target(selected_target or "target"),
                "default_indication": default_indication,
                "verified_comparators": [],
                "comparator_class_label": "auto-selected clinical small-molecule comparators",
            }
        else:
            selected_target = target_choice
            config = TARGET_CONFIGS[selected_target]
            default_indication = config["default_indication"]
        f_indication = st.text_input(
            "Therapeutic context / indication (optional)",
            value=default_indication,
            help="Used for display context only. The app can infer a default from the selected target, but you can override it.",
        )
        f_smiles = st.text_area("Candidate SMILES *", placeholder="e.g. COc1cc2ncnc(Nc3cccc(Cl)c3)c2cc1OCCCN1CCOCC1", height=160)
        st.markdown("---")
        submitted = st.form_submit_button("Analyze Candidate", type="primary")

    if submitted:
        target = str(selected_target).strip().upper()
        if not target:
            st.error("Target is required.")
            st.stop()
        slug = config.get("slug") or _slugify_target(target)
        if not f_name.strip():
            st.error("Candidate name is required.")
            st.stop()
        if not f_smiles.strip():
            st.error("Candidate SMILES is required.")
            st.stop()

        dirs, files, rel = _paths_for(slug)
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        verified_comparators = config.get("verified_comparators", [])
        st.session_state.candidate_info = {
            "name": f_name.strip(),
            "smiles": f_smiles.strip(),
            "target": target,
            "slug": slug,
            "indication": f_indication.strip() or config.get("default_indication", "—"),
            "setup_label": target,
            "comparator_mode": "verified" if verified_comparators else "auto",
            "verified_comparators": verified_comparators,
            "comparator_class_label": config.get("comparator_class_label", "auto-selected clinical small-molecule comparators"),
            "display_modalities": ["Small molecule"],
            "display_min_stage": "Any",
        }
        st.session_state.selected_faers_comparators = []
        st.session_state.selected_trial_comparators = []
        cand = st.session_state.candidate_info

        st.markdown('<hr class="rule-orange"/>', unsafe_allow_html=True)
        with st.status("Running candidate analysis...", expanded=True) as _status:
            st.write(f"Pulling target context from Open Targets for {target}...")
            ok, out, err = _run([sys.executable, _engine_script("open_targets_context.py"), target, "--prefix", rel["ot_prefix"], "--drug-limit", "100"])
            if not ok:
                st.error("open_targets_context.py failed.")
                with st.expander("Details"): st.code(err or out)
                _status.update(label="Analysis failed.", state="error")
                st.stop()

            st.write(f"Curating target-associated molecules for {target}...")
            ok, out, err = _run([sys.executable, _engine_script("curate_target_class.py"), rel["known_drugs"], "--target", target, "--out", rel["curated"]])
            if not ok:
                st.error("curate_target_class.py failed.")
                with st.expander("Details"): st.code(err or out)
                _status.update(label="Analysis failed.", state="error")
                st.stop()

            st.write("Computing candidate similarity across target-associated molecules...")
            sim_cmd = [
                sys.executable, _engine_script("candidate_similarity.py"),
                "--candidate-name", cand["name"],
                "--candidate-smiles", cand["smiles"],
                "--target", target,
                "--known-drugs", rel["known_drugs"],
                "--curated-class", rel["curated"],
                "--out", rel["sim_out"],
            ]
            if files["recommendations"].exists():
                sim_cmd.extend(["--recommendations", rel["recs_out"]])
            ok, out, err = _run(sim_cmd)
            if not ok:
                st.error("candidate_similarity.py failed.")
                with st.expander("Details"): st.code(err or out)
                _status.update(label="Analysis failed.", state="error")
                st.stop()
            _status.update(label="Candidate analysis complete.", state="complete", expanded=False)

        st.session_state.comp_page = 0
        st.session_state.rec_idx = 0
        st.session_state.page = "results"
        st.rerun()
