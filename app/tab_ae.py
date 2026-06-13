from __future__ import annotations

import sys

import pandas as pd
import streamlit as st

from config import MAX_FAERS_COMPARATORS
from utils import _csv, _run, _engine_script, _auto_select_comparators, _option_map_from_similarity
from safety import _spec_and_sev_maps, _render_scaffold_signal_recommendations


def render_ae_tab(files: dict, rel: dict, dirs: dict, cand: dict, df_sim: pd.DataFrame) -> None:
    st.markdown("### Adverse Event Analysis")
    st.markdown(
        '<div class="notice" style="font-size:0.80rem;padding:9px 14px;">'
        'FAERS analysis is run only on the selected comparator set. Class-wide and molecule-specific signals are conditioned on this choice.'
        '</div>',
        unsafe_allow_html=True,
    )

    target = cand["target"]

    if df_sim.empty:
        st.warning("Similarity data not available yet.")
        st.stop()

    options_map = _option_map_from_similarity(df_sim)
    if not options_map:
        st.warning("No FAERS-searchable comparator candidates were found in the similarity table.")
        st.stop()

    labels_by_value = {v.lower(): k for k, v in options_map.items()}
    default_drugs = []
    if cand.get("comparator_mode") == "verified":
        for d in cand.get("verified_comparators", []):
            if d.lower() in labels_by_value:
                default_drugs.append(labels_by_value[d.lower()])
            else:
                label = f"{d}  ·  verified comparator"
                options_map[label] = d
                default_drugs.append(label)
    if not default_drugs:
        auto = _auto_select_comparators(df_sim, MAX_FAERS_COMPARATORS)
        default_drugs = [labels_by_value[d.lower()] for d in auto if d.lower() in labels_by_value]

    selected_labels = st.multiselect(
        f"Select FAERS Comparators (Maximum {MAX_FAERS_COMPARATORS})",
        options=list(options_map.keys()),
        default=default_drugs[:MAX_FAERS_COMPARATORS],
        help="Prefer clinically advanced, target-relevant small molecules with clean FAERS-searchable names.",
    )
    selected_drugs = [options_map[l] for l in selected_labels]

    if len(selected_drugs) > MAX_FAERS_COMPARATORS:
        st.error(f"Please select at most {MAX_FAERS_COMPARATORS} comparator drugs.")
        st.stop()
    if len(selected_drugs) < 2:
        st.warning("Select at least 2 drugs to distinguish class-wide from molecule-specific signals.")

    run_col, clear_col = st.columns([2, 1])
    with run_col:
        run_faers = st.button("Run FAERS Analysis", type="secondary", disabled=len(selected_drugs) < 2, use_container_width=True)
    with clear_col:
        if st.button("Clear FAERS outputs", use_container_width=True):
            for p in [files["event_class"], files["ror_matrix"], files["count_matrix"], files["recommendations"], files["faers_selected_drugs"]]:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
            st.session_state.selected_faers_comparators = []
            st.rerun()

    if run_faers:
        dirs["faers_class"].mkdir(parents=True, exist_ok=True)
        dirs["recs"].mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"faers_search_name": selected_drugs}).to_csv(files["faers_selected_drugs"], index=False)
        st.session_state.selected_faers_comparators = selected_drugs
        with st.status("Running FAERS analysis on selected comparators...", expanded=True) as _ae_st:
            st.write("Building event x drug FAERS matrix...")
            ok, out, err = _run([sys.executable, _engine_script("class_matrix.py"), "--drugs", *selected_drugs, "--prefix", rel["faers_prefix"]])
            if not ok:
                st.error("class_matrix.py failed.")
                with st.expander("Details"): st.code(err or out)
                _ae_st.update(label="FAERS analysis failed.", state="error")
                st.stop()
            st.write("Building recommendation layer...")
            ok, out, err = _run([
                sys.executable, _engine_script("build_recommendations.py"),
                "--event-classification", f"{rel['faers_prefix']}_event_classification.csv",
                "--ror-matrix", f"{rel['faers_prefix']}_ror_matrix.csv",
                "--count-matrix", f"{rel['faers_prefix']}_count_matrix.csv",
                "--target", target,
                "--out", rel["recs_out"],
            ])
            if not ok:
                st.error("build_recommendations.py failed.")
                with st.expander("Details"): st.code(err or out)
                _ae_st.update(label="Recommendation build failed.", state="error")
                st.stop()
            _ae_st.update(label="FAERS analysis complete.", state="complete", expanded=False)
        st.rerun()

    if files["event_class"].exists() and files["recommendations"].exists():
        if not st.session_state.selected_faers_comparators and files["faers_selected_drugs"].exists():
            st.session_state.selected_faers_comparators = (
                _csv(files["faers_selected_drugs"])
                .get("faers_search_name", pd.Series(dtype=str))
                .astype(str).tolist()
            )

        df_ec = _csv(files["event_class"])
        df_recs = _csv(files["recommendations"])
        spec_map, sev_map = _spec_and_sev_maps(df_ec, df_recs)

        severity_order = ["high", "medium", "low", "contextual", "unknown"]
        severity_label_map = {"high": "High", "medium": "Medium", "low": "Low", "contextual": "Contextual", "unknown": "Unknown"}
        observed_severities: set[str] = set()
        if not df_recs.empty and "severity" in df_recs.columns:
            observed_severities = set(df_recs["severity"].astype(str).str.lower().str.strip())
        available_severities = [s for s in severity_order if s in observed_severities] or severity_order

        selected_severity_labels = st.multiselect(
            "Filter displayed FAERS signals by severity",
            options=[severity_label_map[s] for s in available_severities],
            default=[severity_label_map[s] for s in available_severities],
            help="This only filters the displayed class-wide, molecule-specific, and structural safety prompt sections. It does not rerun FAERS.",
        )
        selected_severities = {k for k, v in severity_label_map.items() if v in selected_severity_labels}

        df_ec_display = df_ec.copy()
        if not df_ec_display.empty and "event" in df_ec_display.columns:
            df_ec_display["_severity_filter"] = df_ec_display["event"].astype(str).str.upper().map(sev_map).fillna("unknown")
            df_ec_display = df_ec_display[df_ec_display["_severity_filter"].isin(selected_severities)].copy()

        df_recs_display = df_recs.copy()
        if not df_recs_display.empty and "severity" in df_recs_display.columns:
            df_recs_display["_severity_filter"] = df_recs_display["severity"].astype(str).str.lower().str.strip().replace("", "unknown")
            df_recs_display = df_recs_display[df_recs_display["_severity_filter"].isin(selected_severities)].copy()

        spec_map_display, sev_map_display = _spec_and_sev_maps(df_ec_display, df_recs if not df_recs.empty else df_recs_display)

        col1, col2 = st.columns(2)
        with col1:
            cw: list[str] = []
            if not df_ec_display.empty and "classification" in df_ec_display.columns:
                cw = df_ec_display[df_ec_display["classification"].str.contains("class-wide", case=False, na=False)]["event"].astype(str).str.title().tolist()
            li = "".join(f"<li>{e}</li>" for e in cw[:15]) or '<li style="color:#bbb;font-style:italic;">None detected</li>'
            st.markdown(f'<div class="sig-box" style="background:#FFF3E0; border-left:4px solid #FFA500;"><div class="sig-box-title" style="color:#E65100;">Class-Wide FAERS Signals — {target}</div><ul style="max-height:220px; overflow-y:auto;">{li}</ul></div>', unsafe_allow_html=True)
        with col2:
            mol_rows = (
                df_ec_display[df_ec_display["classification"].str.contains("molecule-specific", case=False, na=False)]
                if not df_ec_display.empty and "classification" in df_ec_display.columns
                else pd.DataFrame()
            )
            if not mol_rows.empty:
                li = "".join(f"<li><b>{str(r.get('driver','')).title()}</b>: {str(r.get('event','')).title()}</li>" for _, r in mol_rows.head(15).iterrows())
            else:
                li = '<li style="color:#bbb;font-style:italic;">None detected</li>'
            st.markdown(f'<div class="sig-box" style="background:#FFF8EE; border-left:4px solid #FFB347;"><div class="sig-box-title" style="color:#D4780A;">Molecule-Specific FAERS Signals</div><ul style="max-height:220px; overflow-y:auto;">{li}</ul></div>', unsafe_allow_html=True)

        _render_scaffold_signal_recommendations(df_sim, spec_map_display, sev_map_display, cand)

        if not df_recs_display.empty:
            with st.expander("Full FAERS recommendation table"):
                show_cols = [c for c in [
                    "event", "event_category", "severity", "classification", "driver",
                    "n_signal_drugs", "max_ror", "total_class_reports", "priority", "stakeholder_owner",
                ] if c in df_recs.columns]
                st.dataframe(df_recs_display[show_cols].reset_index(drop=True), use_container_width=True, hide_index=True)
    else:
        st.info("No FAERS analysis has been run for this candidate/target yet. Select comparator drugs above and click Run FAERS Analysis.")
