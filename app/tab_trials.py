from __future__ import annotations

import html
import re
import sys
from typing import Any

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

from config import MAX_FAERS_COMPARATORS
from utils import _csv, _run, _engine_script, _has_value, _is_small_molecule, _stage_score, _auto_select_comparators


def _parse_age_years(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    s = str(value).strip().lower()
    if not s or s in {"nan", "none", "null", "n/a", "na"}:
        return None
    match = re.search(r"(\d+\.?\d*)", s)
    if not match:
        return None
    val = float(match.group(1))
    if "month" in s:
        return val / 12.0
    if "week" in s:
        return val / 52.0
    if "day" in s:
        return val / 365.0
    return val


def _extract_trial_enrollment(df_locs_in: pd.DataFrame, df_demo_in: pd.DataFrame) -> pd.DataFrame:
    source = df_demo_in.copy() if not df_demo_in.empty else (df_locs_in.copy() if not df_locs_in.empty else pd.DataFrame())
    if source.empty:
        return pd.DataFrame()
    enroll_col = next((c for c in ["enrollment_count", "enrollment", "enrollmentCount", "enrollment_number", "actual_enrollment"] if c in source.columns), None)
    if enroll_col is None:
        return pd.DataFrame()
    source["_enrollment"] = pd.to_numeric(source[enroll_col], errors="coerce")
    status_col = next((c for c in ["overall_status", "trial_status", "status"] if c in source.columns), None)
    if status_col is not None:
        source["_status"] = source[status_col].astype(str).str.upper()
        source = source[source["_status"].str.contains("COMPLETED", na=False)]
    if "nct_id" in source.columns:
        source = source.drop_duplicates(subset=["nct_id"])
    source = source[source["_enrollment"].notna()]
    source = source[source["_enrollment"] > 0]
    return source


def _demographic_metric_source(df_demo_in: pd.DataFrame, df_locs_in: pd.DataFrame) -> pd.DataFrame:
    if not df_demo_in.empty:
        return df_demo_in.copy()
    if not df_locs_in.empty:
        return df_locs_in.copy()
    return pd.DataFrame()


def render_trials_tab(files: dict, dirs: dict, target: str, cand: dict, df_sim: pd.DataFrame) -> None:
    df_locs = _csv(files["trial_locations"])
    df_country = _csv(files["country_summary"])
    df_demo = _csv(files["trial_demographics"])
    df_demo_summary = _csv(files["demographic_summary"])
    df_baseline = _csv(files["baseline_demographics"])
    df_selected_trials = _csv(files["trial_selected_drugs"])

    st.markdown("### Clinical Trial Context")
    st.caption("ClinicalTrials.gov comparator context for trial geography, enrollment benchmarks, and registry-level demographic fields.")

    if df_sim.empty:
        st.warning("Similarity data not found. Run the candidate analysis pipeline first.")
        st.stop()

    # ── Trial comparator selection ──────────────────────────────────────── #
    df_trial_opts = df_sim.copy()
    if "faers_search_name" not in df_trial_opts.columns and "drug_name" in df_trial_opts.columns:
        df_trial_opts["faers_search_name"] = df_trial_opts["drug_name"]
    df_trial_opts = df_trial_opts[df_trial_opts["faers_search_name"].apply(_has_value)].copy()
    if "drug_type" in df_trial_opts.columns:
        df_trial_opts = df_trial_opts[df_trial_opts["drug_type"].apply(_is_small_molecule)].copy()
    df_trial_opts["_tan"] = pd.to_numeric(df_trial_opts.get("tanimoto_similarity", pd.Series([0] * len(df_trial_opts))), errors="coerce").fillna(0)
    df_trial_opts["_stage_score"] = df_trial_opts.get("max_clinical_stage_for_target", pd.Series([""] * len(df_trial_opts))).apply(_stage_score)
    df_trial_opts["_name_norm"] = df_trial_opts["faers_search_name"].astype(str).str.lower().str.strip()
    df_trial_opts = df_trial_opts.sort_values(["_stage_score", "_tan"], ascending=[False, False]).drop_duplicates("_name_norm").reset_index(drop=True)

    trial_option_map: dict[str, str] = {}
    for _, r in df_trial_opts.iterrows():
        faers_name = str(r.get("faers_search_name", "")).strip()
        drug_name = str(r.get("drug_name", faers_name)).strip()
        stage = str(r.get("max_clinical_stage_for_target", "unknown")).strip()
        dtype = str(r.get("drug_type", "unknown")).strip()
        tan = r.get("_tan", None)
        label = f"{drug_name} · {stage} · {dtype}"
        if pd.notna(tan):
            label += f" · Tanimoto {float(tan):.3f}"
        trial_option_map[label] = faers_name

    previous_trial_selection = st.session_state.get("selected_trial_comparators", []) or []
    if not previous_trial_selection and not df_selected_trials.empty:
        first_col = df_selected_trials.columns[0]
        previous_trial_selection = [str(x).strip() for x in df_selected_trials[first_col].dropna().tolist() if str(x).strip()]
    if not previous_trial_selection:
        previous_trial_selection = st.session_state.get("selected_faers_comparators", []) or []
    if not previous_trial_selection:
        previous_trial_selection = cand.get("verified_comparators", []) or []
    if not previous_trial_selection:
        previous_trial_selection = _auto_select_comparators(df_sim, n=MAX_FAERS_COMPARATORS)

    previous_norm = {str(x).lower().strip() for x in previous_trial_selection}
    default_labels = [label for label, value in trial_option_map.items() if str(value).lower().strip() in previous_norm]

    st.markdown(
        '<div class="notice" style="font-size:0.80rem;padding:9px 14px;margin-top:4px;">'
        'Select comparator drugs to query ClinicalTrials.gov. This section provides clinical precedent and rough planning context, not a formal feasibility estimate.'
        '</div>',
        unsafe_allow_html=True,
    )

    sel_col, action_col = st.columns([3, 1])
    with sel_col:
        selected_trial_labels = st.multiselect(
            "Trial comparator drugs",
            options=list(trial_option_map.keys()),
            default=default_labels[:MAX_FAERS_COMPARATORS],
            help=f"Select up to {MAX_FAERS_COMPARATORS} comparator drugs for ClinicalTrials.gov context.",
        )
    selected_trial_drugs = [trial_option_map[x] for x in selected_trial_labels]
    if len(selected_trial_drugs) > MAX_FAERS_COMPARATORS:
        st.warning(f"Please select no more than {MAX_FAERS_COMPARATORS} comparator drugs.")
        selected_trial_drugs = selected_trial_drugs[:MAX_FAERS_COMPARATORS]

    with action_col:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        rebuild_trials = st.button("Build / rebuild trial context", use_container_width=True, type="primary", disabled=(len(selected_trial_drugs) == 0))
        clear_trials = st.button("Clear trial outputs", use_container_width=True, type="secondary")

    if clear_trials:
        for p in [files["trial_locations"], files["country_summary"], files["trial_demographics"], files["demographic_summary"], files["baseline_demographics"], files["trial_selected_drugs"]]:
            try:
                if p.exists(): p.unlink()
            except Exception:
                pass
        st.session_state.selected_trial_comparators = []
        st.rerun()

    if rebuild_trials:
        dirs["trial_geo"].mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"drug_name": selected_trial_drugs}).to_csv(files["trial_selected_drugs"], index=False)
        st.session_state.selected_trial_comparators = selected_trial_drugs
        with st.spinner("Querying ClinicalTrials.gov for comparator trial context..."):
            ok, out, err = _run([sys.executable, _engine_script("clinical_trial_geography.py"), "--target", target, "--drugs", *selected_trial_drugs])
        if ok:
            st.success("Clinical trial context rebuilt.")
            st.rerun()
        else:
            st.error("Clinical trial context build failed.")
            with st.expander("ClinicalTrials.gov command output"):
                st.code(out or ""); st.code(err or "")

    st.markdown('<hr class="rule-orange"/>', unsafe_allow_html=True)

    # Reload after any rebuild/clear action
    df_locs = _csv(files["trial_locations"])
    df_country = _csv(files["country_summary"])
    df_demo = _csv(files["trial_demographics"])
    df_demo_summary = _csv(files["demographic_summary"])
    df_baseline = _csv(files["baseline_demographics"])

    selected_display = selected_trial_drugs or st.session_state.get("selected_trial_comparators", [])
    if selected_display:
        st.markdown(f'<div style="font-size:0.82rem;color:#777;margin-bottom:10px;">Current trial comparator set: <b>{", ".join(selected_display)}</b></div>', unsafe_allow_html=True)

    if df_locs.empty and df_country.empty and df_demo.empty:
        st.info("No clinical trial context has been built yet. Select comparator drugs and click **Build / rebuild trial context**.")
        st.stop()

    # ── Trial Geography ─────────────────────────────────────────────────── #
    st.markdown("### Trial Geography")
    if not df_country.empty and _PLOTLY:
        country_col = "country" if "country" in df_country.columns else None
        count_col = next((c for c in ["n_trials", "trial_count", "count"] if c in df_country.columns), None)
        if country_col and count_col:
            fig = px.choropleth(df_country, locations=country_col, locationmode="country names", color=count_col,
                                hover_name=country_col, color_continuous_scale="Oranges",
                                title="Comparator trial footprint by country", labels={count_col: "Trials"})
            fig.update_layout(height=430, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="#F4F4F4", plot_bgcolor="#F4F4F4")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Country summary exists, but country/count columns were not recognized.")
    elif not df_country.empty:
        st.dataframe(df_country, use_container_width=True, hide_index=True)
    else:
        st.info("Country-level trial summary is not available.")

    # ── Enrollment Benchmarks ────────────────────────────────────────────── #
    st.markdown("### Enrollment Benchmarks")
    st.caption("Enrollment among completed comparator trials with available enrollment counts. Use this as a rough planning benchmark, not as a formal feasibility estimate.")

    enroll_df = _extract_trial_enrollment(df_locs, df_demo)
    if enroll_df.empty:
        st.info("Enrollment benchmark data is not available in the current trial outputs.")
    else:
        enroll_values = enroll_df["_enrollment"].dropna()
        avg_enroll = int(round(enroll_values.mean()))
        med_enroll = int(round(enroll_values.median()))
        q1 = int(round(enroll_values.quantile(0.25)))
        q3 = int(round(enroll_values.quantile(0.75)))
        n_completed = int(enroll_df["nct_id"].nunique()) if "nct_id" in enroll_df.columns else int(len(enroll_df))
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Median enrollment", f"{med_enroll:,}")
        k2.metric("Typical range (IQR)", f"{q1:,}–{q3:,}")
        k3.metric("Avg. enrollment", f"{avg_enroll:,}")
        k4.metric("Completed trials", f"{n_completed:,}")
        if avg_enroll > med_enroll * 1.5:
            st.markdown('<div class="notice" style="font-size:0.80rem;padding:9px 14px;margin-top:8px;">The average enrollment is higher than the median, suggesting a right-skewed distribution. In practice, the median and IQR may be more useful for understanding a typical completed comparator trial, while the average is influenced by larger late-stage or multi-site studies.</div>', unsafe_allow_html=True)
        if "drug_name" in enroll_df.columns:
            with st.expander("Enrollment benchmark by comparator"):
                by_drug = (
                    enroll_df.groupby("drug_name")["_enrollment"]
                    .agg(completed_trials="count", median_enrollment="median", avg_enrollment="mean",
                         q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75))
                    .reset_index()
                )
                for c in ["median_enrollment", "avg_enrollment", "q1", "q3"]:
                    by_drug[c] = by_drug[c].round(0).astype(int)
                by_drug["typical_range_iqr"] = by_drug["q1"].astype(str) + "–" + by_drug["q3"].astype(str)
                st.dataframe(by_drug[["drug_name", "completed_trials", "median_enrollment", "typical_range_iqr", "avg_enrollment"]].sort_values("completed_trials", ascending=False), use_container_width=True, hide_index=True)

    # ── Demographic Benchmarks ───────────────────────────────────────────── #
    st.markdown("### Demographic Benchmarks")
    st.caption("Registry-level eligibility patterns across comparator trials. These reflect trial design constraints, not enrolled patient-level demographics.")

    demo_source = _demographic_metric_source(df_demo, df_locs)
    if demo_source.empty:
        st.info("Demographic benchmark data is not available in the current trial outputs.")
    else:
        demo = demo_source.copy()
        demo_unique = demo.drop_duplicates(subset=["nct_id"]).copy() if "nct_id" in demo.columns else demo.copy()
        total_trials = len(demo_unique)

        healthy_col = next((c for c in ["healthy_volunteers", "accepts_healthy_volunteers", "healthyVolunteers"] if c in demo_unique.columns), None)
        healthy_pct = None
        if healthy_col is not None and total_trials > 0:
            healthy_vals = demo_unique[healthy_col].astype(str).str.upper()
            healthy_pct = 100 * healthy_vals.str.contains("YES|TRUE", regex=True, na=False).sum() / total_trials

        min_age_col = next((c for c in ["minimum_age", "minimumAge", "min_age"] if c in demo_unique.columns), None)
        max_age_col = next((c for c in ["maximum_age", "maximumAge", "max_age"] if c in demo_unique.columns), None)
        min_ages = demo_unique[min_age_col].apply(_parse_age_years).dropna() if min_age_col else pd.Series(dtype=float)
        max_ages = demo_unique[max_age_col].apply(_parse_age_years).dropna() if max_age_col else pd.Series(dtype=float)

        age_range_txt = "—"
        if len(min_ages) > 0 or len(max_ages) > 0:
            min_txt = f"{min_ages.min():.0f}" if len(min_ages) > 0 else "—"
            max_txt = f"{max_ages.max():.0f}" if len(max_ages) > 0 else "—"
            age_range_txt = f"{min_txt}–{max_txt} yrs"

        adult_only_pct = None
        pediatric_pct = None
        median_min_age_txt = "—"
        if len(min_ages) > 0 and total_trials > 0:
            adult_only_pct = 100 * (min_ages >= 18).sum() / total_trials
            pediatric_pct = 100 * (min_ages < 18).sum() / total_trials
            median_min_age_txt = f"{min_ages.median():.0f} yrs"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Healthy volunteers allowed", f"{healthy_pct:.0f}%" if healthy_pct is not None else "—")
        c2.metric("Observed age range", age_range_txt)
        c3.metric("Adult-only trials", f"{adult_only_pct:.0f}%" if adult_only_pct is not None else "—")
        c4.metric("Median minimum age", median_min_age_txt)

        if pediatric_pct is not None and pediatric_pct > 0:
            st.caption(f"Approximately {pediatric_pct:.0f}% of comparator trials appear to allow participants under 18 based on minimum-age eligibility fields.")

        exclusion_col = next((c for c in ["eligibility_criteria", "eligibilityCriteria", "exclusion_criteria", "criteria", "eligibility"] if c in demo_unique.columns), None)
        exclusion_examples: list[dict[str, str]] = []
        if exclusion_col is not None:
            criteria_rows = (
                demo_unique[[col for col in ["drug_name", "nct_id", "brief_title", exclusion_col] if col in demo_unique.columns]]
                .dropna(subset=[exclusion_col]).drop_duplicates()
            )
            health_keywords = ["history", "cardiac", "cardiovascular", "heart", "qt", "liver", "hepatic", "renal", "kidney",
                               "infection", "immun", "pregnan", "brain metast", "cns metast", "interstitial lung",
                               "pneumonitis", "pulmonary", "prior treatment", "previous therapy", "ecog"]
            for _, r in criteria_rows.iterrows():
                txt = str(r.get(exclusion_col, "")).replace("\n", " ").strip()
                if not txt: continue
                lower_txt = txt.lower()
                if "exclusion criteria" in lower_txt:
                    txt_focus = txt[lower_txt.find("exclusion criteria"):]
                elif "exclusion" in lower_txt:
                    txt_focus = txt[lower_txt.find("exclusion"):]
                else:
                    txt_focus = txt
                fragments = []
                for sep in ["; ", ". ", " - ", "•"]:
                    if sep in txt_focus:
                        fragments = [x.strip() for x in txt_focus.split(sep)]
                        break
                if not fragments:
                    fragments = [txt_focus]
                for frag in fragments:
                    frag_clean = frag.strip(" -•:\t")
                    if len(frag_clean) < 30: continue
                    if any(k in frag_clean.lower() for k in health_keywords):
                        drug_txt = str(r.get("drug_name", "")).strip()
                        nct_txt = str(r.get("nct_id", "")).strip()
                        label = ""
                        if drug_txt and drug_txt.lower() not in {"nan", "none"}:
                            label += drug_txt
                        if nct_txt and nct_txt.lower() not in {"nan", "none"}:
                            label += f" ({nct_txt})" if label else nct_txt
                        exclusion_examples.append({
                            "source": label or "Comparator trial",
                            "criterion": frag_clean[:320] + ("..." if len(frag_clean) > 320 else ""),
                        })
                        break
                if len(exclusion_examples) >= 5:
                    break

        st.markdown(
            '<div style="margin-top:14px;background:#FFF8EE;border-left:4px solid #FFB347;padding:12px 14px;border-radius:0 8px 8px 0;">'
            '<div style="font-size:0.72rem;font-weight:900;color:#D4780A;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Health-history exclusion signals</div>'
            '<div style="font-size:0.86rem;color:#444;line-height:1.65;">Extracted examples of health-history, organ-function, prior-therapy, or comorbidity exclusions from comparator trial eligibility text.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if exclusion_examples:
            for ex in exclusion_examples:
                st.markdown(
                    f'<div style="background:#FFFFFF;border:1px solid #E6E6E6;border-radius:8px;padding:10px 12px;margin-top:8px;">'
                    f'<div style="font-size:0.74rem;color:#999;font-weight:800;margin-bottom:4px;">{html.escape(ex["source"])}</div>'
                    f'<div style="font-size:0.84rem;color:#333;line-height:1.55;">{html.escape(ex["criterion"])}</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No health-history exclusion examples were found in the available trial eligibility fields. This may mean the current trial export does not include full eligibility criteria text.")

        with st.expander("Trial demographic evidence"):
            if not df_demo_summary.empty:
                st.markdown("**Demographic summary output**")
                st.dataframe(df_demo_summary.reset_index(drop=True), use_container_width=True, hide_index=True)
            demo_cols = [c for c in ["drug_name", "nct_id", "brief_title", "sex", "gender", "minimum_age", "maximum_age", "healthy_volunteers", "enrollment_count", "overall_status", "eligibility_criteria", "criteria"] if c in demo_source.columns]
            if demo_cols:
                st.dataframe(demo_source[demo_cols].drop_duplicates().reset_index(drop=True), use_container_width=True, hide_index=True)
            else:
                st.dataframe(demo_source.reset_index(drop=True), use_container_width=True, hide_index=True)

    if not df_locs.empty:
        with st.expander("Trial-level evidence"):
            cols = [c for c in ["drug_name", "nct_id", "brief_title", "phase", "overall_status", "enrollment_count", "enrollment", "lead_sponsor", "lead_sponsor_name", "lead_sponsor_class", "country", "city", "state", "facility"] if c in df_locs.columns]
            if cols:
                st.dataframe(df_locs[cols].drop_duplicates().reset_index(drop=True), use_container_width=True, hide_index=True)
            else:
                st.dataframe(df_locs.reset_index(drop=True), use_container_width=True, hide_index=True)
