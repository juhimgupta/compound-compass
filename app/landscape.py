from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import plotly.express as px
    import plotly.graph_objects as go
    _PLOTLY = True
except ImportError:
    _PLOTLY = False


def _phase_buckets(phase_value: Any) -> list[str]:
    s = str(phase_value or "").upper().replace("_", "").replace(" ", "")
    buckets: list[str] = []
    if "EARLYPHASE1" in s:
        buckets.append("Early Phase 1")
    if "PHASE1" in s:
        buckets.append("Phase 1")
    if "PHASE2" in s:
        buckets.append("Phase 2")
    if "PHASE3" in s:
        buckets.append("Phase 3")
    if "PHASE4" in s:
        buckets.append("Phase 4")
    if "NOTAPPLICABLE" in s:
        buckets.append("Not Applicable")
    return buckets or ["Unknown"]


def _ot_stage_bucket(stage_value: Any) -> str:
    s = str(stage_value or "").upper().strip()
    mapping = {
        "APPROVAL": "Approved", "APPROVED": "Approved",
        "PHASE_4": "Phase 4", "PHASE 4": "Phase 4",
        "PHASE_3": "Phase 3", "PHASE 3": "Phase 3",
        "PHASE_2_3": "Phase 2/3", "PHASE 2/3": "Phase 2/3",
        "PHASE_2": "Phase 2", "PHASE 2": "Phase 2",
        "PHASE_1_2": "Phase 1/2", "PHASE 1/2": "Phase 1/2",
        "PHASE_1": "Phase 1", "PHASE 1": "Phase 1",
        "EARLY_PHASE_1": "Early Phase 1", "EARLY PHASE 1": "Early Phase 1",
        "PRECLINICAL": "Preclinical", "DISCOVERY": "Discovery",
        "WITHDRAWN": "Withdrawn", "SUSPENDED": "Suspended", "TERMINATED": "Terminated",
    }
    return mapping.get(s, "Other / Unknown")


def _stage_rank_for_landscape(stage: str) -> int:
    order = [
        "Discovery", "Preclinical", "Early Phase 1", "Phase 1", "Phase 1/2",
        "Phase 2", "Phase 2/3", "Phase 3", "Phase 4", "Approved",
        "Withdrawn", "Suspended", "Terminated", "Other / Unknown",
    ]
    return order.index(stage) if stage in order else len(order)


def _safe_html_text(x: Any) -> str:
    return html.escape(str(x or "").strip())


def _prepare_ot_landscape(df_known: pd.DataFrame) -> pd.DataFrame:
    if df_known.empty:
        return pd.DataFrame()
    df = df_known.copy()
    if "drug_name" not in df.columns:
        return pd.DataFrame()
    stage_col = None
    for c in ["max_clinical_stage_for_target", "max_clinical_stage", "clinical_stage"]:
        if c in df.columns:
            stage_col = c
            break
    df["stage_bucket"] = df[stage_col].apply(_ot_stage_bucket) if stage_col else "Other / Unknown"
    df = df[df["drug_name"].astype(str).str.strip().ne("")]
    df = df[df["drug_name"].astype(str).str.lower().ne("nan")]
    sort_cols, ascending = [], []
    if "n_clinical_reports" in df.columns:
        sort_cols.append("n_clinical_reports"); ascending.append(False)
    if "overall_score" in df.columns:
        sort_cols.append("overall_score"); ascending.append(False)
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending)
    df = df.drop_duplicates(subset=["drug_name"], keep="first").copy()
    df["_stage_rank"] = df["stage_bucket"].apply(_stage_rank_for_landscape)
    sort_cols = ["_stage_rank"]; ascending = [True]
    if "n_clinical_reports" in df.columns:
        sort_cols.append("n_clinical_reports"); ascending.append(False)
    elif "overall_score" in df.columns:
        sort_cols.append("overall_score"); ascending.append(False)
    else:
        sort_cols.append("drug_name"); ascending.append(True)
    return df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def _drug_type_bucket(drug_type: Any) -> str:
    s = str(drug_type or "").lower().strip()
    if not s or s in {"nan", "none", "null"}:
        return "Other / unknown"
    if "small" in s and "molecule" in s:
        return "Small molecule"
    if "antibody" in s and ("drug conjugate" in s or "adc" in s):
        return "Antibody drug conjugate"
    if "antibody" in s:
        return "Antibody"
    if "protein" in s or "enzyme" in s or "peptide" in s or "biologic" in s:
        return "Protein / biologic"
    return "Other / unknown"


def _drug_type_style(bucket: str) -> dict[str, str]:
    styles = {
        "Small molecule": {"bg": "#FFA500", "bd": "#D88400", "fg": "#FFFFFF"},
        "Antibody": {"bg": "#1976D2", "bd": "#0D47A1", "fg": "#FFFFFF"},
        "Antibody drug conjugate": {"bg": "#7B1FA2", "bd": "#4A148C", "fg": "#FFFFFF"},
        "Protein / biologic": {"bg": "#2E7D32", "bd": "#1B5E20", "fg": "#FFFFFF"},
        "Other / unknown": {"bg": "#9E9E9E", "bd": "#757575", "fg": "#FFFFFF"},
    }
    return styles.get(bucket, styles["Other / unknown"])


def _render_ot_development_landscape(df_known: pd.DataFrame, target: str) -> None:
    df = _prepare_ot_landscape(df_known)
    if df.empty:
        st.info("No Open Targets drug-stage data available for the development landscape.")
        return
    stage_order = [
        "Discovery", "Preclinical", "Early Phase 1", "Phase 1", "Phase 1/2",
        "Phase 2", "Phase 2/3", "Phase 3", "Phase 4", "Approved",
        "Withdrawn", "Suspended", "Terminated", "Other / Unknown",
    ]
    stage_counts = df["stage_bucket"].astype(str).value_counts().to_dict()
    max_stack = max(stage_counts.values()) if stage_counts else 1
    total_drugs = len(df)
    df["_modality_bucket"] = df.get("drug_type", pd.Series([""] * len(df))).apply(_drug_type_bucket)
    observed_buckets = [
        b for b in ["Small molecule", "Antibody", "Antibody drug conjugate", "Protein / biologic", "Other / unknown"]
        if b in set(df["_modality_bucket"])
    ]
    legend_html = ""
    for bucket in observed_buckets:
        sty = _drug_type_style(bucket)
        legend_html += (
            f'<span class="cc-legend-item">'
            f'<span class="cc-legend-dot" style="background:{sty["bg"]};border-color:{sty["bd"]};"></span>'
            f'{_safe_html_text(bucket)}</span>\n'
        )
    lanes_html = ""
    for stage in stage_order:
        stage_df = df[df["stage_bucket"].astype(str) == stage].copy()
        pills_html = ""
        for _, r in stage_df.iterrows():
            drug = _safe_html_text(r.get("drug_name", "Unknown drug"))
            raw_type = _safe_html_text(r.get("drug_type", ""))
            bucket = _drug_type_bucket(r.get("drug_type", ""))
            sty = _drug_type_style(bucket)
            moa = _safe_html_text(r.get("mechanism_of_action", ""))
            diseases = _safe_html_text(r.get("disease_names", ""))
            raw_stage = _safe_html_text(r.get("max_clinical_stage_for_target", ""))
            title_parts = [f"Drug: {drug}", f"Stage: {stage}", f"Type: {raw_type or bucket}"]
            if raw_stage and raw_stage.lower() != "nan":
                title_parts.append(f"Open Targets stage value: {raw_stage}")
            if moa and moa.lower() != "nan":
                title_parts.append(f"Mechanism: {moa}")
            if diseases and diseases.lower() != "nan":
                title_parts.append(f"Diseases: {diseases}")
            title = _safe_html_text(" | ".join(title_parts))
            pills_html += (
                f'<div class="cc-pill" title="{title}" '
                f'style="background:{sty["bg"]};border-color:{sty["bd"]};color:{sty["fg"]};">'
                f'{drug}</div>\n'
            )
        if not pills_html:
            pills_html = '<div class="cc-empty-lane">white space</div>'
        lanes_html += (
            f'<div class="cc-stage-lane">'
            f'<div class="cc-stage-head">'
            f'<div class="cc-stage-title">{_safe_html_text(stage)}</div>'
            f'<div class="cc-stage-count">{stage_counts.get(stage, 0)}</div>'
            f'</div><div class="cc-pill-wrap">{pills_html}</div></div>\n'
        )
    iframe_height = min(760, max(390, 185 + max_stack * 28))
    min_width = len(stage_order) * 138
    landscape_html = f"""<!DOCTYPE html>
<html><head><style>
  body {{ margin: 0; padding: 0; background: #FFFFFF; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #222; }}
  .cc-wrap {{ background: #FFFFFF; border: 1px solid #E0E0E0; border-radius: 10px; padding: 14px; box-sizing: border-box; width: 100%; }}
  .cc-eyebrow {{ font-size: 0.68rem; color: #999; font-weight: 850; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }}
  .cc-title-row {{ display: flex; gap: 12px; align-items: baseline; justify-content: space-between; flex-wrap: wrap; margin-bottom: 5px; }}
  .cc-title {{ font-size: 0.98rem; color: #222; font-weight: 850; }}
  .cc-total {{ font-size: 0.76rem; color: #777; font-weight: 750; }}
  .cc-caption {{ font-size: 0.78rem; color: #666; line-height: 1.45; margin-bottom: 10px; }}
  .cc-legend {{ display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: center; margin-bottom: 12px; font-size: 0.72rem; color: #666; font-weight: 700; }}
  .cc-legend-item {{ display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; }}
  .cc-legend-dot {{ width: 10px; height: 10px; border-radius: 999px; border: 1px solid transparent; display: inline-block; }}
  .cc-landscape-scroll {{ overflow-x: auto; padding-bottom: 6px; }}
  .cc-landscape {{ min-width: {min_width}px; display: grid; grid-template-columns: repeat({len(stage_order)}, minmax(128px, 1fr)); align-items: stretch; border-top: 1px solid #E5E5E5; border-bottom: 1px solid #E5E5E5; background: #FFFFFF; }}
  .cc-stage-lane {{ border-left: 1px solid #DCDCDC; min-height: {max(185, 78 + max_stack * 27)}px; padding: 8px 7px 10px 7px; box-sizing: border-box; background: linear-gradient(180deg, #FAFAFA 0%, #FFFFFF 38%); }}
  .cc-stage-lane:last-child {{ border-right: 1px solid #DCDCDC; }}
  .cc-stage-head {{ min-height: 38px; text-align: center; margin-bottom: 6px; }}
  .cc-stage-title {{ color: #555; font-size: 0.72rem; font-weight: 900; line-height: 1.15; }}
  .cc-stage-count {{ color: #999; font-size: 0.66rem; font-weight: 800; margin-top: 2px; }}
  .cc-pill-wrap {{ display: flex; flex-wrap: wrap; align-content: flex-start; gap: 5px; }}
  .cc-pill {{ border: 1px solid transparent; border-radius: 999px; padding: 4px 8px; font-size: 0.64rem; line-height: 1.1; font-weight: 850; max-width: 112px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; box-shadow: 0 1px 2px rgba(0,0,0,0.08); box-sizing: border-box; }}
  .cc-empty-lane {{ color: #B8B8B8; border: 1px dashed #D4D4D4; border-radius: 999px; padding: 4px 8px; font-size: 0.62rem; font-style: italic; text-align: center; width: fit-content; margin: 2px auto 0 auto; }}
  .cc-footnote {{ margin-top: 10px; font-size: 0.72rem; color: #777; line-height: 1.45; }}
</style></head><body>
  <div class="cc-wrap">
    <div class="cc-eyebrow">Open Targets Development Landscape</div>
    <div class="cc-title-row">
      <div class="cc-title">{_safe_html_text(target)} target-associated drug landscape</div>
      <div class="cc-total">{total_drugs} Open Targets drugs shown</div>
    </div>
    <div class="cc-caption">Each pill is one target-associated therapeutic, placed once by maximum clinical stage for this target. Vertical lanes separate phases; sparse lanes suggest potential white space.</div>
    <div class="cc-legend">{legend_html}</div>
    <div class="cc-landscape-scroll"><div class="cc-landscape">{lanes_html}</div></div>
    <div class="cc-footnote">This landscape uses the broad Open Targets drug/candidate set. Candidate Analysis may show fewer compounds because structural similarity requires usable SMILES and applies display filters.</div>
  </div>
</body></html>"""
    components.html(landscape_html, height=iframe_height, scrolling=True)
    with st.expander("Open Targets development landscape table"):
        show_cols = [c for c in [
            "drug_name", "stage_bucket", "drug_type", "max_clinical_stage_for_target",
            "drug_maximum_clinical_stage", "mechanism_of_action", "disease_names",
            "canonical_smiles", "structure_lookup_status",
        ] if c in df.columns]
        st.dataframe(df[show_cols].reset_index(drop=True), use_container_width=True, hide_index=True)


def _trial_landscape_table(df_locs: pd.DataFrame) -> pd.DataFrame:
    if df_locs.empty or "drug_name" not in df_locs.columns or "nct_id" not in df_locs.columns:
        return pd.DataFrame()
    cols = [c for c in ["drug_name", "nct_id", "phase", "overall_status", "lead_sponsor", "lead_sponsor_name", "lead_sponsor_class"] if c in df_locs.columns]
    df = df_locs[cols].drop_duplicates().copy()
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        for phase in _phase_buckets(r.get("phase", "")):
            rows.append({
                "drug_name": r.get("drug_name", ""),
                "nct_id": r.get("nct_id", ""),
                "phase_bucket": phase,
                "overall_status": r.get("overall_status", ""),
                "lead_sponsor": r.get("lead_sponsor", r.get("lead_sponsor_name", "")),
                "lead_sponsor_class": r.get("lead_sponsor_class", ""),
            })
    if not rows:
        return pd.DataFrame()
    expanded = pd.DataFrame(rows)
    return expanded.groupby(["drug_name", "phase_bucket"], as_index=False).agg(n_trials=("nct_id", "nunique"))


def _render_trial_phase_landscape(df_locs: pd.DataFrame, target: str) -> None:
    if not _PLOTLY:
        st.info("Install plotly to view the clinical trial activity landscape.")
        return
    phase_order = ["Early Phase 1", "Phase 1", "Phase 2", "Phase 3", "Phase 4", "Not Applicable", "Unknown"]
    phase_df = _trial_landscape_table(df_locs)
    if phase_df.empty:
        st.info("No phase-level clinical trial activity data available.")
        return
    phase_df = phase_df[phase_df["n_trials"] > 0].copy()
    drugs = phase_df.groupby("drug_name")["n_trials"].sum().sort_values(ascending=True).index.tolist()
    if not drugs:
        st.info("No trials found for the selected comparator drugs.")
        return
    phase_to_x = {p: i for i, p in enumerate(phase_order)}
    drug_to_y = {d: i for i, d in enumerate(drugs)}
    fig = go.Figure()
    for phase, x in phase_to_x.items():
        fig.add_shape(type="rect", x0=x - 0.48, x1=x + 0.48, y0=-0.6, y1=len(drugs) - 0.4,
                      fillcolor="#EFEFEF", line=dict(color="#DDDDDD", width=1), layer="below")
        fig.add_annotation(x=x, y=len(drugs) - 0.15, text=f"<b>{phase}</b>", showarrow=False,
                           font=dict(size=12, color="#666"), yanchor="bottom")
    for _, r in phase_df.iterrows():
        drug = str(r["drug_name"]); phase = str(r["phase_bucket"]); n = int(r["n_trials"])
        if drug not in drug_to_y or phase not in phase_to_x:
            continue
        x = phase_to_x[phase]; y = drug_to_y[drug]
        fig.add_shape(type="rect", x0=x - 0.36, x1=x + 0.36, y0=y - 0.28, y1=y + 0.28,
                      fillcolor="#FFA500", line=dict(color="#D4780A", width=1.5), layer="above")
        fig.add_annotation(x=x, y=y, text=f"{n}", showarrow=False, font=dict(size=12, color="white"))
    fig.update_xaxes(tickmode="array", tickvals=list(phase_to_x.values()), ticktext=phase_order, showgrid=False, zeroline=False)
    fig.update_yaxes(tickmode="array", tickvals=list(drug_to_y.values()), ticktext=drugs, showgrid=False, zeroline=False)
    fig.update_layout(
        title=f"Clinical trial activity landscape — selected {target} comparator drugs",
        height=max(360, 105 + 50 * len(drugs)),
        margin=dict(l=130, r=30, t=70, b=45),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#F4F4F4", showlegend=False,
        xaxis=dict(range=[-0.6, len(phase_order) - 0.4]),
        yaxis=dict(range=[-0.6, len(drugs) - 0.1]),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("This optional chart is trial-count based, so one drug can appear in multiple phase buckets. Numbers represent unique registered ClinicalTrials.gov studies.")


def _render_sponsor_landscape(df_locs: pd.DataFrame) -> None:
    if not _PLOTLY:
        return
    sponsor_col = next((c for c in ["lead_sponsor", "lead_sponsor_name"] if c in df_locs.columns), None)
    if df_locs.empty or sponsor_col is None:
        st.info("Sponsor data is not available yet. Rebuild the clinical trial footprint after updating clinical_trial_geography.py.")
        return
    df = df_locs.drop_duplicates(subset=["nct_id"]).copy()
    df = df[df[sponsor_col].astype(str).str.strip().ne("")]
    if df.empty:
        st.info("No lead sponsor information found in the trial records.")
        return
    sponsor_counts = (
        df.groupby(sponsor_col)["nct_id"].nunique().sort_values(ascending=False)
        .head(12).reset_index(name="n_trials")
    )
    fig = px.bar(
        sponsor_counts.sort_values("n_trials", ascending=True),
        x="n_trials", y=sponsor_col, orientation="h",
        title="Top lead sponsors by registered trial count",
        labels={"n_trials": "Unique trials", sponsor_col: "Lead sponsor"},
    )
    fig.update_traces(marker_color="#FFA500")
    fig.update_layout(height=max(320, 45 * len(sponsor_counts)), margin=dict(l=175, r=30, t=55, b=35),
                      plot_bgcolor="#FFFFFF", paper_bgcolor="#F4F4F4", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    if "lead_sponsor_class" in df.columns:
        cls = (
            df[df["lead_sponsor_class"].astype(str).str.strip().ne("")]
            .groupby("lead_sponsor_class")["nct_id"].nunique().sort_values(ascending=False)
            .reset_index(name="n_trials")
        )
        if not cls.empty:
            fig2 = px.bar(cls, x="lead_sponsor_class", y="n_trials",
                          title="Trial activity by sponsor class",
                          labels={"lead_sponsor_class": "Sponsor class", "n_trials": "Unique trials"})
            fig2.update_traces(marker_color="#FFA500")
            fig2.update_layout(height=320, margin=dict(l=40, r=30, t=55, b=50),
                               plot_bgcolor="#FFFFFF", paper_bgcolor="#F4F4F4", showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
