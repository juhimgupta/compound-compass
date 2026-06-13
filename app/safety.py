from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st

from chemistry import _mcs_scaffold_summary


_SEV_STYLES = {
    "high": {"bg": "#FDECEA", "fg": "#B71C1C", "bd": "#EF9A9A"},
    "medium": {"bg": "#FFF3E0", "fg": "#BF360C", "bd": "#FFCC80"},
    "low": {"bg": "#E8F5E9", "fg": "#1B5E20", "bd": "#A5D6A7"},
    "contextual": {"bg": "#F5F5F5", "fg": "#616161", "bd": "#BDBDBD"},
    "unknown": {"bg": "#F5F5F5", "fg": "#757575", "bd": "#BDBDBD"},
}
_SEV_RANK = {"high": 4, "medium": 3, "low": 2, "contextual": 1, "unknown": 0}


def _sev_badge(severity: str) -> str:
    s = str(severity).lower().strip() if severity else "unknown"
    sty = _SEV_STYLES.get(s, _SEV_STYLES["unknown"])
    return (
        f'<span style="background:{sty["bg"]}; color:{sty["fg"]}; border:1.5px solid {sty["bd"]}; '
        f'padding:5px 16px; border-radius:12px; font-size:0.78rem; font-weight:700; '
        f'white-space:nowrap; display:inline-block;">{s.capitalize()}</span>'
    )


def _worst_sev(events: list[str], severity_map: dict[str, str]) -> str | None:
    if not events:
        return None
    sevs = [severity_map.get(e.strip().upper(), "unknown") for e in events]
    return max(sevs, key=lambda s: _SEV_RANK.get(s, 0))


def _faers_key(row: pd.Series | dict[str, Any]) -> str:
    fn = row.get("faers_search_name")
    dn = row.get("drug_name")
    if pd.notna(fn) and str(fn).strip().lower() not in ("", "nan", "none"):
        return str(fn).strip().lower()
    return str(dn).strip().lower() if pd.notna(dn) else ""


def _spec_and_sev_maps(df_ec: pd.DataFrame, df_recs: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, str]]:
    spec_map: dict[str, list[str]] = {}
    if not df_ec.empty and "classification" in df_ec.columns:
        mol_rows = df_ec[df_ec["classification"].str.contains("molecule-specific", case=False, na=False)]
        for _, r in mol_rows.iterrows():
            drv = str(r.get("driver", "")).lower().strip()
            evt = str(r.get("event", "")).strip()
            if drv and drv != "nan" and evt:
                spec_map.setdefault(drv, []).append(evt)
    sev_map: dict[str, str] = {}
    if not df_recs.empty and "event" in df_recs.columns and "severity" in df_recs.columns:
        sev_map = dict(zip(df_recs["event"].astype(str).str.upper(), df_recs["severity"].astype(str).str.lower()))
    return spec_map, sev_map


def _format_signal_list(events: list[str], max_events: int = 4) -> str:
    clean = [str(e).strip().title() for e in events if str(e).strip()]
    if not clean:
        return ""
    shown = clean[:max_events]
    if len(clean) > max_events:
        return ", ".join(shown) + f", and {len(clean) - max_events} additional signal(s)"
    return ", ".join(shown)


def _build_scaffold_signal_recommendations(
    df_sim: pd.DataFrame,
    spec_map: dict[str, list[str]],
    sev_map: dict[str, str],
    cand: dict[str, Any],
    max_recs: int = 5,
) -> list[dict[str, Any]]:
    if df_sim.empty or not spec_map:
        return []
    candidate_smiles = cand.get("smiles", "")
    df = df_sim.copy()
    df["_tan"] = pd.to_numeric(
        df.get("tanimoto_similarity", pd.Series([0] * len(df))), errors="coerce"
    ).fillna(0)
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        fkey = _faers_key(r)
        events = spec_map.get(fkey, [])
        if not events:
            continue
        smi = r.get("canonical_smiles")
        if not smi or pd.isna(smi):
            continue
        mcs = _mcs_scaffold_summary(candidate_smiles, str(smi))
        rows.append({
            "drug_name": str(r.get("drug_name", "Comparator")),
            "drug_type": str(r.get("drug_type", "")),
            "stage": str(r.get("max_clinical_stage_for_target", "")),
            "tanimoto": float(r.get("_tan", 0)),
            "events": events,
            "severity": _worst_sev(events, sev_map) or "unknown",
            "mcs_atoms": mcs["mcs_atoms"],
            "mcs_bonds": mcs["mcs_bonds"],
            "mcs_smarts": mcs["mcs_smarts"],
        })
    if not rows:
        return []
    recs = pd.DataFrame(rows)
    recs["_sev_rank"] = recs["severity"].map(_SEV_RANK).fillna(0)
    recs = recs.sort_values(["_sev_rank", "tanimoto", "mcs_atoms"], ascending=[False, False, False])
    return recs.head(max_recs).to_dict("records")


def _render_scaffold_signal_recommendations(
    df_sim: pd.DataFrame,
    spec_map: dict[str, list[str]],
    sev_map: dict[str, str],
    cand: dict[str, Any],
) -> None:
    recs = _build_scaffold_signal_recommendations(df_sim, spec_map, sev_map, cand)
    st.markdown("### Candidate-Specific Structural Safety Recommendations")
    if not recs:
        st.info(
            "No scaffold-linked molecule-specific FAERS recommendations are available yet. "
            "Select FAERS comparators and rebuild the FAERS analysis to generate molecule-specific signals."
        )
        return
    st.caption(
        "These recommendations connect candidate structural similarity to comparator-specific FAERS signals. "
        "They are hypothesis-generating safety review recommendations, not causal toxicity claims."
    )
    for rec in recs:
        drug = html.escape(str(rec.get("drug_name", "Comparator")))
        tan = float(rec.get("tanimoto", 0))
        events_txt = html.escape(_format_signal_list(rec.get("events", [])))
        severity = str(rec.get("severity", "unknown"))
        mcs_atoms = int(rec.get("mcs_atoms", 0) or 0)
        mcs_bonds = int(rec.get("mcs_bonds", 0) or 0)
        stage = html.escape(str(rec.get("stage", "")))
        if mcs_atoms >= 6:
            scaffold_phrase = f"shares a {mcs_atoms}-atom / {mcs_bonds}-bond common scaffold with"
        elif tan >= 0.35:
            scaffold_phrase = "has measurable structural similarity to"
        else:
            scaffold_phrase = "has comparator-linked safety context from"
        stage_txt = f" · {stage}" if stage and stage.lower() not in {"nan", "none", ""} else ""
        st.markdown(
            f"""
<div class="sig-box" style="background:#FFF8EE; border-left:4px solid #FFB347;">
  <div class="sig-box-title" style="color:#D4780A;">Structural safety prompt</div>
  <div style="font-size:0.88rem;color:#333;line-height:1.75;">
    Your candidate <b>{scaffold_phrase} {drug}</b>
    <span style="color:#777;">(Tanimoto {tan:.3f}{stage_txt})</span>.
    <b>{drug}</b> has molecule-specific FAERS signals for <b>{events_txt}</b>,
    which may warrant targeted monitoring or follow-up review.
    <div style="margin-top:8px;">{_sev_badge(severity)}</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        if rec.get("mcs_smarts"):
            with st.expander(f"Shared scaffold SMARTS for {drug}"):
                st.code(rec["mcs_smarts"])
