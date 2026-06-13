from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from utils import _csv, _json_file
from landscape import _render_ot_development_landscape


def render_target_tab(files: dict, target: str) -> None:
    summary = _json_file(files["target_summary"])
    df_diseases = _csv(files["associated_diseases"])
    df_known_landscape = _csv(files["known_drugs"])

    target_id = summary.get("target_id", "")
    target_symbol = summary.get("target_symbol", target)
    target_name = summary.get("target_name", "")
    biotype = summary.get("biotype", "")
    n_diseases_total = summary.get("n_associated_diseases_total", "—")
    n_drugs_returned = summary.get("n_drug_candidates_returned", "—")
    n_drugs_total = summary.get("n_drug_candidates_total", "—")
    n_safety = summary.get("n_safety_liabilities", "—")

    ot_target_url = (
        f"https://platform.opentargets.org/target/{target_id}"
        if target_id else "https://platform.opentargets.org/"
    )

    function_descriptions = summary.get("function_descriptions", []) or []
    target_description = (
        str(function_descriptions[0]) if function_descriptions
        else f"{target_symbol} is a target-associated gene/protein with Open Targets evidence available for disease and drug-candidate context."
    )
    if len(target_description) > 475:
        target_description = target_description[:475].rsplit(" ", 1)[0] + "..."

    disease_html = ""
    if not df_diseases.empty:
        chips = []
        for _, r in df_diseases.head(3).iterrows():
            disease_name = html.escape(str(r.get("disease_name", "Associated disease")))
            disease_id = str(r.get("disease_id", "")).strip()
            score = r.get("overall_score", None)
            score_txt = ""
            if pd.notna(score):
                try:
                    score_txt = f" · score {float(score):.2f}"
                except Exception:
                    pass
            if disease_id and disease_id.lower() != "nan":
                disease_url = f"https://platform.opentargets.org/disease/{disease_id}"
                chips.append(
                    f'<a href="{disease_url}" target="_blank" '
                    f'style="display:inline-block;text-decoration:none;background:#FFF8EE;border:1px solid #FFCC80;'
                    f'color:#333;border-radius:999px;padding:7px 12px;margin:4px 6px 4px 0;font-size:0.82rem;font-weight:700;">'
                    f'{disease_name}<span style="color:#999;font-weight:600;">{score_txt}</span></a>'
                )
            else:
                chips.append(
                    f'<span style="display:inline-block;background:#FFF8EE;border:1px solid #FFCC80;'
                    f'color:#333;border-radius:999px;padding:7px 12px;margin:4px 6px 4px 0;font-size:0.82rem;font-weight:700;">'
                    f'{disease_name}<span style="color:#999;font-weight:600;">{score_txt}</span></span>'
                )
        disease_html = "".join(chips)
    else:
        disease_html = '<span style="color:#999;font-size:0.84rem;">No associated disease file available.</span>'

    st.markdown('<div class="card" style="padding:22px 24px;margin-bottom:16px;">', unsafe_allow_html=True)
    left_col, right_col = st.columns([1.45, 1])

    with left_col:
        st.markdown(
            f"""
<div style="font-size:0.72rem;font-weight:900;color:#999;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Target Summary</div>
<div style="font-size:1.75rem;font-weight:900;color:#222;line-height:1.15;margin-bottom:6px;">
  {html.escape(str(target_symbol))} — {html.escape(str(target_name))}
</div>
<div style="font-size:0.88rem;color:#666;line-height:1.55;margin-bottom:12px;">
  <b>Target ID:</b> {html.escape(str(target_id))} &nbsp;·&nbsp; <b>Biotype:</b> {html.escape(str(biotype))}
</div>
<a href="{ot_target_url}" target="_blank" style="display:inline-block;text-decoration:none;background:#333333;color:white;border-radius:8px;padding:8px 12px;font-size:0.82rem;font-weight:800;">
  View target in Open Targets
</a>
""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
<div style="margin-top:22px;max-width:760px;background:#FFF8EE;border-left:4px solid #FFB347;border-radius:0 8px 8px 0;padding:12px 16px;">
  <div style="font-size:0.72rem;font-weight:900;color:#D4780A;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:5px;">Open Targets Description</div>
  <div style="font-size:0.86rem;color:#444;line-height:1.65;">
    {html.escape(target_description)}
    <a href="{ot_target_url}" target="_blank" style="color:#D4780A;font-weight:800;text-decoration:none;">[Open Targets]</a>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with right_col:
        st.markdown(
            f'<div style="font-size:0.72rem;font-weight:900;color:#999;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">Top Associated Diseases</div>'
            f'<div style="margin-bottom:16px;">{disease_html}</div>',
            unsafe_allow_html=True,
        )
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(
                f'<div style="background:#FAFAFA;border:1px solid #E6E6E6;border-radius:8px;padding:12px;margin-bottom:10px;">'
                f'<div style="font-size:0.68rem;color:#999;font-weight:900;text-transform:uppercase;letter-spacing:0.06em;">Known drug candidates</div>'
                f'<div style="font-size:1.35rem;font-weight:900;color:#222;margin-top:4px;">{n_drugs_total}</div>'
                f'<div style="font-size:0.74rem;color:#888;">{n_drugs_returned} retrieved for analysis</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="background:#FAFAFA;border:1px solid #E6E6E6;border-radius:8px;padding:12px;margin-bottom:10px;">'
                f'<div style="font-size:0.68rem;color:#999;font-weight:900;text-transform:uppercase;letter-spacing:0.06em;">Safety liabilities</div>'
                f'<div style="font-size:1.35rem;font-weight:900;color:#222;margin-top:4px;">{n_safety}</div>'
                f'<div style="font-size:0.74rem;color:#888;">Target-level safety context</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div style="background:#FAFAFA;border:1px solid #E6E6E6;border-radius:8px;padding:12px;margin-bottom:10px;">'
                f'<div style="font-size:0.68rem;color:#999;font-weight:900;text-transform:uppercase;letter-spacing:0.06em;">Associated diseases</div>'
                f'<div style="font-size:1.35rem;font-weight:900;color:#222;margin-top:4px;">{n_diseases_total}</div>'
                f'<div style="font-size:0.74rem;color:#888;">Open Targets associations</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div style="background:#FAFAFA;border:1px solid #E6E6E6;border-radius:8px;padding:12px;margin-bottom:10px;">'
                '<div style="font-size:0.68rem;color:#999;font-weight:900;text-transform:uppercase;letter-spacing:0.06em;">Data source</div>'
                '<div style="font-size:1.05rem;font-weight:900;color:#222;margin-top:7px;">Open Targets</div>'
                '<div style="font-size:0.74rem;color:#888;">Target context database</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("### Development Landscape")

    if df_known_landscape.empty:
        st.info("No Open Targets known-drug file found yet. Rebuild target context or rerun candidate analysis.")
    else:
        _render_ot_development_landscape(df_known_landscape, target)

    with st.expander("Full target summary JSON"):
        st.json(summary)

    if not df_diseases.empty:
        with st.expander("All associated diseases"):
            show_cols = [c for c in ["disease_name", "disease_id", "overall_score", "therapeutic_areas"] if c in df_diseases.columns]
            st.dataframe(df_diseases[show_cols].reset_index(drop=True), use_container_width=True, hide_index=True)
