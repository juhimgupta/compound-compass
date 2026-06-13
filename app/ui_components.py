from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from config import APP_TITLE, APP_SUBTITLE, LOGO_PATH
from chemistry import _smiles_to_bytes, _admet_props, _admet_html, render_3d_comparison
from safety import _faers_key, _sev_badge


def _image_to_base64(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return base64.b64encode(path.read_bytes()).decode()
    except Exception:
        return None


def _render_header(
    target: str | None = None,
    indication: str | None = None,
    comparator_label: str | None = None,
    centered: bool = False,
) -> None:
    logo_b64 = _image_to_base64(LOGO_PATH)
    logo_size = 112 if centered else 58
    title_size = "3.2rem" if centered else "2rem"
    subtitle_size = "1.05rem" if centered else "0.86rem"
    if logo_b64:
        logo_html = (
            f'<img src="data:image/png;base64,{logo_b64}" '
            f'style="width:{logo_size}px;height:{logo_size}px;object-fit:contain;border-radius:18px;" />'
        )
    else:
        logo_html = (
            f'<div style="width:{logo_size}px;height:{logo_size}px;border-radius:18px;'
            f'background:linear-gradient(135deg,#FFA500,#FFCC80);display:flex;align-items:center;'
            f'justify-content:center;font-weight:900;color:white;font-size:{2.2 if centered else 1}rem;">CC</div>'
        )
    cfg_line = ""
    if target:
        cfg_line = (
            f'<div style="font-size:0.86rem;color:#888;margin-top:10px;">'
            f'Target: <b>{target}</b>'
            + (f' · Therapeutic context: <b>{indication}</b>' if indication else '')
            + '</div>'
        )
    if centered:
        st.markdown(
            f"""
<div style="text-align:center;margin-top:10px;margin-bottom:22px;">
  <div style="display:flex;justify-content:center;margin-bottom:14px;">{logo_html}</div>
  <div style="font-size:{title_size};font-weight:900;color:#222;line-height:1.05;">{APP_TITLE}</div>
  <div style="font-size:{subtitle_size};color:#666;margin-top:10px;">{APP_SUBTITLE}</div>
  {cfg_line}
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:4px;">
  {logo_html}
  <div>
    <div style="font-size:{title_size};font-weight:800;color:#222;line-height:1.1;">{APP_TITLE}</div>
    <div style="font-size:{subtitle_size};color:#777;margin-top:3px;">{APP_SUBTITLE}</div>
    {cfg_line}
  </div>
</div>
""",
            unsafe_allow_html=True,
        )


def _render_workflow(target: str) -> None:
    st.markdown(f"""
<div style="background:#FFFFFF;border:1px solid #E0E0E0;border-radius:10px;padding:16px 18px;margin:12px 0 14px 0;">
  <div style="font-size:0.72rem;font-weight:800;color:#999;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px;">Candidate triage workflow</div>
  <div style="display:flex;align-items:stretch;gap:8px;flex-wrap:wrap;">
    <div style="flex:1;min-width:145px;background:#FFF8E1;border-left:4px solid #FFA500;border-radius:6px;padding:10px;"><div style="font-weight:800;color:#222;">1. Candidate</div><div style="font-size:0.77rem;color:#666;">SMILES + {target} context</div></div>
    <div style="align-self:center;color:#AAA;font-weight:900;">→</div>
    <div style="flex:1;min-width:145px;background:#F5F5F5;border-left:4px solid #999;border-radius:6px;padding:10px;"><div style="font-weight:800;color:#222;">2. Comparators</div><div style="font-size:0.77rem;color:#666;">Open Targets / ChEMBL molecules</div></div>
    <div style="align-self:center;color:#AAA;font-weight:900;">→</div>
    <div style="flex:1;min-width:145px;background:#FFF3E0;border-left:4px solid #FFB347;border-radius:6px;padding:10px;"><div style="font-weight:800;color:#222;">3. FAERS set</div><div style="font-size:0.77rem;color:#666;">User-selected ≤5 drugs</div></div>
    <div style="align-self:center;color:#AAA;font-weight:900;">→</div>
    <div style="flex:1;min-width:145px;background:#E8F5E9;border-left:4px solid #2E7D32;border-radius:6px;padding:10px;"><div style="font-weight:800;color:#222;">4. Recommendation</div><div style="font-size:0.77rem;color:#666;">Safety review priorities</div></div>
  </div>
</div>
""", unsafe_allow_html=True)


@st.dialog("Compound Detail", width="large")
def _show_compound_dialog(row_dict: dict[str, Any], cand: dict[str, Any], spec_map: dict[str, list[str]], sev_map: dict[str, str]) -> None:
    drug_name = str(row_dict.get("drug_name", ""))
    comp_smi = row_dict.get("canonical_smiles")
    tan = row_dict.get("tanimoto_similarity")
    spec_evts = spec_map.get(_faers_key(row_dict), [])

    h1, h2 = st.columns([4, 1])
    with h1:
        st.markdown(f"### {drug_name}")
    with h2:
        if pd.notna(tan):
            st.metric("Tanimoto", f"{float(tan):.3f}")

    tab_2d, tab_3d = st.tabs(["Structure & ADMET", "3D Structure Comparison"])
    with tab_2d:
        c_col, m_col = st.columns(2)
        with c_col:
            st.markdown(f"**{cand['name']}** *(Candidate)*")
            b = _smiles_to_bytes(cand["smiles"], size=(420, 330))
            if b:
                st.image(b, use_container_width=True)
            st.markdown(_admet_html(_admet_props(cand["smiles"]), compact=True), unsafe_allow_html=True)
        with m_col:
            st.markdown(f"**{drug_name}**")
            b = _smiles_to_bytes(str(comp_smi) if comp_smi and pd.notna(comp_smi) else None, size=(420, 330))
            if b:
                st.image(b, use_container_width=True)
            else:
                st.info("No structure available.")
            st.markdown(_admet_html(_admet_props(str(comp_smi)) if comp_smi and pd.notna(comp_smi) else {}, compact=True), unsafe_allow_html=True)
        if spec_evts:
            st.markdown("**Compound-Specific FAERS Signals:**")
            for evt in spec_evts:
                sev_s = sev_map.get(evt.upper(), "unknown")
                st.markdown(f"• {evt.title()} &nbsp; {_sev_badge(sev_s)}", unsafe_allow_html=True)
        else:
            st.caption("No molecule-specific FAERS signal is available for this compound in the current FAERS comparator set.")
    with tab_3d:
        if comp_smi and pd.notna(comp_smi):
            render_3d_comparison(cand["smiles"], str(comp_smi), cand["name"], drug_name)
        else:
            st.warning("No SMILES available for this compound.")
