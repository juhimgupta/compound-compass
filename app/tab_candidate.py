from __future__ import annotations

import base64
import html
from typing import Any

import pandas as pd
import streamlit as st

from config import PAGE_SIZE, _MIN_STAGE_MAP
from utils import _csv, _has_value, _stage_score
from chemistry import _smiles_to_bytes, _admet_props, _admet_html
from safety import _spec_and_sev_maps, _faers_key, _sev_badge, _worst_sev
from ui_components import _show_compound_dialog

try:
    from rdkit import Chem
    _RDKIT = True
except ImportError:
    _RDKIT = False


def _candidate_modality_from_smiles(smiles: str) -> str:
    if not smiles:
        return "Unknown"
    if _RDKIT:
        try:
            mol = Chem.MolFromSmiles(str(smiles).strip())
            if mol is not None:
                return "Small molecule"
        except Exception:
            pass
    return "Unknown"


def _percent_higher_than(value: Any, reference_values: list[Any]) -> float | None:
    try:
        v = float(value)
    except Exception:
        return None
    refs = []
    for x in reference_values:
        try:
            if pd.notna(x):
                refs.append(float(x))
        except Exception:
            pass
    if not refs:
        return None
    return round(100 * sum(x < v for x in refs) / len(refs), 0)


def _format_candidate_metric(value: Any, key: str) -> str:
    try:
        v = float(value)
        return f"{v:.2f}" if key in {"LogP", "QED"} else f"{v:.1f}"
    except Exception:
        return str(value)


def _candidate_admet_context_verbose(cand_admet: dict[str, Any], df_sim: pd.DataFrame, candidate_smiles: str) -> str:
    if df_sim.empty or "canonical_smiles" not in df_sim.columns:
        return ""
    candidate_modality = _candidate_modality_from_smiles(candidate_smiles)
    df_ref = df_sim.copy()
    if candidate_modality == "Small molecule" and "drug_type" in df_ref.columns:
        df_ref = df_ref[df_ref["drug_type"].astype(str).str.contains("small molecule", case=False, na=False)].copy()
    comparator_props = []
    for smi in df_ref["canonical_smiles"].dropna().astype(str):
        props = _admet_props(smi)
        if props and props.get("MW") is not None:
            comparator_props.append(props)
    if not comparator_props:
        return ""
    metrics = [("molecular weight", "MW"), ("calculated LogP", "LogP"), ("QED drug-likeness", "QED")]
    phrases = []
    for label, key in metrics:
        cand_val = cand_admet.get(key)
        ref_vals = [p.get(key) for p in comparator_props if p.get(key) is not None]
        pct_higher = _percent_higher_than(cand_val, ref_vals)
        if pct_higher is None or pd.isna(cand_val):
            continue
        value_txt = _format_candidate_metric(cand_val, key)
        phrases.append(f"its <b>{label}</b> is <b>{value_txt}</b>, which is in the <b>{int(pct_higher)}%</b> percentile")
    if not phrases:
        return ""
    return (
        f"The submitted structure is treated as a <b>{candidate_modality.lower()}</b> candidate. "
        f"Compared with <b>{len(comparator_props)}</b> small-molecule target-associated compounds with usable structures, "
        + "; ".join(phrases) + "."
    )


def _traceability_link(label: str, value: Any, url: str | None = None) -> str:
    if not _has_value(value):
        return ""
    value_txt = html.escape(str(value).strip())
    label_txt = html.escape(label)
    if url:
        return f'<a href="{url}" target="_blank" style="text-decoration:none;color:#D4780A;font-weight:800;">{label_txt}: {value_txt}</a>'
    return f'<span style="color:#777;font-weight:800;">{label_txt}: {value_txt}</span>'


def _chembl_url(chembl_id: Any) -> str | None:
    if not _has_value(chembl_id):
        return None
    cid = str(chembl_id).strip()
    if not cid.startswith("CHEMBL"):
        return None
    return f"https://www.ebi.ac.uk/chembl/compound_report_card/{cid}/"


def _open_targets_drug_url(drug_id: Any) -> str | None:
    if not _has_value(drug_id):
        return None
    return f"https://platform.opentargets.org/drug/{str(drug_id).strip()}"


def _matches_selected_modality(drug_type: Any, selected: list[str]) -> bool:
    if not selected:
        return True
    s = str(drug_type or "").lower()
    if "Antibody drug conjugate" in selected and ("antibody drug conjugate" in s or "adc" in s):
        return True
    if "Small molecule" in selected and "small molecule" in s:
        return True
    if "Antibody" in selected and "antibody" in s and "drug conjugate" not in s:
        return True
    if "Protein" in selected and ("protein" in s or "biologic" in s or "enzyme" in s):
        return True
    if "Other" in selected:
        known = "small molecule" in s or "antibody" in s or "drug conjugate" in s or "adc" in s or "protein" in s or "biologic" in s or "enzyme" in s
        return not known
    return False


def render_candidate_tab(files: dict, cand: dict[str, Any]) -> None:
    df_sim = _csv(files["similarity"])
    df_ec = _csv(files["event_class"])
    df_recs = _csv(files["recommendations"])
    spec_map, sev_map = _spec_and_sev_maps(df_ec, df_recs)

    if df_sim.empty:
        st.warning("Similarity data not found. Run the pipeline first.")
        st.stop()

    target = cand["target"]
    mol_col, summary_col = st.columns([1.05, 1.95])
    cand_admet = _admet_props(cand["smiles"])
    cand_admet_html = _admet_html(cand_admet, compact=False)

    top = (
        df_sim.assign(_tanimoto_sort=pd.to_numeric(df_sim["tanimoto_similarity"], errors="coerce").fillna(-1))
        .sort_values("_tanimoto_sort", ascending=False).head(1)
    )

    candidate_summary_html = ""
    if not top.empty:
        r = top.iloc[0]
        tan_text = (
            f" with Tanimoto similarity <b>{float(r.get('tanimoto_similarity')):.3f}</b>"
            if pd.notna(r.get("tanimoto_similarity")) else ""
        )
        admet_context = _candidate_admet_context_verbose(cand_admet, df_sim, cand["smiles"])
        candidate_summary_html = f"""
<div class="sig-box" style="background:#FFF8EE; border-left:4px solid #FFB347; margin-top:14px; margin-bottom:8px;">
  <div class="sig-box-title" style="color:#D4780A;">Candidate Summary</div>
  <div style="font-size:0.88rem;color:#333;line-height:1.75;">
    The candidate is most structurally similar to <b>{str(r.get('drug_name', '—'))}</b>{tan_text}
    among target-associated compounds with usable chemical structures.<br>
    {admet_context}<br>
  </div>
</div>
"""

    with mol_col:
        img_bytes = _smiles_to_bytes(cand["smiles"], size=(340, 260))
        if img_bytes:
            b64 = base64.b64encode(img_bytes).decode()
            img_tag = f'<img src="data:image/png;base64,{b64}" style="width:100%; border-radius:4px;"/>'
        else:
            img_tag = '<div style="width:100%;height:220px;background:#F0F0F0;border-radius:4px;display:flex;align-items:center;justify-content:center;color:#BBB;font-size:0.84rem;">Structure unavailable</div>'
        st.markdown(
            f"""
<div class="card" style="min-height:420px;">
  <div style="display:flex; gap:16px; align-items:flex-start; width:100%; min-height:270px;">
    <div style="flex:0 0 34%; min-width:118px; padding-top:8px;">
      <div class="meta-label" style="margin-top:0;">Candidate</div>
      <div class="meta-value">{cand["name"]}</div>
      <div class="meta-label">Target</div>
      <div class="meta-value">{cand["target"]}</div>
      <div class="meta-label">Therapeutic Context</div>
      <div class="meta-value">{cand["indication"] or "Not specified"}</div>
    </div>
    <div style="flex:1; min-width:0;">{img_tag}</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with summary_col:
        st.markdown(
            f'<div class="card" style="min-height:420px; padding:18px 20px;">'
            f'<div style="font-size:0.86rem;font-weight:900;color:#888;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">Candidate ADMET / Developability</div>'
            f'{cand_admet_html}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(candidate_summary_html, unsafe_allow_html=True)
    st.markdown('<hr class="rule-orange"/>', unsafe_allow_html=True)
    st.markdown(f"### Compounds Targeting {target}")
    st.markdown(
        '<div class="notice" style="font-size:0.80rem;padding:9px 14px;margin-top:8px;">'
        'The compounds below were originally retrieved from <a href="https://www.opentargets.org/" target="_blank">Open Targets</a>, '
        'with available chemical structures enriched from <a href="https://www.ebi.ac.uk/chembl/" target="_blank">ChEMBL</a>. '
        'Open Targets provides target-associated drug and clinical-candidate context, while ChEMBL provides curated compound identifiers and SMILES strings where available. '
        'The comparator cards below represent the subset of target-associated compounds with usable chemical structures for similarity comparison.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption("Sorted by Tanimoto similarity to candidate")

    modality_options = ["Small molecule", "Antibody", "Antibody drug conjugate", "Protein", "Other"]
    stage_options = list(_MIN_STAGE_MAP.keys())
    prev_modalities = st.session_state.candidate_info.get("display_modalities", ["Small molecule"])
    prev_min_stage = st.session_state.candidate_info.get("display_min_stage", "Any")
    if prev_min_stage not in stage_options:
        prev_min_stage = "Any"

    filter_col1, filter_col2 = st.columns([1.35, 1])
    with filter_col1:
        selected_modalities = st.multiselect("Drug modality / type", options=modality_options,
            default=[m for m in prev_modalities if m in modality_options] or ["Small molecule"],
            key="candidate_display_modalities")
    with filter_col2:
        selected_min_stage = st.selectbox("Minimum clinical stage", options=stage_options,
            index=stage_options.index(prev_min_stage), key="candidate_display_min_stage")

    if (selected_modalities != st.session_state.candidate_info.get("display_modalities")
            or selected_min_stage != st.session_state.candidate_info.get("display_min_stage")):
        st.session_state.comp_page = 0

    st.session_state.candidate_info["display_modalities"] = selected_modalities
    st.session_state.candidate_info["display_min_stage"] = selected_min_stage

    df_sim["_sort"] = pd.to_numeric(df_sim["tanimoto_similarity"], errors="coerce").fillna(-1)
    df_all_unfiltered = df_sim.sort_values("_sort", ascending=False).drop(columns=["_sort"]).reset_index(drop=True)
    df_all = df_all_unfiltered.copy()

    if "drug_type" in df_all.columns:
        df_all = df_all[df_all["drug_type"].apply(lambda x: _matches_selected_modality(x, selected_modalities))].copy()

    min_stage_score = _MIN_STAGE_MAP.get(selected_min_stage, 0)
    if min_stage_score > 0 and "max_clinical_stage_for_target" in df_all.columns:
        df_all = df_all[df_all["max_clinical_stage_for_target"].apply(_stage_score) >= min_stage_score].copy()
    df_all = df_all.reset_index(drop=True)

    st.markdown(
        f'<div style="font-size:0.82rem;color:#777;margin:8px 0 12px 0;">'
        f'Showing <b>{len(df_all)}</b> of <b>{len(df_all_unfiltered)}</b> compounds with usable chemical structures after filters.'
        f'</div>',
        unsafe_allow_html=True,
    )

    total_rows = len(df_all)
    if total_rows == 0:
        st.info("No comparator compounds match the current filters. Try selecting additional modalities or lowering the minimum clinical stage.")
        st.stop()

    max_page = max(0, (total_rows - 1) // PAGE_SIZE)
    comp_page = min(st.session_state.comp_page, max_page)
    st.session_state.comp_page = comp_page
    start_idx = comp_page * PAGE_SIZE
    end_idx = min(start_idx + PAGE_SIZE, total_rows)

    pg_prev, pg_info, pg_next = st.columns([1, 5, 1])
    with pg_prev:
        if st.button("← Previous", disabled=(comp_page == 0), use_container_width=True):
            st.session_state.comp_page = comp_page - 1
            st.rerun()
    with pg_info:
        st.markdown(f'<div style="text-align:center; padding:8px 0; font-size:0.84rem; color:#888;">Compounds {start_idx + 1}–{end_idx} of {total_rows}</div>', unsafe_allow_html=True)
    with pg_next:
        if st.button("Next →", disabled=(comp_page >= max_page), use_container_width=True):
            st.session_state.comp_page = comp_page + 1
            st.rerun()

    df_page = df_all.iloc[start_idx:end_idx].reset_index(drop=True)
    selected_now = {d.lower() for d in st.session_state.get("selected_faers_comparators", [])}

    for i, (_, row) in enumerate(df_page.iterrows()):
        drug = str(row.get("drug_name", "—"))
        dtype = str(row.get("drug_type", "—"))
        stage = str(row.get("max_clinical_stage_for_target", "—"))
        smi = row.get("canonical_smiles")
        tan = row.get("tanimoto_similarity")
        fkey = _faers_key(row)
        spec = spec_map.get(fkey, []) if fkey in selected_now else []
        wsev = _worst_sev(spec, sev_map)
        tan_s = f"{float(tan):.3f}" if pd.notna(tan) else "—"
        bg = "#FFFFFF" if i % 2 == 0 else "#FAFAFA"

        drug_id = row.get("drug_id", "")
        structure_source = row.get("structure_source", "")
        structure_status = row.get("structure_lookup_status", "")

        trace_links = [
            _traceability_link("Open Targets", drug_id, _open_targets_drug_url(drug_id)),
            _traceability_link("ChEMBL", drug_id, _chembl_url(drug_id)),
        ]
        trace_links = [x for x in trace_links if x]
        traceability_html = " &nbsp;·&nbsp; ".join(trace_links)

        if _has_value(structure_source) or _has_value(structure_status):
            source_txt = html.escape(str(structure_source or "structure source").strip())
            status_txt = html.escape(str(structure_status or "").strip())
            structure_html = (
                f'<div style="font-size:0.68rem;color:#999;margin-top:4px;line-height:1.4;">'
                f'Structure source: <b>{source_txt}</b>'
                + (f' · Status: <b>{status_txt}</b>' if status_txt else "") + '</div>'
            )
        else:
            structure_html = ""

        st.markdown(f'<div style="background:{bg}; border:1px solid #E8E8E8; border-radius:8px; padding:12px 16px; margin-bottom:8px;">', unsafe_allow_html=True)
        img_col, info_col = st.columns([0.75, 2.50])

        with img_col:
            img = _smiles_to_bytes(str(smi), size=(650, 430)) if smi and pd.notna(smi) else None
            if img:
                st.image(img, width=390)
            else:
                st.markdown('<div style="height:255px; background:#F5F5F5; border-radius:6px; display:flex; align-items:center; justify-content:center; font-size:0.78rem; color:#BBB;">No structure</div>', unsafe_allow_html=True)
            if traceability_html or structure_html:
                st.markdown(f'<div style="font-size:0.70rem;color:#777;line-height:1.45;margin-top:4px;text-align:center;">{traceability_html}{structure_html}</div>', unsafe_allow_html=True)

        with info_col:
            st.markdown(f'<div style="font-size:1.05rem; font-weight:700; color:#222; margin-bottom:2px;">{drug}</div><div style="font-size:0.82rem; color:#888; margin-bottom:10px;">{dtype} &nbsp;·&nbsp; {stage} &nbsp;·&nbsp;</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="font-size:0.72rem; font-weight:700; text-transform:uppercase; color:#888; letter-spacing:0.05em; margin-bottom:2px;">Tanimoto Similarity</div>'
                f'<div style="font-family:monospace; font-weight:700; font-size:1.1rem; color:#333; margin-bottom:8px;">{tan_s}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(_admet_html(_admet_props(str(smi)) if smi and pd.notna(smi) else {}, compact=False), unsafe_allow_html=True)
            if spec:
                st.markdown('<div style="font-size:0.72rem; font-weight:700; text-transform:uppercase; color:#888; letter-spacing:0.05em; margin-top:8px; margin-bottom:3px;">Compound-Specific FAERS Signals</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size:0.85rem; color:#333; margin-bottom:8px;">{", ".join(e.title() for e in spec)}</div>', unsafe_allow_html=True)
            view_cols = st.columns([3, 2])
            with view_cols[0]:
                if wsev:
                    st.markdown(_sev_badge(wsev), unsafe_allow_html=True)
            with view_cols[1]:
                if st.button("View & Compare", key=f"view_{comp_page}_{i}", use_container_width=True, type="primary"):
                    _show_compound_dialog(row.to_dict(), cand, spec_map, sev_map)

        st.markdown("</div>", unsafe_allow_html=True)
