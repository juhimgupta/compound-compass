from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem, rdFMCS, rdMolAlign, Descriptors, Crippen, Lipinski, QED
    _RDKIT = True
except ImportError:
    _RDKIT = False


def _smiles_to_bytes(smiles, size=(300, 230)) -> bytes | None:
    if not _RDKIT or not smiles:
        return None
    if isinstance(smiles, float) and pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if not mol:
            return None
        buf = BytesIO()
        Draw.MolToImage(mol, size=size).save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def _admet_props(smiles: str) -> dict[str, Any]:
    out = {"MW": None, "LogP": None, "TPSA": None, "HBD": None, "HBA": None, "RotB": None, "QED": None, "RO5": None, "Flag": "Unavailable"}
    if not _RDKIT or not smiles:
        return out
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return out
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        rotb = Lipinski.NumRotatableBonds(mol)
        qed = QED.qed(mol)
        ro5 = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
        flags = []
        if mw > 500: flags.append("High MW")
        if logp > 5: flags.append("High LogP")
        if tpsa > 140: flags.append("High TPSA")
        if rotb > 10: flags.append("Flexible")
        if ro5 == 0: flags.append("RO5 clean")
        out.update({
            "MW": round(mw, 1), "LogP": round(logp, 2), "TPSA": round(tpsa, 1),
            "HBD": int(hbd), "HBA": int(hba), "RotB": int(rotb), "QED": round(qed, 2),
            "RO5": int(ro5), "Flag": ", ".join(flags) if flags else "Review",
        })
        return out
    except Exception:
        return out


def _percentile_rank(value: Any, reference_values: list[float]) -> float | None:
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
    below_or_equal = sum(x <= v for x in refs)
    return round(100 * below_or_equal / len(refs), 0)


def _admet_percentile_phrase(label: str, value: Any, percentile: float | None) -> str:
    if percentile is None or pd.isna(value):
        return ""
    try:
        value_txt = f"{float(value):.2f}" if label in {"LogP", "QED"} else f"{float(value):.1f}"
    except Exception:
        value_txt = str(value)
    return f"<b>{label}</b> is {value_txt}, around the <b>{int(percentile)}th percentile</b> of structure-comparable target compounds"


def _candidate_admet_context(cand_admet: dict[str, Any], df_sim: pd.DataFrame) -> str:
    if df_sim.empty or "canonical_smiles" not in df_sim.columns:
        return ""
    comparator_props = []
    for smi in df_sim["canonical_smiles"].dropna().astype(str):
        props = _admet_props(smi)
        if props and props.get("MW") is not None:
            comparator_props.append(props)
    if not comparator_props:
        return ""
    metrics = [
        ("Molecular weight", "MW"),
        ("Calculated LogP", "LogP"),
        ("QED drug-likeness", "QED"),
    ]
    phrases = []
    for label, key in metrics:
        cand_val = cand_admet.get(key)
        ref_vals = [p.get(key) for p in comparator_props if p.get(key) is not None]
        pct = _percentile_rank(cand_val, ref_vals)
        phrase = _admet_percentile_phrase(label, cand_val, pct)
        if phrase:
            phrases.append(phrase)
    if not phrases:
        return ""
    return "; ".join(phrases) + "."


def _admet_html(props: dict[str, Any], compact: bool = False) -> str:
    if not props or props.get("MW") is None:
        return '<div style="font-size:0.84rem;color:#BBB;font-style:italic;">ADMET unavailable</div>'
    if compact:
        grid_cols = "repeat(4, 1fr)"
        value_font = "0.82rem"
        label_font = "0.62rem"
        items = [
            ("MW", props["MW"]), ("LogP", props["LogP"]), ("TPSA", props["TPSA"]), ("HBD", props["HBD"]),
            ("HBA", props["HBA"]), ("RotB", props["RotB"]), ("QED", props["QED"]), ("RO5", props["RO5"]),
        ]
    else:
        grid_cols = "repeat(2, 1fr)"
        value_font = "1.02rem"
        label_font = "0.74rem"
        items = [
            ("Molecular weight", props["MW"]),
            ("Calculated LogP", props["LogP"]),
            ("Topological polar surface area", props["TPSA"]),
            ("Hydrogen-bond donors", props["HBD"]),
            ("Hydrogen-bond acceptors", props["HBA"]),
            ("Rotatable bonds", props["RotB"]),
            ("QED drug-likeness", props["QED"]),
            ("Lipinski Rule of Five violations", props["RO5"]),
        ]
    cells = ""
    for k, v in items:
        cells += (
            f'<div style="background:#FAFAFA;border:1px solid #E6E6E6;border-radius:6px;padding:8px 10px;text-align:right;min-height:52px;">'
            f'<div style="font-size:{label_font};color:#888;font-weight:800;text-transform:uppercase;letter-spacing:0.045em;line-height:1.25;">{k}</div>'
            f'<div style="font-size:{value_font};font-weight:900;color:#222;margin-top:3px;">{v}</div></div>'
        )
    return (
        f'<div style="margin-top:8px;"><div style="font-size:0.76rem;color:#999;font-weight:800;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;text-align:right;">Developability Snapshot</div>'
        f'<div style="display:grid;grid-template-columns:{grid_cols};gap:7px;">{cells}</div>'
        f'<div style="font-size:0.78rem;color:#666;margin-top:8px;text-align:right;font-weight:700;">{props.get("Flag", "")}</div></div>'
    )


def _mcs_scaffold_summary(candidate_smiles: str, comparator_smiles: str) -> dict[str, Any]:
    out = {"mcs_atoms": 0, "mcs_bonds": 0, "mcs_smarts": ""}
    if not _RDKIT or not candidate_smiles or not comparator_smiles:
        return out
    try:
        mol_a = Chem.MolFromSmiles(str(candidate_smiles))
        mol_b = Chem.MolFromSmiles(str(comparator_smiles))
        if mol_a is None or mol_b is None:
            return out
        mcs = rdFMCS.FindMCS(
            [mol_a, mol_b],
            timeout=8,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
        )
        if mcs and mcs.numAtoms:
            out["mcs_atoms"] = int(mcs.numAtoms)
            out["mcs_bonds"] = int(mcs.numBonds)
            out["mcs_smarts"] = str(mcs.smartsString or "")
        return out
    except Exception:
        return out


def build_3dmol_html(c_sdf: str, m_sdf: str, c_serial: list[int], m_serial: list[int], width: int = 780, height: int = 480) -> str:
    c_js = json.dumps(c_sdf)
    m_js = json.dumps(m_sdf)
    cs = json.dumps(c_serial)
    ms = json.dumps(m_serial)
    return (
        f"""<!DOCTYPE html><html><head><meta charset="utf-8">"""
        f"""<script src="https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js"></script>"""
        f"""<style>body{{margin:0;padding:0;background:#F4F4F4;overflow:hidden;}}</style></head>"""
        f"""<body><div id="gldiv" style="width:{width}px;height:{height}px;position:relative;"></div>"""
        f"""<script>(function(){{var v=$3Dmol.createViewer(document.getElementById('gldiv'),{{backgroundColor:'#F4F4F4'}});"""
        f"""v.addModel({c_js},'sdf');v.setStyle({{model:0}},{{stick:{{color:'#C0C0C0',radius:0.10}}}});"""
        f"""var cs={cs};if(cs.length>0){{v.addStyle({{model:0,serial:cs}},{{stick:{{color:'#FF8C00',radius:0.22}},sphere:{{color:'#FF8C00',radius:0.28}}}});}}"""
        f"""v.addModel({m_js},'sdf');v.setStyle({{model:1}},{{stick:{{color:'#C0C0C0',radius:0.10}}}});"""
        f"""var ms={ms};if(ms.length>0){{v.addStyle({{model:1,serial:ms}},{{stick:{{color:'#1976D2',radius:0.22}},sphere:{{color:'#1976D2',radius:0.28}}}});}}"""
        f"""v.zoomTo();v.render();}})();</script></body></html>"""
    )


def render_3d_comparison(cand_smi: str, comp_smi: str, cand_name: str, comp_name: str) -> None:
    if not _RDKIT:
        st.warning("RDKit required for 3D comparison.")
        return

    def _make_3d(smi: str, label: str):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            st.error(f"Invalid SMILES for {label}.")
            return None
        mol = Chem.AddHs(mol)
        try:
            params = AllChem.ETKDGv3()
        except AttributeError:
            params = AllChem.ETKDG()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params) == -1:
            st.warning(f"Could not generate 3D conformer for {label}.")
            return None
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            try:
                AllChem.UFFOptimizeMolecule(mol)
            except Exception:
                pass
        return mol

    with st.spinner("Generating 3D conformers…"):
        mol_c = _make_3d(cand_smi, cand_name)
        mol_m = _make_3d(comp_smi, comp_name)
    if mol_c is None or mol_m is None:
        return
    c_noh = Chem.RemoveHs(mol_c)
    m_noh = Chem.RemoveHs(mol_m)
    mcs = rdFMCS.FindMCS(
        [c_noh, m_noh], timeout=8,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
    )
    c_mcs, m_mcs, mcs_n = [], [], 0
    if mcs and mcs.numAtoms >= 3:
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
        if mcs_mol:
            c_mcs = list(c_noh.GetSubstructMatch(mcs_mol))
            m_mcs = list(m_noh.GetSubstructMatch(mcs_mol))
            mcs_n = mcs.numAtoms
            if c_mcs and m_mcs:
                try:
                    rdMolAlign.AlignMolecule(mol_m, mol_c, atomMap=list(zip(m_mcs, c_mcs)))
                except Exception:
                    pass
    st.markdown(
        f'<div style="display:flex;gap:24px;margin:4px 0 6px 0;font-size:0.83rem;">'
        f'<span><span style="display:inline-block;width:11px;height:11px;background:#FF8C00;border-radius:50%;margin-right:5px;vertical-align:middle;"></span><b>Candidate:</b> {cand_name}</span>'
        f'<span><span style="display:inline-block;width:11px;height:11px;background:#1976D2;border-radius:50%;margin-right:5px;vertical-align:middle;"></span><b>Comparator:</b> {comp_name}</span>'
        f'<span style="color:#888;">· {"Shared scaffold: " + str(mcs_n) + " atoms" if mcs_n else "No significant common substructure"}</span></div>',
        unsafe_allow_html=True,
    )
    components.html(
        build_3dmol_html(Chem.MolToMolBlock(mol_c), Chem.MolToMolBlock(mol_m), [i + 1 for i in c_mcs], [i + 1 for i in m_mcs], width=760, height=460),
        height=480, scrolling=False,
    )
    st.markdown(
        '<div class="notice" style="margin-top:6px;font-size:0.79rem;padding:9px 14px;">'
        '3D structures are generated conformers from SMILES and are for visual comparison only. '
        'They are not experimental structures or docked binding poses.</div>',
        unsafe_allow_html=True,
    )
