from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from config import PROJECT_DIR, ENGINE_DIR, _STAGE_RANK, MAX_FAERS_COMPARATORS


def _engine_script(name: str) -> str:
    """Return the absolute path to an engine script."""
    return str(ENGINE_DIR / name)


def _slugify_target(target: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(target).strip()).strip("_") or "target"


def _paths_for(slug: str) -> tuple[dict[str, Path], dict[str, Path], dict[str, str]]:
    dirs = {
        "open_targets": PROJECT_DIR / "open_targets_output" / slug,
        "curate": PROJECT_DIR / "curate_target_class_output" / slug,
        "faers_class": PROJECT_DIR / "faers_class_output" / slug,
        "recs": PROJECT_DIR / "recommendation_output" / slug,
        "similarity": PROJECT_DIR / "candidate_similarity_output" / slug,
        "trial_geo": PROJECT_DIR / "trial_geography_output" / slug,
    }
    files = {
        "known_drugs": dirs["open_targets"] / f"{slug}_known_drugs.csv",
        "target_summary": dirs["open_targets"] / f"{slug}_target_summary.json",
        "associated_diseases": dirs["open_targets"] / f"{slug}_associated_diseases.csv",
        "curated_class": dirs["curate"] / f"{slug}_curated_class_demo.csv",
        "faers_selected_drugs": dirs["faers_class"] / f"{slug}_selected_faers_comparators.csv",
        "event_class": dirs["faers_class"] / f"{slug}_selected_faers_event_classification.csv",
        "ror_matrix": dirs["faers_class"] / f"{slug}_selected_faers_ror_matrix.csv",
        "count_matrix": dirs["faers_class"] / f"{slug}_selected_faers_count_matrix.csv",
        "recommendations": dirs["recs"] / f"{slug}_recommendations.csv",
        "similarity": dirs["similarity"] / f"{slug}_candidate_similarity.csv",
        "trial_locations": dirs["trial_geo"] / f"{slug}_trial_locations.csv",
        "country_summary": dirs["trial_geo"] / f"{slug}_country_summary.csv",
        "trial_demographics": dirs["trial_geo"] / f"{slug}_trial_demographics.csv",
        "demographic_summary": dirs["trial_geo"] / f"{slug}_demographic_summary.csv",
        "baseline_demographics": dirs["trial_geo"] / f"{slug}_baseline_demographics.csv",
        "trial_selected_drugs": dirs["trial_geo"] / f"{slug}_selected_trial_comparators.csv",
    }
    rel = {
        "ot_prefix": f"open_targets_output/{slug}/{slug}",
        "known_drugs": f"open_targets_output/{slug}/{slug}_known_drugs.csv",
        "curated": f"curate_target_class_output/{slug}/{slug}_curated_class_demo.csv",
        "faers_prefix": f"faers_class_output/{slug}/{slug}_selected_faers",
        "recs_out": f"recommendation_output/{slug}/{slug}_recommendations.csv",
        "sim_out": f"candidate_similarity_output/{slug}/{slug}_candidate_similarity.csv",
    }
    return dirs, files, rel


def _run(cmd: list[str]) -> tuple[bool, str, str]:
    r = subprocess.run(cmd, cwd=str(PROJECT_DIR), capture_output=True, text=True)
    return r.returncode == 0, r.stdout, r.stderr


def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception:
        return {}


def _has_value(x: Any) -> bool:
    return pd.notna(x) and str(x).strip().lower() not in {"", "nan", "none", "null"}


def _stage_score(stage: Any) -> int:
    return _STAGE_RANK.get(str(stage).strip().lower(), 0)


def _is_small_molecule(drug_type: Any) -> bool:
    return "small molecule" in str(drug_type).strip().lower()


def _auto_select_comparators(df_sim: pd.DataFrame, n: int = MAX_FAERS_COMPARATORS) -> list[str]:
    if df_sim.empty:
        return []
    df = df_sim.copy()
    if "faers_search_name" not in df.columns and "drug_name" in df.columns:
        df["faers_search_name"] = df["drug_name"]
    df = df[df["faers_search_name"].apply(_has_value)].copy()
    if "canonical_smiles" in df.columns:
        df = df[df["canonical_smiles"].apply(_has_value)].copy()
    if "drug_type" in df.columns:
        df = df[df["drug_type"].apply(_is_small_molecule)].copy()
    if df.empty:
        return []
    df["_stage_score"] = df.get("max_clinical_stage_for_target", pd.Series([""] * len(df))).apply(_stage_score)
    df["_tan"] = pd.to_numeric(df.get("tanimoto_similarity", pd.Series([0] * len(df))), errors="coerce").fillna(0)
    df["_name_norm"] = df["faers_search_name"].astype(str).str.lower().str.strip()
    df = df.sort_values(["_stage_score", "_tan"], ascending=[False, False]).drop_duplicates("_name_norm")
    return df.head(n)["faers_search_name"].astype(str).str.strip().tolist()


def _option_map_from_similarity(df_sim: pd.DataFrame) -> dict[str, str]:
    opts: dict[str, str] = {}
    if df_sim.empty:
        return opts
    df = df_sim.copy()
    if "faers_search_name" not in df.columns and "drug_name" in df.columns:
        df["faers_search_name"] = df["drug_name"]
    df = df[df["faers_search_name"].apply(_has_value)].copy()
    if df.empty:
        return opts
    df["_tan"] = pd.to_numeric(df.get("tanimoto_similarity", pd.Series([0] * len(df))), errors="coerce").fillna(0)
    df["_stage_score"] = df.get("max_clinical_stage_for_target", pd.Series([""] * len(df))).apply(_stage_score)
    df["_name_norm"] = df["faers_search_name"].astype(str).str.lower().str.strip()
    df = df.sort_values(["_stage_score", "_tan"], ascending=[False, False]).drop_duplicates("_name_norm")
    for _, r in df.iterrows():
        faers = str(r.get("faers_search_name", "")).strip()
        drug = str(r.get("drug_name", faers)).strip()
        dtype = str(r.get("drug_type", "unknown")).strip()
        stage = str(r.get("max_clinical_stage_for_target", "unknown")).strip()
        tan = r.get("_tan", None)
        label = f"{drug}  ·  {stage}  ·  {dtype}"
        if pd.notna(tan):
            label += f"  ·  Tanimoto {float(tan):.3f}"
        opts[label] = faers
    return opts
