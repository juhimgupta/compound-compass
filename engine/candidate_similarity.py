#!/usr/bin/env python3
"""
candidate_similarity.py
=======================
Compare an uploaded/internal candidate molecule against known target-associated
molecules from Open Targets / ChEMBL, then optionally attach FAERS-derived
recommendation context.

This script is intentionally positioned as a triage/context tool, not a safety
prediction model:

    candidate SMILES + target
        -> known target-associated molecules
        -> structural similarity ranking
        -> optional safety/recommendation context

Inputs
------
--candidate-name       Friendly name for the uploaded/internal molecule
--candidate-smiles     Candidate SMILES string
--target               Target symbol, e.g. EGFR
--known-drugs          CSV from open_targets_context.py, ideally enriched with
                       canonical_smiles from ChEMBL
--curated-class        CSV from curate_target_class.py, used to mark which
                       molecules were included in the FAERS class matrix
--recommendations      Optional CSV from build_recommendations.py
--out                  Output CSV path for similarity table

Key outputs
-----------
1) <out>
   One row per known target-associated molecule, with Tanimoto similarity.

2) <out stem>_safety_context.csv, if --recommendations is provided
   A compact context table connecting class-wide and molecule-specific FAERS
   recommendation rows to the uploaded candidate.

Example
-------
python candidate_similarity.py ^
  --candidate-name "Internal EGFR Candidate" ^
  --candidate-smiles "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC=CC=C3)OCCCN4CCOCC4" ^
  --target EGFR ^
  --known-drugs open_targets_output/egfr/egfr_known_drugs.csv ^
  --curated-class curate_target_class_output/egfr/egfr_curated_class_demo.csv ^
  --recommendations recommendation_output/egfr/egfr_recommendations.csv ^
  --out candidate_similarity_output/egfr/egfr_candidate_similarity.csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import pandas as pd
import requests

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "RDKit is required for candidate similarity. Install with:\n"
        "  conda install -c conda-forge rdkit"
    ) from exc


# --------------------------------------------------------------------------- #
# Name normalization / duplicate handling
# --------------------------------------------------------------------------- #
SALT_SUFFIXES = [
    "hydrochloride",
    "dihydrochloride",
    "maleate",
    "dimaleate",
    "mesylate",
    "succinate",
    "ditosylate",
    "tosylate",
    "phosphate",
    "sulfate",
    "sulphate",
    "fumarate",
    "tartrate",
    "citrate",
    "malate",
    "besylate",
    "anhydrous",
    "hydrate",
    "monohydrate",
    "dihydrate",
]


def norm_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x).strip().lower())


def normalize_drug_name(name: Any) -> str:
    """Normalize drug name for joins and FAERS-style querying."""
    s = norm_text(name)
    if not s:
        return ""
    s = s.replace("_", " ").replace("-", "-")
    # Remove parenthetical annotations.
    s = re.sub(r"\([^)]*\)", "", s).strip()
    # Drop trailing salt/formulation suffixes iteratively.
    changed = True
    while changed:
        changed = False
        for suffix in SALT_SUFFIXES:
            pattern = r"\b" + re.escape(suffix) + r"$"
            new_s = re.sub(pattern, "", s).strip()
            if new_s != s:
                s = new_s
                changed = True
    s = re.sub(r"\s+", " ", s).strip()
    return s


def first_existing_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# --------------------------------------------------------------------------- #
# Structure handling
# --------------------------------------------------------------------------- #
def mol_from_smiles(smiles: Any) -> Optional[Chem.Mol]:
    if pd.isna(smiles):
        return None
    s = str(smiles).strip()
    if not s or s.lower() in {"nan", "none", "null", "<na>"}:
        return None
    mol = Chem.MolFromSmiles(s)
    return mol


def fingerprint(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048):
    # Use Morgan fingerprint. RDKit may emit deprecation warnings in newer builds;
    # this is fine for a lightweight interview prototype.
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto(candidate_smiles: str, comparator_smiles: Any) -> Optional[float]:
    cand_mol = mol_from_smiles(candidate_smiles)
    comp_mol = mol_from_smiles(comparator_smiles)
    if cand_mol is None:
        raise ValueError("Candidate SMILES could not be parsed by RDKit.")
    if comp_mol is None:
        return None
    return float(DataStructs.TanimotoSimilarity(fingerprint(cand_mol), fingerprint(comp_mol)))


def similarity_band(value: Any) -> str:
    if value is None or pd.isna(value):
        return "unavailable"
    v = float(value)
    if v >= 0.70:
        return "high structural similarity"
    if v >= 0.40:
        return "moderate structural similarity"
    if v >= 0.20:
        return "low structural similarity"
    return "very low structural similarity"


# --------------------------------------------------------------------------- #
# Optional PubChem fallback, kept off by default
# --------------------------------------------------------------------------- #
def load_cache(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def save_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True))


def pubchem_smiles_lookup(name: str, cache: dict[str, Any], pause: float = 0.15) -> tuple[Optional[str], str]:
    """Fallback only. Primary structure source should be Open Targets/ChEMBL output."""
    q = normalize_drug_name(name)
    if not q:
        return None, "no_name"
    if q in cache:
        val = cache[q]
        if isinstance(val, dict):
            return val.get("canonical_smiles"), val.get("status", "cached")
        return val, "cached"

    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(q)}/property/CanonicalSMILES/JSON"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            data = r.json()
            props = data.get("PropertyTable", {}).get("Properties", [])
            if props:
                smiles = props[0].get("CanonicalSMILES")
                cache[q] = {"canonical_smiles": smiles, "status": "pubchem_name_match"}
                time.sleep(pause)
                return smiles, "pubchem_name_match"
        cache[q] = {"canonical_smiles": None, "status": f"pubchem_http_{r.status_code}"}
        time.sleep(pause)
        return None, f"pubchem_http_{r.status_code}"
    except Exception as exc:
        cache[q] = {"canonical_smiles": None, "status": f"pubchem_error:{type(exc).__name__}"}
        return None, f"pubchem_error:{type(exc).__name__}"


def enrich_missing_smiles_with_pubchem(df: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    cache = load_cache(cache_path)
    smiles_col = first_existing_col(df, ["canonical_smiles", "smiles", "comparator_smiles"])
    if smiles_col is None:
        df["canonical_smiles"] = pd.NA
        smiles_col = "canonical_smiles"
    if "structure_source" not in df.columns:
        df["structure_source"] = pd.NA
    if "structure_lookup_status" not in df.columns:
        df["structure_lookup_status"] = pd.NA

    for idx, row in df.iterrows():
        if mol_from_smiles(row.get(smiles_col)) is not None:
            continue
        name = row.get("faers_search_name") or row.get("drug_name")
        smiles, status = pubchem_smiles_lookup(str(name), cache)
        if smiles:
            df.at[idx, smiles_col] = smiles
            df.at[idx, "structure_source"] = "PubChem fallback"
        df.at[idx, "structure_lookup_status"] = status
    save_cache(cache_path, cache)
    return df


# --------------------------------------------------------------------------- #
# Data loading / joins
# --------------------------------------------------------------------------- #
def read_known_drugs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    drug_col = first_existing_col(df, ["drug_name", "name", "drug"])
    if drug_col is None:
        raise ValueError("known-drugs CSV must contain a drug_name/name/drug column.")
    if drug_col != "drug_name":
        df = df.rename(columns={drug_col: "drug_name"})

    if "faers_search_name" not in df.columns:
        df["faers_search_name"] = df["drug_name"].map(normalize_drug_name)
    else:
        # Fill missing names with normalized drug names, then normalize all.
        df["faers_search_name"] = df["faers_search_name"].where(
            df["faers_search_name"].notna(), df["drug_name"]
        )
        df["faers_search_name"] = df["faers_search_name"].map(normalize_drug_name)

    # Make sure structure field exists and has a consistent name.
    smiles_col = first_existing_col(
        df,
        ["canonical_smiles", "smiles", "canonicalSmiles", "canonical_smile", "isomeric_smiles"],
    )
    if smiles_col is None:
        df["canonical_smiles"] = pd.NA
    elif smiles_col != "canonical_smiles":
        df = df.rename(columns={smiles_col: "canonical_smiles"})

    return df


def read_curated_class(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    df = pd.read_csv(path)
    if "faers_search_name" not in df.columns:
        drug_col = first_existing_col(df, ["drug_name", "name", "drug"])
        if drug_col is None:
            raise ValueError("curated-class CSV must contain faers_search_name or drug_name.")
        df["faers_search_name"] = df[drug_col].map(normalize_drug_name)
    else:
        df["faers_search_name"] = df["faers_search_name"].map(normalize_drug_name)

    if "include_in_faers_matrix" in df.columns:
        df["include_in_faers_matrix"] = df["include_in_faers_matrix"].map(parse_bool)
    else:
        df["include_in_faers_matrix"] = pd.NA

    keep_cols = [c for c in [
        "faers_search_name",
        "include_in_faers_matrix",
        "curation_reason",
        "drug_name",
        "drug_type",
        "max_clinical_stage_for_target",
    ] if c in df.columns]

    # One row per normalized FAERS name. Prefer included rows if duplicates exist.
    sort_cols = []
    if "include_in_faers_matrix" in df.columns:
        df["_include_sort"] = df["include_in_faers_matrix"].fillna(False).astype(bool).astype(int)
        sort_cols.append("_include_sort")
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=False)
    df = df.drop_duplicates("faers_search_name", keep="first")
    return df[keep_cols]


def parse_bool(x: Any) -> Any:
    if pd.isna(x):
        return pd.NA
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y", "included", "include"}:
        return True
    if s in {"false", "0", "no", "n", "excluded", "exclude"}:
        return False
    return pd.NA


def attach_curated_metadata(known: pd.DataFrame, curated: Optional[pd.DataFrame]) -> pd.DataFrame:
    if curated is None:
        known["include_in_faers_matrix"] = pd.NA
        known["curation_reason"] = pd.NA
        return known

    # Avoid suffix confusion by only bringing curated columns not already in known,
    # except for the include/reason fields that we explicitly want to preserve.
    bring = ["faers_search_name"]
    for c in ["include_in_faers_matrix", "curation_reason"]:
        if c in curated.columns:
            bring.append(c)
    curated_small = curated[bring].copy()

    # Drop existing stale include/reason columns in known before merge.
    known = known.drop(columns=[c for c in ["include_in_faers_matrix", "curation_reason"] if c in known.columns])
    out = known.merge(curated_small, on="faers_search_name", how="left")
    return out


def collapse_duplicate_faers_names(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the best representative row per faers_search_name.

    Ranking preference:
    1. Valid similarity available
    2. Highest similarity
    3. Included in FAERS matrix
    4. Shorter drug_name, usually base molecule over salt/formulation
    """
    if "faers_search_name" not in df.columns:
        return df
    tmp = df.copy()
    tmp["_has_similarity"] = tmp["tanimoto_similarity"].notna().astype(int)
    tmp["_sim_sort"] = pd.to_numeric(tmp["tanimoto_similarity"], errors="coerce").fillna(-1)
    if "include_in_faers_matrix" in tmp.columns:
        tmp["_include_sort"] = tmp["include_in_faers_matrix"].fillna(False).astype(bool).astype(int)
    else:
        tmp["_include_sort"] = 0
    tmp["_name_len"] = tmp["drug_name"].astype(str).str.len()

    tmp = tmp.sort_values(
        ["faers_search_name", "_has_similarity", "_sim_sort", "_include_sort", "_name_len"],
        ascending=[True, False, False, False, True],
    )
    tmp = tmp.drop_duplicates("faers_search_name", keep="first")
    return tmp.drop(columns=[c for c in tmp.columns if c.startswith("_")])


# --------------------------------------------------------------------------- #
# Similarity + safety context
# --------------------------------------------------------------------------- #
def build_similarity_table(
    candidate_name: str,
    candidate_smiles: str,
    target: str,
    known_drugs_csv: str,
    curated_class_csv: Optional[str] = None,
    only_curated_included: bool = False,
    collapse_duplicates: bool = True,
    use_pubchem_fallback: bool = False,
    cache_path: Optional[str] = None,
) -> pd.DataFrame:
    # Validate candidate early.
    if mol_from_smiles(candidate_smiles) is None:
        raise ValueError("Candidate SMILES could not be parsed by RDKit.")

    known = read_known_drugs(known_drugs_csv)
    curated = read_curated_class(curated_class_csv) if curated_class_csv else None
    df = attach_curated_metadata(known, curated)

    if use_pubchem_fallback:
        cp = Path(cache_path) if cache_path else Path("candidate_similarity_output/pubchem_smiles_cache.json")
        df = enrich_missing_smiles_with_pubchem(df, cp)

    if only_curated_included:
        if "include_in_faers_matrix" not in df.columns:
            raise ValueError("--only-curated-included requires --curated-class with include_in_faers_matrix.")
        mask = df["include_in_faers_matrix"].fillna(False).astype(bool)
        df = df.loc[mask].copy()

    sims: list[Optional[float]] = []
    for smi in df["canonical_smiles"]:
        sims.append(tanimoto(candidate_smiles, smi))
    df["tanimoto_similarity"] = sims
    df["similarity_band"] = df["tanimoto_similarity"].map(similarity_band)

    if collapse_duplicates:
        df = collapse_duplicate_faers_names(df)

    # Useful for Streamlit filtering.
    df["has_structure"] = df["canonical_smiles"].map(lambda x: mol_from_smiles(x) is not None)
    df["candidate_name"] = candidate_name
    df["candidate_smiles"] = candidate_smiles
    df["target"] = target.upper()

    # Friendly column aliases.
    if "max_clinical_stage_for_target" not in df.columns and "clinical_stage" in df.columns:
        df["max_clinical_stage_for_target"] = df["clinical_stage"]

    preferred_cols = [
        "candidate_name",
        "target",
        "drug_name",
        "faers_search_name",
        "tanimoto_similarity",
        "similarity_band",
        "include_in_faers_matrix",
        "curation_reason",
        "has_structure",
        "drug_type",
        "max_clinical_stage_for_target",
        "canonical_smiles",
        "standard_inchi_key",
        "structure_source",
        "structure_lookup_status",
        "candidate_smiles",
    ]
    cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
    df = df[cols]

    # Sort: available/high similarity first; rows without structures last.
    df["_sim_sort"] = pd.to_numeric(df["tanimoto_similarity"], errors="coerce").fillna(-1)
    df = df.sort_values("_sim_sort", ascending=False).drop(columns="_sim_sort").reset_index(drop=True)
    return df


def best_similarity_for_driver(sim_df: pd.DataFrame, driver: Any) -> tuple[Any, Any]:
    if pd.isna(driver) or not str(driver).strip():
        return pd.NA, pd.NA
    d = normalize_drug_name(driver)
    if not d:
        return pd.NA, pd.NA
    matches = sim_df[sim_df["faers_search_name"].map(normalize_drug_name) == d]
    if matches.empty:
        return pd.NA, pd.NA
    row = matches.sort_values("tanimoto_similarity", ascending=False, na_position="last").iloc[0]
    return row.get("tanimoto_similarity", pd.NA), row.get("similarity_band", pd.NA)


def build_safety_context(sim_df: pd.DataFrame, recommendations_csv: str, top_n_each: int = 10) -> pd.DataFrame:
    rec = pd.read_csv(recommendations_csv)

    # Normalize expected fields.
    for col in ["event_category", "classification", "priority", "severity", "driver"]:
        if col not in rec.columns:
            rec[col] = pd.NA

    frames = []

    # Class-level safety context: top high/medium safety toxicity rows that are class-wide.
    class_mask = (
        rec["event_category"].eq("safety_toxicity")
        & rec["classification"].astype(str).str.contains("class-wide", case=False, na=False)
    )
    class_ctx = rec.loc[class_mask].copy()
    class_ctx["context_type"] = "target/class safety context"
    class_ctx["candidate_relevance"] = (
        "Relevant to the candidate as target/class context; structural similarity to one specific drug is not required."
    )
    frames.append(class_ctx.head(top_n_each))

    # Molecule-specific context: attach similarity to the driver molecule.
    mol_mask = (
        rec["event_category"].eq("safety_toxicity")
        & rec["classification"].astype(str).str.contains("molecule-specific", case=False, na=False)
    )
    mol_ctx = rec.loc[mol_mask].copy()
    if not mol_ctx.empty:
        sims = mol_ctx["driver"].map(lambda d: best_similarity_for_driver(sim_df, d))
        mol_ctx["tanimoto_similarity"] = [x[0] for x in sims]
        mol_ctx["similarity_band"] = [x[1] for x in sims]
        mol_ctx["context_type"] = "molecule-specific safety context"
        mol_ctx["candidate_relevance"] = mol_ctx.apply(
            lambda r: molecule_specific_relevance(r.get("driver"), r.get("tanimoto_similarity"), r.get("similarity_band")),
            axis=1,
        )
        mol_ctx = mol_ctx.sort_values(
            ["priority", "tanimoto_similarity"],
            ascending=[True, False],
            na_position="last",
        )
        frames.append(mol_ctx.head(top_n_each))

    # Disease/resistance context: top contextual rows, capped for readability.
    contextual_mask = rec["event_category"].isin(["disease_progression", "resistance_or_efficacy"])
    contextual = rec.loc[contextual_mask].copy()
    contextual["context_type"] = "disease/resistance interpretation context"
    contextual["candidate_relevance"] = (
        "Important context, but not interpreted as direct candidate toxicity from FAERS alone."
    )
    frames.append(contextual.head(top_n_each))

    out = pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True)
    if out.empty:
        return out

    # Ensure similarity fields exist for all rows.
    if "tanimoto_similarity" not in out.columns:
        out["tanimoto_similarity"] = pd.NA
    if "similarity_band" not in out.columns:
        out["similarity_band"] = pd.NA

    preferred = [
        "context_type",
        "event",
        "event_category",
        "severity",
        "classification",
        "priority",
        "driver",
        "tanimoto_similarity",
        "similarity_band",
        "candidate_relevance",
        "interpretation",
        "recommended_action",
        "candidate_implication",
        "stakeholder_owner",
        "future_optimization_extension",
    ]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    return out[cols]


def molecule_specific_relevance(driver: Any, sim: Any, band: Any) -> str:
    if pd.isna(driver) or not str(driver).strip():
        return "Molecule-specific event; driver drug unavailable."
    if pd.isna(sim):
        return f"Molecule-specific event for {driver}; candidate similarity unavailable, so review qualitatively."
    v = float(sim)
    if v >= 0.70:
        return f"Candidate is highly similar to driver drug {driver}; review this molecule-specific signal closely."
    if v >= 0.40:
        return f"Candidate has moderate structural similarity to driver drug {driver}; consider this signal in follow-up review."
    if v >= 0.20:
        return f"Candidate has low structural similarity to driver drug {driver}; signal may be less directly relevant but still worth awareness."
    return f"Candidate has very low structural similarity to driver drug {driver}; signal is less likely to be structure-linked based on this metric alone."


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Candidate-to-known-drug structural similarity with optional FAERS context")
    ap.add_argument("--candidate-name", required=True)
    ap.add_argument("--candidate-smiles", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--known-drugs", required=True, help="CSV from open_targets_context.py")
    ap.add_argument("--curated-class", help="CSV from curate_target_class.py")
    ap.add_argument("--recommendations", help="CSV from build_recommendations.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only-curated-included", action="store_true", help="Restrict similarity table to rows included in FAERS matrix")
    ap.add_argument("--no-collapse-duplicates", action="store_true", help="Keep salt/formulation duplicate rows instead of one row per FAERS name")
    ap.add_argument("--use-pubchem-fallback", action="store_true", help="Fill missing SMILES from PubChem name lookup if Open Targets/ChEMBL structures are missing")
    ap.add_argument("--pubchem-cache", default="candidate_similarity_output/pubchem_smiles_cache.json")
    ap.add_argument("--top-context", type=int, default=10)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sim_df = build_similarity_table(
        candidate_name=args.candidate_name,
        candidate_smiles=args.candidate_smiles,
        target=args.target,
        known_drugs_csv=args.known_drugs,
        curated_class_csv=args.curated_class,
        only_curated_included=args.only_curated_included,
        collapse_duplicates=not args.no_collapse_duplicates,
        use_pubchem_fallback=args.use_pubchem_fallback,
        cache_path=args.pubchem_cache,
    )
    sim_df.to_csv(out_path, index=False)

    display_cols = [
        "drug_name",
        "faers_search_name",
        "tanimoto_similarity",
        "similarity_band",
        "include_in_faers_matrix",
        "drug_type",
        "max_clinical_stage_for_target",
    ]
    display_cols = [c for c in display_cols if c in sim_df.columns]
    print("\n=== most similar known molecules ===")
    print(sim_df[display_cols].head(20).to_string(index=False))
    print(f"\nSaved {out_path} ({len(sim_df)} molecules scored)")

    n_available = int(sim_df["tanimoto_similarity"].notna().sum())
    n_total = len(sim_df)
    if n_available < n_total:
        print(f"Note: {n_total - n_available}/{n_total} rows had unavailable similarity, usually biologics or missing structures.")

    if args.recommendations:
        ctx = build_safety_context(sim_df, args.recommendations, top_n_each=args.top_context)
        ctx_path = out_path.with_name(out_path.stem + "_safety_context.csv")
        ctx.to_csv(ctx_path, index=False)
        print("\n=== safety context summary ===")
        if ctx.empty:
            print("No matching safety context rows found.")
        else:
            context_display = [
                "context_type",
                "event",
                "severity",
                "classification",
                "priority",
                "driver",
                "tanimoto_similarity",
                "similarity_band",
            ]
            context_display = [c for c in context_display if c in ctx.columns]
            print(ctx[context_display].head(30).to_string(index=False))
        print(f"\nSaved {ctx_path} ({len(ctx)} context rows)")

    print("\nNote: structural similarity and FAERS context are triage aids only; they are not causal safety predictions.")


if __name__ == "__main__":
    main()
