#!/usr/bin/env python3
"""
curate_target_class.py
======================
Create a reviewable drug-class curation table from Open Targets known/candidate
-drug output, then optionally export the included drug names for FAERS class
matrix analysis.

Why this exists
---------------
Open Targets can return many drugs/candidates for a target: small molecules,
antibodies, antibody-drug conjugates, combinations, preclinical candidates, and
non-US/investigational agents. For FAERS disproportionality comparisons, you
usually want a coherent comparison set.

This script turns the Open Targets output into a human-reviewable curation file:

    egfr_known_drugs.csv  ->  egfr_curated_class.csv

The output includes:
    - include_in_faers_matrix: True/False
    - curation_reason: why the row was included/excluded
    - faers_search_name: cleaned lowercase name to pass to openFDA/FAERS

Usage
-----
Default heuristic curation:
    python curate_target_class.py egfr_known_drugs.csv --target EGFR

Export the included drugs as a space-separated command argument list:
    python curate_target_class.py egfr_known_drugs.csv --target EGFR --print-drugs

Then run class matrix:
    python class_matrix.py --drugs gefitinib erlotinib afatinib dacomitinib osimertinib

Recommended demo framing
------------------------
"Open Targets proposes the target-associated drug universe. The curation layer
selects a coherent, reviewable comparison class before running FAERS, because
mixing antibodies, ADCs, small molecules, and investigational agents can make
safety comparisons misleading."
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


# A small curated override for the EGFR demo. These are the classic EGFR TKIs that
# make a coherent FAERS comparison class for your MVP. The script still works for
# other targets using the heuristic filter below.
TARGET_OVERRIDES = {
    "EGFR": {
        "include": {
            "gefitinib": "approved small-molecule EGFR TKI; coherent with EGFR TKI class comparison",
            "erlotinib": "approved small-molecule EGFR TKI; coherent with EGFR TKI class comparison",
            "afatinib": "approved small-molecule EGFR/HER-family TKI; coherent with EGFR TKI class comparison",
            "dacomitinib": "approved small-molecule EGFR/HER-family TKI; coherent with EGFR TKI class comparison",
            "osimertinib": "approved small-molecule EGFR TKI; coherent with EGFR TKI class comparison",
        },
        "exclude": {
            "cetuximab": "antibody; excluded from small-molecule TKI FAERS comparison",
            "panitumumab": "antibody; excluded from small-molecule TKI FAERS comparison",
            "necitumumab": "antibody; excluded from small-molecule TKI FAERS comparison",
            "amivantamab": "bispecific antibody; excluded from small-molecule TKI FAERS comparison",
            "depatuxizumab": "antibody/ADC-related program; excluded from small-molecule TKI FAERS comparison",
            "depatuxizumab mafodotin": "antibody-drug conjugate; excluded from small-molecule TKI FAERS comparison",
            "futuximab": "antibody; excluded from small-molecule TKI FAERS comparison",
            "zalutumumab": "antibody; excluded from small-molecule TKI FAERS comparison",
        },
    }
}


EXCLUDE_TYPE_KEYWORDS = (
    "antibody",
    "antibody drug conjugate",
    "adc",
    "protein",
    "oligonucleotide",
    "cell therapy",
    "gene therapy",
    "vaccine",
)

# Heuristic terms that often indicate a record is less likely to be a simple
# comparable small-molecule class member for FAERS analysis.
EXCLUDE_NAME_PATTERNS = (
    r"\bcombination\b",
    r"\+",
    r"/",
)

# Common salt / formulation suffixes that can cause duplicate FAERS queries, e.g.
# "erlotinib" and "erlotinib hydrochloride". For FAERS/openFDA searches, the
# parent/base drug name is usually the cleaner query. This is intentionally
# conservative: it only strips suffixes at the END of a drug name.
SALT_SUFFIXES = (
    "hydrochloride",
    "dihydrochloride",
    "hydrobromide",
    "mesylate",
    "maleate",
    "dimaleate",
    "tosylate",
    "ditosylate",
    "besylate",
    "fumarate",
    "succinate",
    "phosphate",
    "diphosphate",
    "sulfate",
    "sulphate",
    "citrate",
    "tartrate",
    "acetate",
    "lactate",
    "malate",
    "oxalate",
    "sodium",
    "potassium",
    "calcium",
    "magnesium",
    "anhydrous",
)


PREFERRED_COLUMNS = [
    "drug_name",
    "faers_search_name",
    "include_in_faers_matrix",
    "curation_reason",
    "drug_type",
    "max_clinical_stage_for_target",
    "n_diseases",
    "n_clinical_reports",
    "mechanism_of_action",
    "target_symbol",
]


def normalize_name(name: object) -> str:
    """Normalize a drug name for matching and FAERS/openFDA search."""
    if pd.isna(name):
        return ""
    text = str(name).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def strip_salt_suffix(name: object) -> str:
    """Return a conservative parent-drug query by removing terminal salt words."""
    text = normalize_name(name)
    if not text:
        return ""

    # Normalize punctuation that sometimes appears in formulation names.
    text = re.sub(r"[,;()]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Remove one or more salt suffixes at the end. This handles names like
    # "canertinib dihydrochloride" and "afatinib dimaleate".
    salt_pattern = r"(?:\s+(?:" + "|".join(map(re.escape, SALT_SUFFIXES)) + r"))+$"
    base = re.sub(salt_pattern, "", text).strip()
    return base or text


def is_small_molecule(row: pd.Series) -> bool:
    drug_type = normalize_name(row.get("drug_type", ""))
    return drug_type == "small molecule"


def stage_rank(stage: object) -> int:
    """Convert an Open Targets clinical stage label to a rough ordinal."""
    if pd.isna(stage):
        return -1

    s = str(stage).upper().strip()

    # Open Targets may use APPROVAL rather than APPROVED.
    # Treat approval/Phase 4 as above Phase 3.
    if "APPROVAL" in s or "APPROVED" in s or "PHASE_4" in s:
        return 4
    if "PHASE_3" in s:
        return 3
    if "PHASE_2" in s:
        return 2
    if "PHASE_1" in s:
        return 1
    return 0


def has_excluded_type(row: pd.Series) -> bool:
    drug_type = normalize_name(row.get("drug_type", ""))
    return any(k in drug_type for k in EXCLUDE_TYPE_KEYWORDS)


def has_excluded_name_pattern(name: str) -> bool:
    return any(re.search(pattern, name) for pattern in EXCLUDE_NAME_PATTERNS)


def curate_row(row: pd.Series, target: str | None = None) -> tuple[bool, str]:
    """Return (include, reason) for one Open Targets drug/candidate row."""
    name = normalize_name(row.get("drug_name", ""))
    target_key = target.upper() if target else ""

    # Hard-coded demo override: for EGFR, choose a clinically coherent small-molecule
    # TKI class. This is deliberately reviewable rather than pretending full automation.
    if target_key in TARGET_OVERRIDES:
        inc = TARGET_OVERRIDES[target_key]["include"]
        exc = TARGET_OVERRIDES[target_key]["exclude"]
        if name in inc:
            return True, inc[name]
        if name in exc:
            return False, exc[name]

    if not name:
        return False, "missing drug name"

    if has_excluded_type(row):
        return False, "excluded modality/type for this FAERS class comparison"

    if has_excluded_name_pattern(name):
        return False, "possible combination or non-simple drug name; requires manual review before FAERS"

    if not is_small_molecule(row):
        return False, "not labeled as small molecule; excluded for coherent small-molecule comparison"

    if stage_rank(row.get("max_clinical_stage_for_target")) < 3:
        return False, "clinical stage below Phase 3; excluded from default FAERS class comparison"

    return True, "small-molecule late-stage/approved target-associated candidate; included by heuristic"


def load_known_drugs(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    df = pd.read_csv(p)
    if "drug_name" not in df.columns:
        raise ValueError("Input CSV must contain a 'drug_name' column from open_targets_context.py")
    return df


def _safe_numeric(value: object) -> float:
    """Convert a possibly missing value to a sortable number."""
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _deduplicate_included_rows(out: pd.DataFrame) -> pd.DataFrame:
    """For included rows with the same FAERS parent query, keep one representative."""
    included = out.index[out["include_in_faers_matrix"]].tolist()
    by_query: dict[str, list[int]] = {}
    for idx in included:
        query = out.at[idx, "faers_search_name"]
        if query:
            by_query.setdefault(query, []).append(idx)

    for query, idxs in by_query.items():
        if len(idxs) <= 1:
            continue

        # Prefer the row with the highest stage, then the most clinical reports,
        # then the shortest drug name, which usually selects the parent name over
        # the salt name when both are present.
        def score(idx: int) -> tuple[int, float, int]:
            row = out.loc[idx]
            return (
                stage_rank(row.get("max_clinical_stage_for_target")),
                _safe_numeric(row.get("n_clinical_reports")),
                -len(normalize_name(row.get("drug_name", ""))),
            )

        keep_idx = max(idxs, key=score)
        keep_name = out.at[keep_idx, "drug_name"]
        for idx in idxs:
            if idx == keep_idx:
                continue
            out.at[idx, "include_in_faers_matrix"] = False
            out.at[idx, "curation_reason"] = (
                f"duplicate salt/formulation or alias for FAERS query '{query}'; "
                f"keeping representative row '{keep_name}'"
            )

    return out


def curate_known_drugs(df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    out = df.copy()

    # Use the parent/base drug name for FAERS searching so salts like
    # "erlotinib hydrochloride" collapse to "erlotinib".
    out["faers_search_name"] = out["drug_name"].map(strip_salt_suffix)

    decisions = out.apply(lambda row: curate_row(row, target=target), axis=1)
    out["include_in_faers_matrix"] = [d[0] for d in decisions]
    out["curation_reason"] = [d[1] for d in decisions]
    out = _deduplicate_included_rows(out)

    if target and "target_symbol" not in out.columns:
        out["target_symbol"] = target.upper()

    # Make important columns appear first, but preserve all Open Targets columns.
    first = [c for c in PREFERRED_COLUMNS if c in out.columns]
    rest = [c for c in out.columns if c not in first]
    out = out[first + rest]

    return out.sort_values(
        by=["include_in_faers_matrix", "max_clinical_stage_for_target", "n_clinical_reports"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)


def included_drugs(curated: pd.DataFrame) -> list[str]:
    return (
        curated.loc[curated["include_in_faers_matrix"], "faers_search_name"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a reviewable FAERS drug-class curation table from Open Targets known drugs."
    )
    parser.add_argument("known_drugs_csv", help="CSV produced by open_targets_context.py, e.g. egfr_known_drugs.csv")
    parser.add_argument("--target", default=None, help="Target symbol, e.g. EGFR. Enables target-specific demo overrides.")
    parser.add_argument("--out", default=None, help="Output curated CSV path. Default: <target>_curated_class.csv or curated_class.csv")
    parser.add_argument("--print-drugs", action="store_true", help="Print included FAERS drug names as a space-separated list")
    args = parser.parse_args()

    df = load_known_drugs(args.known_drugs_csv)
    curated = curate_known_drugs(df, target=args.target)

    if args.out:
        out_path = Path(args.out)
    elif args.target:
        out_path = Path(f"{args.target.lower()}_curated_class.csv")
    else:
        out_path = Path("curated_class.csv")

    curated.to_csv(out_path, index=False)

    drugs = included_drugs(curated)
    print("\n=== Curated target class ===")
    print(f"Input rows: {len(df)}")
    print(f"Included for FAERS matrix: {len(drugs)}")
    if drugs:
        print("Included drugs:")
        for d in drugs:
            print(f"  - {d}")
    else:
        print("No drugs were included by the default heuristic. Review the CSV and set include_in_faers_matrix manually.")

    print(f"\nSaved: {out_path}")

    preview_cols = [
        c for c in [
            "drug_name",
            "drug_type",
            "max_clinical_stage_for_target",
            "include_in_faers_matrix",
            "curation_reason",
        ]
        if c in curated.columns
    ]
    if preview_cols:
        print("\nPreview:")
        print(curated[preview_cols].head(20).to_string(index=False))

    if args.print_drugs and drugs:
        print("\nFAERS/class_matrix argument list:")
        print(" ".join(drugs))


if __name__ == "__main__":
    main()
