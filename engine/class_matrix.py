#!/usr/bin/env python3
"""
class_matrix.py
===============
Build an EVENT x DRUG disproportionality matrix for a class of drugs that share
a target (e.g. EGFR inhibitors), then tag each event as class-wide vs
molecule-specific.

This script is the FAERS class-level analysis layer. It can run in three ways:

1) Manual drugs, useful for quick tests:
    python class_matrix.py --drugs osimertinib erlotinib afatinib

2) Demo/canonical EGFR class, useful for the interview video:
    python class_matrix.py --mode demo --target EGFR --canonical-demo \
      --prefix faers_class_output/egfr/egfr_canonical_tki

3) Curated class CSV from curate_target_class.py, useful for the broader
   Open Targets-derived workflow:
    python class_matrix.py --mode auto --target EGFR \
      --prefix faers_class_output/egfr/egfr_open_targets_auto

Expected curated CSV columns:
    include_in_faers_matrix, faers_search_name, drug_name

Reuses the openFDA helpers from faers_signals.py, so keep that file in the same
folder. If yours is named faers_signal.py, either rename it to faers_signals.py
(recommended) or change the import line below.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from faers_signals import (
    total_reports,
    drug_total,
    drug_event_counts,
    event_total,
    disproportionality,
)

# Canonical approved small-molecule EGFR TKIs for the clean demo matrix.
# The broader auto/curated class should come from curate_target_class.py.
DEFAULT_EGFR_CANONICAL_TKIS = [
    "gefitinib",
    "erlotinib",
    "afatinib",
    "dacomitinib",
    "osimertinib",
]


def _as_bool(x: object) -> bool:
    """Robust bool parser for CSV values like True, TRUE, 1, yes."""
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {"true", "1", "yes", "y"}


def _clean_drug_name(x: object) -> str:
    """Normalize a drug name for openFDA querying and duplicate cleanup."""
    if pd.isna(x):
        return ""
    return " ".join(str(x).strip().lower().split())


def default_curated_csv_path(target: str, mode: str) -> Path:
    """Return the default curation output path for a target + mode."""
    t = target.strip().lower()
    return Path("curate_target_class_output") / t / f"{t}_curated_class_{mode}.csv"


def read_drugs_from_curated_csv(
    csv_path: str | Path,
    max_drugs: int | None = None,
) -> list[str]:
    """
    Read included FAERS search names from a curate_target_class.py output file.

    Uses faers_search_name if present; falls back to drug_name. Only rows with
    include_in_faers_matrix=True are used when that column exists.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Curated class CSV not found: {path}\n"
            "Run curate_target_class.py first, or pass --drug-class-csv with the correct path."
        )

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Curated class CSV is empty: {path}")

    if "include_in_faers_matrix" in df.columns:
        df = df[df["include_in_faers_matrix"].map(_as_bool)].copy()

    name_col = "faers_search_name" if "faers_search_name" in df.columns else "drug_name"
    if name_col not in df.columns:
        raise ValueError(
            f"{path} must contain either 'faers_search_name' or 'drug_name'. "
            f"Found columns: {list(df.columns)}"
        )

    drugs: list[str] = []
    seen: set[str] = set()
    for raw in df[name_col].tolist():
        drug = _clean_drug_name(raw)
        if not drug or drug in seen:
            continue
        seen.add(drug)
        drugs.append(drug)

    if max_drugs is not None:
        drugs = drugs[:max_drugs]

    if not drugs:
        raise ValueError(f"No included FAERS drugs found in {path}")
    return drugs


# --------------------------------------------------------------------------- #
#  Pure assembly logic (no network -> unit-testable)
# --------------------------------------------------------------------------- #
def build_matrices(
    n: int,
    drug_totals: dict[str, int],
    drug_counts: dict[str, dict[str, int]],
    event_totals: dict[str, int],
    events: Iterable[str],
    drugs: list[str],
    classwide_frac: float = 0.6,
):
    """
    Build ROR / signal / count matrices and a per-event class summary from
    already-fetched counts.

      n            : total reports (N)
      drug_totals  : {drug: a+b}
      drug_counts  : {drug: {event: a}}
      event_totals : {event: a+c}
      events       : matrix rows
      drugs        : matrix columns
    """
    ror, sig, cnt = {}, {}, {}
    for event in events:
        et = event_totals[event]
        ror_row, sig_row, cnt_row = {}, {}, {}
        for drug in drugs:
            a = drug_counts.get(drug, {}).get(event, 0)
            dt = drug_totals[drug]
            b, c = dt - a, et - a
            d = n - a - b - c
            s = disproportionality(a, b, c, d)
            ror_row[drug], sig_row[drug], cnt_row[drug] = s["ror"], s["is_signal"], a
        ror[event], sig[event], cnt[event] = ror_row, sig_row, cnt_row

    ror_df = pd.DataFrame(ror).T[drugs]
    sig_df = pd.DataFrame(sig).T[drugs]
    cnt_df = pd.DataFrame(cnt).T[drugs]

    n_drugs = len(drugs)
    classwide_cut = max(2, round(classwide_frac * n_drugs))
    summary = []
    for event in events:
        flags = sig_df.loc[event]
        hit_drugs = list(flags[flags].index)
        n_sig = len(hit_drugs)
        if n_sig == 0:
            label, driver = "none", ""
        elif n_sig >= classwide_cut:
            # Keep wording cautious: FAERS suggests patterns; it does not prove target causality.
            label, driver = "class-wide / possible target-class signal", ""
        elif n_sig == 1:
            label, driver = "molecule-specific / possible compound liability", hit_drugs[0]
        else:
            label, driver = "partial", ", ".join(hit_drugs)
        summary.append(
            {
                "event": event,
                "n_signal_drugs": n_sig,
                "n_drugs": n_drugs,
                "classification": label,
                "driver": driver,
            }
        )

    summary_df = (
        pd.DataFrame(summary)
        .sort_values(["n_signal_drugs", "event"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return ror_df, sig_df, cnt_df, summary_df


# --------------------------------------------------------------------------- #
#  Fetching (network)
# --------------------------------------------------------------------------- #
def fetch_class_data(
    drugs: list[str],
    top_k_per_drug: int = 40,
    min_count: int = 10,
    pause: float = 0.2,
):
    print("[*] total reports in FAERS ...", file=sys.stderr)
    n = total_reports()

    drug_totals, drug_counts = {}, {}
    for drug in drugs:
        print(f"[*] {drug}: report count + reaction profile ...", file=sys.stderr)
        dt = drug_total(drug)
        if dt == 0:
            print(f"    ! no reports for '{drug}', skipping", file=sys.stderr)
            continue
        drug_totals[drug] = dt
        drug_counts[drug] = drug_event_counts(drug)
        time.sleep(pause)

    kept = list(drug_totals.keys())
    if not kept:
        raise RuntimeError("No drugs had FAERS reports. Check drug names / FAERS search terms.")

    # Union of each kept drug's top-K most-reported events above the noise floor.
    union = set()
    for drug in kept:
        top = sorted(
            ((e, a) for e, a in drug_counts[drug].items() if a >= min_count),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k_per_drug]
        union.update(e for e, _ in top)
    events = sorted(union)

    print(f"[*] background totals for {len(events)} union events ...", file=sys.stderr)
    event_totals = {}
    for e in events:
        event_totals[e] = event_total(e)
        time.sleep(pause)

    return n, drug_totals, drug_counts, event_totals, events, kept


# --------------------------------------------------------------------------- #
#  Offline self-test of the assembly logic
# --------------------------------------------------------------------------- #
def selftest():
    n = 1_000_000
    drugs = ["drugA", "drugB"]
    drug_totals = {"drugA": 1000, "drugB": 1000}
    drug_counts = {
        "drugA": {"ON": 200, "MOL": 200, "QUIET": 1},
        "drugB": {"ON": 200, "MOL": 1, "QUIET": 1},
    }
    event_totals = {"ON": 400, "MOL": 250, "QUIET": 100000}
    events = ["ON", "MOL", "QUIET"]
    _, _, _, summary = build_matrices(
        n, drug_totals, drug_counts, event_totals, events, drugs, classwide_frac=0.6
    )
    print(summary.to_string(index=False))
    lab = dict(zip(summary.event, summary.classification))
    assert lab["ON"].startswith("class-wide"), lab["ON"]
    assert lab["MOL"].startswith("molecule-specific"), lab["MOL"]
    assert lab["QUIET"] == "none", lab["QUIET"]
    print("\nself-test passed ✓")


def resolve_drugs(args: argparse.Namespace) -> tuple[list[str], str]:
    """Resolve the class drug list from args and return (drugs, source_label)."""
    target = args.target.strip().upper()

    if args.canonical_demo:
        if target != "EGFR":
            raise ValueError("--canonical-demo is currently implemented only for --target EGFR")
        return DEFAULT_EGFR_CANONICAL_TKIS, "canonical EGFR TKI demo list"

    if args.drug_class_csv:
        path = Path(args.drug_class_csv)
        return read_drugs_from_curated_csv(path, args.max_drugs), f"curated CSV: {path}"

    if args.mode in {"demo", "auto"}:
        path = default_curated_csv_path(target, args.mode)
        return read_drugs_from_curated_csv(path, args.max_drugs), f"{args.mode} curated CSV: {path}"

    if args.drugs:
        return [_clean_drug_name(d) for d in args.drugs], "manual --drugs list"

    # Backward-compatible default.
    return DEFAULT_EGFR_CANONICAL_TKIS, "default EGFR canonical TKI list"


def main():
    ap = argparse.ArgumentParser(
        description="Build an event x drug FAERS disproportionality matrix for a target/class."
    )
    ap.add_argument(
        "--mode",
        choices=["manual", "demo", "auto"],
        default="manual",
        help=(
            "manual = use --drugs or default EGFR canonical list; "
            "demo/auto = read curate_target_class_output/<target>/<target>_curated_class_<mode>.csv"
        ),
    )
    ap.add_argument("--target", default="EGFR", help="target label used for default curated CSV paths")
    ap.add_argument("--drug-class-csv", help="curate_target_class.py output CSV to read included FAERS drugs from")
    ap.add_argument("--canonical-demo", action="store_true", help="use the 5 canonical approved EGFR TKIs for a clean demo matrix")
    ap.add_argument("--drugs", nargs="+", help="manual drug list; ignored if --drug-class-csv, --mode demo/auto, or --canonical-demo is used")
    ap.add_argument("--max-drugs", type=int, help="optional cap when reading a curated CSV, useful for quick tests")
    ap.add_argument("--top", type=int, default=40, help="top events per drug to union")
    ap.add_argument("--min-count", type=int, default=10, help="noise floor on report count")
    ap.add_argument("--classwide-frac", type=float, default=0.6, help="fraction of drugs needed to call an event class-wide")
    ap.add_argument("--prefix", default="egfr_class", help="output filename prefix")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    drugs, source_label = resolve_drugs(args)
    if not drugs:
        raise RuntimeError("No drugs resolved for FAERS class matrix.")

    out_dir = os.path.dirname(args.prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("\n=== FAERS class matrix setup ===")
    print(f"Target: {args.target.upper()}")
    print(f"Drug source: {source_label}")
    print(f"Drugs requested ({len(drugs)}): {', '.join(drugs)}")

    n, drug_totals, drug_counts, event_totals, events, kept_drugs = fetch_class_data(
        drugs, top_k_per_drug=args.top, min_count=args.min_count
    )

    ror_df, sig_df, cnt_df, summary = build_matrices(
        n,
        drug_totals,
        drug_counts,
        event_totals,
        events,
        kept_drugs,
        classwide_frac=args.classwide_frac,
    )

    ror_df.round(2).to_csv(f"{args.prefix}_ror_matrix.csv")
    sig_df.to_csv(f"{args.prefix}_signal_matrix.csv")
    cnt_df.to_csv(f"{args.prefix}_count_matrix.csv")
    summary.to_csv(f"{args.prefix}_event_classification.csv", index=False)

    print("\n=== event classification (top 25 by # signalling drugs) ===")
    print(summary.head(25).to_string(index=False))
    print(f"\nDrugs included after FAERS lookup: {', '.join(kept_drugs)}")
    if len(kept_drugs) < len(drugs):
        missing = [d for d in drugs if d not in kept_drugs]
        print(f"Drugs skipped because no FAERS reports were found: {', '.join(missing)}")
    print(f"Events scored: {len(events)}")
    print(f"Saved {args.prefix}_ror_matrix.csv (+ signal, count, classification CSVs)")


if __name__ == "__main__":
    main()
