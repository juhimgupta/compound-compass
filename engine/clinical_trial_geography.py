#!/usr/bin/env python3
"""
clinical_trial_geography.py
============================
Query ClinicalTrials.gov v2 API for registered trial locations for a selected
list of drug interventions and produce geography + trial-demographic CSVs for
the Streamlit app.

Usage:
    python clinical_trial_geography.py --target EGFR --drugs gefitinib erlotinib afatinib
    python clinical_trial_geography.py --target BRAF --drugs vemurafenib dabrafenib encorafenib
    python clinical_trial_geography.py --target EGFR --include-baseline-demographics

Outputs:
    trial_geography_output/<target>/<target>_trial_locations.csv
    trial_geography_output/<target>/<target>_country_summary.csv
    trial_geography_output/<target>/<target>_trial_demographics.csv
    trial_geography_output/<target>/<target>_demographic_summary.csv
    trial_geography_output/<target>/<target>_baseline_demographics.csv  # optional, if available

Important interpretation notes:
    - Geography shows registered ClinicalTrials.gov study-site locations.
    - Trial demographics here are registry-level fields: planned/actual enrollment,
      eligibility sex, age limits, healthy-volunteer eligibility, and stdAges.
    - Baseline demographic results are included only when posted in ClinicalTrials.gov
      results records; many studies do not have these data available.
    - None of these outputs represent adverse-event incidence or treatment safety rates.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

API_BASE = "https://clinicaltrials.gov/api/v2/studies"
PAGE_SIZE = 100
MAX_LOCS_PER_TRIAL = 150   # cap site rows per trial so large Phase 3 studies don't dominate

DEFAULT_EGFR_TKIS = [
    "gefitinib",
    "erlotinib",
    "afatinib",
    "dacomitinib",
    "osimertinib",
]

DEMOGRAPHIC_KEYWORDS = (
    "age", "sex", "gender", "race", "ethnic", "ethnicity", "ancestry", "region"
)


# ── Small helpers ─────────────────────────────────────────────────────────── #

def slugify(x: str) -> str:
    """Filesystem-safe target slug."""
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(x).strip()).strip("_") or "target"


def _list_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        return "; ".join(str(v) for v in x if v is not None)
    return str(x)


def _safe_json(x: Any) -> str:
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def _phases_str(design_mod: dict) -> str:
    phases = design_mod.get("phases") or []
    return ", ".join(str(p) for p in phases)


def _intervention_names(proto: dict) -> str:
    arm_mod = proto.get("armsInterventionsModule") or {}
    interventions = arm_mod.get("interventions") or []
    names = []
    for intr in interventions:
        name = intr.get("name")
        if name:
            names.append(str(name))
    return "; ".join(dict.fromkeys(names))


def _sponsor_name(proto: dict) -> str:
    sponsor_mod = proto.get("sponsorCollaboratorsModule") or {}
    lead = sponsor_mod.get("leadSponsor") or {}
    return str(lead.get("name") or "")


def _conditions(proto: dict) -> str:
    cond_mod = proto.get("conditionsModule") or {}
    return _list_str(cond_mod.get("conditions") or [])


# ── API helpers ────────────────────────────────────────────────────────────── #

def _get_page(drug: str, page_token: str | None, pause: float, condition: str | None = None) -> dict:
    params: dict[str, Any] = {
        "query.intr": drug,
        "pageSize": PAGE_SIZE,
        "format": "json",
    }
    if condition:
        params["query.cond"] = condition
    if page_token:
        params["pageToken"] = page_token

    time.sleep(pause)
    resp = requests.get(API_BASE, params=params, timeout=45)
    resp.raise_for_status()
    return resp.json()


def fetch_studies_for_drug(
    drug: str,
    max_studies: int = 300,
    pause: float = 0.6,
    condition: str | None = None,
) -> list[dict]:
    """Return raw study records for trials involving `drug`."""
    studies_all: list[dict] = []
    page_token: str | None = None

    while len(studies_all) < max_studies:
        try:
            data = _get_page(drug, page_token, pause=pause, condition=condition)
        except Exception as exc:
            print(f"  Warning: API error for '{drug}': {exc}")
            break

        studies = data.get("studies") or []
        if not studies:
            break

        remaining = max_studies - len(studies_all)
        studies_all.extend(studies[:remaining])

        page_token = data.get("nextPageToken")
        if not page_token or len(studies_all) >= max_studies:
            break

    return studies_all


# ── Record extraction ─────────────────────────────────────────────────────── #

def extract_location_records(drug: str, studies: list[dict]) -> list[dict]:
    """Return location-level records for a drug's studies."""
    records: list[dict] = []

    for study in studies:
        proto = study.get("protocolSection") or {}
        id_mod = proto.get("identificationModule") or {}
        st_mod = proto.get("statusModule") or {}
        des_mod = proto.get("designModule") or {}
        loc_mod = proto.get("contactsLocationsModule") or {}

        nct_id = id_mod.get("nctId", "")
        title = id_mod.get("briefTitle", "")
        status = st_mod.get("overallStatus", "")
        phase = _phases_str(des_mod)
        locs = loc_mod.get("locations") or []

        base = {
            "drug_name": drug,
            "nct_id": nct_id,
            "brief_title": title,
            "phase": phase,
            "overall_status": status,
        }

        if locs:
            for loc in locs[:MAX_LOCS_PER_TRIAL]:
                records.append({
                    **base,
                    "country": loc.get("country", ""),
                    "city": loc.get("city", ""),
                    "state": loc.get("state", ""),
                    "facility": loc.get("facility", ""),
                })
        else:
            records.append({**base, "country": "", "city": "", "state": "", "facility": ""})

    return records


def extract_trial_demographic_records(drug: str, studies: list[dict]) -> list[dict]:
    """Return one trial-level demographic/eligibility summary row per study."""
    rows: list[dict] = []

    for study in studies:
        proto = study.get("protocolSection") or {}
        results = study.get("resultsSection") or {}

        id_mod = proto.get("identificationModule") or {}
        st_mod = proto.get("statusModule") or {}
        des_mod = proto.get("designModule") or {}
        elig_mod = proto.get("eligibilityModule") or {}

        enrollment = des_mod.get("enrollmentInfo") or {}
        baseline_mod = results.get("baselineCharacteristicsModule") or {}
        has_baseline = bool(baseline_mod)

        rows.append({
            "drug_name": drug,
            "nct_id": id_mod.get("nctId", ""),
            "brief_title": id_mod.get("briefTitle", ""),
            "overall_status": st_mod.get("overallStatus", ""),
            "start_date": (st_mod.get("startDateStruct") or {}).get("date", ""),
            "completion_date": (st_mod.get("completionDateStruct") or {}).get("date", ""),
            "phase": _phases_str(des_mod),
            "study_type": des_mod.get("studyType", ""),
            "enrollment_count": enrollment.get("count", ""),
            "enrollment_type": enrollment.get("type", ""),
            "eligibility_sex": elig_mod.get("sex", ""),
            "minimum_age": elig_mod.get("minimumAge", ""),
            "maximum_age": elig_mod.get("maximumAge", ""),
            "std_ages": _list_str(elig_mod.get("stdAges") or []),
            "healthy_volunteers": elig_mod.get("healthyVolunteers", ""),
            "conditions": _conditions(proto),
            "interventions": _intervention_names(proto),
            "lead_sponsor": _sponsor_name(proto),
            "has_posted_baseline_demographics": has_baseline,
        })

    return rows


def extract_baseline_demographic_records(drug: str, studies: list[dict]) -> list[dict]:
    """Extract posted baseline demographic result measures when available.

    ClinicalTrials.gov baseline result schemas can vary. This function captures
    common Age/Sex/Gender/Race/Ethnicity-style measures in a long table.
    """
    rows: list[dict] = []

    for study in studies:
        proto = study.get("protocolSection") or {}
        results = study.get("resultsSection") or {}
        id_mod = proto.get("identificationModule") or {}
        nct_id = id_mod.get("nctId", "")
        title = id_mod.get("briefTitle", "")

        baseline_mod = results.get("baselineCharacteristicsModule") or {}
        if not baseline_mod:
            continue

        groups = baseline_mod.get("groups") or []
        group_names = {g.get("id", ""): g.get("title", "") for g in groups}
        measures = baseline_mod.get("measures") or []

        for measure in measures:
            m_title = str(measure.get("title") or "")
            m_title_l = m_title.lower()
            if not any(k in m_title_l for k in DEMOGRAPHIC_KEYWORDS):
                continue

            param_type = measure.get("paramType", "")
            dispersion_type = measure.get("dispersionType", "")
            unit = measure.get("unitOfMeasure", "")
            classes = measure.get("classes") or []

            # Some measures are nested class -> category -> measurements.
            # Others may have measurements directly. Handle both.
            direct_measurements = measure.get("measurements") or []
            for m in direct_measurements:
                gid = m.get("groupId", "")
                rows.append({
                    "drug_name": drug,
                    "nct_id": nct_id,
                    "brief_title": title,
                    "measure": m_title,
                    "class_title": "",
                    "category_title": "",
                    "group_id": gid,
                    "group_title": group_names.get(gid, ""),
                    "value": m.get("value", ""),
                    "unit": unit,
                    "param_type": param_type,
                    "dispersion_type": dispersion_type,
                    "raw_measurement": _safe_json(m),
                })

            for cls in classes:
                class_title = cls.get("title", "")
                categories = cls.get("categories") or []
                for cat in categories:
                    cat_title = cat.get("title", "")
                    measurements = cat.get("measurements") or []
                    for m in measurements:
                        gid = m.get("groupId", "")
                        rows.append({
                            "drug_name": drug,
                            "nct_id": nct_id,
                            "brief_title": title,
                            "measure": m_title,
                            "class_title": class_title,
                            "category_title": cat_title,
                            "group_id": gid,
                            "group_title": group_names.get(gid, ""),
                            "value": m.get("value", ""),
                            "unit": unit,
                            "param_type": param_type,
                            "dispersion_type": dispersion_type,
                            "raw_measurement": _safe_json(m),
                        })

    return rows


# ── Aggregation ────────────────────────────────────────────────────────────── #

def build_country_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per country: n_trials (unique NCTIds), n_drugs, drugs_tested."""
    if df.empty:
        return pd.DataFrame(columns=["country", "n_trials", "n_drugs", "drugs_tested"])

    df_loc = df[df["country"].fillna("").astype(str).str.strip().ne("")].copy()
    if df_loc.empty:
        return pd.DataFrame(columns=["country", "n_trials", "n_drugs", "drugs_tested"])

    deduped = df_loc.drop_duplicates(subset=["country", "nct_id", "drug_name"])

    country_trials = deduped.groupby("country")["nct_id"].nunique().rename("n_trials")
    country_drugs = deduped.groupby("country").agg(
        n_drugs=("drug_name", "nunique"),
        drugs_tested=("drug_name", lambda x: ", ".join(sorted(set(map(str, x))))),
    )

    return (
        country_trials.to_frame()
        .join(country_drugs)
        .reset_index()
        .sort_values("n_trials", ascending=False)
        .reset_index(drop=True)
    )


def build_demographic_summary(df_demo: pd.DataFrame) -> pd.DataFrame:
    """Small summary table by drug from trial-level demographic/eligibility rows."""
    cols = [
        "drug_name", "n_trials", "total_enrollment_reported", "n_trials_with_enrollment",
        "sex_values", "age_groups", "min_age_values", "max_age_values",
        "n_trials_with_posted_baseline_demographics",
    ]
    if df_demo.empty:
        return pd.DataFrame(columns=cols)

    df = df_demo.copy()
    df["enrollment_count_num"] = pd.to_numeric(df.get("enrollment_count"), errors="coerce")

    rows = []
    for drug, g in df.groupby("drug_name", dropna=False):
        rows.append({
            "drug_name": drug,
            "n_trials": int(g["nct_id"].nunique()),
            "total_enrollment_reported": int(g["enrollment_count_num"].dropna().sum()) if g["enrollment_count_num"].notna().any() else "",
            "n_trials_with_enrollment": int(g["enrollment_count_num"].notna().sum()),
            "sex_values": "; ".join(sorted(set(v for v in g["eligibility_sex"].dropna().astype(str) if v.strip()))),
            "age_groups": "; ".join(sorted(set(v for v in g["std_ages"].dropna().astype(str) if v.strip()))),
            "min_age_values": "; ".join(sorted(set(v for v in g["minimum_age"].dropna().astype(str) if v.strip()))),
            "max_age_values": "; ".join(sorted(set(v for v in g["maximum_age"].dropna().astype(str) if v.strip()))),
            "n_trials_with_posted_baseline_demographics": int(g["has_posted_baseline_demographics"].fillna(False).astype(bool).sum()),
        })

    return pd.DataFrame(rows, columns=cols).sort_values("n_trials", ascending=False).reset_index(drop=True)


# ── Main ───────────────────────────────────────────────────────────────────── #

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fetch ClinicalTrials.gov registered trial locations and trial demographics for selected drug interventions."
    )
    ap.add_argument(
        "--drugs", nargs="+", default=None,
        help="Drug/intervention names to query. Required for non-EGFR targets. If omitted for EGFR, defaults to canonical EGFR TKIs.",
    )
    ap.add_argument(
        "--target", default="EGFR",
        help="Target label used for output directory naming.",
    )
    ap.add_argument(
        "--condition", default=None,
        help="Optional ClinicalTrials.gov condition query, e.g. NSCLC or melanoma.",
    )
    ap.add_argument(
        "--max-studies", type=int, default=300,
        help="Maximum trials to fetch per drug.",
    )
    ap.add_argument(
        "--pause", type=float, default=0.6,
        help="Pause (s) between API pages — be polite.",
    )
    ap.add_argument(
        "--include-baseline-demographics", action="store_true",
        help="Also extract posted baseline demographic result measures when available. This can create a longer CSV.",
    )
    args = ap.parse_args()

    if args.drugs is None:
        if args.target.strip().upper() == "EGFR":
            args.drugs = DEFAULT_EGFR_TKIS
            print("No --drugs provided; using default EGFR TKI set.")
        else:
            raise ValueError(
                "No --drugs provided. For non-EGFR targets, pass selected comparator drugs, e.g. "
                "python clinical_trial_geography.py --target BRAF --drugs vemurafenib dabrafenib encorafenib"
            )

    target_slug = slugify(args.target)
    out_dir = Path("trial_geography_output") / target_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    locations_path = out_dir / f"{target_slug}_trial_locations.csv"
    summary_path = out_dir / f"{target_slug}_country_summary.csv"
    demographics_path = out_dir / f"{target_slug}_trial_demographics.csv"
    demographic_summary_path = out_dir / f"{target_slug}_demographic_summary.csv"
    baseline_path = out_dir / f"{target_slug}_baseline_demographics.csv"

    all_location_records: list[dict] = []
    all_demo_records: list[dict] = []
    all_baseline_records: list[dict] = []

    # Track seen (drug, NCT) pairs only for exact duplicate API pages, not to remove
    # multi-drug overlap across different selected comparators.
    for drug in args.drugs:
        print(f"Querying ClinicalTrials.gov for: {drug} ...")
        studies = fetch_studies_for_drug(
            drug,
            max_studies=args.max_studies,
            pause=args.pause,
            condition=args.condition,
        )
        print(f"  {len(studies)} studies fetched")

        loc_records = extract_location_records(drug, studies)
        demo_records = extract_trial_demographic_records(drug, studies)
        baseline_records = extract_baseline_demographic_records(drug, studies) if args.include_baseline_demographics else []

        with_loc = sum(1 for r in loc_records if r.get("country"))
        with_enrollment = sum(1 for r in demo_records if str(r.get("enrollment_count", "")).strip())
        print(f"  {len(loc_records)} location records ({with_loc} with country data)")
        print(f"  {len(demo_records)} demographic summary rows ({with_enrollment} with enrollment count)")
        if args.include_baseline_demographics:
            print(f"  {len(baseline_records)} posted baseline demographic rows")

        all_location_records.extend(loc_records)
        all_demo_records.extend(demo_records)
        all_baseline_records.extend(baseline_records)
        time.sleep(args.pause)

    df_locs = pd.DataFrame(all_location_records)
    df_locs.to_csv(locations_path, index=False)
    print(f"\nSaved: {locations_path}  ({len(df_locs)} rows)")

    df_country = build_country_summary(df_locs)
    df_country.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}  ({len(df_country)} countries)")

    df_demo = pd.DataFrame(all_demo_records)
    df_demo.to_csv(demographics_path, index=False)
    print(f"Saved: {demographics_path}  ({len(df_demo)} rows)")

    df_demo_summary = build_demographic_summary(df_demo)
    df_demo_summary.to_csv(demographic_summary_path, index=False)
    print(f"Saved: {demographic_summary_path}  ({len(df_demo_summary)} drugs)")

    if args.include_baseline_demographics:
        df_base = pd.DataFrame(all_baseline_records)
        df_base.to_csv(baseline_path, index=False)
        print(f"Saved: {baseline_path}  ({len(df_base)} rows)")

    print("\nNote: geography shows registered study-site locations, not adverse-event incidence.")
    print("Note: demographic outputs are registry-level trial/eligibility summaries unless posted baseline result data are available.")


if __name__ == "__main__":
    main()
