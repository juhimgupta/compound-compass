#!/usr/bin/env python3
"""
build_recommendations.py
========================
Translate FAERS class-level signal patterns into a reviewable recommendation
layer for pharma program teams.

This script consumes the outputs from class_matrix.py:
  - *_event_classification.csv
  - *_ror_matrix.csv
  - *_count_matrix.csv

It adds:
  - event category: safety toxicity vs disease progression vs resistance/efficacy
  - rough severity
  - priority score with clinically calibrated priority labels
  - interpretation
  - recommended follow-up action
  - candidate implication
  - future optimization extension
  - likely stakeholder owner

IMPORTANT:
  This is a triage layer, not a causal safety model. FAERS is spontaneous-report
  data and cannot estimate true incidence or prove causality. The goal is to
  help teams decide what to investigate next.

Example:
  python build_recommendations.py \
    --event-classification faers_class_output/egfr/egfr_canonical_tki_event_classification.csv \
    --ror-matrix faers_class_output/egfr/egfr_canonical_tki_ror_matrix.csv \
    --count-matrix faers_class_output/egfr/egfr_canonical_tki_count_matrix.csv \
    --target EGFR \
    --out recommendation_output/egfr/egfr_recommendations.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


# --------------------------------------------------------------------------- #
#  Keyword dictionaries
# --------------------------------------------------------------------------- #

DISEASE_PROGRESSION_KEYWORDS = [
    "METASTASES",
    "METASTASIS",
    "NEOPLASM PROGRESSION",
    "MALIGNANT NEOPLASM PROGRESSION",
    "DISEASE PROGRESSION",
    "CANCER PROGRESSION",
    "BREAST CANCER",
    "TUMOUR PROGRESSION",
    "TUMOR PROGRESSION",
]

RESISTANCE_OR_EFFICACY_KEYWORDS = [
    "DRUG RESISTANCE",
    "DRUG INEFFECTIVE",
    "THERAPEUTIC RESPONSE DECREASED",
    "TREATMENT FAILURE",
    "ACQUIRED GENE MUTATION",
    "EGFR GENE MUTATION",
    "GENE MUTATION",
    "BIOMARKER",
    "CARCINOEMBRYONIC ANTIGEN INCREASED",
]


BACKGROUND_OR_INFECTION_CONTEXT_KEYWORDS = [
    "COVID-19",
    "COVID 19",
    "SARS-COV-2",
]

REPORTING_OR_USE_KEYWORDS = [
    "OFF LABEL USE",
    "PRODUCT USE",
    "MEDICATION ERROR",
    "OVERDOSE",
    "UNDERDOSE",
    "WRONG DOSE",
    "INCORRECT DOSE",
    "DRUG ADMINISTRATION ERROR",
    "PRODUCT QUALITY",
]

# Safety/toxicity keywords. These are intentionally broad and pragmatic for an MVP.
SAFETY_TOXICITY_KEYWORDS = [
    "PNEUMONITIS",
    "INTERSTITIAL LUNG DISEASE",
    "PULMONARY",
    "PLEURAL EFFUSION",
    "DYSPNOEA",
    "PNEUMONIA",
    "COUGH",
    "THROMBOSIS",
    "EMBOLISM",
    "CEREBROVASCULAR ACCIDENT",
    "CARDIAC",
    "HEART",
    "MYELOSUPPRESSION",
    "NEUTROPENIA",
    "THROMBOCYTOPENIA",
    "PLATELET COUNT DECREASED",
    "WHITE BLOOD CELL COUNT DECREASED",
    "ANAEMIA",
    "ANEMIA",
    "HEPATIC",
    "LIVER",
    "RENAL",
    "KIDNEY",
    "DIARRHOEA",
    "DIARRHEA",
    "VOMITING",
    "NAUSEA",
    "STOMATITIS",
    "MOUTH ULCERATION",
    "DECREASED APPETITE",
    "WEIGHT DECREASED",
    "RASH",
    "DRY SKIN",
    "PRURITUS",
    "ACNE",
    "DERMATITIS",
    "PARONYCHIA",
    "NAIL DISORDER",
    "SKIN DISORDER",
    "DRUG ERUPTION",
    "PALMAR-PLANTAR",
    "FATIGUE",
    "ASTHENIA",
    "MALAISE",
    "PYREXIA",
    "PAIN",
    "HEADACHE",
    "DIZZINESS",
    "DEATH",
    "FALL",
    "ABDOMINAL DISTENSION",
    "ABDOMINAL PAIN",
    "ABDOMINAL PAIN UPPER",
    "GASTROINTESTINAL DISORDER",
    "URINARY TRACT INFECTION",
    "ERYTHEMA",
    "BLOOD CREATINE PHOSPHOKINASE INCREASED",
    "BLOOD CREATINE KINASE INCREASED",
    "HYPOTENSION",
    "INFUSION RELATED REACTION",
    "OXYGEN SATURATION DECREASED",
    "PARAESTHESIA",
    "PARESTHESIA",
    "HOSPITALISATION",
    "HOSPITALIZATION",
    "COVID-19",
    "COVID 19",
]

HIGH_SEVERITY_KEYWORDS = [
    "DEATH",
    "PNEUMONITIS",
    "INTERSTITIAL LUNG DISEASE",
    "PULMONARY EMBOLISM",
    "THROMBOSIS",
    "CEREBROVASCULAR ACCIDENT",
    "CARDIAC FAILURE",
    "HEART FAILURE",
    "HEPATIC FAILURE",
    "LIVER FAILURE",
    "RENAL FAILURE",
    "KIDNEY FAILURE",
    "MYELOSUPPRESSION",
    "NEUTROPENIA",
    "THROMBOCYTOPENIA",
    "PLATELET COUNT DECREASED",
    "SEPSIS",
    "HOSPITALISATION",
    "HOSPITALIZATION",
    "OXYGEN SATURATION DECREASED",
]

MEDIUM_SEVERITY_KEYWORDS = [
    "HEPATIC FUNCTION ABNORMAL",
    "PLEURAL EFFUSION",
    "PNEUMONIA",
    "DYSPNOEA",
    "STOMATITIS",
    "MOUTH ULCERATION",
    "DIARRHOEA",
    "DIARRHEA",
    "RASH",
    "DRUG ERUPTION",
    "DECREASED APPETITE",
    "WEIGHT DECREASED",
    "VOMITING",
    "NAUSEA",
    "PYREXIA",
    "FATIGUE",
    "ASTHENIA",
    "ABDOMINAL DISTENSION",
    "ABDOMINAL PAIN",
    "ABDOMINAL PAIN UPPER",
    "GASTROINTESTINAL DISORDER",
    "URINARY TRACT INFECTION",
    "BLOOD CREATINE PHOSPHOKINASE INCREASED",
    "BLOOD CREATINE KINASE INCREASED",
    "HYPOTENSION",
    "INFUSION RELATED REACTION",
    "PARAESTHESIA",
    "PARESTHESIA",
]

LOW_SEVERITY_KEYWORDS = [
    "DRY SKIN",
    "PRURITUS",
    "ACNE",
    "DERMATITIS ACNEIFORM",
    "NAIL DISORDER",
    "PARONYCHIA",
    "MUSCLE SPASMS",
    "HEADACHE",
    "DIZZINESS",
    "ERYTHEMA",
]


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def contains_any(text: str, keywords: Iterable[str]) -> bool:
    t = str(text).upper()
    return any(k in t for k in keywords)


def categorize_event(event: str) -> str:
    """Assign a pragmatic event category for interpretation."""
    e = str(event).upper()

    # Order matters: progression/resistance terms can otherwise look like safety terms.
    if contains_any(e, DISEASE_PROGRESSION_KEYWORDS):
        return "disease_progression"
    if contains_any(e, RESISTANCE_OR_EFFICACY_KEYWORDS):
        return "resistance_or_efficacy"
    if contains_any(e, BACKGROUND_OR_INFECTION_CONTEXT_KEYWORDS):
        return "reporting_or_use_context"
    if contains_any(e, REPORTING_OR_USE_KEYWORDS):
        return "reporting_or_use_context"
    if contains_any(e, SAFETY_TOXICITY_KEYWORDS):
        return "safety_toxicity"
    return "unknown"


def assign_severity(event: str, category: str) -> str:
    """Assign rough severity. This is for prioritization, not clinical adjudication."""
    e = str(event).upper()

    if category in {"disease_progression", "resistance_or_efficacy", "reporting_or_use_context"}:
        # These can be clinically serious or operationally relevant, but are not interpreted as direct toxicity.
        return "contextual"

    if contains_any(e, HIGH_SEVERITY_KEYWORDS):
        return "high"
    if contains_any(e, MEDIUM_SEVERITY_KEYWORDS):
        return "medium"
    if contains_any(e, LOW_SEVERITY_KEYWORDS):
        return "low"
    return "unknown"


def classification_bucket(classification: str) -> str:
    c = str(classification).lower()
    if "class-wide" in c:
        return "class_wide"
    if "molecule-specific" in c:
        return "molecule_specific"
    if "partial" in c:
        return "partial"
    if "none" in c:
        return "none"
    return "unknown"


def safe_numeric_frame(path: str | Path) -> pd.DataFrame:
    """Read a matrix CSV with events as index and numeric drug columns."""
    df = pd.read_csv(path, index_col=0)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def event_matrix_metrics(event: str, ror_df: pd.DataFrame, count_df: pd.DataFrame) -> dict:
    """Compute matrix-derived metrics for an event."""
    metrics = {
        "max_ror": 0.0,
        "mean_ror": 0.0,
        "total_class_reports": 0,
        "max_count_drug": "",
        "max_count": 0,
    }

    if event in ror_df.index:
        ror_values = pd.to_numeric(ror_df.loc[event], errors="coerce").fillna(0)
        metrics["max_ror"] = round(float(ror_values.max()), 3)
        metrics["mean_ror"] = round(float(ror_values.mean()), 3)

    if event in count_df.index:
        count_values = pd.to_numeric(count_df.loc[event], errors="coerce").fillna(0)
        metrics["total_class_reports"] = int(count_values.sum())
        metrics["max_count"] = int(count_values.max())
        if len(count_values) > 0:
            metrics["max_count_drug"] = str(count_values.idxmax())

    return metrics


def priority_score(row: pd.Series) -> int:
    """Transparent triage score used to sort recommendations.

    This score is intentionally used for sorting only. The final priority label is
    calibrated separately so that low-severity class effects and contextual
    resistance/progression terms do not get presented as high-priority toxicity
    just because their ROR is large.
    """
    bucket = classification_bucket(row.get("classification", ""))
    category = row.get("event_category", "unknown")
    severity = row.get("severity", "unknown")
    max_ror = float(row.get("max_ror", 0) or 0)
    total_reports = int(row.get("total_class_reports", 0) or 0)

    score = 0

    # Pattern score
    if bucket == "class_wide":
        score += 3
    elif bucket in {"molecule_specific", "partial"}:
        score += 2

    # Severity score
    if severity == "high":
        score += 3
    elif severity == "medium":
        score += 2
    elif severity == "low":
        score += 1
    elif severity == "contextual":
        score += 1

    # Signal strength score. Cap the ROR contribution for low-severity events so
    # expected/manageable EGFR effects like acne or paronychia do not outrank
    # serious pulmonary or hematologic concerns solely due to very high ROR.
    if max_ror >= 10:
        signal_points = 3
    elif max_ror >= 5:
        signal_points = 2
    elif max_ror >= 2:
        signal_points = 1
    else:
        signal_points = 0
    if category == "safety_toxicity" and severity == "low":
        signal_points = min(signal_points, 1)
    score += signal_points

    # Evidence count score
    if total_reports >= 500:
        score += 2
    elif total_reports >= 100:
        score += 1

    # Avoid over-prioritizing non-toxicity context as safety risk.
    if category == "disease_progression":
        score -= 2
    elif category in {"resistance_or_efficacy", "reporting_or_use_context"}:
        score -= 1

    return max(score, 0)


def raw_priority_label(score: int) -> str:
    if score >= 8:
        return "high"
    if score >= 5:
        return "medium"
    return "low"


def cap_priority(priority: str, max_priority: str) -> str:
    ranks = {"low": 1, "medium": 2, "high": 3}
    reverse = {v: k for k, v in ranks.items()}
    return reverse[min(ranks.get(priority, 1), ranks.get(max_priority, 3))]


def calibrated_priority_label(row: pd.Series) -> str:
    """Convert priority score to a clinically calibrated label."""
    priority = raw_priority_label(int(row.get("priority_score", 0) or 0))
    category = row.get("event_category", "unknown")
    severity = row.get("severity", "unknown")
    bucket = classification_bucket(row.get("classification", ""))

    # Contextual oncology/resistance terms are important, but should not appear
    # as high-priority toxicity concerns in the primary triage table.
    if category in {"disease_progression", "resistance_or_efficacy"}:
        priority = cap_priority(priority, "medium")

    # Product-use/reporting terms are usually operational review items.
    if category == "reporting_or_use_context":
        priority = cap_priority(priority, "low")

    # Low-severity EGFR class effects can be biologically characteristic, but are
    # usually manageable and should be framed as expected/monitorable class effects.
    if category == "safety_toxicity" and severity == "low":
        priority = cap_priority(priority, "medium")

    # Unknown events should not be high unless they are broad class-wide signals
    # with enough evidence to justify SME review.
    if category == "unknown" and bucket != "class_wide":
        priority = cap_priority(priority, "medium")

    return priority


def stakeholder_owner(category: str, bucket: str, severity: str) -> str:
    if category == "safety_toxicity":
        if bucket == "molecule_specific":
            return "Safety pharmacology / medicinal chemistry"
        if severity == "high":
            return "Safety / translational medicine / clinical development"
        return "Safety / clinical development"
    if category == "disease_progression":
        return "Clinical development / safety epidemiology"
    if category == "resistance_or_efficacy":
        return "Translational medicine / biomarker strategy"
    if category == "reporting_or_use_context":
        return "Pharmacovigilance operations"
    return "Program team / subject-matter expert review"


def build_interpretation(row: pd.Series, target: str) -> str:
    event = row["event"]
    category = row["event_category"]
    bucket = classification_bucket(row["classification"])
    driver = str(row.get("driver", "") or "")
    n_sig = int(row.get("n_signal_drugs", 0) or 0)
    n_drugs = int(row.get("n_drugs", 0) or 0)

    if category == "safety_toxicity":
        if bucket == "class_wide":
            return (
                f"{event} appears as a signal across {n_sig}/{n_drugs} comparator drugs, "
                f"suggesting a possible {target} target-class or pathway-level safety pattern. "
                "FAERS cannot prove causality, so this should be treated as a follow-up signal."
            )
        if bucket == "molecule_specific":
            return (
                f"{event} appears primarily for {driver}, suggesting a possible compound-specific "
                "liability, exposure difference, indication mix, or reporting artifact rather than a whole-class conclusion."
            )
        if bucket == "partial":
            return (
                f"{event} appears in a subset of comparator drugs, which may reflect subclass biology, "
                "drug generation, exposure, indication differences, or data sparsity."
            )
        return f"{event} is categorized as a safety/toxicity event but is not a disproportionality signal in this class matrix."

    if category == "disease_progression":
        return (
            f"{event} is more consistent with disease progression or oncology treatment context than direct drug toxicity. "
            "It should be interpreted with indication, disease stage, and line of therapy in mind."
        )

    if category == "resistance_or_efficacy":
        return (
            f"{event} is more consistent with resistance, efficacy loss, or biomarker context than direct toxicity. "
            "It is useful for translational strategy but should not be treated as a safety liability by itself."
        )

    if category == "reporting_or_use_context":
        return (
            f"{event} appears related to reporting, product use, or administration context. "
            "It should be reviewed operationally before making biological conclusions."
        )

    return (
        f"{event} does not match a predefined interpretation category. "
        "Flag for expert review before using it in candidate decisions."
    )


def build_recommended_action(row: pd.Series, target: str) -> str:
    category = row["event_category"]
    severity = row["severity"]
    bucket = classification_bucket(row["classification"])
    driver = str(row.get("driver", "") or "")

    if category == "safety_toxicity":
        if bucket == "class_wide" and severity == "high":
            return (
                "Prioritize class-level safety review. Review comparator labels and literature, assess target/pathway biology, "
                "and define monitoring, exclusion, or translational biomarker strategy before advancing an internal candidate."
            )
        if bucket == "class_wide":
            return (
                "Treat as a likely manageable or monitorable class-pattern signal. Define mitigation, dose-modification, "
                "patient counseling, or routine monitoring considerations."
            )
        if bucket == "molecule_specific":
            return (
                f"Investigate the driver drug ({driver}) as a potential compound-specific liability. Compare chemistry, "
                "off-target profile, PK/exposure, dose, formulation, and treated population against backup compounds."
            )
        if bucket == "partial":
            return (
                "Investigate why the signal appears in only part of the class. Compare drug generation, selectivity, exposure, "
                "indication, geography, and reporting volume before generalizing to the whole target class."
            )
        return "Monitor but deprioritize relative to events with stronger class-level or molecule-specific signal patterns."

    if category == "disease_progression":
        return (
            "Do not interpret as direct toxicity. Stratify FAERS interpretation by disease, disease stage, line of therapy, "
            "and comparator population before using it in safety decision-making."
        )

    if category == "resistance_or_efficacy":
        return (
            "Treat as efficacy/resistance context. Review resistance mutations, biomarker strategy, patient selection, "
            "and whether the internal program should include mutation-panel experiments."
        )

    if category == "reporting_or_use_context":
        return (
            "Review as a pharmacovigilance operations or product-use signal. Do not assign biological meaning unless supported "
            "by additional clinical or mechanistic evidence."
        )

    return "Flag for subject-matter expert review; insufficient rule-based context to recommend a specific action."


def build_candidate_implication(row: pd.Series, target: str) -> str:
    category = row["event_category"]
    bucket = classification_bucket(row["classification"])
    severity = row["severity"]
    driver = str(row.get("driver", "") or "")

    if category == "safety_toxicity":
        if bucket == "class_wide":
            return (
                f"For a new internal {target} candidate, this signal should be considered during candidate risk assessment "
                "even before large clinical exposure."
            )
        if bucket == "molecule_specific":
            return (
                f"This does not automatically implicate the whole {target} class. Assess whether the internal candidate resembles "
                f"{driver} in structure, off-target profile, exposure, or clinical setting."
            )
        if bucket == "partial":
            return (
                "Candidate relevance depends on whether the internal molecule resembles the signaling subset in modality, generation, "
                "selectivity, exposure, or indication."
            )
        return "Low immediate candidate relevance from this matrix unless other evidence supports concern."

    if category == "disease_progression":
        return "Avoid using this as a direct safety penalty for an internal candidate; interpret in the disease and treatment context."

    if category == "resistance_or_efficacy":
        return (
            f"For a new {target} candidate, this may inform mutation coverage, resistance experiments, and biomarker strategy "
            "rather than safety monitoring alone."
        )

    if category == "reporting_or_use_context":
        return "Candidate relevance is indirect; review operational context before assigning scientific significance."

    return "Candidate implication requires expert review."


def build_future_optimization_extension(row: pd.Series) -> str:
    category = row["event_category"]
    bucket = classification_bucket(row["classification"])

    if category == "safety_toxicity" and bucket == "molecule_specific":
        return (
            "Future extension: use internal compound descriptors, off-target predictions, potency, ADMET, and uncertainty "
            "estimates to prioritize backup compounds or analogs via Bayesian optimization."
        )
    if category == "safety_toxicity" and bucket == "class_wide":
        return (
            "Not primarily a molecule-optimization problem alone; future work should combine target biology, patient selection, "
            "monitoring strategy, and candidate-specific exposure/off-target data."
        )
    if category == "resistance_or_efficacy":
        return (
            "Future extension: optimize candidates against resistance mutation panels while balancing potency, selectivity, ADMET, "
            "and uncertainty."
        )
    if category == "disease_progression":
        return "No compound optimization recommendation from FAERS alone; improve disease/indication stratification first."
    return "Future extension depends on additional candidate-specific molecular and assay data."


# --------------------------------------------------------------------------- #
#  Main build function
# --------------------------------------------------------------------------- #

def build_recommendations(
    event_classification_csv: str | Path,
    ror_matrix_csv: str | Path,
    count_matrix_csv: str | Path,
    target: str,
) -> pd.DataFrame:
    events = pd.read_csv(event_classification_csv)
    ror_df = safe_numeric_frame(ror_matrix_csv)
    count_df = safe_numeric_frame(count_matrix_csv)

    required = {"event", "n_signal_drugs", "n_drugs", "classification", "driver"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"event classification CSV is missing required columns: {sorted(missing)}")

    rows = []
    for _, base in events.iterrows():
        row = base.to_dict()
        event = row["event"]
        row.update(event_matrix_metrics(event, ror_df, count_df))
        row["event_category"] = categorize_event(event)
        row["severity"] = assign_severity(event, row["event_category"])
        row["priority_score"] = priority_score(pd.Series(row))
        row["priority"] = calibrated_priority_label(pd.Series(row))
        bucket = classification_bucket(row["classification"])
        row["stakeholder_owner"] = stakeholder_owner(row["event_category"], bucket, row["severity"])
        row["interpretation"] = build_interpretation(pd.Series(row), target)
        row["recommended_action"] = build_recommended_action(pd.Series(row), target)
        row["candidate_implication"] = build_candidate_implication(pd.Series(row), target)
        row["future_optimization_extension"] = build_future_optimization_extension(pd.Series(row))
        rows.append(row)

    out = pd.DataFrame(rows)

    preferred_cols = [
        "event",
        "event_category",
        "severity",
        "classification",
        "n_signal_drugs",
        "n_drugs",
        "driver",
        "max_ror",
        "mean_ror",
        "total_class_reports",
        "max_count_drug",
        "max_count",
        "priority_score",
        "priority",
        "interpretation",
        "recommended_action",
        "candidate_implication",
        "stakeholder_owner",
        "future_optimization_extension",
    ]
    remaining = [c for c in out.columns if c not in preferred_cols]
    out = out[[c for c in preferred_cols if c in out.columns] + remaining]

    priority_rank = {"high": 3, "medium": 2, "low": 1}
    severity_rank = {"high": 4, "medium": 3, "low": 2, "contextual": 1, "unknown": 0}
    category_rank = {
        "safety_toxicity": 4,
        "resistance_or_efficacy": 3,
        "disease_progression": 2,
        "reporting_or_use_context": 1,
        "unknown": 0,
    }
    out["_priority_rank"] = out["priority"].map(priority_rank).fillna(0)
    out["_severity_rank"] = out["severity"].map(severity_rank).fillna(0)
    out["_category_rank"] = out["event_category"].map(category_rank).fillna(0)
    out = out.sort_values(
        ["_priority_rank", "_severity_rank", "_category_rank", "priority_score", "n_signal_drugs", "max_ror"],
        ascending=[False, False, False, False, False, False],
    ).drop(columns=["_priority_rank", "_severity_rank", "_category_rank"]).reset_index(drop=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build reviewable recommendations from FAERS class-level signal outputs.")
    p.add_argument("--event-classification", required=True, help="*_event_classification.csv from class_matrix.py")
    p.add_argument("--ror-matrix", required=True, help="*_ror_matrix.csv from class_matrix.py")
    p.add_argument("--count-matrix", required=True, help="*_count_matrix.csv from class_matrix.py")
    p.add_argument("--target", default="target", help="Target symbol/name for recommendation text, e.g. EGFR")
    p.add_argument("--out", required=True, help="Output recommendations CSV")
    p.add_argument("--top", type=int, default=25, help="Rows to print to terminal")
    args = p.parse_args()

    recs = build_recommendations(args.event_classification, args.ror_matrix, args.count_matrix, args.target.upper())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    recs.to_csv(out_path, index=False)

    print("\n=== top recommendations ===")
    display_cols = [
        "event",
        "event_category",
        "severity",
        "classification",
        "n_signal_drugs",
        "max_ror",
        "total_class_reports",
        "priority",
        "stakeholder_owner",
    ]
    print(recs[display_cols].head(args.top).to_string(index=False))
    print(f"\nSaved {out_path} ({len(recs)} events)")
    print("Note: recommendations are triage guidance from FAERS patterns, not causal safety conclusions.")


if __name__ == "__main__":
    main()
