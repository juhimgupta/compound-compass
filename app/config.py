from __future__ import annotations
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
ENGINE_DIR = PROJECT_DIR / "engine"
LOGO_PATH = PROJECT_DIR / "assets" / "logo.png"

APP_TITLE = "Compound Compass"
APP_SUBTITLE = "Therapeutic Candidate Intelligence Platform"

TARGET_CONFIGS = {
    "EGFR": {
        "slug": "egfr",
        "default_indication": "Non-small cell lung cancer",
        "verified_comparators": ["gefitinib", "erlotinib", "afatinib", "dacomitinib", "osimertinib"],
        "comparator_class_label": "verified EGFR TKI comparator set",
    },
    "BRAF": {
        "slug": "braf",
        "default_indication": "Melanoma",
        "verified_comparators": [],
        "comparator_class_label": "auto-selected clinical small-molecule comparators",
    },
    "KRAS": {
        "slug": "kras",
        "default_indication": "Solid tumors",
        "verified_comparators": [],
        "comparator_class_label": "auto-selected clinical small-molecule comparators",
    },
}
TARGET_OPTIONS = list(TARGET_CONFIGS.keys()) + ["Other / custom target"]

PAGE_SIZE = 5
MAX_FAERS_COMPARATORS = 5

_STAGE_RANK = {
    "phase_1": 1, "phase_1_2": 1,
    "phase_2": 2, "phase_2_3": 2,
    "phase_3": 3,
    "phase_4": 4,
    "approval": 5,
}
_MIN_STAGE_MAP = {
    "Any": 0,
    "Phase II+": 2,
    "Phase III+ (late-stage)": 3,
    "Approved only": 5,
}
