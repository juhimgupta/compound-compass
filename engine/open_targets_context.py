#!/usr/bin/env python3
"""
open_targets_context.py
=======================
Fetch target/class context from the current Open Targets Platform GraphQL API.

Role in the project:
  - Open Targets = target biology / clinical-candidate context
  - FAERS/openFDA = real-world safety signal context
  - Your app/notebook joins them into decision support

Example:
  python open_targets_context.py EGFR

Outputs:
  egfr_target_context.json
  egfr_target_summary.json
  egfr_associated_diseases.csv
  egfr_tractability.csv
  egfr_known_drugs.csv
  egfr_safety_liabilities.csv

Notes:
  - The Open Targets schema changed: target.knownDrugs was replaced by
    target.drugAndClinicalCandidates in the current API schema.
  - This script intentionally keeps the GraphQL query conservative and flattens
    clinical candidates into a human-reviewable drug-class CSV.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

BASE_URL = "https://api.platform.opentargets.org/api/v4/graphql"
CHEMBL_MOLECULE_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"

TARGET_SYMBOL_TO_ENSEMBL = {
    "EGFR": "ENSG00000146648",
    "JAK1": "ENSG00000162434",
    "JAK2": "ENSG00000096968",
    "BTK": "ENSG00000010671",
    "CDK4": "ENSG00000135446",
    "CDK6": "ENSG00000105810",
    "PIK3CA": "ENSG00000121879",
    "BRAF": "ENSG00000157764",
    "KRAS": "ENSG00000133703",
}

TARGET_CONTEXT_QUERY = """
query targetContext($ensemblId: String!, $diseasePage: Pagination!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    approvedName
    biotype
    functionDescriptions
    subcellularLocations {
      location
      source
      labelSL
      termSL
    }
    proteinIds {
      id
      source
    }
    tractability {
      label
      modality
      value
    }
    safetyLiabilities {
      event
      eventId
      datasource
      literature
      url
      biosamples {
        tissueLabel
        tissueId
        cellLabel
        cellFormat
      }
      effects {
        dosing
        direction
      }
      studies {
        name
        type
        description
      }
    }
    associatedDiseases(page: $diseasePage) {
      count
      rows {
        score
        disease {
          id
          name
          therapeuticAreas {
            id
            name
          }
        }
        datatypeScores {
          id
          score
        }
      }
    }
    drugAndClinicalCandidates {
      count
      rows {
        id
        maxClinicalStage
        drug {
          id
          name
          drugType
          maximumClinicalStage
          description
          tradeNames
          synonyms
          crossReferences {
            source
            ids
          }
          parentMolecule {
            id
            name
          }
        }
        diseases {
          diseaseFromSource
          disease {
            id
            name
          }
        }
        clinicalReports {
          id
          type
          source
          clinicalStage
          trialPhase
          phaseFromSource
          trialOverallStatus
          title
          url
          year
        }
      }
    }
  }
}
"""


def _post_graphql(query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    """POST a GraphQL query and return data. Print GraphQL errors clearly."""
    response = requests.post(
        BASE_URL,
        json={"query": query, "variables": variables},
        timeout=60,
        headers={"Content-Type": "application/json"},
    )

    # Open Targets often returns useful GraphQL errors in the body even with 400.
    try:
        payload = response.json()
    except ValueError:
        response.raise_for_status()
        raise

    if response.status_code >= 400 or payload.get("errors"):
        details = json.dumps(payload.get("errors", payload), indent=2)
        raise RuntimeError(
            f"Open Targets GraphQL request failed with HTTP {response.status_code}.\n"
            f"Details:\n{details}"
        )

    return payload["data"]


def get_target_context(symbol_or_ensembl: str, disease_limit: int = 25, drug_limit: int = 100) -> Dict[str, Any]:
    """Fetch target context for a curated symbol or an Ensembl target ID."""
    query = symbol_or_ensembl.strip().upper()
    ensembl_id = TARGET_SYMBOL_TO_ENSEMBL.get(query, symbol_or_ensembl.strip())
    variables = {
        "ensemblId": ensembl_id,
        "diseasePage": {"index": 0, "size": disease_limit},
    }
    data = _post_graphql(TARGET_CONTEXT_QUERY, variables)
    target = data.get("target")
    if not target:
        raise ValueError(
            f"No Open Targets target returned for '{symbol_or_ensembl}'. "
            "Try an Ensembl target ID, e.g. ENSG00000146648 for EGFR."
        )

    # The current target.drugAndClinicalCandidates endpoint has no page arg.
    # Keep drug_limit as a user-facing output cap so the CSV stays readable.
    candidates = target.get("drugAndClinicalCandidates") or {}
    rows = candidates.get("rows") or []
    candidates["rows"] = rows[:drug_limit]
    target["drugAndClinicalCandidates"] = candidates
    return target


def _safe_disease_name(item: Dict[str, Any]) -> str:
    disease = (item or {}).get("disease") or {}
    return disease.get("name") or item.get("diseaseFromSource") or ""


def flatten_associated_diseases(target: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for row in (target.get("associatedDiseases") or {}).get("rows") or []:
        disease = row.get("disease") or {}
        therapeutic_areas = disease.get("therapeuticAreas") or []
        datatype_scores = row.get("datatypeScores") or []
        rows.append({
            "target_id": target.get("id"),
            "target_symbol": target.get("approvedSymbol"),
            "disease_id": disease.get("id"),
            "disease_name": disease.get("name"),
            "overall_score": row.get("score"),
            "therapeutic_areas": "; ".join(t.get("name", "") for t in therapeutic_areas),
            "datatype_scores": json.dumps({d.get("id"): d.get("score") for d in datatype_scores}),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("overall_score", ascending=False, ignore_index=True)
    return df


def flatten_tractability(target: Dict[str, Any]) -> pd.DataFrame:
    rows = [{
        "target_id": target.get("id"),
        "target_symbol": target.get("approvedSymbol"),
        "label": row.get("label"),
        "modality": row.get("modality"),
        "value": row.get("value"),
    } for row in target.get("tractability") or []]
    return pd.DataFrame(rows)



def _load_json_cache(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _write_json_cache(path: Optional[str], cache: Dict[str, Any]) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def _chembl_structure_from_id(chembl_id: Optional[str], cache: Dict[str, Any], pause: float = 0.05) -> Dict[str, Any]:
    """
    Fetch structure fields from ChEMBL for a ChEMBL molecule ID.

    Open Targets exposes ChEMBL drug IDs and cross-references, but the current
    Open Targets GraphQL Drug type does not expose SMILES directly. ChEMBL's
    molecule endpoint provides molecule_structures.canonical_smiles for many
    small molecules. Antibodies/ADCs/biologics commonly have no small-molecule
    SMILES, which is expected.
    """
    if not chembl_id or not str(chembl_id).startswith("CHEMBL"):
        return {
            "canonical_smiles": None,
            "standard_inchi_key": None,
            "structure_source": "not_chembl_id",
            "structure_lookup_status": "skipped",
        }

    chembl_id = str(chembl_id)
    if chembl_id in cache:
        return cache[chembl_id]

    out = {
        "canonical_smiles": None,
        "standard_inchi_key": None,
        "structure_source": "ChEMBL",
        "structure_lookup_status": "not_found",
    }
    try:
        r = requests.get(CHEMBL_MOLECULE_URL.format(chembl_id=chembl_id), timeout=30)
        if r.status_code == 200:
            payload = r.json()
            structures = payload.get("molecule_structures") or {}
            out = {
                "canonical_smiles": structures.get("canonical_smiles"),
                "standard_inchi_key": structures.get("standard_inchi_key"),
                "structure_source": "ChEMBL",
                "structure_lookup_status": "ok" if structures.get("canonical_smiles") else "no_small_molecule_structure",
            }
        elif r.status_code == 404:
            out["structure_lookup_status"] = "not_found"
        else:
            out["structure_lookup_status"] = f"http_{r.status_code}"
    except requests.RequestException as exc:
        out["structure_lookup_status"] = f"request_error: {exc.__class__.__name__}"

    cache[chembl_id] = out
    time.sleep(pause)
    return out


def _format_cross_refs(drug: Dict[str, Any]) -> str:
    refs = drug.get("crossReferences") or []
    parts = []
    for ref in refs:
        source = ref.get("source")
        ids = ref.get("ids") or []
        if source and ids:
            parts.append(f"{source}:" + ",".join(ids[:10]))
    return "; ".join(parts)


def flatten_known_drugs(target: Dict[str, Any], enrich_chembl_smiles: bool = True, chembl_cache: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Flatten current Open Targets drugAndClinicalCandidates into a file named
    *_known_drugs.csv so downstream project code can stay intuitive.
    """
    rows: List[Dict[str, Any]] = []
    chembl_cache = chembl_cache if chembl_cache is not None else {}
    for row in (target.get("drugAndClinicalCandidates") or {}).get("rows") or []:
        drug = row.get("drug") or {}
        structure = (_chembl_structure_from_id(drug.get("id"), chembl_cache)
                     if enrich_chembl_smiles else {
                         "canonical_smiles": None,
                         "standard_inchi_key": None,
                         "structure_source": "not_requested",
                         "structure_lookup_status": "skipped",
                     })
        parent = drug.get("parentMolecule") or {}
        disease_items = row.get("diseases") or []
        reports = row.get("clinicalReports") or []
        disease_names = sorted({name for name in (_safe_disease_name(x) for x in disease_items) if name})
        report_types = sorted({r.get("type") for r in reports if r.get("type")})
        statuses = sorted({r.get("trialOverallStatus") for r in reports if r.get("trialOverallStatus")})
        phases = sorted({r.get("clinicalStage") or r.get("trialPhase") or r.get("phaseFromSource") for r in reports if (r.get("clinicalStage") or r.get("trialPhase") or r.get("phaseFromSource"))})
        rows.append({
            "target_id": target.get("id"),
            "target_symbol": target.get("approvedSymbol"),
            "candidate_id": row.get("id"),
            "drug_id": drug.get("id"),
            "drug_name": drug.get("name"),
            "parent_molecule_id": parent.get("id"),
            "parent_molecule_name": parent.get("name"),
            "canonical_smiles": structure.get("canonical_smiles"),
            "standard_inchi_key": structure.get("standard_inchi_key"),
            "structure_source": structure.get("structure_source"),
            "structure_lookup_status": structure.get("structure_lookup_status"),
            "drug_type": drug.get("drugType"),
            "max_clinical_stage_for_target": row.get("maxClinicalStage"),
            "drug_maximum_clinical_stage": drug.get("maximumClinicalStage"),
            "disease_names": "; ".join(disease_names),
            "n_diseases": len(disease_names),
            "n_clinical_reports": len(reports),
            "clinical_report_types": "; ".join(report_types),
            "clinical_stages_seen": "; ".join(phases),
            "trial_statuses_seen": "; ".join(statuses),
            "trade_names": "; ".join(drug.get("tradeNames") or []),
            "synonyms": "; ".join((drug.get("synonyms") or [])[:20]),
            "cross_references": _format_cross_refs(drug),
            "description": drug.get("description"),
            # Useful for a human review step before passing names to FAERS.
            "include_in_faers_matrix": "",
            "exclusion_reason": "",
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["max_clinical_stage_for_target", "drug_name"], ascending=[False, True], ignore_index=True)
    return df


def flatten_safety_liabilities(target: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for row in target.get("safetyLiabilities") or []:
        rows.append({
            "target_id": target.get("id"),
            "target_symbol": target.get("approvedSymbol"),
            "event": row.get("event"),
            "event_id": row.get("eventId"),
            "datasource": row.get("datasource"),
            "biosamples": json.dumps(row.get("biosamples") or []),
            "effects": json.dumps(row.get("effects") or []),
            "studies": json.dumps(row.get("studies") or []),
            "literature": row.get("literature"),
            "url": row.get("url"),
        })
    return pd.DataFrame(rows)


def build_target_summary(target: Dict[str, Any]) -> Dict[str, Any]:
    associated = target.get("associatedDiseases") or {}
    candidates = target.get("drugAndClinicalCandidates") or {}
    safety = target.get("safetyLiabilities") or []
    return {
        "target_id": target.get("id"),
        "target_symbol": target.get("approvedSymbol"),
        "target_name": target.get("approvedName"),
        "biotype": target.get("biotype"),
        "n_associated_diseases_returned": len(associated.get("rows") or []),
        "n_associated_diseases_total": associated.get("count"),
        "n_drug_candidates_returned": len(candidates.get("rows") or []),
        "n_drug_candidates_total": candidates.get("count"),
        "n_safety_liabilities": len(safety),
        "function_descriptions": target.get("functionDescriptions") or [],
    }


def write_outputs(target: Dict[str, Any], out_prefix: str, enrich_chembl_smiles: bool = True, chembl_cache_path: Optional[str] = None) -> Dict[str, Path]:
    prefix = Path(out_prefix)
    outputs = {
        "raw_json": prefix.with_name(prefix.name + "_target_context.json"),
        "summary_json": prefix.with_name(prefix.name + "_target_summary.json"),
        "associated_diseases": prefix.with_name(prefix.name + "_associated_diseases.csv"),
        "tractability": prefix.with_name(prefix.name + "_tractability.csv"),
        "known_drugs": prefix.with_name(prefix.name + "_known_drugs.csv"),
        "safety_liabilities": prefix.with_name(prefix.name + "_safety_liabilities.csv"),
    }
    outputs["raw_json"].write_text(json.dumps(target, indent=2), encoding="utf-8")
    outputs["summary_json"].write_text(json.dumps(build_target_summary(target), indent=2), encoding="utf-8")
    flatten_associated_diseases(target).to_csv(outputs["associated_diseases"], index=False)
    flatten_tractability(target).to_csv(outputs["tractability"], index=False)
    cache = _load_json_cache(chembl_cache_path)
    flatten_known_drugs(target, enrich_chembl_smiles=enrich_chembl_smiles, chembl_cache=cache).to_csv(outputs["known_drugs"], index=False)
    _write_json_cache(chembl_cache_path, cache)
    flatten_safety_liabilities(target).to_csv(outputs["safety_liabilities"], index=False)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch target context from Open Targets")
    parser.add_argument("target", help="Target symbol in curated map, e.g. EGFR, or Ensembl ID")
    parser.add_argument("--disease-limit", type=int, default=25)
    parser.add_argument("--drug-limit", type=int, default=100, help="Cap rows written from drugAndClinicalCandidates")
    parser.add_argument("--prefix", help="Output prefix. Defaults to lowercase target symbol.")
    parser.add_argument("--no-chembl-smiles", action="store_true", help="Do not enrich Open Targets drugs with ChEMBL canonical SMILES")
    parser.add_argument("--chembl-cache", default="open_targets_output/chembl_structure_cache.json", help="Local JSON cache for ChEMBL structure lookups")
    args = parser.parse_args()

    target = get_target_context(args.target, args.disease_limit, args.drug_limit)
    prefix = args.prefix or target.get("approvedSymbol", args.target).lower()
    outputs = write_outputs(target, prefix, enrich_chembl_smiles=not args.no_chembl_smiles, chembl_cache_path=args.chembl_cache)

    print("\n=== Open Targets target context ===")
    print(json.dumps(build_target_summary(target), indent=2))
    print("\nSaved files:")
    for label, path in outputs.items():
        print(f"  {label}: {path}")

    drugs = pd.read_csv(outputs["known_drugs"])
    if not drugs.empty:
        print("\nTop drug/candidate rows:")
        show_cols = ["drug_name", "drug_type", "max_clinical_stage_for_target", "canonical_smiles", "structure_lookup_status"]
        existing = [c for c in show_cols if c in drugs.columns]
        print(drugs[existing].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
