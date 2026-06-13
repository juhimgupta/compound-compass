#!/usr/bin/env python3
"""
faers_signals.py
================
Pull adverse-event reports for one drug from the openFDA FAERS API and compute
disproportionality signals (PRR, ROR, Yates chi-square).

HOW THE DATA PULL WORKS
-----------------------
For a drug D and adverse event E we need a 2x2 table:

                    event E      not E
    drug D            a            b
    not drug D        c            d

openFDA gives counts, not a table, so we reconstruct it from marginals using
only two query types: `search=` (filter + read meta.results.total) and
`count=<field>.exact` (frequency distribution).

    1. N           = all reports                      ->  ?limit=1
    2. drug_total  = a + b  (reports with the drug)   ->  ?search=<drug>&limit=1
    3. a per event = drug's whole reaction profile    ->  ?search=<drug>&count=reaction.exact
    4. event_total = a + c  (reports with the event)  ->  ?search=reaction.exact:"E"&limit=1

Then  b = drug_total - a ,  c = event_total - a ,  d = N - a - b - c.

Only step 4 scales (one call per event), so we cap it at the drug's top-K events.

IMPORTANT CAVEATS (FAERS is spontaneous-report data):
    - `a` counts REPORTS mentioning the pair, not patients; no true incidence.
    - Duplicates and reporting biases exist; a signal != causation.
    - Names are messy, so we match brand + generic + free-text product fields.

Usage:
    python faers_signals.py osimertinib
    python faers_signals.py "lenalidomide" --top 60 --out lena.csv
    python faers_signals.py --selftest          # verify the math, no network needed

Optional: set FDA_API_KEY in the environment to raise the rate limit.

References for the statistics:
    van Puijenbroek et al. 2002 (PRR/ROR); Evans et al. 2001 (signalling criteria).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import requests
import pandas as pd

BASE = "https://api.fda.gov/drug/event.json"
BASE_URL = BASE
API_KEY = os.environ.get("OPENFDA_API_KEY") or os.environ.get("FDA_API_KEY")  # optional; raises rate limit if set


# --------------------------------------------------------------------------- #
#  Disproportionality statistics (textbook formulae, nothing copied)
# --------------------------------------------------------------------------- #
def _nz(x, eps=0.5):
    """Haldane-Anscombe continuity correction so we never divide by zero."""
    return x if x > 0 else eps


def disproportionality(a, b, c, d):
    """Return PRR, ROR, their 95% CIs, the Yates chi-square, and a signal flag."""
    a_, b_, c_, d_ = _nz(a), _nz(b), _nz(c), _nz(d)

    prr = (a_ / (a_ + b_)) / (c_ / (c_ + d_))
    se_prr = math.sqrt(1 / a_ - 1 / (a_ + b_) + 1 / c_ - 1 / (c_ + d_))
    prr_lo = math.exp(math.log(prr) - 1.96 * se_prr)
    prr_hi = math.exp(math.log(prr) + 1.96 * se_prr)

    ror = (a_ * d_) / (b_ * c_)
    se_ror = math.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
    ror_lo = math.exp(math.log(ror) - 1.96 * se_ror)
    ror_hi = math.exp(math.log(ror) + 1.96 * se_ror)

    n = a_ + b_ + c_ + d_
    den = (a_ + b_) * (c_ + d_) * (a_ + c_) * (b_ + d_)
    chi2 = (n * (abs(a_ * d_ - b_ * c_) - n / 2) ** 2) / den if den > 0 else 0.0

    # Evans (2001) criteria for flagging a signal
    is_signal = (prr >= 2.0) and (chi2 >= 4.0) and (a >= 3)

    return {
        "a": a, "b": b, "c": c, "d": d,
        "prr": round(prr, 3), "prr_lower": round(prr_lo, 3), "prr_upper": round(prr_hi, 3),
        "ror": round(ror, 3), "ror_lower": round(ror_lo, 3), "ror_upper": round(ror_hi, 3),
        "chi2_yates": round(chi2, 2), "is_signal": is_signal,
    }


# --------------------------------------------------------------------------- #
#  openFDA access
# --------------------------------------------------------------------------- #
def _get(params, retries=4):
    """GET with API-key injection, exponential backoff, and 404-as-zero handling."""
    if API_KEY:
        params = {**params, "api_key": API_KEY}
    for attempt in range(retries):
        r = requests.get(BASE, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:                       # openFDA: zero matches
            return {"meta": {"results": {"total": 0}}, "results": []}
        if r.status_code in (429, 500, 502, 503):      # transient -> back off
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
    raise RuntimeError(f"openFDA request failed after {retries} retries: {params}")


def _drug_filter(drug):
    """Match a drug across brand, generic, and free-text product name fields."""
    d = drug.strip()
    return (f'(patient.drug.medicinalproduct:"{d}"'
            f' OR patient.drug.openfda.generic_name:"{d}"'
            f' OR patient.drug.openfda.brand_name:"{d}")')


def total_reports():
    """N: every report in the openFDA FAERS subset."""
    return _get({"limit": 1})["meta"]["results"]["total"]


def drug_total(drug):
    """a + b: reports mentioning this drug."""
    return _get({"search": _drug_filter(drug), "limit": 1})["meta"]["results"]["total"]


def drug_event_counts(drug, limit=1000):
    """a for every reaction reported with this drug -> {EVENT: count}."""
    data = _get({"search": _drug_filter(drug),
                 "count": "patient.reaction.reactionmeddrapt.exact",
                 "limit": limit})
    return {row["term"]: row["count"] for row in data.get("results", [])}


def event_total(event):
    """a + c: reports mentioning this reaction term across all drugs."""
    return _get({"search": f'patient.reaction.reactionmeddrapt.exact:"{event}"',
                 "limit": 1})["meta"]["results"]["total"]


def signals_for_drug(drug, top_k=50, min_count=3, pause=0.2, verbose=True):
    """Return a disproportionality table (one row per event) for one drug."""
    if verbose:
        print(f"[1/4] total reports in FAERS ...", file=sys.stderr)
    n = total_reports()

    if verbose:
        print(f"[2/4] reports mentioning '{drug}' ...", file=sys.stderr)
    dt = drug_total(drug)
    if dt == 0:
        raise ValueError(
            f"No FAERS reports for '{drug}'. Try the generic name, or check spelling."
        )

    if verbose:
        print(f"[3/4] reaction profile for '{drug}' ...", file=sys.stderr)
    counts = drug_event_counts(drug)
    events = [(e, a) for e, a in counts.items() if a >= min_count][:top_k]

    if verbose:
        print(f"[4/4] background totals for {len(events)} events ...", file=sys.stderr)
    rows = []
    for event, a in events:
        et = event_total(event)
        b = dt - a
        c = et - a
        d = n - a - b - c
        rows.append({"drug": drug, "event": event, "drug_event_count": a,
                     "drug_total": dt, "event_total": et, "n_total": n,
                     **disproportionality(a, b, c, d)})
        time.sleep(pause)

    return (pd.DataFrame(rows)
            .sort_values("ror", ascending=False)
            .reset_index(drop=True))


# --------------------------------------------------------------------------- #
#  Offline self-test (verifies the math)
# --------------------------------------------------------------------------- #
def selftest():
    # a=20 reports of an event among 500 drug reports; event seen 1000x in 1e6 total
    a, dt, et, n = 20, 500, 1000, 1_000_000
    b, c, d = dt - a, et - a, n - a - (dt - a) - (et - a)
    s = disproportionality(a, b, c, d)
    print(f"cells a,b,c,d = {a},{b},{c},{d}")
    print(f"PRR = {s['prr']}  ({s['prr_lower']}-{s['prr_upper']})")
    print(f"ROR = {s['ror']}  ({s['ror_lower']}-{s['ror_upper']})")
    print(f"chi2 (Yates) = {s['chi2_yates']}   signal = {s['is_signal']}")
    assert abs(s["ror"] - 42.454) < 0.01, "ROR mismatch"
    assert abs(s["chi2_yates"] - 723.08) < 0.1, "chi2 mismatch"
    assert s["is_signal"] is True
    print("\nself-test passed ✓")


def main():
    p = argparse.ArgumentParser(description="openFDA FAERS disproportionality signals")
    p.add_argument("drug", nargs="?", help="drug name (brand or generic)")
    p.add_argument("--top", type=int, default=50, help="number of top events to score")
    p.add_argument("--min-count", type=int, default=3, help="ignore events below this report count")
    p.add_argument("--out", help="CSV output path")
    p.add_argument("--selftest", action="store_true", help="verify the math, no network")
    args = p.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.drug:
        p.error("provide a drug name (or use --selftest)")

    df = signals_for_drug(args.drug, top_k=args.top, min_count=args.min_count)
    cols = ["event", "drug_event_count", "prr", "ror", "ror_lower", "chi2_yates", "is_signal"]
    print(df[cols].to_string(index=False))

    out = args.out or f"signals_{args.drug.lower().replace(' ', '_')}.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}  ({len(df)} events, {int(df['is_signal'].sum())} flagged as signals)")


if __name__ == "__main__":
    main()