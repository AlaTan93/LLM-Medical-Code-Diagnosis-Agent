"""Evaluator: measure ICD-10-CM coding accuracy of the /diagnose pipeline.

Runs on the host outside the container. Loads test cases from
``data/icd10_cm_cases.json``, calls the ``/diagnose`` endpoint for each case,
and reports top-1 through top-3 accuracy, recall, precision@GT, F1, diagnosis
count metrics, and model agreement.

Usage::

    python evaluate.py              # all 50 cases
    python evaluate.py 5            # first 5 cases (quick test)
    python evaluate.py --save data/eval_results.json
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

BASE_URL = "http://localhost:8000"
CASES_PATH = "data/icd10_cm_cases.json"

MODEL_A = "ii-medical-q8"
MODEL_B = "gemma-4-medical-q6"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize(code: str) -> str:
    """Strip periods from an ICD-10-CM code for comparison.

    The database stores codes without periods (e.g. ``E119`` for ``E11.9``).
    Ground truth uses standard notation with periods.
    """
    return code.replace(".", "")


def load_cases(path: str = CASES_PATH) -> list[dict]:
    """Load test cases from the JSON file."""
    with open(path) as f:
        return json.load(f)["cases"]


def call_diagnose(text: str, timeout: float = 300) -> dict:
    """POST a medical note to /diagnose and return the parsed response."""
    payload = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/diagnose",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


# ---------------------------------------------------------------------------
# Per-case evaluation
# ---------------------------------------------------------------------------


def _evaluate_case(case: dict, index: int, total: int) -> dict | None:
    """Evaluate a single case against the /diagnose endpoint.

    Calls the API, extracts per-model results, computes hit/recall/precision
    metrics, prints a one-line summary, and returns a details dict for
    aggregation and JSON export.

    Returns ``None`` if the API call fails (error is printed).
    """
    gt_raw = case["icd10_cm"]["codes"]
    gt = {normalize(c) for c in gt_raw}
    label = f"[{index + 1:2d}/{total}]"

    try:
        result = call_diagnose(case["medical_note"])
    except Exception as e:
        print(f"{label} ERROR: {e}")
        return None

    elapsed = result.get("elapsed_s", 0)
    res_a = result["results"][0]["codes"]
    res_b = result["results"][1]["codes"]

    # Extract codes (normalized) and diagnosis counts for each model.
    codes_a = [c["code"] for c in res_a]
    codes_b = [c["code"] for c in res_b]
    dx_a = result["results"][0].get("diagnoses", [])
    dx_b = result["results"][1].get("diagnoses", [])
    reasoning_a = result["results"][0].get("reasoning", "")
    reasoning_b = result["results"][1].get("reasoning", "")
    dx_cnt_a = len(dx_a)
    dx_cnt_b = len(dx_b)
    gt_cnt = len(gt_raw)

    # Set-based overlap for recall/precision.
    set_a = set(codes_a)
    set_b = set(codes_b)
    tp_a = set_a & gt
    tp_b = set_b & gt

    # Top-N hit checks: is the correct code within the first N results?
    hit_1a = bool(codes_a) and codes_a[0] in gt
    hit_1b = bool(codes_b) and codes_b[0] in gt
    hit_2a = any(c in gt for c in codes_a[:2])
    hit_2b = any(c in gt for c in codes_b[:2])
    hit_3a = any(c in gt for c in codes_a[:3])
    hit_3b = any(c in gt for c in codes_b[:3])
    same_top1 = bool(codes_a) and bool(codes_b) and codes_a[0] == codes_b[0]

    # Recall: fraction of GT codes found in the model's results.
    rec_a = len(tp_a) / len(gt) if gt else 0.0
    rec_b = len(tp_b) / len(gt) if gt else 0.0

    # Precision@GT: denominator is max(dx_count, gt_count) so that
    # overdiagnosis penalises precision, but search breadth (k per
    # diagnosis) does not.  Underdiagnosis is also penalised because
    # the denominator stays at gt_count when dx_count < gt_count.
    denom_a = max(dx_cnt_a, gt_cnt)
    denom_b = max(dx_cnt_b, gt_cnt)
    prec_a = len(tp_a) / denom_a if denom_a else 0.0
    prec_b = len(tp_b) / denom_b if denom_b else 0.0

    # Print per-case summary.  '+' = top-2 hit, '-' = miss.
    a_str = ", ".join(f"{c['code']} ({c['similarity']:.0%})" for c in res_a) or "(none)"
    b_str = ", ".join(f"{c['code']} ({c['similarity']:.0%})" for c in res_b) or "(none)"
    mark_a = "+" if hit_2a else "-"
    mark_b = "+" if hit_2b else "-"
    gt_word = "code" if gt_cnt == 1 else "codes"
    print(f"{label} {elapsed:5.1f}s  GT: {', '.join(gt_raw)} ({gt_cnt} {gt_word})")
    print(f"{'':9s}A [{mark_a}] {dx_cnt_a}dx  {a_str}  R:{rec_a:.0%} P:{prec_a:.0%}")
    if reasoning_a:
        print(f"{'':9s}    \"{reasoning_a[:120]}\"")
    print(f"{'':9s}B [{mark_b}] {dx_cnt_b}dx  {b_str}  R:{rec_b:.0%} P:{prec_b:.0%}")
    if reasoning_b:
        print(f"{'':9s}    \"{reasoning_b[:120]}\"")

    return {
        "case": index + 1,
        "ground_truth": gt_raw,
        "elapsed_s": elapsed,
        "model_a": {
            "diagnoses": dx_a,
            "dx_count": dx_cnt_a,
            "codes": codes_a,
            "reasoning": reasoning_a,
            "top1_hit": hit_1a,
            "top2_hit": hit_2a,
            "top3_hit": hit_3a,
            "recall": round(rec_a, 4),
            "precision": round(prec_a, 4),
        },
        "model_b": {
            "diagnoses": dx_b,
            "dx_count": dx_cnt_b,
            "codes": codes_b,
            "reasoning": reasoning_b,
            "top1_hit": hit_1b,
            "top2_hit": hit_2b,
            "top3_hit": hit_3b,
            "recall": round(rec_b, 4),
            "precision": round(prec_b, 4),
        },
        "agreement": same_top1,
    }


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------


def _compute_summary(details: list[dict]) -> dict:
    """Aggregate per-case metrics into summary statistics.

    Computes top-N hit rates, average recall/precision/F1, diagnosis count
    metrics (match/over/under), and model agreement rate.

    Args:
        details: List of per-case dicts from :func:`_evaluate_case`.

    Returns:
        A flat dict of summary metrics ready for table display and JSON.
    """
    total = len(details)

    top1_a = sum(1 for d in details if d["model_a"]["top1_hit"])
    top1_b = sum(1 for d in details if d["model_b"]["top1_hit"])
    top2_a = sum(1 for d in details if d["model_a"]["top2_hit"])
    top2_b = sum(1 for d in details if d["model_b"]["top2_hit"])
    top3_a = sum(1 for d in details if d["model_a"]["top3_hit"])
    top3_b = sum(1 for d in details if d["model_b"]["top3_hit"])

    avg_rec_a = sum(d["model_a"]["recall"] for d in details) / total
    avg_rec_b = sum(d["model_b"]["recall"] for d in details) / total
    avg_prec_a = sum(d["model_a"]["precision"] for d in details) / total
    avg_prec_b = sum(d["model_b"]["precision"] for d in details) / total

    denom_a = avg_rec_a + avg_prec_a
    denom_b = avg_rec_b + avg_prec_b
    f1_a = 2 * avg_rec_a * avg_prec_a / denom_a if denom_a else 0.0
    f1_b = 2 * avg_rec_b * avg_prec_b / denom_b if denom_b else 0.0

    # Diagnosis count metrics.
    dx_cnts_a = [d["model_a"]["dx_count"] for d in details]
    dx_cnts_b = [d["model_b"]["dx_count"] for d in details]
    gt_cnts = [len(d["ground_truth"]) for d in details]
    avg_dx_a = sum(dx_cnts_a) / total
    avg_dx_b = sum(dx_cnts_b) / total
    avg_gt = sum(gt_cnts) / total

    match_a = sum(1 for da, g in zip(dx_cnts_a, gt_cnts) if da == g)
    match_b = sum(1 for db, g in zip(dx_cnts_b, gt_cnts) if db == g)
    over_a = sum(1 for da, g in zip(dx_cnts_a, gt_cnts) if da > g)
    over_b = sum(1 for db, g in zip(dx_cnts_b, gt_cnts) if db > g)
    under_a = sum(1 for da, g in zip(dx_cnts_a, gt_cnts) if da < g)
    under_b = sum(1 for db, g in zip(dx_cnts_b, gt_cnts) if db < g)

    agree = sum(1 for d in details if d["agreement"])

    return {
        "total": total,
        "top1_a": top1_a, "top1_b": top1_b,
        "top2_a": top2_a, "top2_b": top2_b,
        "top3_a": top3_a, "top3_b": top3_b,
        "avg_rec_a": avg_rec_a, "avg_rec_b": avg_rec_b,
        "avg_prec_a": avg_prec_a, "avg_prec_b": avg_prec_b,
        "f1_a": f1_a, "f1_b": f1_b,
        "avg_gt": avg_gt, "avg_dx_a": avg_dx_a, "avg_dx_b": avg_dx_b,
        "match_a": match_a, "match_b": match_b,
        "over_a": over_a, "over_b": over_b,
        "under_a": under_a, "under_b": under_b,
        "agree": agree,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

# Column format for the summary table: (label, hits, rate).
# Empty string means the cell is blank for that row.
def _build_rows(m: dict) -> list[tuple[str, str, str]]:
    """Build summary table rows from aggregated metrics.

    Each row is a (label, hits, rate) triple where *hits* is a count
    fraction (e.g. ``"3/10"``) and *rate* is a formatted percentage or value.
    """
    t = m["total"]
    return [
        (f"Top-1  ({MODEL_A})", f"{m['top1_a']}/{t}", f"{m['top1_a'] / t:.1%}"),
        (f"Top-1  ({MODEL_B})", f"{m['top1_b']}/{t}", f"{m['top1_b'] / t:.1%}"),
        (f"Top-2  ({MODEL_A})", f"{m['top2_a']}/{t}", f"{m['top2_a'] / t:.1%}"),
        (f"Top-2  ({MODEL_B})", f"{m['top2_b']}/{t}", f"{m['top2_b'] / t:.1%}"),
        (f"Top-3  ({MODEL_A})", f"{m['top3_a']}/{t}", f"{m['top3_a'] / t:.1%}"),
        (f"Top-3  ({MODEL_B})", f"{m['top3_b']}/{t}", f"{m['top3_b'] / t:.1%}"),
        (f"Recall  ({MODEL_A})", "", f"{m['avg_rec_a']:.1%}"),
        (f"Recall  ({MODEL_B})", "", f"{m['avg_rec_b']:.1%}"),
        (f"Prec@GT ({MODEL_A})", "", f"{m['avg_prec_a']:.1%}"),
        (f"Prec@GT ({MODEL_B})", "", f"{m['avg_prec_b']:.1%}"),
        (f"F1  ({MODEL_A})", "", f"{m['f1_a']:.1%}"),
        (f"F1  ({MODEL_B})", "", f"{m['f1_b']:.1%}"),
        ("Avg GT codes", "", f"{m['avg_gt']:.1f}"),
        (f"Avg diagnoses  ({MODEL_A})", "", f"{m['avg_dx_a']:.1f}"),
        (f"Avg diagnoses  ({MODEL_B})", "", f"{m['avg_dx_b']:.1f}"),
        (f"Count match    ({MODEL_A})", f"{m['match_a']}/{t}", f"{m['match_a'] / t:.1%}"),
        (f"Count match    ({MODEL_B})", f"{m['match_b']}/{t}", f"{m['match_b'] / t:.1%}"),
        (f"Over-dx        ({MODEL_A})", f"{m['over_a']}/{t}", f"{m['over_a'] / t:.1%}"),
        (f"Over-dx        ({MODEL_B})", f"{m['over_b']}/{t}", f"{m['over_b'] / t:.1%}"),
        (f"Under-dx       ({MODEL_A})", f"{m['under_a']}/{t}", f"{m['under_a'] / t:.1%}"),
        (f"Under-dx       ({MODEL_B})", f"{m['under_b']}/{t}", f"{m['under_b'] / t:.1%}"),
        ("Agreement (same top-1)", f"{m['agree']}/{t}", f"{m['agree'] / t:.1%}"),
    ]


def _print_summary(m: dict) -> None:
    """Print the formatted summary table to stdout."""
    rows = _build_rows(m)
    w = max(len(label) for label, _, _ in rows)
    total = m["total"]

    print()
    print("=" * (w + 18))
    print(f"Cases evaluated: {total}")
    print(f"{'':{w}s}{'hits':>8s}  {'rate':>6s}")
    print("-" * (w + 18))
    for label, hits, rate in rows:
        print(f"{label:<{w}s}{hits:>8s}  {rate:>6s}")
    print("=" * (w + 18))


def _build_metrics_json(m: dict) -> dict:
    """Convert summary metrics into the JSON-serialisable metrics dict."""
    t = m["total"]
    return {
        "top1_a": m["top1_a"], "top1_a_pct": round(m["top1_a"] / t, 4),
        "top1_b": m["top1_b"], "top1_b_pct": round(m["top1_b"] / t, 4),
        "top2_a": m["top2_a"], "top2_a_pct": round(m["top2_a"] / t, 4),
        "top2_b": m["top2_b"], "top2_b_pct": round(m["top2_b"] / t, 4),
        "top3_a": m["top3_a"], "top3_a_pct": round(m["top3_a"] / t, 4),
        "top3_b": m["top3_b"], "top3_b_pct": round(m["top3_b"] / t, 4),
        "recall_a": round(m["avg_rec_a"], 4),
        "recall_b": round(m["avg_rec_b"], 4),
        "precision_a": round(m["avg_prec_a"], 4),
        "precision_b": round(m["avg_prec_b"], 4),
        "f1_a": round(m["f1_a"], 4),
        "f1_b": round(m["f1_b"], 4),
        "avg_gt_codes": round(m["avg_gt"], 2),
        "avg_dx_a": round(m["avg_dx_a"], 2),
        "avg_dx_b": round(m["avg_dx_b"], 2),
        "count_match_a": m["match_a"], "count_match_a_pct": round(m["match_a"] / t, 4),
        "count_match_b": m["match_b"], "count_match_b_pct": round(m["match_b"] / t, 4),
        "over_dx_a": m["over_a"], "over_dx_a_pct": round(m["over_a"] / t, 4),
        "over_dx_b": m["over_b"], "over_dx_b_pct": round(m["over_b"] / t, 4),
        "under_dx_a": m["under_a"], "under_dx_a_pct": round(m["under_a"] / t, 4),
        "under_dx_b": m["under_b"], "under_dx_b_pct": round(m["under_b"] / t, 4),
        "agreement": m["agree"], "agreement_pct": round(m["agree"] / t, 4),
    }


def _save_json(path: str, m: dict, details: list[dict]) -> None:
    """Write evaluation results (metrics + per-case details) to a JSON file."""
    with open(path, "w") as f:
        json.dump(
            {
                "cases_evaluated": m["total"],
                "metrics": _build_metrics_json(m),
                "details": details,
            },
            f,
            indent=2,
        )
    print(f"\nDetailed results saved to {path}")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def evaluate(limit: int | None, save_path: str | None) -> None:
    """Run the evaluation: load cases, call /diagnose per case, report results.

    Args:
        limit: Evaluate only the first *limit* cases (``None`` = all).
        save_path: If set, save detailed JSON results to this path.
    """
    cases = load_cases()
    if limit is not None:
        cases = cases[:limit]
    n = len(cases)

    details: list[dict] = []
    for i, case in enumerate(cases):
        d = _evaluate_case(case, i, n)
        if d is not None:
            details.append(d)

    if not details:
        print("No cases evaluated.")
        return

    metrics = _compute_summary(details)
    _print_summary(metrics)

    if save_path:
        _save_json(save_path, metrics, details)


def main() -> int:
    """CLI entry point — parse args and run the evaluator."""
    parser = argparse.ArgumentParser(
        description="Evaluate ICD-10-CM coding accuracy of the /diagnose pipeline."
    )
    parser.add_argument(
        "limit",
        type=int,
        nargs="?",
        default=None,
        help="Number of cases to evaluate (default: all).",
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        default=None,
        help="Save detailed per-case results to this JSON file.",
    )
    args = parser.parse_args()

    started = time.time()
    evaluate(args.limit, args.save)
    print(f"\nTotal wall time: {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
