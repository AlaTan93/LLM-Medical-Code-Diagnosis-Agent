"""Evaluator: measure ICD-10-CM coding accuracy of the /diagnose pipeline.

Runs on the host outside the container. Loads test cases from
``data/icd10_cm_cases.json``, calls the ``/diagnose`` endpoint for each case,
and reports top-1 through top-3 accuracy, recall, precision@GT, F1, diagnosis
count metrics, model agreement, and critic recall/precision/F1 (over the
subset of cases where the debate-critic was triggered).  Category-level
(3-char prefix) recall, precision, and F1 are reported alongside exact-match
metrics to distinguish "right disease, wrong specificity" from total misses.

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

MODEL_A = "medgemma-27b-q4_k_s"
MODEL_B = "deepseek-r1-medical-cot"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize(code: str) -> str:
    """Strip periods from an ICD-10-CM code for comparison.

    The database stores codes without periods (e.g. ``E119`` for ``E11.9``).
    Ground truth uses standard notation with periods.
    """
    return code.replace(".", "")


def _cat_recall(gt: set[str], codes: list[str]) -> float:
    """Fraction of GT codes whose 3-char category appears in *codes*."""
    if not gt:
        return 0.0
    pred_cats = {c[:3] for c in codes}
    return sum(1 for g in gt if g[:3] in pred_cats) / len(gt)


def _cat_precision(gt: set[str], codes: list[str]) -> float:
    """Fraction of returned codes whose 3-char category matches a GT code."""
    if not codes:
        return 0.0
    gt_cats = {c[:3] for c in gt}
    return sum(1 for c in codes if c[:3] in gt_cats) / len(codes)


def _f1(recall: float, precision: float) -> float:
    """Harmonic mean of recall and precision (0 when both are 0)."""
    denom = recall + precision
    return 2 * recall * precision / denom if denom else 0.0


def _model_metrics(gt: set[str], codes: list[str], dx_cnt: int,
                   gt_cnt: int) -> dict:
    """Compute all per-case metrics for one model's codes.

    Calculates top-1/2/3 hits, exact recall/precision, and category-level
    (3-char prefix) recall/precision in one pass.

    Args:
        gt: Set of normalised ground-truth codes.
        codes: List of normalised predicted codes (ranked by similarity).
        dx_cnt: Number of diagnoses the model produced.
        gt_cnt: Number of ground-truth codes.

    Returns:
        A dict with keys: ``top1_hit``, ``top2_hit``, ``top3_hit``,
        ``recall``, ``precision``, ``cat_recall``, ``cat_precision``
        (all floats rounded to 4 dp).
    """
    tp = set(codes) & gt
    denom = max(dx_cnt, gt_cnt)
    return {
        "top1_hit": bool(codes) and codes[0] in gt,
        "top2_hit": any(c in gt for c in codes[:2]),
        "top3_hit": any(c in gt for c in codes[:3]),
        "recall": round(len(tp) / len(gt), 4) if gt else 0.0,
        "precision": round(len(tp) / denom, 4) if denom else 0.0,
        "cat_recall": round(_cat_recall(gt, codes), 4),
        "cat_precision": round(_cat_precision(gt, codes), 4),
    }


def _empty_metrics() -> dict:
    """Return a zero-valued metrics dict (used when a model has no codes)."""
    return {
        "top1_hit": False,
        "top2_hit": False,
        "top3_hit": False,
        "recall": 0.0,
        "precision": 0.0,
        "cat_recall": 0.0,
        "cat_precision": 0.0,
    }


def load_cases(path: str = CASES_PATH) -> list[dict]:
    """Load test cases from the JSON file."""
    with open(path) as f:
        return json.load(f)["cases"]


def call_diagnose(text: str, timeout: float = 600) -> dict:
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


def _print_model_line(tag: str, mark: str, dx_cnt: int, res: list[dict],
                      m: dict) -> None:
    """Print one model's per-case summary line (A, B, or C)."""
    codes_str = ", ".join(f"{c['code']} ({c['similarity']:.0%})" for c in res) or "(none)"
    print(f"{'':9s}{tag} [{mark}] {dx_cnt}dx  {codes_str}  "
          f"R:{m['recall']:.0%} P:{m['precision']:.0%} "
          f"CatR:{m['cat_recall']:.0%} CatP:{m['cat_precision']:.0%}")


def _evaluate_case(case: dict, index: int, total: int) -> dict | None:
    """Evaluate a single case against the /diagnose endpoint.

    Calls the API, computes metrics for both models (and the critic if
    triggered), prints a summary, and returns a details dict.

    Returns ``None`` if the API call fails (error is printed).
    """
    gt_raw = case["icd10_cm"]["codes"]
    gt = {normalize(c) for c in gt_raw}
    gt_cnt = len(gt_raw)
    label = f"[{index + 1:2d}/{total}]"

    try:
        result = call_diagnose(case["medical_note"])
    except Exception as e:
        print(f"{label} ERROR: {e}")
        return None

    elapsed = result.get("elapsed_s", 0)
    ra, rb = result["results"]
    res_a, res_b = ra["codes"], rb["codes"]

    codes_a = [c["code"] for c in res_a]
    codes_b = [c["code"] for c in res_b]
    dx_a, dx_b = ra.get("diagnoses", []), rb.get("diagnoses", [])
    reasoning_a, reasoning_b = ra.get("reasoning", ""), rb.get("reasoning", "")

    ma = _model_metrics(gt, codes_a, len(dx_a), gt_cnt)
    mb = _model_metrics(gt, codes_b, len(dx_b), gt_cnt)

    # Print header and model lines.
    gt_word = "code" if gt_cnt == 1 else "codes"
    print(f"{label} {elapsed:5.1f}s  GT: {', '.join(gt_raw)} ({gt_cnt} {gt_word})")
    _print_model_line("A", "+" if ma["top2_hit"] else "-", len(dx_a), res_a, ma)
    if reasoning_a:
        print(f"{'':9s}    \"{reasoning_a[:120]}\"")
    _print_model_line("B", "+" if mb["top2_hit"] else "-", len(dx_b), res_b, mb)
    if reasoning_b:
        print(f"{'':9s}    \"{reasoning_b[:120]}\"")

    # Critic results (if triggered).
    critic_triggered = result.get("critic_triggered", False)
    critic_rounds = result.get("critic_rounds", [])
    critic_codes: list[str] = []
    critic_dx: list[str] = []
    critic_reasoning = ""
    mc = _empty_metrics()

    if critic_triggered and critic_rounds:
        final = critic_rounds[-1]
        critic_codes = [c["code"] for c in final.get("codes", [])]
        critic_dx = final.get("diagnoses", [])
        critic_reasoning = final.get("reasoning", "")
        mc = _model_metrics(gt, critic_codes, len(critic_dx), gt_cnt)

        n_rounds = len(critic_rounds)
        c_str = ", ".join(critic_codes[:5]) or "(none)"
        mark_c = "+" if mc["top2_hit"] else "-"
        print(f"{'':9s}C [{mark_c}] {len(critic_dx)}dx  {c_str}  "
              f"R:{mc['recall']:.0%} P:{mc['precision']:.0%} "
              f"CatR:{mc['cat_recall']:.0%} CatP:{mc['cat_precision']:.0%}  "
              f"({n_rounds} round{'s' if n_rounds != 1 else ''})")
        if critic_reasoning:
            print(f"{'':9s}    \"{critic_reasoning[:120]}\"")

    return {
        "case": index + 1,
        "ground_truth": gt_raw,
        "elapsed_s": elapsed,
        "model_a": {
            "diagnoses": dx_a, "dx_count": len(dx_a), "codes": codes_a,
            "reasoning": reasoning_a, **ma,
        },
        "model_b": {
            "diagnoses": dx_b, "dx_count": len(dx_b), "codes": codes_b,
            "reasoning": reasoning_b, **mb,
        },
        "agreement": bool(codes_a) and bool(codes_b) and codes_a[0] == codes_b[0],
        "critic": {
            "triggered": critic_triggered,
            "rounds": len(critic_rounds),
            "diagnoses": critic_dx,
            "dx_count": len(critic_dx),
            "codes": critic_codes,
            "reasoning": critic_reasoning,
            **mc,
        },
    }


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------


def _compute_summary(details: list[dict]) -> dict:
    """Aggregate per-case metrics into summary statistics.

    Computes top-N hit rates, average recall/precision/F1 (exact + category),
    diagnosis count metrics (match/over/under), and model agreement rate.
    """
    total = len(details)

    # --- helpers for the A/B pair pattern ---
    def _hits(key: str) -> tuple[int, int]:
        return (sum(1 for d in details if d["model_a"][key]),
                sum(1 for d in details if d["model_b"][key]))

    def _avg(key: str) -> tuple[float, float]:
        return (sum(d["model_a"][key] for d in details) / total,
                sum(d["model_b"][key] for d in details) / total)

    top1_a, top1_b = _hits("top1_hit")
    top2_a, top2_b = _hits("top2_hit")
    top3_a, top3_b = _hits("top3_hit")

    avg_rec_a, avg_rec_b = _avg("recall")
    avg_prec_a, avg_prec_b = _avg("precision")
    f1_a, f1_b = _f1(avg_rec_a, avg_prec_a), _f1(avg_rec_b, avg_prec_b)

    avg_cat_rec_a, avg_cat_rec_b = _avg("cat_recall")
    avg_cat_prec_a, avg_cat_prec_b = _avg("cat_precision")
    cat_f1_a = _f1(avg_cat_rec_a, avg_cat_prec_a)
    cat_f1_b = _f1(avg_cat_rec_b, avg_cat_prec_b)

    # Diagnosis count metrics.
    dx_a = [d["model_a"]["dx_count"] for d in details]
    dx_b = [d["model_b"]["dx_count"] for d in details]
    gt_c = [len(d["ground_truth"]) for d in details]
    avg_gt = sum(gt_c) / total

    def _cnt_cmp(dx: list[int]) -> tuple[int, int, int]:
        match = sum(1 for d, g in zip(dx, gt_c) if d == g)
        over = sum(1 for d, g in zip(dx, gt_c) if d > g)
        under = sum(1 for d, g in zip(dx, gt_c) if d < g)
        return match, over, under

    match_a, over_a, under_a = _cnt_cmp(dx_a)
    match_b, over_b, under_b = _cnt_cmp(dx_b)
    agree = sum(1 for d in details if d["agreement"])

    # Critic metrics — averaged only over cases where the critic was triggered.
    cd = [d for d in details if d["critic"]["triggered"]]
    n_critic = len(cd)

    if n_critic > 0:
        cn = n_critic
        top1_c = sum(1 for d in cd if d["critic"]["top1_hit"])
        top2_c = sum(1 for d in cd if d["critic"]["top2_hit"])
        top3_c = sum(1 for d in cd if d["critic"]["top3_hit"])
        avg_rec_c = sum(d["critic"]["recall"] for d in cd) / cn
        avg_prec_c = sum(d["critic"]["precision"] for d in cd) / cn
        f1_c = _f1(avg_rec_c, avg_prec_c)
        avg_cat_rec_c = sum(d["critic"]["cat_recall"] for d in cd) / cn
        avg_cat_prec_c = sum(d["critic"]["cat_precision"] for d in cd) / cn
        cat_f1_c = _f1(avg_cat_rec_c, avg_cat_prec_c)
        avg_dx_c = sum(d["critic"]["dx_count"] for d in cd) / cn
    else:
        top1_c = top2_c = top3_c = 0
        avg_rec_c = avg_prec_c = f1_c = avg_dx_c = 0.0
        avg_cat_rec_c = avg_cat_prec_c = cat_f1_c = 0.0

    return {
        "total": total,
        "n_critic": n_critic,
        "top1_a": top1_a, "top1_b": top1_b,
        "top2_a": top2_a, "top2_b": top2_b,
        "top3_a": top3_a, "top3_b": top3_b,
        "avg_rec_a": avg_rec_a, "avg_rec_b": avg_rec_b,
        "avg_prec_a": avg_prec_a, "avg_prec_b": avg_prec_b,
        "f1_a": f1_a, "f1_b": f1_b,
        "avg_cat_rec_a": avg_cat_rec_a, "avg_cat_rec_b": avg_cat_rec_b,
        "avg_cat_prec_a": avg_cat_prec_a, "avg_cat_prec_b": avg_cat_prec_b,
        "cat_f1_a": cat_f1_a, "cat_f1_b": cat_f1_b,
        "avg_gt": avg_gt,
        "avg_dx_a": sum(dx_a) / total, "avg_dx_b": sum(dx_b) / total,
        "match_a": match_a, "match_b": match_b,
        "over_a": over_a, "over_b": over_b,
        "under_a": under_a, "under_b": under_b,
        "agree": agree,
        "top1_c": top1_c, "top2_c": top2_c, "top3_c": top3_c,
        "avg_rec_c": avg_rec_c, "avg_prec_c": avg_prec_c, "f1_c": f1_c,
        "avg_cat_rec_c": avg_cat_rec_c, "avg_cat_prec_c": avg_cat_prec_c,
        "cat_f1_c": cat_f1_c,
        "avg_dx_c": avg_dx_c,
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
    rows = [
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
        (f"Cat Recall  ({MODEL_A})", "", f"{m['avg_cat_rec_a']:.1%}"),
        (f"Cat Recall  ({MODEL_B})", "", f"{m['avg_cat_rec_b']:.1%}"),
        (f"Cat Prec@GT ({MODEL_A})", "", f"{m['avg_cat_prec_a']:.1%}"),
        (f"Cat Prec@GT ({MODEL_B})", "", f"{m['avg_cat_prec_b']:.1%}"),
        (f"Cat F1  ({MODEL_A})", "", f"{m['cat_f1_a']:.1%}"),
        (f"Cat F1  ({MODEL_B})", "", f"{m['cat_f1_b']:.1%}"),
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

    # Critic rows (only when the critic triggered at least once).
    nc = m.get("n_critic", 0)
    if nc > 0:
        rows.extend([
            (f"Top-1  (Critic, {nc} trig.)", f"{m['top1_c']}/{nc}", f"{m['top1_c'] / nc:.1%}"),
            (f"Top-2  (Critic, {nc} trig.)", f"{m['top2_c']}/{nc}", f"{m['top2_c'] / nc:.1%}"),
            (f"Top-3  (Critic, {nc} trig.)", f"{m['top3_c']}/{nc}", f"{m['top3_c'] / nc:.1%}"),
            (f"Recall  (Critic, {nc} trig.)", "", f"{m['avg_rec_c']:.1%}"),
            (f"Prec@GT (Critic, {nc} trig.)", "", f"{m['avg_prec_c']:.1%}"),
            (f"F1  (Critic, {nc} trig.)", "", f"{m['f1_c']:.1%}"),
            (f"Cat Recall  (Critic, {nc} trig.)", "", f"{m['avg_cat_rec_c']:.1%}"),
            (f"Cat Prec@GT (Critic, {nc} trig.)", "", f"{m['avg_cat_prec_c']:.1%}"),
            (f"Cat F1  (Critic, {nc} trig.)", "", f"{m['cat_f1_c']:.1%}"),
            (f"Avg diagnoses  (Critic)", "", f"{m['avg_dx_c']:.1f}"),
        ])

    return rows


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
    metrics = {
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
        "cat_recall_a": round(m["avg_cat_rec_a"], 4),
        "cat_recall_b": round(m["avg_cat_rec_b"], 4),
        "cat_precision_a": round(m["avg_cat_prec_a"], 4),
        "cat_precision_b": round(m["avg_cat_prec_b"], 4),
        "cat_f1_a": round(m["cat_f1_a"], 4),
        "cat_f1_b": round(m["cat_f1_b"], 4),
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

    # Critic metrics (only when triggered at least once).
    nc = m.get("n_critic", 0)
    if nc > 0:
        metrics["critic_triggered"] = nc
        metrics["critic_top1"] = m["top1_c"]
        metrics["critic_top1_pct"] = round(m["top1_c"] / nc, 4)
        metrics["critic_top2"] = m["top2_c"]
        metrics["critic_top2_pct"] = round(m["top2_c"] / nc, 4)
        metrics["critic_top3"] = m["top3_c"]
        metrics["critic_top3_pct"] = round(m["top3_c"] / nc, 4)
        metrics["critic_recall"] = round(m["avg_rec_c"], 4)
        metrics["critic_precision"] = round(m["avg_prec_c"], 4)
        metrics["critic_f1"] = round(m["f1_c"], 4)
        metrics["critic_cat_recall"] = round(m["avg_cat_rec_c"], 4)
        metrics["critic_cat_precision"] = round(m["avg_cat_prec_c"], 4)
        metrics["critic_cat_f1"] = round(m["cat_f1_c"], 4)
        metrics["critic_avg_dx"] = round(m["avg_dx_c"], 2)

    return metrics


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
