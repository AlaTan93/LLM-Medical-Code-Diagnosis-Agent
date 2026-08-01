"""Evaluator: measure ICD-10-CM coding accuracy of the /diagnose pipeline.

Runs on the host outside the container. Loads test cases from
``data/icd10_cm_cases.json``, calls the ``/diagnose`` endpoint for each case,
and reports top-1 through top-3 accuracy, recall, precision@GT, F1, diagnosis
count metrics, model agreement, and critic recall/precision/F1 (over the
subset of cases where the debate-critic was triggered).  Category-level
(3-char prefix) recall, precision, and F1 are reported alongside exact-match
metrics to distinguish "right disease, wrong specificity" from total misses.

Usage::

    python evaluate.py              # all 54 cases
    python evaluate.py 5            # first 5 cases (quick test)
    python evaluate.py --save eval_output.json
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

    Args:
        gt: Set of normalised ground-truth codes.
        codes: List of normalised predicted codes (ranked by similarity).
        dx_cnt: Number of diagnoses the model produced.
        gt_cnt: Number of ground-truth codes.

    Returns:
        Dict with ``top1_hit``, ``top2_hit``, ``top3_hit``, ``recall``,
        ``precision``, ``f1``, ``cat_recall``, ``cat_precision``
        (floats rounded to 4 dp).
    """
    tp = set(codes) & gt
    denom = max(dx_cnt, gt_cnt)
    recall = round(len(tp) / len(gt), 4) if gt else 0.0
    precision = round(len(tp) / denom, 4) if denom else 0.0
    return {
        "top1_hit": bool(codes) and codes[0] in gt,
        "top2_hit": any(c in gt for c in codes[:2]),
        "top3_hit": any(c in gt for c in codes[:3]),
        "recall": recall,
        "precision": precision,
        "f1": round(_f1(recall, precision), 4),
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
        "f1": 0.0,
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
# Per-case console output
# ---------------------------------------------------------------------------


def _print_model_block(
    model_name: str, dx_cnt: int, codes: list[dict], m: dict,
    suffix: str = "",
) -> None:
    """Print one model's per-case block with metrics and codes.

    Args:
        model_name: Display name for the model.
        dx_cnt: Number of diagnoses produced.
        codes: List of code dicts from the API (``code``, ``similarity``).
        m: Metrics dict with ``recall``, ``precision``, ``f1``.
        suffix: Optional parenthetical to append to the model name.
    """
    label = f"{model_name} ({suffix})" if suffix else model_name
    f1 = m.get("f1", 0.0)
    print(f"  {label:<34s} {dx_cnt:>2} dx   "
          f"R:{m['recall']:>4.0%}  P:{m['precision']:>4.0%}  F1:{f1:>4.0%}")
    if codes:
        parts = [f"{c['code']} ({c['similarity']:.0%})" for c in codes[:8]]
        trail = "  ..." if len(codes) > 8 else ""
        print(f"    {'  '.join(parts)}{trail}")
    else:
        print("    (none)")


def _evaluate_case(case: dict, index: int, total: int) -> dict | None:
    """Evaluate a single case against the /diagnose endpoint.

    Calls the API, computes metrics for both models (and the critic if
    triggered), prints a summary, and returns a details dict.

    Returns ``None`` if the API call fails (error is printed).
    """
    gt_raw = case["icd10_cm"]["codes"]
    gt = {normalize(c) for c in gt_raw}
    gt_cnt = len(gt_raw)

    try:
        result = call_diagnose(case["medical_note"])
    except Exception as e:
        print(f"[{index + 1:2d}/{total}] ERROR: {e}")
        return None

    elapsed = result.get("elapsed_s", 0)
    ra, rb = result["results"]
    res_a, res_b = ra["codes"], rb["codes"]

    codes_a = [c["code"] for c in res_a]
    codes_b = [c["code"] for c in res_b]
    dx_a, dx_b = ra.get("diagnoses", []), rb.get("diagnoses", [])
    reasoning_a, reasoning_b = ra.get("reasoning", ""), rb.get("reasoning", "")
    name_a = ra.get("model", MODEL_A)
    name_b = rb.get("model", MODEL_B)

    ma = _model_metrics(gt, codes_a, len(dx_a), gt_cnt)
    mb = _model_metrics(gt, codes_b, len(dx_b), gt_cnt)

    # -- Print per-case block -----------------------------------------------
    print(f"-- {index + 1}/{total} {'-' * 50}")
    gt_word = "code" if gt_cnt == 1 else "codes"
    print(f"  GT: {', '.join(gt_raw)}  ({gt_cnt} {gt_word})  {elapsed:5.1f}s\n")

    _print_model_block(name_a, len(dx_a), res_a, ma)
    _print_model_block(name_b, len(dx_b), res_b, mb)

    # -- Critic (if triggered) ----------------------------------------------
    critic_triggered = result.get("critic_triggered", False)
    critic_rounds = result.get("critic_rounds", [])
    critic_codes_raw: list[dict] = []
    critic_codes: list[str] = []
    critic_dx: list[str] = []
    critic_reasoning = ""
    tool_calls_count = 0
    mc = _empty_metrics()

    if critic_triggered and critic_rounds:
        final = critic_rounds[-1]
        critic_codes_raw = final.get("codes", [])
        critic_codes = [c["code"] for c in critic_codes_raw]
        critic_dx = final.get("diagnoses", [])
        critic_reasoning = final.get("reasoning", "")
        mc = _model_metrics(gt, critic_codes, len(critic_dx), gt_cnt)

        n_rounds = len(critic_rounds)
        tool_calls_count = sum(
            len(r.get("tool_calls", [])) for r in critic_rounds
        )

        parts = [f"{n_rounds} round{'s' if n_rounds != 1 else ''}"]
        if tool_calls_count:
            parts.append(f"{tool_calls_count} tool calls")
        _print_model_block(
            "critic", len(critic_dx), critic_codes_raw, mc,
            suffix=", ".join(parts),
        )

    print()

    # -- Return detail dict --------------------------------------------------
    return {
        "case": index + 1,
        "ground_truth": gt_raw,
        "elapsed_s": elapsed,
        "model_a": {
            "diagnoses": dx_a,
            "dx_count": len(dx_a),
            "codes": codes_a,
            "reasoning": reasoning_a,
            "metrics": ma,
        },
        "model_b": {
            "diagnoses": dx_b,
            "dx_count": len(dx_b),
            "codes": codes_b,
            "reasoning": reasoning_b,
            "metrics": mb,
        },
        "agreement": bool(codes_a) and bool(codes_b) and codes_a[0] == codes_b[0],
        "critic": {
            "triggered": critic_triggered,
            "rounds": len(critic_rounds),
            "tool_calls": tool_calls_count,
            "diagnoses": critic_dx,
            "dx_count": len(critic_dx),
            "codes": critic_codes,
            "reasoning": critic_reasoning,
            "metrics": mc,
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

    def _hits(key: str) -> tuple[int, int]:
        return (sum(1 for d in details if d["model_a"]["metrics"][key]),
                sum(1 for d in details if d["model_b"]["metrics"][key]))

    def _avg(key: str) -> tuple[float, float]:
        return (sum(d["model_a"]["metrics"][key] for d in details) / total,
                sum(d["model_b"]["metrics"][key] for d in details) / total)

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
        top1_c = sum(1 for d in cd if d["critic"]["metrics"]["top1_hit"])
        top2_c = sum(1 for d in cd if d["critic"]["metrics"]["top2_hit"])
        top3_c = sum(1 for d in cd if d["critic"]["metrics"]["top3_hit"])
        avg_rec_c = sum(d["critic"]["metrics"]["recall"] for d in cd) / cn
        avg_prec_c = sum(d["critic"]["metrics"]["precision"] for d in cd) / cn
        f1_c = _f1(avg_rec_c, avg_prec_c)
        avg_cat_rec_c = sum(d["critic"]["metrics"]["cat_recall"] for d in cd) / cn
        avg_cat_prec_c = sum(d["critic"]["metrics"]["cat_precision"] for d in cd) / cn
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
# Console summary table
# ---------------------------------------------------------------------------


def _print_summary(m: dict) -> None:
    """Print a side-by-side summary table to stdout."""
    t = m["total"]
    name_a = MODEL_A
    name_b = MODEL_B
    cw = max(len(name_a), len(name_b), 14)

    W = cw * 2 + 38

    def _pct(h: int, total: int) -> str:
        return f"{h}/{total} ({h / total:.1%})" if total else ""

    def _val(v: float | int) -> str:
        return f"{v:.1%}" if isinstance(v, float) else str(v)

    print()
    print("=" * W)
    print(f"  {t} cases evaluated")
    print("=" * W)

    # -- Model A vs Model B side-by-side ------------------------------------
    print(f"  {'':<28s}  {name_a:>{cw}s}    {name_b:>{cw}s}")
    print(f"  {'-' * 28}  {'-' * cw}    {'-' * cw}")

    def _row2(label: str, a: str, b: str) -> None:
        print(f"  {label:<28s}  {a:>{cw}s}    {b:>{cw}s}")

    _row2("Top-1", _pct(m["top1_a"], t), _pct(m["top1_b"], t))
    _row2("Top-2", _pct(m["top2_a"], t), _pct(m["top2_b"], t))
    _row2("Top-3", _pct(m["top3_a"], t), _pct(m["top3_b"], t))
    _row2("Recall", _val(m["avg_rec_a"]), _val(m["avg_rec_b"]))
    _row2("Precision@GT", _val(m["avg_prec_a"]), _val(m["avg_prec_b"]))
    _row2("F1", _val(m["f1_a"]), _val(m["f1_b"]))

    print(f"  -- Category level (3-char prefix) --")
    _row2("Cat Recall", _val(m["avg_cat_rec_a"]), _val(m["avg_cat_rec_b"]))
    _row2("Cat Precision", _val(m["avg_cat_prec_a"]), _val(m["avg_cat_prec_b"]))
    _row2("Cat F1", _val(m["cat_f1_a"]), _val(m["cat_f1_b"]))

    print(f"  -- Diagnosis count --")
    _row2("Avg diagnoses", f"{m['avg_dx_a']:.1f}", f"{m['avg_dx_b']:.1f}")
    _row2("Match", _pct(m["match_a"], t), _pct(m["match_b"], t))
    _row2("Over-diagnose", _pct(m["over_a"], t), _pct(m["over_b"], t))
    _row2("Under-diagnose", _pct(m["under_a"], t), _pct(m["under_b"], t))

    print()
    print(f"  Avg GT codes: {m['avg_gt']:.1f}    "
          f"Agreement (same top-1): {m['agree']}/{t} "
          f"({m['agree'] / t:.1%})")

    # -- Critic section ------------------------------------------------------
    nc = m.get("n_critic", 0)
    if nc > 0:
        print()
        print(f"  {'-' * 35}")
        print(f"  Critic ({nc} of {t} triggered)")
        print(f"  {'-' * 35}")

        def _row1(label: str, v: str) -> None:
            print(f"  {label:<28s}  {v:>{cw}s}")

        _row1("Top-1", _pct(m["top1_c"], nc))
        _row1("Top-2", _pct(m["top2_c"], nc))
        _row1("Top-3", _pct(m["top3_c"], nc))
        _row1("Recall", _val(m["avg_rec_c"]))
        _row1("Precision@GT", _val(m["avg_prec_c"]))
        _row1("F1", _val(m["f1_c"]))
        _row1("Cat Recall", _val(m["avg_cat_rec_c"]))
        _row1("Cat F1", _val(m["cat_f1_c"]))

    print("=" * W)


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def _build_metrics_json(m: dict) -> dict:
    """Convert summary metrics into the nested JSON-serialisable structure."""
    t = m["total"]

    def _model_block(p: str) -> dict:
        """Build nested metrics for one model (prefix 'a' or 'b')."""
        return {
            "top1": {"hits": m[f"top1_{p}"], "total": t,
                     "rate": round(m[f"top1_{p}"] / t, 4)},
            "top2": {"hits": m[f"top2_{p}"], "total": t,
                     "rate": round(m[f"top2_{p}"] / t, 4)},
            "top3": {"hits": m[f"top3_{p}"], "total": t,
                     "rate": round(m[f"top3_{p}"] / t, 4)},
            "recall": round(m[f"avg_rec_{p}"], 4),
            "precision": round(m[f"avg_prec_{p}"], 4),
            "f1": round(m[f"f1_{p}"], 4),
            "category": {
                "recall": round(m[f"avg_cat_rec_{p}"], 4),
                "precision": round(m[f"avg_cat_prec_{p}"], 4),
                "f1": round(m[f"cat_f1_{p}"], 4),
            },
            "diagnoses": {
                "avg_count": round(m[f"avg_dx_{p}"], 2),
                "match": {"hits": m[f"match_{p}"], "total": t,
                          "rate": round(m[f"match_{p}"] / t, 4)},
                "over": {"hits": m[f"over_{p}"], "total": t,
                         "rate": round(m[f"over_{p}"] / t, 4)},
                "under": {"hits": m[f"under_{p}"], "total": t,
                          "rate": round(m[f"under_{p}"] / t, 4)},
            },
        }

    metrics: dict = {
        "model_a": _model_block("a"),
        "model_b": _model_block("b"),
        "avg_gt_codes": round(m["avg_gt"], 2),
        "agreement": {"hits": m["agree"], "total": t,
                      "rate": round(m["agree"] / t, 4)},
    }

    nc = m.get("n_critic", 0)
    if nc > 0:
        metrics["critic"] = {
            "triggered": nc,
            "top1": {"hits": m["top1_c"], "total": nc,
                     "rate": round(m["top1_c"] / nc, 4)},
            "top2": {"hits": m["top2_c"], "total": nc,
                     "rate": round(m["top2_c"] / nc, 4)},
            "top3": {"hits": m["top3_c"], "total": nc,
                     "rate": round(m["top3_c"] / nc, 4)},
            "recall": round(m["avg_rec_c"], 4),
            "precision": round(m["avg_prec_c"], 4),
            "f1": round(m["f1_c"], 4),
            "category": {
                "recall": round(m["avg_cat_rec_c"], 4),
                "precision": round(m["avg_cat_prec_c"], 4),
                "f1": round(m["cat_f1_c"], 4),
            },
            "avg_dx": round(m["avg_dx_c"], 2),
        }

    return metrics


def _save_json(path: str, m: dict, details: list[dict]) -> None:
    """Write evaluation results (metrics + per-case details) to a JSON file."""
    with open(path, "w") as f:
        json.dump(
            {
                "cases_evaluated": m["total"],
                "models": {"a": MODEL_A, "b": MODEL_B},
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
    """CLI entry point -- parse args and run the evaluator."""
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
