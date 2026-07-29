"""Evaluator: measure ICD-10-CM coding accuracy of the /diagnose pipeline.

Runs on the host outside the container. Loads test cases from
``data/icd10_cm_cases.json``, calls the ``/diagnose`` endpoint for each case,
and reports top-1 through top-5 accuracy, recall, precision, F1, and model
agreement.

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
MODEL_B = "deepseek-r1-medical-cot"


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


def evaluate(limit: int | None, save_path: str | None) -> None:
    """Run the evaluation and print results."""
    cases = load_cases()
    if limit is not None:
        cases = cases[:limit]
    n = len(cases)

    top1_a = top1_b = 0
    top2_a = top2_b = 0
    top3_a = top3_b = 0
    agree = 0
    recall_a_sum = recall_b_sum = 0.0
    precision_a_sum = precision_b_sum = 0.0
    dx_count_a_sum = dx_count_b_sum = gt_count_sum = 0
    match_a = match_b = over_a = over_b = under_a = under_b = 0
    details: list[dict] = []

    for i, case in enumerate(cases):
        gt_raw = case["icd10_cm"]["codes"]
        gt = {normalize(c) for c in gt_raw}

        label = f"[{i + 1:2d}/{n}]"
        try:
            result = call_diagnose(case["medical_note"])
        except Exception as e:
            print(f"{label} ERROR: {e}")
            details.append({"case": i + 1, "error": str(e)})
            continue

        elapsed = result.get("elapsed_s", 0)
        res_a = result["results"][0]["codes"]
        res_b = result["results"][1]["codes"]
        codes_a = [c["code"] for c in res_a]
        codes_b = [c["code"] for c in res_b]
        dx_a = result["results"][0].get("diagnoses", [])
        dx_b = result["results"][1].get("diagnoses", [])
        dx_cnt_a = len(dx_a)
        dx_cnt_b = len(dx_b)
        gt_cnt = len(gt_raw)

        set_a = set(codes_a)
        set_b = set(codes_b)
        tp_a = set_a & gt
        tp_b = set_b & gt

        hit_1a = bool(codes_a) and codes_a[0] in gt
        hit_1b = bool(codes_b) and codes_b[0] in gt
        hit_2a = any(c in gt for c in codes_a[:2])
        hit_2b = any(c in gt for c in codes_b[:2])
        hit_3a = any(c in gt for c in codes_a[:3])
        hit_3b = any(c in gt for c in codes_b[:3])
        same_top1 = (
            bool(codes_a)
            and bool(codes_b)
            and codes_a[0] == codes_b[0]
        )

        rec_a = len(tp_a) / len(gt) if gt else 0.0
        rec_b = len(tp_b) / len(gt) if gt else 0.0
        denom_a = max(dx_cnt_a, gt_cnt)
        denom_b = max(dx_cnt_b, gt_cnt)
        prec_a = len(tp_a) / denom_a if denom_a else 0.0
        prec_b = len(tp_b) / denom_b if denom_b else 0.0

        if hit_1a:
            top1_a += 1
        if hit_1b:
            top1_b += 1
        if hit_2a:
            top2_a += 1
        if hit_2b:
            top2_b += 1
        if hit_3a:
            top3_a += 1
        if hit_3b:
            top3_b += 1
        if same_top1:
            agree += 1

        recall_a_sum += rec_a
        recall_b_sum += rec_b
        precision_a_sum += prec_a
        precision_b_sum += prec_b

        dx_count_a_sum += dx_cnt_a
        dx_count_b_sum += dx_cnt_b
        gt_count_sum += gt_cnt
        if dx_cnt_a == gt_cnt:
            match_a += 1
        elif dx_cnt_a > gt_cnt:
            over_a += 1
        else:
            under_a += 1
        if dx_cnt_b == gt_cnt:
            match_b += 1
        elif dx_cnt_b > gt_cnt:
            over_b += 1
        else:
            under_b += 1

        a_str = ", ".join(f"{c['code']} ({c['similarity']:.0%})" for c in res_a) or "(none)"
        b_str = ", ".join(f"{c['code']} ({c['similarity']:.0%})" for c in res_b) or "(none)"
        mark_a = "+" if hit_2a else "-"
        mark_b = "+" if hit_2b else "-"
        gt_word = "code" if gt_cnt == 1 else "codes"
        print(f"{label} {elapsed:5.1f}s  GT: {', '.join(gt_raw)} ({gt_cnt} {gt_word})")
        print(f"{'':9s}A [{mark_a}] {dx_cnt_a}dx  {a_str}  R:{rec_a:.0%} P:{prec_a:.0%}")
        print(f"{'':9s}B [{mark_b}] {dx_cnt_b}dx  {b_str}  R:{rec_b:.0%} P:{prec_b:.0%}")
        details.append(
            {
                "case": i + 1,
                "ground_truth": gt_raw,
                "elapsed_s": elapsed,
                "model_a": {
                    "diagnoses": result["results"][0].get("diagnoses", []),
                    "dx_count": dx_cnt_a,
                    "codes": codes_a,
                    "top1_hit": hit_1a,
                    "top2_hit": hit_2a,
                    "top3_hit": hit_3a,
                    "recall": round(rec_a, 4),
                    "precision": round(prec_a, 4),
                },
                "model_b": {
                    "diagnoses": result["results"][1].get("diagnoses", []),
                    "dx_count": dx_cnt_b,
                    "codes": codes_b,
                    "top1_hit": hit_1b,
                    "top2_hit": hit_2b,
                    "top3_hit": hit_3b,
                    "recall": round(rec_b, 4),
                    "precision": round(prec_b, 4),
                },
                "agreement": same_top1,
            }
        )

    total = len(details)
    if total == 0:
        print("No cases evaluated.")
        return

    avg_rec_a = recall_a_sum / total
    avg_rec_b = recall_b_sum / total
    avg_prec_a = precision_a_sum / total
    avg_prec_b = precision_b_sum / total
    denom_a = avg_rec_a + avg_prec_a
    denom_b = avg_rec_b + avg_prec_b
    f1_a = 2 * avg_rec_a * avg_prec_a / denom_a if denom_a else 0.0
    f1_b = 2 * avg_rec_b * avg_prec_b / denom_b if denom_b else 0.0
    avg_dx_a = dx_count_a_sum / total
    avg_dx_b = dx_count_b_sum / total
    avg_gt = gt_count_sum / total

    rows = [
        (f"Top-1  ({MODEL_A})", f"{top1_a}/{total}", f"{top1_a / total:.1%}"),
        (f"Top-1  ({MODEL_B})", f"{top1_b}/{total}", f"{top1_b / total:.1%}"),
        (f"Top-2  ({MODEL_A})", f"{top2_a}/{total}", f"{top2_a / total:.1%}"),
        (f"Top-2  ({MODEL_B})", f"{top2_b}/{total}", f"{top2_b / total:.1%}"),
        (f"Top-3  ({MODEL_A})", f"{top3_a}/{total}", f"{top3_a / total:.1%}"),
        (f"Top-3  ({MODEL_B})", f"{top3_b}/{total}", f"{top3_b / total:.1%}"),
        (f"Recall  ({MODEL_A})", "", f"{avg_rec_a:.1%}"),
        (f"Recall  ({MODEL_B})", "", f"{avg_rec_b:.1%}"),
        (f"Prec@GT ({MODEL_A})", "", f"{avg_prec_a:.1%}"),
        (f"Prec@GT ({MODEL_B})", "", f"{avg_prec_b:.1%}"),
        (f"F1  ({MODEL_A})", "", f"{f1_a:.1%}"),
        (f"F1  ({MODEL_B})", "", f"{f1_b:.1%}"),
        ("Avg GT codes", "", f"{avg_gt:.1f}"),
        (f"Avg diagnoses  ({MODEL_A})", "", f"{avg_dx_a:.1f}"),
        (f"Avg diagnoses  ({MODEL_B})", "", f"{avg_dx_b:.1f}"),
        (f"Count match    ({MODEL_A})", f"{match_a}/{total}", f"{match_a / total:.1%}"),
        (f"Count match    ({MODEL_B})", f"{match_b}/{total}", f"{match_b / total:.1%}"),
        (f"Over-dx        ({MODEL_A})", f"{over_a}/{total}", f"{over_a / total:.1%}"),
        (f"Over-dx        ({MODEL_B})", f"{over_b}/{total}", f"{over_b / total:.1%}"),
        (f"Under-dx       ({MODEL_A})", f"{under_a}/{total}", f"{under_a / total:.1%}"),
        (f"Under-dx       ({MODEL_B})", f"{under_b}/{total}", f"{under_b / total:.1%}"),
        ("Agreement (same top-1)", f"{agree}/{total}", f"{agree / total:.1%}"),
    ]
    w = max(len(label) for label, _, _ in rows)

    print()
    print("=" * (w + 18))
    print(f"Cases evaluated: {total}")
    print(f"{'':{w}s}{'hits':>8s}  {'rate':>6s}")
    print("-" * (w + 18))
    for label, hits, rate in rows:
        print(f"{label:<{w}s}{hits:>8s}  {rate:>6s}")
    print("=" * (w + 18))

    if save_path:
        with open(save_path, "w") as f:
            json.dump(
                {
                    "cases_evaluated": total,
                    "metrics": {
                        "top1_a": top1_a,
                        "top1_a_pct": round(top1_a / total, 4),
                        "top1_b": top1_b,
                        "top1_b_pct": round(top1_b / total, 4),
                        "top2_a": top2_a,
                        "top2_a_pct": round(top2_a / total, 4),
                        "top2_b": top2_b,
                        "top2_b_pct": round(top2_b / total, 4),
                        "top3_a": top3_a,
                        "top3_a_pct": round(top3_a / total, 4),
                        "top3_b": top3_b,
                        "top3_b_pct": round(top3_b / total, 4),
                        "recall_a": round(avg_rec_a, 4),
                        "recall_b": round(avg_rec_b, 4),
                        "precision_a": round(avg_prec_a, 4),
                        "precision_b": round(avg_prec_b, 4),
                        "f1_a": round(f1_a, 4),
                        "f1_b": round(f1_b, 4),
                        "avg_gt_codes": round(avg_gt, 2),
                        "avg_dx_a": round(avg_dx_a, 2),
                        "avg_dx_b": round(avg_dx_b, 2),
                        "count_match_a": match_a,
                        "count_match_a_pct": round(match_a / total, 4),
                        "count_match_b": match_b,
                        "count_match_b_pct": round(match_b / total, 4),
                        "over_dx_a": over_a,
                        "over_dx_a_pct": round(over_a / total, 4),
                        "over_dx_b": over_b,
                        "over_dx_b_pct": round(over_b / total, 4),
                        "under_dx_a": under_a,
                        "under_dx_a_pct": round(under_a / total, 4),
                        "under_dx_b": under_b,
                        "under_dx_b_pct": round(under_b / total, 4),
                        "agreement": agree,
                        "agreement_pct": round(agree / total, 4),
                    },
                    "details": details,
                },
                f,
                indent=2,
            )
        print(f"\nDetailed results saved to {save_path}")


def main() -> int:
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
