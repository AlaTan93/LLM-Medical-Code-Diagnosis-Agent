"""Analyze debate-critic performance vs Model A and Model B.

For every case where the debate-critic was triggered, compares the critic's
recall, top-1 accuracy, and precision against both medical models to
determine whether the reconciliation loop added value.

Usage::

    python analyze_critic.py                           # all eval_output*.json
    python analyze_critic.py eval_output_reembed.json  # single file
    python analyze_critic.py file1.json file2.json     # explicit files
"""

from __future__ import annotations

import glob
import json
import sys

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _f1(recall: float, precision: float) -> float:
    """Harmonic mean of recall and precision (0 when both are 0)."""
    denom = recall + precision
    return 2 * recall * precision / denom if denom else 0.0


def _extract(detail: dict, label: str) -> dict:
    """Pull metrics, codes, and diagnoses from a detail entry.

    Args:
        detail: A per-case detail dict from an eval_output JSON file.
        label: One of ``"model_a"``, ``"model_b"``, ``"critic"``.

    Returns:
        Dict with ``recall``, ``precision``, ``f1``, ``top1``, ``codes``,
        ``diagnoses``, ``dx_count`` (all zeroed if the entry is absent).
    """
    e = detail.get(label, {})
    m = e.get("metrics", e)
    rec = m.get("recall", 0.0)
    prec = m.get("precision", 0.0)
    return {
        "recall": rec,
        "precision": prec,
        "f1": m.get("f1", _f1(rec, prec)),
        "top1": m.get("top1_hit", False),
        "codes": e.get("codes", []),
        "diagnoses": e.get("diagnoses", []),
        "dx_count": e.get("dx_count", 0),
    }


def _compare(
    c_rec: float, c_top1: bool, m_rec: float, m_top1: bool
) -> str:
    """Compare critic vs model on recall (primary) then top-1 (tiebreaker).

    Returns ``"+"`` (improved), ``"-"`` (regressed), or ``"="`` (same).
    """
    if c_rec > m_rec + 0.001:
        return "+"
    if c_rec < m_rec - 0.001:
        return "-"
    if c_top1 and not m_top1:
        return "+"
    if not c_top1 and m_top1:
        return "-"
    return "="


def _t1(hit: bool) -> str:
    """Compact top-1 indicator: ``"+"`` for hit, ``"-"`` for miss."""
    return "+" if hit else "-"


def _codes_str(codes: list[str], limit: int = 10) -> str:
    """Format a code list for display, truncating with an ellipsis."""
    if not codes:
        return "(none)"
    head = ", ".join(codes[:limit])
    suffix = ", ..." if len(codes) > limit else ""
    return f"[{head}{suffix}]"


def _dx_str(diagnoses: list[str], limit: int = 3) -> str:
    """Format a diagnosis list for display."""
    if not diagnoses:
        return "(none)"
    head = "; ".join(diagnoses[:limit])
    suffix = "; ..." if len(diagnoses) > limit else ""
    return f"[{head}{suffix}]"


def _short_model(name: str) -> str:
    """Shorten an Ollama registry tag for display.

    ``hf.co/unsloth/medgemma-27b-text-it-GGUF:Q4_K_S`` → ``medgemma-27b-text-it``
    """
    name = name.replace("hf.co/", "")
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    name = name.split("-GGUF")[0]
    if ":" in name:
        name = name.rsplit(":", 1)[0]
    return name[:30]


def _collect_perf_stats(details: list[dict]) -> dict:
    """Collect elapsed time and per-call token data from all cases.

    Returns a dict with:
    - ``elapsed``: list of ``elapsed_s`` values (all cases)
    - ``model_tokens``: ``{model: [(prompt_per_call, completion_per_call), ...]}``
    """
    elapsed = [
        d.get("elapsed_s", 0) for d in details if d.get("elapsed_s")
    ]

    model_tokens: dict[str, list[tuple[float, float]]] = {}
    for d in details:
        tu = d.get("token_usage")
        if not tu or not isinstance(tu, dict):
            continue
        for model, stats in tu.items():
            if model == "_combined":
                continue
            calls = stats.get("calls", 0)
            if not calls:
                continue
            model_tokens.setdefault(model, []).append((
                stats.get("prompt_tokens", 0) / calls,
                stats.get("completion_tokens", 0) / calls,
            ))

    return {"elapsed": elapsed, "model_tokens": model_tokens}


def _print_perf_stats(stats: dict) -> None:
    """Print time and per-call token statistics."""
    elapsed = stats.get("elapsed", [])
    model_tokens = stats.get("model_tokens", {})

    if not elapsed and not model_tokens:
        return

    print(f"\n{'─' * 60}")
    print("Performance stats")
    print(f"{'─' * 60}")

    if elapsed:
        avg = sum(elapsed) / len(elapsed)
        print(
            f"  Time per case ({len(elapsed)} cases):  "
            f"avg {avg:.1f}s   min {min(elapsed):.1f}s   "
            f"max {max(elapsed):.1f}s"
        )

    for model, values in sorted(model_tokens.items()):
        short = _short_model(model)
        prompts = [v[0] for v in values]
        completions = [v[1] for v in values]
        n = len(values)
        print(f"\n  Tokens per call ({short}, {n} cases):")
        print(
            f"    Input:    "
            f"avg {sum(prompts) / n:,.0f}   "
            f"min {min(prompts):,.0f}   "
            f"max {max(prompts):,.0f}"
        )
        print(
            f"    Output:   "
            f"avg {sum(completions) / n:,.0f}   "
            f"min {min(completions):,.0f}   "
            f"max {max(completions):,.0f}"
        )

    print(f"{'─' * 60}")


# ---------------------------------------------------------------------------
# Per-file analysis
# ---------------------------------------------------------------------------


def analyze_file(path: str) -> dict:
    """Analyze one eval_output file and print a per-case comparison table.

    Returns a summary dict for cross-file aggregation.
    """
    with open(path) as f:
        data = json.load(f)

    details = data.get("details", [])
    total = len(details)
    triggered = [
        d for d in details if d.get("critic", {}).get("triggered")
    ]
    n = len(triggered)

    print(f"\n{'=' * 90}")
    print(f"{path} — {n}/{total} cases triggered")
    print(f"{'=' * 90}")

    summary = {
        "file": path,
        "triggered": n,
        "total": total,
        "vs_a": {"+": 0, "-": 0, "=": 0},
        "vs_b": {"+": 0, "-": 0, "=": 0},
        "vs_best": {"+": 0, "-": 0, "=": 0},
    }

    if n == 0:
        print("  No critic-triggered cases.\n")
        return summary

    # -- Table header -------------------------------------------------------
    hdr = (
        f"{'Case':>4}  {'GT':<24} "
        f"{'A_rec':>5} {'B_rec':>5} {'C_rec':>5}  "
        f"{'A_t1':>4} {'B_t1':>4} {'C_t1':>4}  "
        f"{'vsA':>4} {'vsB':>4} {'vBst':>4}  "
        f"{'rnd':>3}"
    )
    print(f"\n{hdr}")
    print("-" * len(hdr))

    notable_up: list[dict] = []
    notable_down: list[dict] = []
    verdict_pairs: list[tuple[str, str]] = []

    for d in triggered:
        case_num = d["case"]
        gt = d.get("ground_truth", [])
        gt_str = ", ".join(gt)[:24]

        a = _extract(d, "model_a")
        b = _extract(d, "model_b")
        c = _extract(d, "critic")
        rounds = d.get("critic", {}).get("rounds", 0)

        best_rec = max(a["recall"], b["recall"])
        best_t1 = a["top1"] or b["top1"]

        va = _compare(c["recall"], c["top1"], a["recall"], a["top1"])
        vb = _compare(c["recall"], c["top1"], b["recall"], b["top1"])
        vbest = _compare(
            c["recall"], c["top1"], best_rec, best_t1
        )

        summary["vs_a"][va] += 1
        summary["vs_b"][vb] += 1
        summary["vs_best"][vbest] += 1
        verdict_pairs.append((va, vb))

        delta_best = c["recall"] - best_rec
        entry = {
            "case": case_num,
            "gt": gt,
            "a": a,
            "b": b,
            "c": c,
            "rounds": rounds,
            "va": va,
            "vb": vb,
            "vbest": vbest,
            "delta_best": delta_best,
        }
        if delta_best > 0.3:
            notable_up.append(entry)
        elif delta_best < -0.3:
            notable_down.append(entry)

        print(
            f"{case_num:>4}  {gt_str:<24} "
            f"{a['recall']:>5.3f} {b['recall']:>5.3f} {c['recall']:>5.3f}  "
            f"   {_t1(a['top1'])}    {_t1(b['top1'])}    {_t1(c['top1'])}  "
            f"{va:>4} {vb:>4} {vbest:>4}  "
            f"{rounds:>3}"
        )

    # -- Summary ------------------------------------------------------------
    both_up = sum(1 for va, vb in verdict_pairs if va == "+" and vb == "+")
    both_down = sum(1 for va, vb in verdict_pairs if va == "-" and vb == "-")

    print(f"\nSummary ({n} triggered cases):")
    print(
        f"  {'':>16}"
        f"{'improved':>10}{'regressed':>11}{'same':>6}{'net':>6}"
    )
    for label, key in [
        ("vs Model A", "vs_a"),
        ("vs Model B", "vs_b"),
        ("vs Best(A,B)", "vs_best"),
    ]:
        cc = summary[key]
        net = cc["+"] - cc["-"]
        print(
            f"  {label:>16}"
            f"{cc['+']:>10}{cc['-']:>11}{cc['=']:>6}{net:>+6}"
        )

    print(f"\n  Improved over BOTH models: {both_up} cases")
    print(f"  Regressed vs BOTH models:  {both_down} cases")

    # -- Notable cases ------------------------------------------------------
    _print_notable("Notable improvements (recall delta > 0.3 vs Best):",
                   notable_up)
    _print_notable("Notable regressions (recall delta < -0.3 vs Best):",
                   notable_down)

    # -- Performance stats --------------------------------------------------
    perf = _collect_perf_stats(details)
    _print_perf_stats(perf)

    summary["both_up"] = both_up
    summary["both_down"] = both_down
    return summary


def _print_notable(title: str, entries: list[dict]) -> None:
    """Print detail for cases where the critic made a big difference."""
    if not entries:
        return
    print(f"\n{title}")
    for e in sorted(entries, key=lambda x: -abs(x["delta_best"])):
        a, b, c = e["a"], e["b"], e["c"]
        gt = ", ".join(e["gt"])
        print(f"\n  Case {e['case']}  GT: {gt}  "
              f"(rounds: {e['rounds']}, delta: {e['delta_best']:>+.3f})")
        print(f"    Model A  R={a['recall']:.3f}  P={a['precision']:.3f}  "
              f"codes: {_codes_str(a['codes'])}")
        print(f"    Model B  R={b['recall']:.3f}  P={b['precision']:.3f}  "
              f"codes: {_codes_str(b['codes'])}")
        print(f"    Critic   R={c['recall']:.3f}  P={c['precision']:.3f}  "
              f"codes: {_codes_str(c['codes'])}")
        print(f"    Critic dx: {_dx_str(c['diagnoses'])}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    paths: list[str] = []
    for arg in sys.argv[1:]:
        matched = glob.glob(arg)
        paths.extend(matched if matched else [arg])
    if not paths:
        paths = sorted(glob.glob("eval_output*.json"))

    if not paths:
        print("No eval_output*.json files found.")
        sys.exit(1)

    summaries = [analyze_file(p) for p in paths]

    # -- Cross-file aggregate ----------------------------------------------
    if len(summaries) > 1:
        total_trig = sum(s["triggered"] for s in summaries)
        if total_trig == 0:
            print("\nNo critic-triggered cases in any file.")
            return

        print(f"\n{'#' * 90}")
        print(f"Cross-file aggregate ({len(summaries)} files, "
              f"{total_trig} triggered cases)")
        print(f"{'#' * 90}\n")

        print(
            f"  {'File':<52} {'trig':>5} "
            f"{'+A':>4} {'-A':>4} {'+B':>4} {'-B':>4} "
            f"{'+bst':>5} {'-bst':>5} {'both+':>6} {'both-':>6}"
        )
        print("  " + "-" * 88)
        for s in summaries:
            if s["triggered"] == 0:
                continue
            print(
                f"  {s['file']:<52} {s['triggered']:>5} "
                f"{s['vs_a']['+']:>4} {s['vs_a']['-']:>4} "
                f"{s['vs_b']['+']:>4} {s['vs_b']['-']:>4} "
                f"{s['vs_best']['+']:>5} {s['vs_best']['-']:>5} "
                f"{s.get('both_up', 0):>6} {s.get('both_down', 0):>6}"
            )

        # Totals
        agg = {"vs_a": {"+": 0, "-": 0, "=": 0},
               "vs_b": {"+": 0, "-": 0, "=": 0},
               "vs_best": {"+": 0, "-": 0, "=": 0}}
        for s in summaries:
            for key in agg:
                for sym in "+-=":
                    agg[key][sym] += s[key][sym]
        total_up = sum(s.get("both_up", 0) for s in summaries)
        total_down = sum(s.get("both_down", 0) for s in summaries)

        print("  " + "-" * 88)
        print(
            f"  {'TOTAL':<52} {total_trig:>5} "
            f"{agg['vs_a']['+']:>4} {agg['vs_a']['-']:>4} "
            f"{agg['vs_b']['+']:>4} {agg['vs_b']['-']:>4} "
            f"{agg['vs_best']['+']:>5} {agg['vs_best']['-']:>5} "
            f"{total_up:>6} {total_down:>6}"
        )


if __name__ == "__main__":
    main()
