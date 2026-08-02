"""Rank test cases by identification difficulty across all eval runs.

Loads every ``eval_output*.json`` from the repo root, aggregates per-case
results across all model evaluations (model A, model B, and critic when
triggered), and prints a table sorted from most-likely-to-succeed to
most-likely-to-fail.

Usage::

    python analyze_cases.py
"""

from __future__ import annotations

import glob
import json
from collections import Counter
from dataclasses import dataclass

CASES_PATH = "data/icd10_cm_cases.json"


@dataclass
class CaseAgg:
    """Aggregated metrics for one test case across all eval files."""

    case: int
    gt_codes: list[str]
    note: str
    top1_hits: int = 0
    top3_hits: int = 0
    recall_sum: float = 0.0
    num_evals: int = 0

    @property
    def top1_pct(self) -> float:
        return self.top1_hits / self.num_evals if self.num_evals else 0.0

    @property
    def top3_pct(self) -> float:
        return self.top3_hits / self.num_evals if self.num_evals else 0.0

    @property
    def avg_recall(self) -> float:
        return self.recall_sum / self.num_evals if self.num_evals else 0.0

    @property
    def status(self) -> str:
        p = self.top1_pct
        if p >= 0.75:
            return "Easy"
        if p >= 0.25:
            return "Moderate"
        if p > 0.0:
            return "Hard"
        return "Failed"


def _load_eval_files() -> list[dict]:
    """Load all eval_output*.json files from the repo root.

    Returns:
        List of parsed eval-file dicts (each with ``details``).
    """
    files = sorted(glob.glob("eval_output*.json"))
    data = []
    for f in files:
        with open(f) as fh:
            data.append(json.load(fh))
    print(f"Loaded {len(files)} eval files: {', '.join(files)}\n")
    return data


def _load_cases() -> list[dict]:
    """Load test cases for GT codes and note snippets.

    Returns:
        List of case dicts from ``data/icd10_cm_cases.json``.
    """
    with open(CASES_PATH) as f:
        return json.load(f)["cases"]


def aggregate() -> list[CaseAgg]:
    """Aggregate per-case metrics across all eval files.

    Returns:
        List of :class:`CaseAgg` instances, one per test case.
    """
    eval_files = _load_eval_files()
    test_cases = _load_cases()
    aggregated: dict[int, CaseAgg] = {}

    for ef in eval_files:
        for detail in ef.get("details", []):
            num = detail["case"]

            if num not in aggregated:
                tc = test_cases[num - 1]
                aggregated[num] = CaseAgg(
                    case=num,
                    gt_codes=detail.get("ground_truth", tc["icd10_cm"]["codes"]),
                    note=tc["medical_note"],
                )

            agg = aggregated[num]

            entries = []
            if "model_a" in detail:
                entries.append(detail["model_a"])
            if "model_b" in detail:
                entries.append(detail["model_b"])
            if detail.get("critic", {}).get("triggered"):
                entries.append(detail["critic"])

            for e in entries:
                m = e.get("metrics", e)
                agg.num_evals += 1
                if m.get("top1_hit"):
                    agg.top1_hits += 1
                if m.get("top3_hit"):
                    agg.top3_hits += 1
                agg.recall_sum += m.get("recall", 0.0)

    return list(aggregated.values())


def _truncate(s: str, n: int) -> str:
    """Collapse whitespace and truncate to *n* chars with ellipsis.

    Args:
        s: Input string.
        n: Maximum length before truncation.

    Returns:
        Whitespace-collapsed string, truncated with ``"..."`` if needed.
    """
    s = " ".join(s.split())
    return s[:n] + "..." if len(s) > n else s


def print_table(cases: list[CaseAgg]) -> None:
    """Print the analysis table sorted best-to-worst.

    Args:
        cases: List of :class:`CaseAgg` instances to display.
    """
    cases.sort(key=lambda c: (-c.top1_pct, -c.avg_recall))

    total_evals = sum(c.num_evals for c in cases)
    print(f"{len(cases)} cases, {total_evals} total model-evaluations\n")

    hdr = (
        f"{'Case':>4}  {'GT Codes':<22} {'Note':<46} "
        f"{'#Evals':>6} {'Top1':>5} {'Top3':>5} {'Recall':>6}  Status"
    )
    print(hdr)
    print("-" * len(hdr))

    prev_status = None
    for c in cases:
        if prev_status and prev_status != c.status:
            print("-" * len(hdr))

        gt = ", ".join(c.gt_codes)[:22]
        note = _truncate(c.note, 44)
        print(
            f"{c.case:>4}  {gt:<22} {note:<46} "
            f"{c.num_evals:>6} {c.top1_pct * 100:>4.0f}% "
            f"{c.top3_pct * 100:>4.0f}% {c.avg_recall:>6.3f}  {c.status}"
        )
        prev_status = c.status

    counts = Counter(c.status for c in cases)
    print()
    print(
        f"Summary: {counts.get('Easy', 0)} Easy (top1>=75%) | "
        f"{counts.get('Moderate', 0)} Moderate (25-75%) | "
        f"{counts.get('Hard', 0)} Hard (<25%) | "
        f"{counts.get('Failed', 0)} Failed (0%)"
    )


def main() -> None:
    """CLI entry point -- aggregate metrics and print the case difficulty table."""
    cases = aggregate()
    print_table(cases)


if __name__ == "__main__":
    main()
