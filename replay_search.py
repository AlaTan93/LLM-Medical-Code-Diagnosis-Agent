"""Replay stored diagnoses through the new hybrid search to measure impact.

Reads every ``eval_output*.json`` file, extracts the diagnoses that models
A, B, and the critic produced in past eval runs, sends them to the
``POST /search`` endpoint (which uses the current hybrid FTS+vector search),
and compares the new codes/metrics against the original results.

This isolates the **search-only** effect — prompt changes are not tested
because old diagnoses came from old prompts.

Usage::

    python replay_search.py
"""

from __future__ import annotations

import glob
import json
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field

BASE_URL = "http://localhost:8000"
CASES_PATH = "data/icd10_cm_cases.json"


def normalize(code: str) -> str:
    return code.replace(".", "")


def _compute_metrics(
    gt: set[str], codes: list[str]
) -> tuple[bool, bool, bool, float]:
    """Return (top1_hit, top2_hit, top3_hit, recall)."""
    if not codes:
        return False, False, False, 0.0
    top1 = codes[0] in gt
    top2 = any(c in gt for c in codes[:2])
    top3 = any(c in gt for c in codes[:3])
    hits = sum(1 for c in codes if c in gt)
    recall = hits / len(gt) if gt else 0.0
    return top1, top2, top3, recall


def call_search(diagnoses: list[str], k: int = 3) -> list[str]:
    """POST diagnoses to /search, return code strings."""
    payload = json.dumps({"diagnoses": diagnoses, "k": k}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/search",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        matches = json.loads(resp.read())
    return [m["code"] for m in matches]


@dataclass
class CaseAgg:
    """Aggregated old-vs-new metrics for one case across all eval entries."""

    case: int
    gt_codes: list[str]
    note: str
    old_top1: list[bool] = field(default_factory=list)
    new_top1: list[bool] = field(default_factory=list)
    old_top3: list[bool] = field(default_factory=list)
    new_top3: list[bool] = field(default_factory=list)
    old_recall: list[float] = field(default_factory=list)
    new_recall: list[float] = field(default_factory=list)
    new_codes_samples: list[tuple[str, list[str]]] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.old_top1)

    @property
    def old_t1_pct(self) -> float:
        return sum(self.old_top1) / self.n if self.n else 0.0

    @property
    def new_t1_pct(self) -> float:
        return sum(self.new_top1) / self.n if self.n else 0.0

    @property
    def old_t3_pct(self) -> float:
        return sum(self.old_top3) / self.n if self.n else 0.0

    @property
    def new_t3_pct(self) -> float:
        return sum(self.new_top3) / self.n if self.n else 0.0

    @property
    def old_rec(self) -> float:
        return sum(self.old_recall) / self.n if self.n else 0.0

    @property
    def new_rec(self) -> float:
        return sum(self.new_recall) / self.n if self.n else 0.0

    @property
    def delta_recall(self) -> float:
        return self.new_rec - self.old_rec

    @property
    def verdict(self) -> str:
        d = self.delta_recall
        t1_delta = self.new_t1_pct - self.old_t1_pct
        if d > 0.01 or t1_delta > 0.01:
            return "IMPROVED"
        if d < -0.01 or t1_delta < -0.01:
            return "REGRESSED"
        return "same"


def _truncate(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s[:n] + "..." if len(s) > n else s


def main() -> None:
    with open(CASES_PATH) as f:
        test_cases = json.load(f)["cases"]

    eval_files = sorted(glob.glob("eval_output*.json"))
    print(f"Loaded {len(eval_files)} eval files")

    # Phase 1: collect all entries and unique diagnosis sets.
    entries: list[tuple[int, str, list[str], list[str], bool, bool, float]] = []
    search_cache: dict[tuple[str, ...], list[str]] = {}
    unique_queries: set[tuple[str, ...]] = set()

    for ef_path in eval_files:
        with open(ef_path) as f:
            ef = json.load(f)
        for detail in ef.get("details", []):
            case_num = detail["case"]
            for label in ("model_a", "model_b", "critic"):
                entry = detail.get(label, {})
                if not entry:
                    continue
                if label == "critic" and not entry.get("triggered"):
                    continue
                diagnoses = entry.get("diagnoses", [])
                if not diagnoses:
                    continue
                old_codes = [normalize(c) for c in entry.get("codes", [])]
                entries.append(
                    (case_num, label, diagnoses, old_codes)
                )
                unique_queries.add(tuple(diagnoses))

    print(f"{len(entries)} entries, {len(unique_queries)} unique diagnosis sets\n")

    # Phase 2: search unique diagnosis sets only.
    errors = 0
    for i, diag_tuple in enumerate(sorted(unique_queries), 1):
        if i % 10 == 0 or i == len(unique_queries):
            print(f"  searching... {i}/{len(unique_queries)}", flush=True)
        try:
            search_cache[diag_tuple] = call_search(list(diag_tuple), k=3)
        except Exception as e:
            print(f"  ERROR: {e}")
            search_cache[diag_tuple] = []
            errors += 1

    # Phase 3: compute metrics using cached results.
    aggregated: dict[int, CaseAgg] = {}
    total_entries = 0

    for case_num, label, diagnoses, old_codes in entries:
        if case_num not in aggregated:
            tc = test_cases[case_num - 1]
            aggregated[case_num] = CaseAgg(
                case=case_num,
                gt_codes=tc["icd10_cm"]["codes"],
                note=tc["medical_note"],
            )
        agg = aggregated[case_num]
        gt = {normalize(c) for c in agg.gt_codes}

        old_t1, _, old_t3, old_rec = _compute_metrics(gt, old_codes)
        new_raw = search_cache.get(tuple(diagnoses), [])
        new_codes = [normalize(c) for c in new_raw]
        new_t1, _, new_t3, new_rec = _compute_metrics(gt, new_codes)

        agg.old_top1.append(old_t1)
        agg.new_top1.append(new_t1)
        agg.old_top3.append(old_t3)
        agg.new_top3.append(new_t3)
        agg.old_recall.append(old_rec)
        agg.new_recall.append(new_rec)
        total_entries += 1

        if agg.verdict != "same" and len(agg.new_codes_samples) < 2:
            agg.new_codes_samples.append((label, new_raw[:5]))

    if errors:
        print(f"\n{errors} search errors (skipped)\n")

    # --- Print table ---
    cases = sorted(aggregated.values(), key=lambda c: -abs(c.delta_recall))

    hdr = (
        f"{'Case':>4}  {'GT':<20} {'Old T1':>6} {'New T1':>6} "
        f"{'Old T3':>6} {'New T3':>6} "
        f"{'Old Rec':>7} {'New Rec':>7} {'dRec':>6}  Verdict"
    )
    print(hdr)
    print("-" * len(hdr))

    n_improved = 0
    n_regressed = 0
    n_unchanged = 0

    for c in cases:
        if c.n == 0:
            continue
        if c.verdict == "IMPROVED":
            n_improved += 1
        elif c.verdict == "REGRESSED":
            n_regressed += 1
        else:
            n_unchanged += 1

        gt = ", ".join(c.gt_codes)[:20]
        print(
            f"{c.case:>4}  {gt:<20} "
            f"{c.old_t1_pct * 100:>5.0f}% {c.new_t1_pct * 100:>5.0f}% "
            f"{c.old_t3_pct * 100:>5.0f}% {c.new_t3_pct * 100:>5.0f}% "
            f"{c.old_rec:>7.3f} {c.new_rec:>7.3f} "
            f"{c.delta_recall:>+6.3f}  {c.verdict}"
        )

    print(f"\nSummary: {n_improved} improved | {n_regressed} regressed | "
          f"{n_unchanged} unchanged ({total_entries} entries replayed)")

    # --- Detail for changed cases ---
    changed = [c for c in cases if c.verdict != "same" and c.new_codes_samples]
    if changed:
        print(f"\n{'=' * 80}")
        print("Sample new codes for changed cases:")
        print("=" * 80)
        for c in changed:
            gt = ", ".join(c.gt_codes)
            print(f"\n  Case {c.case} (GT: {gt})")
            print(f"  {_truncate(c.note, 76)}")
            for tag, codes in c.new_codes_samples:
                print(f"    [{tag}] -> {', '.join(codes)}")


if __name__ == "__main__":
    main()
