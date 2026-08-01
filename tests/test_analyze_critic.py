"""Tests for ``analyze_critic`` — verdict comparison logic.

Tests ``_compare`` and ``_f1`` — the pure functions that determine whether
the critic improved, regressed, or stayed the same vs each model.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyze_critic import _compare, _f1


class TestCompare:
    def test_improved_recall(self):
        assert _compare(0.8, False, 0.5, False) == "+"

    def test_regressed_recall(self):
        assert _compare(0.3, False, 0.5, False) == "-"

    def test_same_recall_same_top1(self):
        assert _compare(0.5, False, 0.5, False) == "="

    def test_same_recall_critic_top1_better(self):
        assert _compare(0.5, True, 0.5, False) == "+"

    def test_same_recall_critic_top1_worse(self):
        assert _compare(0.5, False, 0.5, True) == "-"

    def test_improved_recall_ignores_top1_regression(self):
        """Recall is primary — if it improves, verdict is '+' regardless of top1."""
        assert _compare(0.8, False, 0.5, True) == "+"

    def test_both_zero(self):
        assert _compare(0.0, False, 0.0, False) == "="

    def test_float_tolerance(self):
        """Tiny differences within 0.001 are treated as equal."""
        assert _compare(0.5001, False, 0.5000, False) == "="


class TestAnalyzeCriticF1:
    def test_perfect(self):
        assert _f1(1.0, 1.0) == 1.0

    def test_zero(self):
        assert _f1(0.0, 0.0) == 0.0

    def test_known_value(self):
        result = _f1(0.6, 0.4)
        expected = 2 * 0.6 * 0.4 / 1.0
        assert abs(result - expected) < 0.001
