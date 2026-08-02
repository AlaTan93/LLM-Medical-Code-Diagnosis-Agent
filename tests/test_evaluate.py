# Written by AI
"""Tests for ``evaluate`` — code normalization and metric computations.

Tests the pure helpers used by the host-side evaluator that don't need a
running server: ``normalize``, ``_f1``, ``_cat_recall``, ``_cat_precision``,
``_model_metrics``, ``_empty_metrics``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import (
    _cat_recall,
    _empty_metrics,
    _f1,
    _model_metrics,
    normalize,
)


class TestNormalize:
    def test_strips_period(self):
        assert normalize("E11.9") == "E119"

    def test_multiple_periods(self):
        assert normalize("T18.128A") == "T18128A"

    def test_no_period(self):
        assert normalize("E119") == "E119"

    def test_empty(self):
        assert normalize("") == ""


class TestEvaluateF1:
    def test_perfect(self):
        assert _f1(1.0, 1.0) == 1.0

    def test_zero_both(self):
        assert _f1(0.0, 0.0) == 0.0

    def test_known_value(self):
        result = _f1(0.5, 0.5)
        assert abs(result - 0.5) < 0.001


class TestEvaluateCatRecall:
    def test_all_categories_found(self):
        gt = {"I1019", "E1190"}
        codes = ["I1011", "E1100", "C509"]
        assert _cat_recall(gt, codes) == 1.0

    def test_some_categories_found(self):
        gt = {"I1019", "C509"}
        codes = ["I1011"]
        assert abs(_cat_recall(gt, codes) - 0.5) < 0.001

    def test_nothing_found(self):
        gt = {"I1019"}
        codes = ["C509", "A419"]
        assert _cat_recall(gt, codes) == 0.0


class TestEvaluateModelMetrics:
    def test_top1_hit(self):
        m = _model_metrics({"E119"}, ["E119", "I10"], 2, 1)
        assert m["top1_hit"] is True

    def test_top1_miss_top3_hit(self):
        m = _model_metrics({"E119"}, ["I10", "C509", "E119"], 3, 1)
        assert m["top1_hit"] is False
        assert m["top3_hit"] is True

    def test_f1_present(self):
        m = _model_metrics({"E119"}, ["E119"], 1, 1)
        assert "f1" in m

    def test_recall_fraction(self):
        gt = {"E119", "I10", "C509"}
        m = _model_metrics(gt, ["E119", "I10"], 2, 3)
        assert abs(m["recall"] - 2 / 3) < 0.01


class TestEmptyMetrics:
    def test_has_all_keys(self):
        m = _empty_metrics()
        assert set(m.keys()) == {
            "top1_hit", "top2_hit", "top3_hit",
            "recall", "precision", "f1",
            "cat_recall", "cat_precision",
        }

    def test_all_zero(self):
        m = _empty_metrics()
        assert all(v == 0.0 or v is False for v in m.values())
