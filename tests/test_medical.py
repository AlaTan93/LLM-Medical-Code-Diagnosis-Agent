# Written by AI
"""Tests for ``medicoder.medical`` — JSON extraction and diagnosis parsing.

Covers the pure-function helpers that don't require a DB or LLM connection:
``_extract_json``, ``_parse_diagnoses``, ``_try_parse_output``,
``clean_str_list``, and ``_clean_embed_text`` (from ``db.embed_icd10``).
"""

from __future__ import annotations

from medicoder.db.embed_icd10 import _clean_embed_text
from medicoder.medical import (
    _extract_json,
    _parse_diagnoses,
    _try_parse_output,
    clean_str_list,
)

# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_direct_parse(self):
        result = _extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_markdown_fence(self):
        result = _extract_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_embedded_after_prose(self):
        result = _extract_json('Here is my answer: {"diagnoses": ["flu"]}')
        assert result == {"diagnoses": ["flu"]}

    def test_no_json(self):
        assert _extract_json("just plain text") is None

    def test_empty_string(self):
        assert _extract_json("") is None

    def test_nested_object(self):
        result = _extract_json('{"a": {"b": [1, 2]}}')
        assert result == {"a": {"b": [1, 2]}}

    def test_multiple_fences(self):
        """Should try each fence segment until one parses."""
        result = _extract_json('```python\nnot json\n```\n```json\n{"x": 1}\n```')
        assert result == {"x": 1}

    def test_brace_substring(self):
        """Strategy 3: grab outermost { ... } from noisy text."""
        result = _extract_json('blah {"deep": {"nested": true}} trailing')
        assert result == {"deep": {"nested": True}}


# ---------------------------------------------------------------------------
# _parse_diagnoses
# ---------------------------------------------------------------------------


class TestParseDiagnoses:
    def test_plain_lines(self):
        result = _parse_diagnoses("Flu\nCold\nFever")
        assert result == ["Flu", "Cold", "Fever"]

    def test_numbered(self):
        result = _parse_diagnoses("1. Flu\n2. Cold\n3. Fever")
        assert result == ["Flu", "Cold", "Fever"]

    def test_bulleted(self):
        result = _parse_diagnoses("- Flu\n- Cold\n* Fever")
        assert result == ["Flu", "Cold", "Fever"]

    def test_boxed(self):
        result = _parse_diagnoses("\\boxed{Diagnosis line}")
        assert result == ["Diagnosis line"]

    def test_empty_lines_filtered(self):
        result = _parse_diagnoses("Flu\n\n\nCold")
        assert result == ["Flu", "Cold"]

    def test_max_diagnoses_cap(self):
        lines = "\n".join(f"dx{i}" for i in range(20))
        result = _parse_diagnoses(lines)
        assert len(result) == 10

    def test_empty_input(self):
        assert _parse_diagnoses("") == []
        assert _parse_diagnoses("   ") == []


# ---------------------------------------------------------------------------
# _try_parse_output
# ---------------------------------------------------------------------------


class TestTryParseOutput:
    def test_json_with_reasoning(self):
        raw = '{"reasoning": "patient has fever", "diagnoses": ["Fever"]}'
        result = _try_parse_output(raw)
        assert result is not None
        diagnoses, reasoning = result
        assert diagnoses == ["Fever"]
        assert reasoning == "patient has fever"

    def test_json_without_reasoning(self):
        raw = '{"diagnoses": ["Flu"]}'
        result = _try_parse_output(raw)
        assert result is not None
        diagnoses, reasoning = result
        assert diagnoses == ["Flu"]
        assert reasoning == ""

    def test_fallback_to_regex(self):
        raw = "1. Some diagnosis\n2. Another one"
        result = _try_parse_output(raw)
        assert result is not None
        diagnoses, reasoning = result
        assert diagnoses == ["Some diagnosis", "Another one"]
        assert reasoning == ""

    def test_empty_output(self):
        result = _try_parse_output("")
        assert result is None


# ---------------------------------------------------------------------------
# clean_str_list
# ---------------------------------------------------------------------------


class TestCleanStrList:
    def test_strips_whitespace(self):
        assert clean_str_list(["  hello  ", "world"]) == ["hello", "world"]

    def test_filters_empty(self):
        assert clean_str_list(["a", "", "  ", "b"]) == ["a", "b"]

    def test_caps_at_max(self):
        result = clean_str_list([f"x{i}" for i in range(20)])
        assert len(result) == 10

    def test_empty_input(self):
        assert clean_str_list([]) == []


# ---------------------------------------------------------------------------
# _clean_embed_text (from db.embed_icd10)
# ---------------------------------------------------------------------------


class TestCleanEmbedText:
    def test_suffix_unspecified(self):
        assert _clean_embed_text(
            "Malignant neoplasm of cervix uteri, unspecified"
        ) == "Malignant neoplasm of cervix uteri"

    def test_prefix_unspecified(self):
        assert _clean_embed_text(
            "Unspecified viral encephalitis"
        ) == "viral encephalitis"

    def test_middle_unspecified(self):
        result = _clean_embed_text("Disorder, unspecified, right side")
        assert "unspecified" not in result.lower()

    def test_no_unspecified(self):
        assert _clean_embed_text("Essential hypertension") == "Essential hypertension"

    def test_case_insensitive(self):
        assert _clean_embed_text("Test, UNSPECIFIED") == "Test"

    def test_multiple_occurrences(self):
        result = _clean_embed_text("Unspecified disorder, unspecified type")
        assert "unspecified" not in result.lower()
