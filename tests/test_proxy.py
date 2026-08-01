"""Tests for ``medicoder.proxy`` — thinking block separation.

Tests the three ``<think>`` block patterns produced by local reasoning models:
complete blocks, unclosed blocks, and orphaned closers.
"""

from __future__ import annotations

from medicoder.proxy import extract_thinking, strip_thinking


class TestCompleteBlock:
    def test_basic(self):
        text = "<think>reasoning here</think>visible output"
        thinking, output = extract_thinking(text)
        assert thinking == "reasoning here"
        assert output == "visible output"

    def test_multiline(self):
        text = "<think>line1\nline2\nline3</think>result"
        thinking, output = extract_thinking(text)
        assert "line1" in thinking
        assert "line3" in thinking
        assert output == "result"

    def test_multiple_blocks(self):
        text = "<think>first</think>mid<think>second</think>end"
        thinking, output = extract_thinking(text)
        assert "first" in thinking
        assert "second" in thinking
        assert output == "midend"

    def test_no_block(self):
        thinking, output = extract_thinking("just output")
        assert thinking == ""
        assert output == "just output"

    def test_empty_block(self):
        thinking, output = extract_thinking("<think></think>output")
        assert thinking == ""
        assert output == "output"


class TestUnclosedBlock:
    def test_truncated(self):
        """Model ran out of tokens mid-thought — no closing tag."""
        text = "visible<think>truncated reasoning without end"
        thinking, output = extract_thinking(text)
        assert "truncated reasoning" in thinking
        assert output == "visible"

    def text_leading_think_truncated(self):
        """Starts with <think>, never closes."""
        text = "<think>all thinking, no close"
        thinking, output = extract_thinking(text)
        assert "all thinking" in thinking
        assert output == ""


class TestOrphanedCloser:
    def test_orphaned_close(self):
        """Ollama stripped the opening tag, leaving only the closer."""
        text = "reasoning content</think>visible output"
        thinking, output = extract_thinking(text)
        assert "reasoning content" in thinking
        assert output == "visible output"

    def test_orphaned_close_multiline(self):
        text = "step1\nstep2\nstep3</think>answer"
        thinking, output = extract_thinking(text)
        assert "step1" in thinking
        assert "step3" in thinking
        assert output == "answer"


class TestStripThinking:
    def test_returns_output_only(self):
        assert strip_thinking("<think>x</think>hello") == "hello"

    def test_no_block(self):
        assert strip_thinking("hello") == "hello"

    def test_preserves_content(self):
        assert strip_thinking("<think>secret</think>keep this") == "keep this"
