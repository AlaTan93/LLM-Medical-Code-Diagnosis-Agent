"""Thin client for the in-network LiteLLM proxy.

Every route calls these functions instead of building raw urllib
requests — one place to change timeout, error handling, or transport.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

CHAT_TIMEOUT = float(os.environ.get("CHAT_TIMEOUT", "300.0"))
EMBED_TIMEOUT = float(os.environ.get("EMBED_TIMEOUT", "120.0"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "8192"))

_THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)


def base_url() -> str:
    """Return the LiteLLM proxy base URL.

    Returns:
        The base URL from ``LLM_BASE_URL`` env var (default
        ``http://litellm:4000/v1``), trailing slash stripped.
    """
    return os.environ.get("LLM_BASE_URL", "http://litellm:4000/v1").rstrip("/")


def chat_completion(
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.1,
    max_tokens: int = MAX_TOKENS,
    timeout: float = CHAT_TIMEOUT,
) -> tuple[str, str]:
    """Send a chat completion request and return the assistant's reply.

    Any ``<think>…</think>`` reasoning blocks are separated from the
    visible output (local reasoning models like DeepSeek-R1 emit them
    inline).

    Args:
        model: A LiteLLM alias (e.g. ``medgemma-27b-q4_k_s``).
        messages: OpenAI-format message list.
        temperature: Sampling temperature (default ``0.1`` — low randomness
            appropriate for medical coding).
        max_tokens: Maximum tokens to generate (default 8192).  Caps runaway
            generation before it wastes GPU time.
        timeout: Request timeout in seconds.

    Returns:
        A ``(content, thinking)`` tuple where *content* is the visible
        reply text (thinking stripped) and *thinking* is the raw
        ``<think>`` block content (empty string if the model didn't
        produce one).  Returns ``("", "")`` if the response has no
        choices.

    Raises:
        urllib.error.HTTPError: If LiteLLM returns an HTTP error.
        urllib.error.URLError: If the proxy is unreachable or times out.
    """
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url()}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    choices = data.get("choices") or []
    if not choices:
        return "", ""
    content = choices[0].get("message", {}).get("content", "")
    thinking, output = extract_thinking(content)
    return output, thinking


def embed(
    texts: list[str],
    *,
    model: str = "embed",
    timeout: float = EMBED_TIMEOUT,
) -> list[list[float]]:
    """Embed texts via the LiteLLM proxy.

    Args:
        texts: Input strings to embed.
        model: LiteLLM embedding alias (default ``embed`` → zembed-1).
        timeout: Request timeout in seconds.

    Returns:
        One embedding vector per input text, in order.

    Raises:
        urllib.error.HTTPError: If LiteLLM returns an HTTP error.
        urllib.error.URLError: If the proxy is unreachable or times out.
    """
    payload = json.dumps({"model": model, "input": texts}).encode()
    req = urllib.request.Request(
        f"{base_url()}/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    items = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
    return [item["embedding"] for item in items]


def extract_thinking(text: str) -> tuple[str, str]:
    """Split a model response into ``(thinking, output)``.

    Extracts content from ``<think>…</think>`` reasoning blocks and
    returns it separately from the visible output.  Handles three
    patterns produced by local reasoning models:

    1. Complete ``<think>…</think>`` blocks (most common).
    2. Unclosed ``<think>`` — truncated output where the model ran
       out of tokens mid-thought.
    3. Orphaned ``</think>`` — Ollama sometimes strips the opening
       tag, leaving only the closer.

    Args:
        text: Raw model output that may contain think blocks.

    Returns:
        A ``(thinking, output)`` tuple where *thinking* is the
        concatenated reasoning content (empty string if none) and
        *output* is the remaining visible text, trimmed.
    """
    thinking_parts: list[str] = []

    def _capture(m: re.Match) -> str:
        thinking_parts.append(m.group(1).strip())
        return ""

    cleaned = _THINK_RE.sub(_capture, text)

    # Unclosed <think> (truncated output).
    if "<think>" in cleaned:
        before, after = cleaned.split("<think>", 1)
        thinking_parts.append(after.strip())
        cleaned = before

    # Orphaned </think> (Ollama stripped opening tag).
    if "</think>" in cleaned:
        before, after = cleaned.rsplit("</think>", 1)
        thinking_parts.append(before.strip())
        cleaned = after

    thinking = "\n".join(t for t in thinking_parts if t)
    return thinking, cleaned.strip()


def strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks from a model response.

    Backward-compatible wrapper around :func:`extract_thinking` that
    returns only the visible output.

    Args:
        text: Raw model output that may contain think blocks.

    Returns:
        The visible output text with all think blocks removed.
    """
    return extract_thinking(text)[1]
