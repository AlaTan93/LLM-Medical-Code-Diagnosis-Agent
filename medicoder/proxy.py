"""Thin client for the in-network LiteLLM proxy.

Every route calls these functions instead of building raw urllib
requests — one place to change timeout, error handling, or transport.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

CHAT_TIMEOUT = 300.0
EMBED_TIMEOUT = 60.0
MAX_TOKENS = 8192

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def base_url() -> str:
    """The LiteLLM proxy base URL (e.g. ``http://litellm:4000/v1``)."""
    return os.environ.get("LLM_BASE_URL", "http://litellm:4000/v1").rstrip("/")


def chat_completion(
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.0,
    max_tokens: int = MAX_TOKENS,
    timeout: float = CHAT_TIMEOUT,
) -> str:
    """Send a chat completion request and return the assistant's reply text.

    Any ``<think>…</think>`` reasoning blocks are stripped from the response
    (local Qwen3-based models emit them inline).

    Args:
        model: A LiteLLM alias (e.g. ``ii-medical-q8``).
        messages: OpenAI-format message list.
        temperature: Sampling temperature (default ``0.0`` for deterministic
            output appropriate for medical coding).
        max_tokens: Maximum tokens to generate (default 8192).  Caps runaway
            generation before it wastes GPU time.
        timeout: Request timeout in seconds.

    Returns:
        The content text from the first choice (thinking stripped), or ``""``
        if the response has no choices.

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
        return ""
    content = choices[0].get("message", {}).get("content", "")
    return strip_thinking(content)


def embed(
    texts: list[str],
    *,
    model: str = "embed",
    timeout: float = EMBED_TIMEOUT,
) -> list[list[float]]:
    """Embed texts via the LiteLLM proxy.

    Args:
        texts: Input strings to embed.
        model: LiteLLM embedding alias (default ``embed`` → bge-m3).
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


def strip_thinking(text: str) -> str:
    """Remove ``<think>…</think>`` reasoning blocks from a model response.

    Strips complete think blocks; if a block is unclosed (truncated output),
    everything from the opening ``<think>`` onward is removed.  Orphaned
    ``</think>`` closing tags (left behind when Ollama strips opening tags)
    are handled by keeping only the text after the last ``</think>``.

    Args:
        text: Raw model output that may contain think blocks.

    Returns:
        The visible content, trimmed.
    """
    cleaned = _THINK_RE.sub("", text)
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>")[0]
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[-1]
    return cleaned.strip()
