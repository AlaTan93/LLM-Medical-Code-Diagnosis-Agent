"""LLM testing route: call any LiteLLM alias with a prompt.

LiteLLM has no host port (in-network only), so this endpoint on the app
(port 8000) is the sole way to exercise the models from the host. ``model`` is a
LiteLLM alias from ``docker/litellm/config.yaml`` (e.g. "ii-medical-q8",
"medical-grpo", "qwen35-medical", "A").
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from fastapi import APIRouter, HTTPException

from medicoder.schemas import TestRequest, TestResponse

router = APIRouter()

# Default prompt used when the POST body omits one.
DEFAULT_PROMPT = "What is the ICD-10-CM code for Type 2 diabetes mellitus without complications?"
# First call to a model loads it into VRAM (can take 10-60s); allow generous time.
_LLM_TIMEOUT = 120.0


# curl -X POST 'http://localhost:8000/test/ii-medical-q8' \
#   -H 'Content-Type: application/json' \
#   -d '{"prompt":"What is the ICD-10-CM code for essential hypertension?"}'
# (omit the -d body to use the default medical prompt)
@router.post("/test/{model}", response_model=TestResponse)
def test_model(model: str, body: TestRequest | None = None) -> TestResponse:
    """Send a single prompt to a LiteLLM model alias and return its reply.

    Args:
        model: A LiteLLM alias from ``docker/litellm/config.yaml`` (e.g.
            "ii-medical-q8", "qwen35-medical").
        body: Optional request body carrying the prompt; ``None`` uses the
            default medical prompt.

    Returns:
        The model's reply with the resolved prompt and elapsed wall-clock time.

    Raises:
        HTTPException: 404 if the alias is unknown to LiteLLM, 504 on timeout,
            502 if the proxy is unreachable or returns no choices.
    """
    prompt = body.prompt if body and body.prompt else DEFAULT_PROMPT
    base = os.environ.get("LLM_BASE_URL", "http://litellm:4000/v1").rstrip("/")
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=_LLM_TIMEOUT) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        # LiteLLM returns 400/404 for an unknown model alias, 429 rate-limit, etc.
        detail = e.read().decode(errors="replace")[:500]
        status = 404 if e.code in (400, 404) else e.code
        raise HTTPException(
            status_code=status,
            detail=f"model '{model}' not available via LiteLLM ({e.code}): {detail}",
        )
    except urllib.error.URLError as e:
        reason = str(e.reason)
        if "timed out" in reason.lower():
            raise HTTPException(status_code=504, detail=f"model '{model}' timed out")
        raise HTTPException(status_code=502, detail=f"LLM proxy unreachable: {reason}")
    elapsed = time.time() - started
    choices = data.get("choices") or []
    if not choices:
        raise HTTPException(status_code=502, detail=f"no choices in LiteLLM response: {data}")
    content = choices[0].get("message", {}).get("content", "")
    return TestResponse(
        model=model, prompt=prompt, response=content, elapsed_s=round(elapsed, 2)
    )
