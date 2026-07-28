"""LLM testing route: call any LiteLLM alias with a prompt.

LiteLLM has no host port (in-network only), so this endpoint on the app
(port 8000) is the sole way to exercise the models from the host. ``model`` is a
LiteLLM alias from ``docker/litellm/config.yaml`` (e.g. ``ii-medical-q8``,
``qwen35-medical``, ``A``).
"""

from __future__ import annotations

import time
import urllib.error

from fastapi import APIRouter, HTTPException

from medicoder import proxy
from medicoder.schemas import TestRequest, TestResponse

router = APIRouter()

DEFAULT_PROMPT = "What is the ICD-10-CM code for Type 2 diabetes mellitus without complications?"


# curl -X POST 'http://localhost:8000/test/ii-medical-q8' \
#   -H 'Content-Type: application/json' \
#   -d '{"prompt":"What is the ICD-10-CM code for essential hypertension?"}'
@router.post("/test/{model}", response_model=TestResponse)
def test_model(model: str, body: TestRequest | None = None) -> TestResponse:
    """Send a single prompt to a LiteLLM model alias and return its reply.

    Args:
        model: A LiteLLM alias from ``docker/litellm/config.yaml``.
        body: Optional request body carrying the prompt; ``None`` uses the
            default medical prompt.

    Returns:
        The model's reply with the resolved prompt and elapsed wall-clock time.

    Raises:
        HTTPException: 404 if the alias is unknown to LiteLLM, 504 on timeout,
            502 if the proxy is unreachable.
    """
    prompt = body.prompt if body and body.prompt else DEFAULT_PROMPT
    started = time.time()
    try:
        content = proxy.chat_completion(
            model, [{"role": "user", "content": prompt}]
        )
    except urllib.error.HTTPError as e:
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
    return TestResponse(
        model=model, prompt=prompt, response=content, elapsed_s=round(elapsed, 2)
    )
