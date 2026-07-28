"""DeepAgent route: run a ``deepagents`` agent on a prompt.

The agent is built with :func:`deepagents.create_deep_agent` and routed through
the same LiteLLM proxy as ``/test/{model}`` (model is a LiteLLM alias, e.g.
``qwen35-medical``). LangChain's ``openai:`` provider reads ``OPENAI_BASE_URL``
and ``OPENAI_API_KEY`` from the environment, so the agent needs no hard-coded
endpoint — it inherits the in-network LiteLLM URL set on the container.

Tool surface is locked down: a provider-level harness profile excludes the
default filesystem / todos / subagent tools so the model only sees ``echo``.
"""

from __future__ import annotations

import time
from functools import lru_cache

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    ProviderProfile,
    create_deep_agent,
    register_harness_profile,
    register_provider_profile,
)
from fastapi import APIRouter, HTTPException

from medicoder.schemas import TestRequest, TestResponse

router = APIRouter()

# All LiteLLM aliases are reached via the OpenAI-compatible provider prefix, so
# a single provider-level profile governs every model this route can select.
register_harness_profile(
    "openai",
    HarnessProfile(
        excluded_tools=frozenset(
            {
                "ls",
                "read_file",
                "write_file",
                "edit_file",
                "delete",
                "glob",
                "grep",
                "execute",
                "write_todos",
            }
        ),
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
    ),
)

# langchain-openai auto-selects OpenAI's Responses API, which LiteLLM's
# /v1/chat/completions endpoint does not implement (it returns malformed text
# blocks that break response parsing). Force the Chat Completions API for every
# openai:* model routed through LiteLLM.
register_provider_profile(
    "openai",
    ProviderProfile(init_kwargs={"use_responses_api": False}),
)

_SYSTEM_PROMPT = "You are a medical coding assistant. Use the available tools when asked."


def echo(text: str) -> str:
    """Echo back the provided text verbatim."""
    return text


# Building a DeepAgent compiles a LangGraph; cache one instance per model alias
# so repeated calls don't pay that cost.
@lru_cache(maxsize=8)
def _build_agent(model: str):  # type: ignore[no-untyped-def]
    return create_deep_agent(
        model=f"openai:{model}",
        tools=[echo],
        system_prompt=_SYSTEM_PROMPT,
    )


# curl -X POST 'http://localhost:8000/agent/qwen35-medical' \
#   -H 'Content-Type: application/json' \
#   -d '{"prompt":"Use the echo tool to repeat: hello"}'
# (omit the -d body to use the default prompt, which exercises the tool)
@router.post("/agent/{model}", response_model=TestResponse)
def run_agent(model: str, body: TestRequest | None = None) -> TestResponse:
    # Default prompt nudges the agent to actually invoke the echo tool.
    prompt = (
        body.prompt
        if body and body.prompt
        else "Use the echo tool to repeat exactly: hello from medicoder"
    )
    try:
        agent = _build_agent(model)
    except Exception as e:  # model spec / provider misconfiguration.
        raise HTTPException(
            status_code=500,
            detail=f"could not build agent for '{model}': {e}",
        )
    started = time.time()
    try:
        # First call to a model loads it into VRAM (10-60s); the agent may make
        # several LLM round-trips, so allow generous wall-clock.
        result = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )
    except Exception as e:
        detail = str(e)
        lowered = detail.lower()
        if "timed out" in lowered or "timeout" in lowered:
            raise HTTPException(status_code=504, detail=f"agent timed out: {detail}")
        if "model" in lowered and ("not found" in lowered or "404" in lowered or "400" in lowered):
            raise HTTPException(
                status_code=404,
                detail=f"model '{model}' not available via LiteLLM: {detail}",
            )
        raise HTTPException(status_code=502, detail=f"agent run failed: {detail}")
    elapsed = time.time() - started
    messages = result.get("messages") or []
    content = ""
    if messages:
        raw = getattr(messages[-1], "content", "")
        # Chat models may return content as a string or as a list of content
        # blocks (e.g. [{"type": "text", "text": "..."}]); collapse to text.
        if isinstance(raw, str):
            content = raw
        elif isinstance(raw, list):
            parts = []
            for block in raw:
                if isinstance(block, dict):
                    parts.append(block.get("text", "") or "")
                elif isinstance(block, str):
                    parts.append(block)
            content = "".join(parts)
        else:
            content = str(raw)
    return TestResponse(
        model=model, prompt=prompt, response=content, elapsed_s=round(elapsed, 2)
    )
