"""Agent testing route: run a tool-calling agent on a prompt.

Uses langchain's ``create_agent`` routed through the LiteLLM proxy.
The agent has two simple tools (``echo`` and ``get_flag``) for testing
tool-calling behaviour. ``model`` is a LiteLLM alias; only tool-calling-capable
models are usable (``ii-medical-q8`` and ``deepseek-r1-medical-cot`` are known
to emit structured tool calls).
"""

from __future__ import annotations

import time
from functools import lru_cache

from fastapi import APIRouter, HTTPException
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from medicoder.messages import extract_tool_results, last_content
from medicoder.schemas import TestRequest, TestResponse

router = APIRouter()

_SYSTEM_PROMPT = "You are a medical coding assistant. Use the available tools when asked."


def echo(text: str) -> str:
    """Echo back the provided text verbatim."""
    return text


def get_flag(text: str) -> str:
    """Gives the flag if the argument is 'hello'."""
    if text == "hello":
        return "{FLAG}_6cfc6bd484e8ff8301657eb4447f9eee71599bdb07ac98f4cf8e4d5d2ec07ccf"
    return "nope"


@lru_cache(maxsize=8)
def _build_agent(model: str):  # type: ignore[no-untyped-def]
    """Build (and cache) a tool-calling agent for a LiteLLM alias.

    Args:
        model: A LiteLLM alias (e.g. ``ii-medical-q8``). Only tool-calling-capable
            models are usable.

    Returns:
        A compiled langchain agent.
    """
    llm = ChatOpenAI(model=model, use_responses_api=False)
    return create_agent(model = llm, 
                        tools=[echo, get_flag], 
                        system_prompt=_SYSTEM_PROMPT)


# curl -X POST 'http://localhost:8000/agent/ii-medical-q8' \
#   -H 'Content-Type: application/json' \
#   -d '{"prompt":"Use the echo tool to repeat: hello"}'
@router.post("/agent/{model}", response_model=TestResponse)
def run_agent(model: str, body: TestRequest | None = None) -> TestResponse:
    """Run a tool-calling agent on a prompt and return its final reply + tool trace.

    Args:
        model: A LiteLLM alias for a tool-calling-capable model.
        body: Optional request body carrying the prompt; ``None`` uses a default
            prompt that exercises the ``echo`` tool.

    Returns:
        The agent's final answer with the resolved prompt, tool results, and
        elapsed time.

    Raises:
        HTTPException: 500 if the agent cannot be built, 504 on timeout, 404 if
            the alias is unknown, 502 on any other agent run failure.
    """
    prompt = (
        body.prompt
        if body and body.prompt
        else "Use the echo tool to repeat exactly: hello from medicoder"
    )
    try:
        agent = _build_agent(model)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"could not build agent for '{model}': {e}",
        )
    started = time.time()
    try:
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
    tool_results = extract_tool_results(messages)
    return TestResponse(
        model=model,
        prompt=prompt,
        response=last_content(messages),
        elapsed_s=round(elapsed, 2),
        tool_results=tool_results or None,
    )
