"""Coding orchestrator route: a generalist agent delegates to a medical model
and a vector-search tool to produce billable ICD-10-CM codes.

The orchestrator (the ``orchestrator`` LiteLLM alias — a tool-calling-capable
generalist, e.g. ``qwen2.5:7b``) drives a two-step pipeline:

1. ``diagnose(text)`` — forwards the clinical text *verbatim* to a medical model
   (a plain chat completion, no tools) and returns a one-sentence diagnosis.
2. ``search_icd10(query, k)`` — embeds the diagnosis and runs a pgvector cosine
   similarity search over billable ICD-10 codes, returning the top-k matches.

The medical models are used only as text generators (no tool-calling required),
sidestepping the fact that most locally-hosted medical fine-tunes cannot call
tools reliably.
"""

from __future__ import annotations

import json
import time
import urllib.error
from functools import lru_cache

from fastapi import APIRouter, HTTPException
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from medicoder import proxy
from medicoder.db.pool import get_pool
from medicoder.messages import extract_tool_results
from medicoder.schemas import CodeRequest, CodeResponse, ICD10Match

router = APIRouter()

_SYSTEM_PROMPT = """You are a medical coding orchestrator. You MUST follow these steps in order:
1. Call the diagnose tool with the user's text verbatim (do not summarize or paraphrase).
2. After receiving the diagnosis, call the search_icd10 tool with that diagnosis.
3. Report ONLY the codes returned by search_icd10 — do not invent or report codes from any other source.

CRITICAL: You must ALWAYS call search_icd10 in step 2. Never skip it, even if the diagnosis already mentions a code."""

_DIAGNOSE_SYSTEM = (
    "Produce a single concise diagnostic sentence describing the patient's "
    "condition. Do NOT include ICD-10 codes, code numbers, explanations, or "
    "markdown. Just one sentence."
)


def _make_diagnose(medical_model: str):
    """Build a ``diagnose`` tool that forwards text to ``medical_model``.

    Args:
        medical_model: A LiteLLM alias for the medical model (e.g.
            ``ii-medical-q8``) that will generate the diagnosis.

    Returns:
        A tool function ``(text: str) -> str`` that calls the medical model via
        a plain chat completion and returns the diagnosis.
    """

    def diagnose(text: str) -> str:
        """Forward clinical text to a medical model for a one-sentence diagnosis."""
        messages = [
            {"role": "system", "content": _DIAGNOSE_SYSTEM},
            {"role": "user", "content": text},
        ]
        try:
            return proxy.chat_completion(medical_model, messages)
        except urllib.error.HTTPError as e:
            return f"[diagnose error] model '{medical_model}' ({e.code}): {e.read().decode(errors='replace')[:300]}"
        except urllib.error.URLError as e:
            return f"[diagnose error] unreachable: {e.reason}"

    return diagnose


def search_icd10(query: str, k: int = 3) -> str:
    """Find the top-k billable ICD-10-CM codes matching a diagnosis description.

    Embeds the query with bge-m3 and runs a pgvector cosine-similarity search
    over all billable codes that have embeddings.

    Args:
        query: A diagnosis description to match against ICD-10 code descriptions.
        k: Maximum number of codes to return (default 3).
    """
    try:
        vectors = proxy.embed([query])
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        return json.dumps({"error": f"embedding failed: {e}"})

    if not vectors:
        return json.dumps({"error": "no embedding returned"})
    query_vec = json.dumps(vectors[0])

    with get_pool().connection() as conn:
        rows = conn.execute(
            """SELECT code, short_desc, long_desc,
                      1 - (embedding <=> %s::vector) AS similarity
               FROM icd10_codes
               WHERE is_billable AND embedding IS NOT NULL
               ORDER BY embedding <=> %s::vector
               LIMIT %s""",
            (query_vec, query_vec, k),
        ).fetchall()  # type: ignore

    return json.dumps(
        [
            {
                "code": r["code"],
                "short_desc": r["short_desc"],
                "long_desc": r["long_desc"],
                "similarity": round(float(r["similarity"]), 4),
            }
            for r in rows
        ]
    )


@lru_cache(maxsize=8)
def _build_orchestrator(medical_model: str):  # type: ignore[no-untyped-def]
    """Build (and cache) the orchestrator agent for a given medical model.

    Uses langgraph's ``create_react_agent``. The model is a ``ChatOpenAI``
    instance with ``use_responses_api=False`` so langchain-openai uses the Chat
    Completions API (LiteLLM does not implement the Responses API).

    Args:
        medical_model: A LiteLLM alias for the medical model used by the
            ``diagnose`` tool.

    Returns:
        A compiled langgraph ReAct agent.
    """
    diagnose = _make_diagnose(medical_model)
    llm = ChatOpenAI(model="orchestrator", use_responses_api=False)
    return create_react_agent(
        llm,
        tools=[diagnose, search_icd10],
        prompt=_SYSTEM_PROMPT,
    )


# curl -X POST 'http://localhost:8000/code' \
#   -H 'Content-Type: application/json' \
#   -d '{"text":"Patient has type 2 diabetes mellitus without complications"}'
@router.post("/code", response_model=CodeResponse)
def run_code_pipeline(body: CodeRequest) -> CodeResponse:
    """Run the coding orchestrator: medical diagnosis + ICD-10 vector search.

    Args:
        body: Request body with the clinical text and optional medical model.

    Returns:
        The diagnosis, top matching billable codes, and full tool-call trace.

    Raises:
        HTTPException: 500 if the orchestrator cannot be built, 504 on timeout,
            502 on any other agent run failure.
    """
    try:
        agent = _build_orchestrator(body.medical_model)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"could not build orchestrator: {e}",
        )
    started = time.time()
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": body.text}]}
        )
    except Exception as e:
        detail = str(e)
        lowered = detail.lower()
        if "timed out" in lowered or "timeout" in lowered:
            raise HTTPException(status_code=504, detail=f"orchestrator timed out: {detail}")
        raise HTTPException(status_code=502, detail=f"orchestrator failed: {detail}")
    elapsed = time.time() - started

    messages = result.get("messages") or []
    tool_results = extract_tool_results(messages)

    diagnosis = ""
    codes: list[ICD10Match] = []
    for tr in tool_results:
        if tr.tool == "diagnose":
            diagnosis = tr.result
        elif tr.tool == "search_icd10":
            try:
                codes = [ICD10Match(**m) for m in json.loads(tr.result)]
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

    return CodeResponse(
        diagnosis=diagnosis,
        codes=codes,
        tool_results=tool_results or None,
        elapsed_s=round(elapsed, 2),
    )
