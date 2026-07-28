"""Deterministic coding pipeline: medical diagnosis + ICD-10 vector search.

Two-step pipeline that runs in fixed order (no LLM orchestration needed):

1. **Diagnose** — forwards the clinical text to a medical model (a plain
   chat completion, no tools) and returns a one-sentence diagnosis.
2. **Search** — embeds the diagnosis with bge-m3 and runs a pgvector
   cosine-similarity search over billable ICD-10 codes, returning the
   top-k matches.

The medical models are used only as text generators (no tool-calling required),
sidestepping the fact that most locally-hosted medical fine-tunes cannot call
tools reliably.
"""

from __future__ import annotations

import json
import time

from fastapi import APIRouter

from medicoder import proxy
from medicoder.db.pool import get_pool
from medicoder.schemas import (
    CodeRequest,
    CodeResponse,
    ICD10Match,
    ToolResult,
)

router = APIRouter()

_DIAGNOSE_SYSTEM = (
    "Produce a single concise diagnostic sentence describing the patient's "
    "condition. Do NOT include ICD-10 codes, code numbers, explanations, or "
    "markdown. Just one sentence."
)


def diagnose(text: str, model: str) -> str:
    """Forward clinical text to a medical model for a one-sentence diagnosis.

    Args:
        text: Clinical text or patient description.
        model: LiteLLM alias for the medical model (e.g. ``ii-medical-q8``).

    Returns:
        A one-sentence diagnosis (thinking blocks stripped).
    """
    messages = [
        {"role": "system", "content": _DIAGNOSE_SYSTEM},
        {"role": "user", "content": text},
    ]
    return proxy.chat_completion(model, messages)


def search_icd10(query: str, k: int = 3) -> list[ICD10Match]:
    """Find the top-k billable ICD-10-CM codes matching a description.

    Embeds the query with bge-m3 and runs a pgvector cosine-similarity search
    over billable codes that have embeddings.

    Args:
        query: A diagnosis description to match against ICD-10 codes.
        k: Maximum number of codes to return (default 3).

    Returns:
        Matching codes ranked by similarity (highest first).
    """
    vectors = proxy.embed([query])
    query_vec = json.dumps(vectors[0])

    with get_pool().connection() as conn:
        conn.execute("SET LOCAL hnsw.ef_search = 200")
        rows = conn.execute(
            """SELECT code, short_desc, long_desc,
                      1 - (embedding <=> %s::vector) AS similarity
               FROM icd10_codes
               WHERE is_billable AND embedding IS NOT NULL
               ORDER BY embedding <=> %s::vector
               LIMIT %s""",
            (query_vec, query_vec, k),
        ).fetchall()  # type: ignore

    return [
        ICD10Match(
            code=r["code"],
            short_desc=r["short_desc"],
            long_desc=r["long_desc"],
            similarity=round(float(r["similarity"]), 4),
        )
        for r in rows
    ]


# curl -X POST 'http://localhost:8000/code' \
#   -H 'Content-Type: application/json' \
#   -d '{"text":"Patient has type 2 diabetes mellitus without complications"}'
@router.post("/code", response_model=CodeResponse)
def run_code_pipeline(body: CodeRequest) -> CodeResponse:
    """Run the coding pipeline: medical diagnosis + ICD-10 vector search.

    Args:
        body: Request body with the clinical text and optional medical model.

    Returns:
        The diagnosis, top matching billable codes, and a step-by-step trace.
    """
    started = time.time()
    tool_results: list[ToolResult] = []
    diagnosis = ""

    # Step 1 — diagnose: medical model produces a one-sentence summary.
    try:
        diagnosis = diagnose(body.text, body.medical_model)
        tool_results.append(
            ToolResult(tool="diagnose", args={"text": body.text}, result=diagnosis)
        )
    except Exception as e:
        tool_results.append(
            ToolResult(tool="diagnose", args={"text": body.text}, result=f"[error] {e}")
        )

    # Step 2 — search: embed the diagnosis, find matching ICD-10 codes.
    codes: list[ICD10Match] = []
    if diagnosis:
        try:
            codes = search_icd10(diagnosis)
            summary = (
                "no matches"
                if not codes
                else ", ".join(f"{c.code} ({c.similarity})" for c in codes)
            )
            tool_results.append(
                ToolResult(
                    tool="search_icd10",
                    args={"query": diagnosis, "k": 3},
                    result=summary,
                )
            )
        except Exception as e:
            tool_results.append(
                ToolResult(
                    tool="search_icd10",
                    args={"query": diagnosis, "k": 3},
                    result=f"[error] {e}",
                )
            )

    elapsed = time.time() - started
    return CodeResponse(
        diagnosis=diagnosis,
        codes=codes,
        tool_results=tool_results,
        elapsed_s=round(elapsed, 2),
    )
