"""Deterministic coding pipeline: medical diagnoses + ICD-10 vector search.

Two-step pipeline that runs in fixed order (no LLM orchestration needed):

1. **Diagnose** — forwards the clinical text to a medical model (a plain
   chat completion, no tools) and returns 1-10 independent diagnoses.
2. **Search** — embeds each diagnosis with bge-m3 and runs a pgvector
   cosine-similarity search over billable ICD-10 codes, returning the
   best match per diagnosis (deduplicated, sorted by similarity).

The medical models are used only as text generators (no tool-calling required),
sidestepping the fact that most locally-hosted medical fine-tunes cannot call
tools reliably.
"""

from __future__ import annotations

import time

from fastapi import APIRouter

from medicoder.medical import diagnose, search_icd10
from medicoder.schemas import (
    CodeRequest,
    CodeResponse,
    ICD10Match,
    ToolResult,
)

router = APIRouter()


# curl -X POST 'http://localhost:8000/code' \
#   -H 'Content-Type: application/json' \
#   -d '{"text":"Patient has type 2 diabetes mellitus without complications"}'
@router.post("/code", response_model=CodeResponse)
def run_code_pipeline(body: CodeRequest) -> CodeResponse:
    """Run the coding pipeline: medical diagnoses + ICD-10 vector search.

    Args:
        body: Request body with the clinical text and optional medical model.

    Returns:
        The diagnoses, matching billable codes, and a step-by-step trace.
    """
    started = time.time()
    tool_results: list[ToolResult] = []
    diagnoses: list[str] = []

    # Step 1 — diagnose: medical model produces 1-10 independent diagnoses.
    try:
        diagnoses = diagnose(body.text, body.medical_model)
        tool_results.append(
            ToolResult(
                tool="diagnose",
                args={"text": body.text},
                result=" | ".join(diagnoses) if diagnoses else "(none)",
            )
        )
    except Exception as e:
        tool_results.append(
            ToolResult(tool="diagnose", args={"text": body.text}, result=f"[error] {e}")
        )

    # Step 2 — search: embed each diagnosis, find matching ICD-10 codes.
    codes: list[ICD10Match] = []
    if diagnoses:
        try:
            codes = search_icd10(diagnoses, k=body.k)
            summary = (
                "no matches"
                if not codes
                else ", ".join(f"{c.code} ({c.similarity})" for c in codes)
            )
            tool_results.append(
                ToolResult(
                    tool="search_icd10",
                    args={"diagnoses": diagnoses, "k": body.k},
                    result=summary,
                )
            )
        except Exception as e:
            tool_results.append(
                ToolResult(
                    tool="search_icd10",
                    args={"diagnoses": diagnoses, "k": body.k},
                    result=f"[error] {e}",
                )
            )

    elapsed = time.time() - started
    return CodeResponse(
        diagnoses=diagnoses,
        codes=codes,
        tool_results=tool_results,
        elapsed_s=round(elapsed, 2),
    )
