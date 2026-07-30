"""Dual-diagnosis pipeline: two medical models + ICD-10 vector search.

Uses a LangGraph :class:`StateGraph` to run two medical models in parallel
(fan-out), then deterministically search both models' diagnoses for matching
billable ICD-10-CM codes (fan-in). The graph topology guarantees both models
are called before the search step — no prompt engineering needed.

    START
      +--> diagnose_a (ii-medical-q8)      --+
      +--> diagnose_b (gemma-4-medical-q6) --+  parallel
                                               |
                                            fan-in
                                               |
                                       search_both (k per dx)
                                               |
                                              END
"""

from __future__ import annotations

import time
from typing import TypedDict

from fastapi import APIRouter
from langgraph.graph import END, START, StateGraph

from medicoder.medical import diagnose, search_icd10
from medicoder.schemas import (
    DiagnosisResult,
    DiagnoseRequest,
    DualDiagnoseResponse,
    ICD10Match,
)

router = APIRouter()

MODEL_A = "ii-medical-q8"
MODEL_B = "gemma-4-medical-q6"


class DiagnoseState(TypedDict):
    """Mutable state passed between graph nodes.

    Attributes:
        text: The original clinical text (set at invocation).
        k: Maximum ICD-10 codes to return per diagnosis (default 3, set
            at invocation from the request body).
        diagnoses_a: Diagnoses from model A (set by ``diagnose_a``).
        diagnoses_b: Diagnoses from model B (set by ``diagnose_b``).
        reasoning_a: Clinical reasoning from model A (set by ``diagnose_a``).
        reasoning_b: Clinical reasoning from model B (set by ``diagnose_b``).
        codes_a: ICD-10 matches for model A (set by ``search_both``).
        codes_b: ICD-10 matches for model B (set by ``search_both``).
    """

    text: str
    k: int
    diagnoses_a: list[str]
    diagnoses_b: list[str]
    reasoning_a: str
    reasoning_b: str
    codes_a: list[ICD10Match]
    codes_b: list[ICD10Match]


# -- Node functions ---------------------------------------------------------


def diagnose_a(state: DiagnoseState) -> dict:
    """Call the first medical model (``ii-medical-q8``) for diagnoses."""
    try:
        diagnoses, reasoning = diagnose(state["text"], MODEL_A)
        return {"diagnoses_a": diagnoses, "reasoning_a": reasoning}
    except Exception as e:
        return {"diagnoses_a": [f"[error] {e}"], "reasoning_a": ""}


def diagnose_b(state: DiagnoseState) -> dict:
    """Call the second medical model (``gemma-4-medical-q6``)."""
    try:
        diagnoses, reasoning = diagnose(state["text"], MODEL_B)
        return {"diagnoses_b": diagnoses, "reasoning_b": reasoning}
    except Exception as e:
        return {"diagnoses_b": [f"[error] {e}"], "reasoning_b": ""}


def search_both(state: DiagnoseState) -> dict:
    """Run ICD-10 vector search for each model's diagnoses."""
    k = state.get("k", 3)
    return {
        "codes_a": _safe_search(state.get("diagnoses_a", []), k),
        "codes_b": _safe_search(state.get("diagnoses_b", []), k),
    }


def _safe_search(diagnoses: list[str], k: int = 3) -> list[ICD10Match]:
    """Run search_icd10, returning [] on error or when all diagnoses are invalid.

    Filters out empty strings and ``[error]``-prefixed entries (emitted by
    ``diagnose_a``/``diagnose_b`` when the model call fails) so that error
    placeholders don't pollute the vector search results.
    """
    valid = [d for d in diagnoses if d and not d.startswith("[error]")]
    if not valid:
        return []
    try:
        return search_icd10(valid, k=k)
    except Exception:
        return []


# -- Graph construction -----------------------------------------------------


def _build_graph():  # type: ignore[no-untyped-def]
    """Compile the LangGraph state graph for dual diagnosis."""
    g = StateGraph(DiagnoseState)
    g.add_node("diagnose_a", diagnose_a)
    g.add_node("diagnose_b", diagnose_b)
    g.add_node("search", search_both)

    # Fan-out: both diagnose nodes run in parallel from START.
    g.add_edge(START, "diagnose_a")
    g.add_edge(START, "diagnose_b")

    # Fan-in: search runs after both diagnoses complete.
    g.add_edge("diagnose_a", "search")
    g.add_edge("diagnose_b", "search")

    g.add_edge("search", END)
    return g.compile()


_graph = _build_graph()


# -- Route ------------------------------------------------------------------

# curl -X POST 'http://localhost:8000/diagnose' \
#   -H 'Content-Type: application/json' \
#   -d '{"text":"Patient has type 2 diabetes mellitus without complications"}'
@router.post("/diagnose", response_model=DualDiagnoseResponse)
def run_diagnose(body: DiagnoseRequest) -> DualDiagnoseResponse:
    """Run two medical models in parallel and search both for codes.

    Args:
        body: Request body with the clinical text and optional k.

    Returns:
        One :class:`DiagnosisResult` per model (diagnoses + ICD-10 matches),
        plus elapsed time.
    """
    started = time.time()
    result = _graph.invoke(
        {
            "text": body.text,
            "k": body.k,
            "diagnoses_a": [],
            "diagnoses_b": [],
            "reasoning_a": "",
            "reasoning_b": "",
        }  # type: ignore
    )
    elapsed = time.time() - started

    return DualDiagnoseResponse(
        results=[
            DiagnosisResult(
                model=MODEL_A,
                diagnoses=result.get("diagnoses_a", []),
                codes=result.get("codes_a", []),
                reasoning=result.get("reasoning_a", ""),
            ),
            DiagnosisResult(
                model=MODEL_B,
                diagnoses=result.get("diagnoses_b", []),
                codes=result.get("codes_b", []),
                reasoning=result.get("reasoning_b", ""),
            ),
        ],
        elapsed_s=round(elapsed, 2),
    )
