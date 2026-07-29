"""Dual-diagnosis pipeline: two medical models + ICD-10 vector search.

Uses a LangGraph :class:`StateGraph` to run two medical models in parallel
(fan-out), then deterministically search both diagnoses for matching billable
ICD-10-CM codes (fan-in). The graph topology guarantees both models are called
before the search step — no prompt engineering needed.

    START
      ├──→ diagnose_a (ii-medical-q8)       ──┐
      ├──→ diagnose_b (deepseek-r1-medical-cot) ─┤  parallel
      │                                        ↓
      │                                   fan-in
      │                                        ↓
      │                               search_both (k=2 each)
      │                                        ↓
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
MODEL_B = "deepseek-r1-medical-cot"


class DiagnoseState(TypedDict):
    """Mutable state passed between graph nodes.

    Attributes:
        text: The original clinical text (set at invocation).
        diagnosis_a: Diagnosis from model A (set by ``diagnose_a``).
        diagnosis_b: Diagnosis from model B (set by ``diagnose_b``).
        codes_a: Top-2 ICD-10 matches for diagnosis A (set by ``search_both``).
        codes_b: Top-2 ICD-10 matches for diagnosis B (set by ``search_both``).
    """

    text: str
    diagnosis_a: str
    diagnosis_b: str
    codes_a: list[ICD10Match]
    codes_b: list[ICD10Match]


# ── Node functions ──────────────────────────────────────────────────────────


def diagnose_a(state: DiagnoseState) -> dict:
    """Call the first medical model (``ii-medical-q8``) for a diagnosis."""
    try:
        return {"diagnosis_a": diagnose(state["text"], MODEL_A)}
    except Exception as e:
        return {"diagnosis_a": f"[error] {e}"}


def diagnose_b(state: DiagnoseState) -> dict:
    """Call the second medical model (``deepseek-r1-medical-cot``)."""
    try:
        return {"diagnosis_b": diagnose(state["text"], MODEL_B)}
    except Exception as e:
        return {"diagnosis_b": f"[error] {e}"}


def search_both(state: DiagnoseState) -> dict:
    """Run ICD-10 vector search (k=2) for each non-error diagnosis."""
    return {
        "codes_a": _safe_search(state.get("diagnosis_a", "")),
        "codes_b": _safe_search(state.get("diagnosis_b", "")),
    }


def _safe_search(diagnosis: str) -> list[ICD10Match]:
    """Run search_icd10, returning [] on error or empty/error diagnoses."""
    if not diagnosis or diagnosis.startswith("[error]"):
        return []
    try:
        return search_icd10(diagnosis, k=2)
    except Exception:
        return []


# ── Graph construction ──────────────────────────────────────────────────────


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


# ── Route ───────────────────────────────────────────────────────────────────

# curl -X POST 'http://localhost:8000/diagnose' \
#   -H 'Content-Type: application/json' \
#   -d '{"text":"Patient has type 2 diabetes mellitus without complications"}'
@router.post("/diagnose", response_model=DualDiagnoseResponse)
def run_diagnose(body: DiagnoseRequest) -> DualDiagnoseResponse:
    """Run two medical models in parallel and search both diagnoses for codes.

    Args:
        body: Request body with the clinical text.

    Returns:
        One :class:`DiagnosisResult` per model (diagnosis + top-2 ICD-10
        matches), plus elapsed time.
    """
    started = time.time()
    result = _graph.invoke(
        {"text": body.text, "diagnosis_a": "", "diagnosis_b": ""}
    )
    elapsed = time.time() - started

    return DualDiagnoseResponse(
        results=[
            DiagnosisResult(
                model=MODEL_A,
                diagnosis=result.get("diagnosis_a", ""),
                codes=result.get("codes_a", []),
            ),
            DiagnosisResult(
                model=MODEL_B,
                diagnosis=result.get("diagnosis_b", ""),
                codes=result.get("codes_b", []),
            ),
        ],
        elapsed_s=round(elapsed, 2),
    )
