# Written by AI
"""Dual-diagnosis pipeline: two medical models + ICD-10 vector search.

Uses a LangGraph :class:`StateGraph` to run two medical models in parallel
(fan-out), then deterministically search both models' diagnoses for matching
billable ICD-10-CM codes (fan-in).  When the two models disagree, an optional
debate-critic loop (in :mod:`medicoder.critic`) reconciles their outputs.

    START
      +--> diagnose_a (medgemma-27b-q4_k_s)       --+
      +--> diagnose_b (deepseek-r1-medical-cot) --+  parallel
                                                   |
                                                fan-in
                                                   |
                                           search_both (k per dx)
                                                   |
                                           should_critic?
                                              ╱        ╲
                                           no            yes
                                            │              │
                                           END     critic_think ──→ critic_search
                                                                ╱          ╲
                                                       should_continue_critic?
                                                           ╱          ╲
                                                        loop            end
                                                         │                │
                                                  critic_think          END
                                                     (next round)
"""

from __future__ import annotations

import os
import time

from fastapi import APIRouter
from langgraph.graph import END, START, StateGraph

from medicoder.critic import (
    critic_agent,
    should_continue_critic,
    should_critic,
)
from medicoder.medical import diagnose, safe_search
from medicoder.schemas import (
    DiagnoseRequest,
    DiagnoseState,
    DiagnosisResult,
    DualDiagnoseResponse,
)

router = APIRouter()

MODEL_A = os.environ.get("MODEL_A", "medgemma-27b-q4_k_s")
MODEL_B = os.environ.get("MODEL_B", "deepseek-r1-medical-cot")


# -- Node functions ---------------------------------------------------------


def diagnose_a(state: DiagnoseState) -> dict:
    """Call the first medical model (``MODEL_A``) for diagnoses.

    Args:
        state: Current LangGraph state containing the clinical text.

    Returns:
        State update with ``diagnoses_a``, ``reasoning_a``, ``thinking_a``.
    """
    try:
        diagnoses, reasoning, thinking = diagnose(state["text"], MODEL_A)
        return {"diagnoses_a": diagnoses, "reasoning_a": reasoning, "thinking_a": thinking}
    except Exception as e:
        return {"diagnoses_a": [f"[error] {e}"], "reasoning_a": "", "thinking_a": ""}


def diagnose_b(state: DiagnoseState) -> dict:
    """Call the second medical model (``MODEL_B``) for diagnoses.

    Args:
        state: Current LangGraph state containing the clinical text.

    Returns:
        State update with ``diagnoses_b``, ``reasoning_b``, ``thinking_b``.
    """
    try:
        diagnoses, reasoning, thinking = diagnose(state["text"], MODEL_B)
        return {"diagnoses_b": diagnoses, "reasoning_b": reasoning, "thinking_b": thinking}
    except Exception as e:
        return {"diagnoses_b": [f"[error] {e}"], "reasoning_b": "", "thinking_b": ""}


def search_both(state: DiagnoseState) -> dict:
    """Run ICD-10 vector search for each model's diagnoses.

    Args:
        state: LangGraph state with both models' diagnoses populated.

    Returns:
        State update with ``codes_a`` and ``codes_b`` (list of
        :class:`~medicoder.schemas.ICD10Match`).
    """
    k = state.get("k", 3)
    return {
        "codes_a": safe_search(state.get("diagnoses_a", []), k),
        "codes_b": safe_search(state.get("diagnoses_b", []), k),
    }


# -- Graph construction -----------------------------------------------------


def _build_graph():  # type: ignore[no-untyped-def]
    """Compile the LangGraph state graph for dual diagnosis + critic loop.

    Returns:
        Compiled LangGraph ready for ``.invoke()``.
    """
    g = StateGraph(DiagnoseState)
    g.add_node("diagnose_a", diagnose_a)
    g.add_node("diagnose_b", diagnose_b)
    g.add_node("search", search_both)
    g.add_node("critic_agent", critic_agent)

    # Fan-out: both diagnose nodes run in parallel from START.
    g.add_edge(START, "diagnose_a")
    g.add_edge(START, "diagnose_b")

    # Fan-in: search runs after both diagnoses complete.
    g.add_edge("diagnose_a", "search")
    g.add_edge("diagnose_b", "search")

    # Conditional: trigger critic agent or skip to END.
    g.add_conditional_edges("search", should_critic, {
        "critic": "critic_agent",
        "end": END,
    })

    # Critic loop: agent runs (with internal tool iterations) then check.
    g.add_conditional_edges("critic_agent", should_continue_critic, {
        "loop": "critic_agent",
        "end": END,
    })

    return g.compile()


_graph = _build_graph()


# -- Route ------------------------------------------------------------------

# curl -X POST 'http://localhost:8000/diagnose' \
# -H 'Content-Type: application/json' \
# -d '{"text":"Patient has type 2 diabetes mellitus without complications"}'
@router.post("/diagnose", response_model=DualDiagnoseResponse)
def run_diagnose(body: DiagnoseRequest) -> DualDiagnoseResponse:
    """Run two medical models in parallel and search both for codes.

    When the models disagree, an optional debate-critic loop reconciles
    their outputs over up to ``MAX_CRITIC_ROUNDS`` rounds (configurable
    via environment variable; set ``enable_critic=false`` in the request
    to disable).

    Args:
        body: Request body with the clinical text, optional k, and
            ``enable_critic`` flag.

    Returns:
        One :class:`DiagnosisResult` per model (diagnoses + ICD-10
        matches), the full critic round trace if triggered, and elapsed
        time.
    """
    started = time.time()
    result = _graph.invoke(
        {
            "text": body.text,
            "k": body.k,
            "enable_critic": body.enable_critic,
            "model_a": MODEL_A,
            "model_b": MODEL_B,
            "diagnoses_a": [],
            "diagnoses_b": [],
            "reasoning_a": "",
            "reasoning_b": "",
            "thinking_a": "",
            "thinking_b": "",
            "codes_a": [],
            "codes_b": [],
            "critic_rounds": [],
        }  # type: ignore
    )
    elapsed = time.time() - started

    critic_rounds = result.get("critic_rounds", [])
    critic_triggered = len(critic_rounds) > 0

    return DualDiagnoseResponse(
        results=[
            DiagnosisResult(
                model=MODEL_A,
                diagnoses=result.get("diagnoses_a", []),
                codes=result.get("codes_a", []),
                reasoning=result.get("reasoning_a", ""),
                thinking=result.get("thinking_a", ""),
            ),
            DiagnosisResult(
                model=MODEL_B,
                diagnoses=result.get("diagnoses_b", []),
                codes=result.get("codes_b", []),
                reasoning=result.get("reasoning_b", ""),
                thinking=result.get("thinking_b", ""),
            ),
        ],
        critic_triggered=critic_triggered,
        critic_rounds=critic_rounds,
        elapsed_s=round(elapsed, 2),
    )
