"""Dual-diagnosis pipeline: two medical models + ICD-10 vector search.

Uses a LangGraph :class:`StateGraph` to run two medical models in parallel
(fan-out), then deterministically search both models' diagnoses for matching
billable ICD-10-CM codes (fan-in).  When the two models disagree, an optional
debate-critic loop reconciles their outputs over one or more rounds.

    START
      +--> diagnose_a (ii-medical-q8)           --+
      +--> diagnose_b (deepseek-r1-medical-cot) --+  parallel
                                                   |
                                                fan-in
                                                   |
                                           search_both (k per dx)
                                                   |
                                           _should_critic?
                                              ╱        ╲
                                           no            yes
                                            │              │
                                           END     critic_think ──→ critic_search
                                                                ╱          ╲
                                                       _should_continue?
                                                           ╱          ╲
                                                        loop            end
                                                         │                │
                                                  critic_think          END
                                                      (next round)
"""

from __future__ import annotations

import os
import time
from typing import TypedDict

from fastapi import APIRouter
from langgraph.graph import END, START, StateGraph

from medicoder.medical import diagnose, diagnose_critic, search_icd10
from medicoder.schemas import (
    CriticRound,
    DiagnosisResult,
    DiagnoseRequest,
    DualDiagnoseResponse,
    ICD10Match,
)

router = APIRouter()

MODEL_A = "ii-medical-q8"
MODEL_B = "deepseek-r1-medical-cot"

CRITIC_MODEL = "ii-medical-q8"
CRITIC_TEMP = 0.25
CRITIC_MAX_TOKENS = 8192
_MAX_CRITIC_ROUNDS = int(os.environ.get("MAX_CRITIC_ROUNDS", "2"))


class DiagnoseState(TypedDict):
    """Mutable state passed between graph nodes.

    Attributes:
        text: The original clinical text (set at invocation).
        k: Maximum ICD-10 codes to return per diagnosis (default 3).
        enable_critic: Whether the debate-critic loop may trigger.
        diagnoses_a: Diagnoses from model A (set by ``diagnose_a``).
        diagnoses_b: Diagnoses from model B (set by ``diagnose_b``).
        reasoning_a: Clinical reasoning from model A (set by ``diagnose_a``).
        reasoning_b: Clinical reasoning from model B (set by ``diagnose_b``).
        codes_a: ICD-10 matches for model A (set by ``search_both``).
        codes_b: ICD-10 matches for model B (set by ``search_both``).
        critic_rounds: Accumulated critic round history (set by critic nodes).
    """

    text: str
    k: int
    enable_critic: bool
    diagnoses_a: list[str]
    diagnoses_b: list[str]
    reasoning_a: str
    reasoning_b: str
    codes_a: list[ICD10Match]
    codes_b: list[ICD10Match]
    critic_rounds: list[CriticRound]


# -- Node functions ---------------------------------------------------------


def diagnose_a(state: DiagnoseState) -> dict:
    """Call the first medical model (``ii-medical-q8``) for diagnoses."""
    try:
        diagnoses, reasoning = diagnose(state["text"], MODEL_A)
        return {"diagnoses_a": diagnoses, "reasoning_a": reasoning}
    except Exception as e:
        return {"diagnoses_a": [f"[error] {e}"], "reasoning_a": ""}


def diagnose_b(state: DiagnoseState) -> dict:
    """Call the second medical model (``deepseek-r1-medical-cot``)."""
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


# -- Critic nodes -----------------------------------------------------------


def _should_critic(state: DiagnoseState) -> str:
    """Conditional router: trigger the critic if the models disagree.

    Returns ``"critic"`` when the models have a different diagnosis count
    or a different top-1 ICD-10 code; otherwise returns ``"end"``.
    """
    if not state.get("enable_critic", True):
        return "end"
    if _MAX_CRITIC_ROUNDS <= 0:
        return "end"

    dx_a = state.get("diagnoses_a", [])
    dx_b = state.get("diagnoses_b", [])
    codes_a = state.get("codes_a", [])
    codes_b = state.get("codes_b", [])

    # Different number of diagnoses → disagreement.
    if len(dx_a) != len(dx_b):
        print(f"[critic] triggered: dx count {len(dx_a)} vs {len(dx_b)}")
        return "critic"

    # Different top-1 code (or one has codes and the other doesn't).
    top_a = codes_a[0].code if codes_a else None
    top_b = codes_b[0].code if codes_b else None
    if top_a != top_b:
        print(f"[critic] triggered: top-1 code {top_a} vs {top_b}")
        return "critic"

    return "end"


def critic_think(state: DiagnoseState) -> dict:
    """Call the critic model to reconcile the two models' outputs.

    Passes both models' diagnoses, codes, reasoning, and any previous
    round history to the critic.  The critic returns its own reconciled
    diagnoses, search queries, reasoning, and a done flag.
    """
    rounds = state.get("critic_rounds", [])
    round_num = len(rounds)

    print(f"[critic] round {round_num} (model={CRITIC_MODEL}, "
          f"temp={CRITIC_TEMP})")

    diagnoses, queries, reasoning, done = diagnose_critic(
        state["text"],
        MODEL_A, state.get("diagnoses_a", []),
        state.get("codes_a", []),
        state.get("reasoning_a", ""),
        MODEL_B, state.get("diagnoses_b", []),
        state.get("codes_b", []),
        state.get("reasoning_b", ""),
        previous_rounds=[r.model_dump() for r in rounds],
        critic_model=CRITIC_MODEL,
        temperature=CRITIC_TEMP,
        max_tokens=CRITIC_MAX_TOKENS,
    )

    new_round = CriticRound(
        round=round_num,
        reasoning=reasoning,
        diagnoses=diagnoses,
        queries=queries,
        codes=[],
        done=done,
    )
    return {"critic_rounds": rounds + [new_round]}


def critic_search(state: DiagnoseState) -> dict:
    """Search ICD-10 for the latest critic round's diagnoses + queries."""
    rounds = state.get("critic_rounds", [])
    if not rounds:
        return {}

    latest = rounds[-1]
    k = state.get("k", 3)

    # Search using diagnoses + critic-provided queries for broader coverage.
    search_terms = list(latest.diagnoses) + list(latest.queries)
    codes = _safe_search(search_terms, k)

    # Update the latest round's codes in place.
    updated_round = latest.model_copy(update={"codes": codes})
    updated_rounds = list(rounds[:-1]) + [updated_round]
    return {"critic_rounds": updated_rounds}


def _should_continue_critic(state: DiagnoseState) -> str:
    """Conditional router: continue the critic loop or end.

    Returns ``"loop"`` if the critic hasn't signalled done, has non-empty
    diagnoses, and hasn't exceeded ``_MAX_CRITIC_ROUNDS``; otherwise
    returns ``"end"``.
    """
    rounds = state.get("critic_rounds", [])
    if not rounds:
        return "end"

    latest = rounds[-1]

    if latest.done:
        print(f"[critic] done after round {latest.round}")
        return "end"

    if not latest.diagnoses:
        print(f"[critic] empty diagnoses on round {latest.round}; stopping")
        return "end"

    if len(rounds) >= _MAX_CRITIC_ROUNDS:
        print(f"[critic] max rounds ({_MAX_CRITIC_ROUNDS}) reached")
        return "end"

    print(f"[critic] continuing to round {latest.round + 1}")
    return "loop"


# -- Graph construction -----------------------------------------------------


def _build_graph():  # type: ignore[no-untyped-def]
    """Compile the LangGraph state graph for dual diagnosis + critic loop."""
    g = StateGraph(DiagnoseState)
    g.add_node("diagnose_a", diagnose_a)
    g.add_node("diagnose_b", diagnose_b)
    g.add_node("search", search_both)
    g.add_node("critic_think", critic_think)
    g.add_node("critic_search", critic_search)

    # Fan-out: both diagnose nodes run in parallel from START.
    g.add_edge(START, "diagnose_a")
    g.add_edge(START, "diagnose_b")

    # Fan-in: search runs after both diagnoses complete.
    g.add_edge("diagnose_a", "search")
    g.add_edge("diagnose_b", "search")

    # Conditional: trigger critic loop or skip to END.
    g.add_conditional_edges("search", _should_critic, {
        "critic": "critic_think",
        "end": END,
    })

    # Critic loop: think → search → continue?
    g.add_edge("critic_think", "critic_search")
    g.add_conditional_edges("critic_search", _should_continue_critic, {
        "loop": "critic_think",
        "end": END,
    })

    return g.compile()


_graph = _build_graph()


# -- Route ------------------------------------------------------------------

# curl -X POST 'http://localhost:8000/diagnose' \
#   -H 'Content-Type: application/json' \
#   -d '{"text":"Patient has type 2 diabetes mellitus without complications"}'
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
            "diagnoses_a": [],
            "diagnoses_b": [],
            "reasoning_a": "",
            "reasoning_b": "",
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
            ),
            DiagnosisResult(
                model=MODEL_B,
                diagnoses=result.get("diagnoses_b", []),
                codes=result.get("codes_b", []),
                reasoning=result.get("reasoning_b", ""),
            ),
        ],
        critic_triggered=critic_triggered,
        critic_rounds=critic_rounds,
        elapsed_s=round(elapsed, 2),
    )
