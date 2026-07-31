"""Debate-critic loop: reconcile two models' diagnoses via an LLM critic.

When the two medical models in the dual-diagnosis pipeline disagree (different
diagnosis count or different top-1 ICD-10 code), a critic model reviews both
models' outputs and produces reconciled diagnoses over one or more rounds.

This module contains everything critic-related:

* **Business logic** — :func:`diagnose_critic` calls the critic model;
  :func:`build_critic_context` assembles its prompt.
* **Graph nodes** — :func:`should_critic`, :func:`critic_think`,
  :func:`critic_search`, :func:`should_continue_critic` are LangGraph node /
  routing functions wired into the state graph by ``routes/diagnose.py``.

The graph wiring itself lives in ``routes/diagnose.py``::

    search ──→ should_critic?
                 ╱        ╲
               no          yes
                │            │
               END   critic_think ──→ critic_search
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

from pydantic import BaseModel, ValidationError, field_validator

from medicoder import proxy
from medicoder.medical import (
    ModelOutput,
    clean_str_list,
    safe_search,
    _extract_json,
    _parse_diagnoses,
)
from medicoder.schemas import CriticRound, DiagnoseState, ICD10Match

# -- Configuration ----------------------------------------------------------

CRITIC_MODEL = os.environ.get("CRITIC_MODEL", "medgemma-27b-q4_k_s")
CRITIC_TEMP = 0.25
CRITIC_MAX_TOKENS = 8192
_MAX_CRITIC_ROUNDS = int(os.environ.get("MAX_CRITIC_ROUNDS", "2"))

# -- System prompt ----------------------------------------------------------

_CRITIC_SYSTEM = (
    "You are a senior clinical coding reviewer. Two independent medical "
    "models analyzed the same patient case and produced different "
    "diagnoses.\n\n"
    "Carefully analyze where the models agree and disagree:\n"
    "- Which diagnoses are well-supported by the clinical text?\n"
    "- Are there missing diagnoses the models overlooked?\n"
    "- Are there redundant or incorrect diagnoses?\n"
    "- Which specific ICD-10 codes are most appropriate?\n\n"
    "Use standard ICD-10-CM diagnostic terminology in your diagnoses and "
    "queries:\n"
    '- "Malignant neoplasm of [site]" — not "cancer" or "carcinoma"\n'
    '- "Unspecified" when the site or type is not documented\n\n'
    "Reason freely and thoroughly.  After your analysis, output a JSON "
    "object with these fields:\n"
    '- "reasoning": Your detailed clinical reasoning\n'
    '- "diagnoses": Your reconciled list of 1-10 concise ICD-10-CM '
    'diagnostic phrases\n'
    '- "queries": Search terms to find ICD-10 codes for any new or changed '
    'diagnoses (these will be embedded and matched against the code '
    'database)\n'
    '- "done": true if you are confident in your reconciled diagnoses, '
    'false if you need another round of review\n'
)

_MAX_DIAGNOSES = 10


# -- Pydantic model for critic output --------------------------------------


class CriticOutput(BaseModel):
    """Pydantic schema for validating debate-critic JSON output.

    The critic reconciles two models' diagnoses and produces its own
    reasoning, reconciled diagnoses, search queries, and a done flag.

    Attributes:
        reasoning: The critic's clinical reasoning.
        diagnoses: 1-10 reconciled diagnosis strings.
        queries: Free-form search terms for ICD-10 vector search.
        done: Whether the critic is confident (``True`` ends the loop).
    """

    reasoning: str = ""
    diagnoses: list[str]
    queries: list[str] = []
    done: bool = False

    @field_validator("diagnoses", mode="after")
    @classmethod
    def clean_diagnoses(cls, v: list[str]) -> list[str]:
        return clean_str_list(v)

    @field_validator("queries", mode="after")
    @classmethod
    def clean_queries(cls, v: list[str]) -> list[str]:
        return clean_str_list(v)


# -- Business logic ---------------------------------------------------------


def build_critic_context(
    text: str,
    a: ModelOutput,
    b: ModelOutput,
    previous_rounds: list[dict],
) -> str:
    """Build the user message giving the critic both models' context.

    Renders a structured summary of each model's diagnoses, reasoning, and
    ICD-10 code matches, plus any previous critic rounds so the critic can
    iterate rather than repeat itself.
    """
    lines: list[str] = [f"Patient: {text}\n"]

    def _model_section(out: ModelOutput) -> list[str]:
        name = f"Model ({out.model})"
        sect = [f"{name} diagnoses:"]
        for i, d in enumerate(out.diagnoses, 1):
            sect.append(f"  {i}. {d}")
        if out.reasoning:
            sect.append(f"{name} reasoning: {out.reasoning}")
        sect.append(f"{name} ICD-10 codes:")
        if out.codes:
            for c in out.codes:
                sect.append(f"  {c.code} - {c.short_desc} "
                            f"(similarity {c.similarity:.2f})")
        else:
            sect.append("  (none)")
        return sect

    lines.extend(_model_section(a))
    lines.append("")
    lines.extend(_model_section(b))

    if previous_rounds:
        lines.append("\nPrevious critic rounds:")
        for r in previous_rounds:
            lines.append(f"  Round {r['round']}:")
            if r.get("reasoning"):
                lines.append(f"    Reasoning: {r['reasoning']}")
            lines.append(f"    Diagnoses: {', '.join(r['diagnoses'])}")
            if r.get("codes"):
                code_strs = [c["code"] for c in r["codes"]]
                lines.append(f"    Codes: {', '.join(code_strs)}")

    return "\n".join(lines)


def diagnose_critic(
    text: str,
    a: ModelOutput,
    b: ModelOutput,
    previous_rounds: list[dict],
    critic_model: str,
    *,
    temperature: float = 0.25,
    max_tokens: int = 8192,
) -> tuple[list[str], list[str], str, bool, str]:
    """Run the debate-critic agent to reconcile two models' diagnoses.

    The critic receives both models' diagnoses, ICD-10 codes, reasoning,
    and any previous round history, then produces its own reconciled
    diagnoses plus optional search queries.

    Uses a generous token budget (8192) to allow the critic to reason
    freely before producing structured JSON output.

    Args:
        text: Original clinical text.
        a: :class:`ModelOutput` from model A.
        b: :class:`ModelOutput` from model B.
        previous_rounds: Prior critic rounds (list of dicts with keys
            ``round``, ``reasoning``, ``diagnoses``, ``codes``).
        critic_model: LiteLLM alias for the critic model.
        temperature: Sampling temperature (default 0.25 — moderate
            creativity for reconciliation).
        max_tokens: Token budget (default 8192).

    Returns:
        A ``(diagnoses, queries, reasoning, done, thinking)`` tuple.
        On parse failure, ``diagnoses`` is empty and ``done`` is ``False``.
    """
    user_msg = build_critic_context(text, a, b, previous_rounds)
    messages = [
        {"role": "system", "content": _CRITIC_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    try:
        raw, thinking = proxy.chat_completion(
            critic_model, messages,
            temperature=temperature, max_tokens=max_tokens,
        )
    except Exception as e:
        print(f"[critic] model call failed: {e}")
        return [], [], "", False, ""

    data = _extract_json(raw)
    if data is not None:
        try:
            result = CriticOutput(**data)
            return result.diagnoses, result.queries, result.reasoning, result.done, thinking
        except (ValidationError, TypeError):
            pass

    diagnoses = _parse_diagnoses(raw)
    return diagnoses, [], "", False, thinking


# -- Graph nodes & routing --------------------------------------------------


def should_critic(state: DiagnoseState) -> str:
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

    # Different number of diagnoses -> disagreement.
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

    a = ModelOutput(
        model=state.get("model_a", ""),
        diagnoses=state.get("diagnoses_a", []),
        codes=state.get("codes_a", []),
        reasoning=state.get("reasoning_a", ""),
    )
    b = ModelOutput(
        model=state.get("model_b", ""),
        diagnoses=state.get("diagnoses_b", []),
        codes=state.get("codes_b", []),
        reasoning=state.get("reasoning_b", ""),
    )
    diagnoses, queries, reasoning, done, thinking = diagnose_critic(
        state["text"],
        a, b,
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
        thinking=thinking,
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
    codes = safe_search(search_terms, k)

    # Update the latest round's codes in place.
    updated_round = latest.model_copy(update={"codes": codes})
    updated_rounds = list(rounds[:-1]) + [updated_round]
    return {"critic_rounds": updated_rounds}


def should_continue_critic(state: DiagnoseState) -> str:
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
