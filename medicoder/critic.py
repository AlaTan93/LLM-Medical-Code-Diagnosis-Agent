"""Agentic debate-critic with read-only database access.

When the two medical models in the dual-diagnosis pipeline disagree (different
diagnosis count or different top-1 ICD-10 code), a critic agent reviews both
models' outputs and iteratively queries the ICD-10-CM code database to find
the best matching codes before producing reconciled diagnoses.

The critic has access to three **read-only** tools:

* **search** — semantic + FTS search by diagnosis description
* **lookup** — browse codes by prefix (explore a code family)
* **get** — get full details for a specific code

The agent loops internally (up to ``CRITIC_MAX_TOOL_CALLS`` iterations per
round), calling tools and seeing results before committing to a final answer.
The outer ``MAX_CRITIC_ROUNDS`` loop is retained as a safety net.

Graph wiring lives in ``routes/diagnose.py``::

    search ──→ should_critic?
                 ╱        ╲
               no          yes
                │            │
               END     critic_agent ──→ should_continue_critic?
                                 ╱          ╲
                              loop            end
                               │                │
                        critic_agent          END
                         (next round)
"""

from __future__ import annotations

import os

from pydantic import BaseModel, ValidationError, field_validator

from medicoder import proxy
from medicoder.db.pool import get_pool
from medicoder.medical import (
    ModelOutput,
    _extract_json,
    _parse_diagnoses,
    clean_str_list,
    safe_search,
)
from medicoder.schemas import CriticRound, DiagnoseState, ICD10Match

# -- Configuration ----------------------------------------------------------

CRITIC_MODEL = os.environ.get("CRITIC_MODEL", "medgemma-27b-q4_k_s")
CRITIC_TEMP = float(os.environ.get("CRITIC_TEMP", "0.25"))
CRITIC_MAX_TOKENS = int(os.environ.get("CRITIC_MAX_TOKENS", "8192"))
_MAX_CRITIC_ROUNDS = int(os.environ.get("MAX_CRITIC_ROUNDS", "2"))
_MAX_TOOL_CALLS = int(os.environ.get("CRITIC_MAX_TOOL_CALLS", "8"))

_MAX_K = 10
_MAX_LOOKUP = 50

# -- System prompt ----------------------------------------------------------

_CRITIC_AGENT_SYSTEM = (
    "You are a senior clinical coding reviewer with direct read-only access "
    "to an ICD-10-CM code database. Two independent medical models analyzed "
    "the same patient case and produced different diagnoses.\n\n"
    "You can query the code database at any time using these tools:\n\n"
    '1. search — Find codes matching diagnosis descriptions (semantic + FTS):\n'
    '   {"tool": "search", "diagnoses": ["Malignant neoplasm of prostate"], '
    '"k": 5}\n\n'
    '2. lookup — Browse codes by prefix (explore a code family):\n'
    '   {"tool": "lookup", "prefix": "C61", "limit": 20}\n\n'
    '3. get — Get full details for a specific code:\n'
    '   {"tool": "get", "code": "C619"}\n\n'
    "Use these tools to explore, verify, and refine before committing to a "
    "final answer. You can call multiple tools in one response.\n\n"
    "Output a JSON object with these fields:\n"
    '- "reasoning": Your detailed clinical reasoning\n'
    '- "actions": List of tool calls to execute (see above). Empty when '
    "done.\n"
    '- "diagnoses": Your reconciled list of 1-10 concise ICD-10-CM '
    "diagnostic phrases (include only when done=true)\n"
    '- "done": true when confident in your final diagnoses, false to '
    "continue exploring\n\n"
    "ICD-10-CM naming conventions:\n"
    '- Use "Malignant neoplasm of [site]" — not "cancer" or "carcinoma"\n'
    '- Use "Unspecified" when the documentation does not specify the site\n'
    '- Include clinical qualifiers where documented (e.g., "acute", '
    '"in remission")\n\n'
    "Do NOT include ICD-10 code numbers in your diagnoses — the search "
    "will find them."
)

# -- Pydantic model for agent output ----------------------------------------


class CriticAgentOutput(BaseModel):
    """Schema for the agentic critic's JSON output.

    The critic either requests tool calls (``actions`` non-empty,
    ``done=false``) or commits to a final answer (``done=true`` with
    ``diagnoses``).

    Attributes:
        reasoning: The critic's clinical reasoning.
        actions: Tool calls to execute. Each dict has a ``tool`` key
            (``"search"``, ``"lookup"``, or ``"get"``) plus tool-specific
            parameters.
        diagnoses: 1-10 reconciled diagnosis strings (when ``done``).
        done: Whether the critic is confident (``True`` ends the loop).
    """

    reasoning: str = ""
    actions: list[dict] = []
    diagnoses: list[str] = []
    done: bool = False

    @field_validator("diagnoses", mode="after")
    @classmethod
    def clean_diagnoses(cls, v: list[str]) -> list[str]:
        return clean_str_list(v)


# -- Read-only tools --------------------------------------------------------


def _tool_search(diagnoses: list[str], k: int = 3) -> str:
    """Execute a vector + FTS search and format results as a string.

    Args:
        diagnoses: Diagnosis descriptions to search for.
        k: Max codes per diagnosis (capped at ``_MAX_K``).

    Returns:
        Formatted code list, or an error message.
    """
    k = min(k, _MAX_K)
    try:
        matches = safe_search(diagnoses, k)
    except Exception as e:
        return f"ERROR: {e}"
    if not matches:
        return "No codes found."
    lines = [
        f"  {m.code:<8} ({m.similarity:.2f}) {m.short_desc} — {m.long_desc}"
        for m in matches
    ]
    return "\n".join(lines)


def _tool_lookup(prefix: str, limit: int = 20) -> str:
    """Browse ICD-10-CM codes by prefix.

    Args:
        prefix: Code prefix (e.g. ``"C61"``, ``"F11"``). Periods are stripped.
        limit: Max rows (capped at ``_MAX_LOOKUP``).

    Returns:
        Formatted code list, or an error message.
    """
    prefix = prefix.replace(".", "").upper()
    limit = min(limit, _MAX_LOOKUP)
    try:
        with get_pool().connection() as conn:
            rows = conn.execute(
                "SELECT code, short_desc, long_desc, is_billable "
                "FROM icd10_codes WHERE starts_with(code, %s) "
                "ORDER BY code LIMIT %s",
                (prefix, limit),
            ).fetchall()  # type: ignore
    except Exception as e:
        return f"ERROR: {e}"
    if not rows:
        return f"No codes starting with '{prefix}'."
    lines = []
    for r in rows:
        tag = "[billable]" if r["is_billable"] else "(header)"
        lines.append(
            f"  {r['code']:<8} {tag:<12} {r['short_desc']} — {r['long_desc']}"
        )
    return "\n".join(lines)


def _tool_get(code: str) -> str:
    """Get full details for a single ICD-10-CM code.

    Args:
        code: Code string (with or without period).

    Returns:
        Formatted code details, or a not-found message.
    """
    code = code.replace(".", "").upper()
    try:
        with get_pool().connection() as conn:
            row = conn.execute(
                "SELECT code, short_desc, long_desc, is_billable "
                "FROM icd10_codes WHERE code = %s",
                (code,),
            ).fetchone()  # type: ignore
    except Exception as e:
        return f"ERROR: {e}"
    if not row:
        return f"Code '{code}' not found."
    tag = "[billable]" if row["is_billable"] else "(header)"
    return (
        f"  {row['code']} {tag} {row['short_desc']} — {row['long_desc']}"
    )


def _execute_action(action: dict) -> str:
    """Dispatch a single tool call and return the formatted result.

    Args:
        action: Dict with a ``tool`` key and tool-specific parameters.

    Returns:
        Tool result as a string (or an error message).
    """
    tool = action.get("tool", "")

    if tool == "search":
        diagnoses = action.get("diagnoses", [])
        k = action.get("k", 3)
        if not diagnoses:
            return "ERROR: 'diagnoses' is required for search."
        return _tool_search(diagnoses, k)

    if tool == "lookup":
        prefix = action.get("prefix", "")
        limit = action.get("limit", 20)
        if not prefix:
            return "ERROR: 'prefix' is required for lookup."
        return _tool_lookup(prefix, limit)

    if tool == "get":
        code = action.get("code", "")
        if not code:
            return "ERROR: 'code' is required for get."
        return _tool_get(code)

    return f"ERROR: Unknown tool '{tool}'. Available: search, lookup, get."


# -- Context building (unchanged from previous version) ---------------------


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

    if len(dx_a) != len(dx_b):
        print(f"[critic] triggered: dx count {len(dx_a)} vs {len(dx_b)}")
        return "critic"

    top_a = codes_a[0].code if codes_a else None
    top_b = codes_b[0].code if codes_b else None
    if top_a != top_b:
        print(f"[critic] triggered: top-1 code {top_a} vs {top_b}")
        return "critic"

    return "end"


def critic_agent(state: DiagnoseState) -> dict:
    """Run the agentic critic: iteratively query the code database and reconcile.

    The critic loops internally (up to ``_MAX_TOOL_CALLS`` iterations):
    each iteration calls the LLM, which either requests tool calls (search,
    lookup, get) or signals ``done`` with final diagnoses.  Tool results are
    appended to the conversation so the critic can see them and refine.

    After the loop, a final ``safe_search`` produces the official ICD-10
    codes for the round (consistent with how Model A/B codes are produced).
    """
    rounds = state.get("critic_rounds", [])
    round_num = len(rounds)

    print(f"[critic] round {round_num} (model={CRITIC_MODEL}, "
          f"temp={CRITIC_TEMP}, max_tool_calls={_MAX_TOOL_CALLS})")

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
    user_msg = build_critic_context(
        state["text"], a, b,
        [r.model_dump() for r in rounds],
    )

    messages: list[dict] = [
        {"role": "system", "content": _CRITIC_AGENT_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    all_tool_calls: list[dict] = []
    diagnoses: list[str] = []
    reasoning = ""
    thinking = ""
    done = False

    for iteration in range(_MAX_TOOL_CALLS):
        try:
            raw, think = proxy.chat_completion(
                CRITIC_MODEL, messages,
                temperature=CRITIC_TEMP,
                max_tokens=CRITIC_MAX_TOKENS,
            )
            thinking = think or thinking
        except Exception as e:
            print(f"[critic] model call failed at iteration {iteration}: {e}")
            break

        data = _extract_json(raw)
        if data is None:
            print(f"[critic] iteration {iteration}: JSON parse failed, "
                  "falling back to regex")
            diagnoses = _parse_diagnoses(raw)
            done = True
            break

        try:
            result = CriticAgentOutput(**data)
        except (ValidationError, TypeError):
            print(f"[critic] iteration {iteration}: validation failed, "
                  "falling back to regex")
            diagnoses = _parse_diagnoses(raw)
            done = True
            break

        reasoning = result.reasoning or reasoning
        done = result.done

        if done or not result.actions:
            diagnoses = result.diagnoses
            print(f"[critic] iteration {iteration}: done={done}, "
                  f"{len(diagnoses)} diagnoses")
            break

        messages.append({"role": "assistant", "content": raw})

        for action in result.actions:
            tool_name = action.get("tool", "unknown")
            tool_result = _execute_action(action)
            all_tool_calls.append({
                "tool": tool_name,
                "params": {k: v for k, v in action.items() if k != "tool"},
                "result_preview": tool_result[:200],
            })
            messages.append({
                "role": "user",
                "content": f"Tool result ({tool_name}):\n{tool_result}",
            })
            print(f"[critic]   tool: {tool_name} → {len(tool_result)} chars")
    else:
        print(f"[critic] max tool calls ({_MAX_TOOL_CALLS}) reached; "
              "exiting agent loop")

    if not diagnoses and data:
        try:
            result = CriticAgentOutput(**data)
            diagnoses = result.diagnoses
        except (ValidationError, TypeError):
            pass

    k = state.get("k", 3)
    codes: list[ICD10Match] = []
    if diagnoses:
        codes = safe_search(diagnoses, k)

    new_round = CriticRound(
        round=round_num,
        reasoning=reasoning,
        diagnoses=diagnoses,
        queries=[],
        codes=codes,
        done=done,
        thinking=thinking,
        tool_calls=all_tool_calls,
    )
    return {"critic_rounds": rounds + [new_round]}


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
