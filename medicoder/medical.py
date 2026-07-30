"""Shared medical pipeline functions used by ``/code`` and ``/diagnose``.

- :func:`diagnose` — forwards clinical text to a medical model and returns
  1-10 independent diagnoses plus the model's clinical reasoning.
- :func:`search_icd10` — batch-embeds multiple diagnosis queries and runs
  pgvector cosine-similarity searches over billable ICD-10-CM codes.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, ValidationError, field_validator

from medicoder import proxy
from medicoder.db.pool import get_pool
from medicoder.schemas import ICD10Match

_MAX_DIAGNOSES = 10

_MAX_RETRIES = 4
_TEMP_INCREMENT = 0.05
_MAX_TEMP = 0.3

_DIAGNOSE_SYSTEM = (
    "Analyze the patient's clinical presentation, then produce ICD-10 type diagnoses.\n"
    "Output a JSON object with two fields:\n"
    '- "reasoning": Your brief clinical reasoning (analysis of symptoms, '
    "findings, and conclusions).\n"
    '- "diagnoses": An array of 1-10 concise, ICD-10 type diagnostic '
    "sentences.\n"
    "Do NOT include ICD-10 codes, code numbers, or 'patient has'.\n\n"
    "Example:\n"
    '{"reasoning": "The patient has a history of opioid dependence and is '
    'currently in remission after detoxification. No acute withdrawal '
    'symptoms are noted.", "diagnoses": ["Opioid dependence, in remission"]}'
)

_DIAGNOSE_SYSTEM_CONCISE = (
    "Output a JSON object with a \"diagnoses\" array of 1-10 concise "
    "ICD-10 type diagnostic sentences.\n"
    "Do NOT include ICD-10 codes, code numbers, or 'patient has'.\n"
    "Do NOT include reasoning or explanation.\n\n"
    "Example: {\"diagnoses\": [\"Opioid dependence, in remission\"]}"
)

_BOXED_RE = re.compile(r"\\boxed\{(.*)\}", re.DOTALL)
_NUM_RE = re.compile(r"^[\d]+[.)]?\s*")
_BULLET_RE = re.compile(r"^[-*]\s*")


class DiagnosisOutput(BaseModel):
    """Pydantic schema for validating JSON diagnosis output from models.

    Captures both the model's clinical reasoning and its resulting
    diagnoses in a structured format.  The ``reasoning`` field makes the
    model's thought process observable — useful for debugging, the critic
    agent's context, and evaluation transparency.

    Attributes:
        reasoning: The model's clinical reasoning (empty string if the
            model didn't include it).
        diagnoses: 1-10 clean diagnosis strings.
    """

    reasoning: str = ""
    diagnoses: list[str]

    @field_validator("diagnoses", mode="after")
    @classmethod
    def clean_diagnoses(cls, v: list[str]) -> list[str]:
        return [d.strip() for d in v if d.strip()][:_MAX_DIAGNOSES]


def _extract_json(raw: str) -> dict | None:
    """Try to extract a JSON object from raw model output.

    Models don't always return clean JSON — some wrap it in markdown
    fences, prepend prose, or embed it mid-sentence.  This function
    tries three strategies in order of decreasing strictness:

    1. Direct ``json.loads`` on the full text (best case).
    2. Extraction from a ``\\`\\`\\`json`` markdown code fence.
    3. Substring from the first ``{`` to the last ``}`` (catches JSON
       embedded after prose or thinking blocks).

    Returns ``None`` if no valid JSON object is found.
    """
    raw = raw.strip()

    # Strategy 1 — direct parse.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strategy 2 — extract from markdown code fences.
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                try:
                    return json.loads(part)
                except json.JSONDecodeError:
                    continue

    # Strategy 3 — grab the outermost { … } substring.
    first = raw.find("{")
    last = raw.rfind("}")
    if first != -1 and last > first:
        try:
            return json.loads(raw[first : last + 1])
        except json.JSONDecodeError:
            pass

    return None


def _parse_diagnoses(raw: str) -> list[str]:
    """Parse raw model output into a list of clean diagnosis strings.

    Strips ``\\boxed{...}`` wrappers, leading numbers/bullets, and blank lines.
    Caps at :data:`_MAX_DIAGNOSES` entries.

    Args:
        raw: The raw text returned by the model (thinking blocks already
            stripped by :func:`medicoder.proxy.chat_completion`).

    Returns:
        A list of 0-_MAX_DIAGNOSES diagnosis strings.
    """
    text = raw.strip()

    m = _BOXED_RE.search(text)
    if m:
        text = m.group(1).strip()

    lines: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = _NUM_RE.sub("", line)
        line = _BULLET_RE.sub("", line)
        line = line.strip()
        if line:
            lines.append(line)
    return lines[:_MAX_DIAGNOSES]


def diagnose(
    text: str,
    model: str,
    *,
    temperature: float = 0.1,
) -> tuple[list[str], str]:
    """Forward clinical text to a medical model for diagnoses and reasoning.

    Uses an incremental retry strategy: the first attempt uses the full
    system prompt (with reasoning) and a 4096-token budget.  If the model
    produces zero diagnoses (e.g. thinking loop consumed all tokens or the
    output was unparseable), subsequent retries switch to a concise prompt
    that suppresses thinking, a 2048-token budget, and a slightly higher
    temperature (+0.05 per attempt, capped at 0.3).

    Pipeline per attempt: system prompt → model call → strip thinking →
    JSON extraction → Pydantic validation → fallback to regex parsing.

    Args:
        text: Clinical text or patient description.
        model: LiteLLM alias for the medical model (e.g. ``ii-medical-q8``).
        temperature: Base sampling temperature (default 0.1).  Retries
            increment this by 0.05 per attempt up to :data:`_MAX_TEMP`.

    Returns:
        A ``(diagnoses, reasoning)`` tuple where *diagnoses* is a list of
        0-_MAX_DIAGNOSES strings and *reasoning* is the model's clinical
        explanation (empty string on fallback or concise-prompt retries).
    """
    for attempt in range(_MAX_RETRIES):
        temp = min(temperature + attempt * _TEMP_INCREMENT, _MAX_TEMP)

        # First attempt: full prompt (with reasoning), generous token budget.
        # Retries: concise prompt (no reasoning), reduced token budget to
        # cap damage from thinking loops.
        if attempt == 0:
            messages = [
                {"role": "system", "content": _DIAGNOSE_SYSTEM},
                {"role": "user", "content": text},
            ]
            max_tokens = 4096
        else:
            print(
                f"[diagnose] retry {attempt}/{_MAX_RETRIES - 1} "
                f"temp={temp:.2f}"
            )
            messages = [
                {"role": "system", "content": _DIAGNOSE_SYSTEM_CONCISE},
                {"role": "user", "content": text},
            ]
            max_tokens = 2048

        try:
            raw = proxy.chat_completion(
                model, messages, temperature=temp, max_tokens=max_tokens
            )
        except Exception:
            if attempt < _MAX_RETRIES - 1:
                continue
            raise

        # Try structured JSON first (preferred path for models that follow
        # the system prompt's format instructions).
        data = _extract_json(raw)
        if data is not None:
            try:
                result = DiagnosisOutput(**data)
                if result.diagnoses:
                    return result.diagnoses, result.reasoning
            except (ValidationError, TypeError):
                pass

        # Fallback: line-based regex parsing for models that ignore JSON.
        diagnoses = _parse_diagnoses(raw)
        if diagnoses:
            return diagnoses, ""

    # All retries exhausted.
    print(f"[diagnose] all {_MAX_RETRIES} attempts failed; returning empty")
    return [], ""


def search_icd10(queries: list[str], k: int = 3) -> list[ICD10Match]:
    """Find billable ICD-10-CM codes matching multiple diagnosis queries.

    Embeds all queries in one batch, then runs one pgvector
    cosine-similarity search per query (each returning up to *k* matches).
    Results are deduplicated by code (keeping the highest similarity) and
    sorted by similarity descending.

    Args:
        queries: Diagnosis descriptions to match against ICD-10 codes.
        k: Maximum codes to return per query (default 3).

    Returns:
        Deduplicated matching codes ranked by similarity (highest first).
    """
    if not queries:
        return []

    # Batch-embed all queries in one LiteLLM call to amortise network latency.
    vectors = proxy.embed(queries)

    # Deduplicate by code: when multiple queries match the same code, keep
    # the one with the highest similarity score.
    seen: dict[str, ICD10Match] = {}
    with get_pool().connection() as conn:
        # Widen the HNSW search frontier for better recall at the cost of
        # a small latency increase (default is 40; 200 is near-exact).
        conn.execute("SET LOCAL hnsw.ef_search = 200")
        for vec in vectors:
            query_vec = json.dumps(vec)
            rows = conn.execute(
                """SELECT code, short_desc, long_desc,
                          1 - (embedding <=> %s::vector) AS similarity
                   FROM icd10_codes
                   WHERE is_billable AND embedding IS NOT NULL
                   ORDER BY embedding <=> %s::vector
                   LIMIT %s""",
                (query_vec, query_vec, k),
            ).fetchall()  # type: ignore
            for r in rows:
                code = r["code"]  # type: ignore
                sim = round(float(r["similarity"]), 4)  # type: ignore
                if code not in seen or sim > seen[code].similarity:
                    seen[code] = ICD10Match(
                        code=code,
                        short_desc=r["short_desc"],  # type: ignore
                        long_desc=r["long_desc"],  # type: ignore
                        similarity=sim,
                    )

    # Return results sorted by similarity descending (best matches first).
    return sorted(seen.values(), key=lambda m: m.similarity, reverse=True)
