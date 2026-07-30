"""Shared medical pipeline functions used by ``/code`` and ``/diagnose``.

- :func:`diagnose` — forwards clinical text to a medical model and returns
  1-10 independent diagnoses.
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

_DIAGNOSE_SYSTEM = (
    "Produce concise, ICD-10 type diagnostic sentences describing the "
    "patient's conditions.\n"
    "Output a JSON object with a \"diagnoses\" array containing 1-10 "
    "diagnosis strings. Do NOT include ICD-10 codes, code numbers, "
    "explanations, markdown, or 'patient has'.\n\n"
    "Example:\n"
    '{"diagnoses": ["Type 2 diabetes mellitus without complications", '
    '"Essential (primary) hypertension"]}'
)

_BOXED_RE = re.compile(r"\\boxed\{(.*)\}", re.DOTALL)
_NUM_RE = re.compile(r"^[\d]+[.)]?\s*")
_BULLET_RE = re.compile(r"^[-*]\s*")


class DiagnosisOutput(BaseModel):
    """Pydantic schema for validating JSON diagnosis output from models.

    Coerces the model's response into a clean list of diagnosis strings,
    filtering out empty entries and capping at :data:`_MAX_DIAGNOSES`.
    """

    diagnoses: list[str]

    @field_validator("diagnoses", mode="after")
    @classmethod
    def clean_diagnoses(cls, v: list[str]) -> list[str]:
        return [d.strip() for d in v if d.strip()][:_MAX_DIAGNOSES]


def _extract_json(raw: str) -> dict | None:
    """Try to extract a JSON object from raw model output.

    Attempts, in order:
    1. Direct ``json.loads`` on the full text.
    2. Extraction from a markdown code fence (``\\`\\`\\`json … \\`\\`\\```).
    3. Substring from the first ``{`` to the last ``}``.

    Returns ``None`` if no valid JSON is found.
    """
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

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


def diagnose(text: str, model: str) -> list[str]:
    """Forward clinical text to a medical model for independent diagnoses.

    Asks the model for JSON output (``{"diagnoses": [...]}``) and validates
    it with :class:`DiagnosisOutput`.  Falls back to :func:`_parse_diagnoses`
    (line-based regex parsing) when the model doesn't produce valid JSON.

    Args:
        text: Clinical text or patient description.
        model: LiteLLM alias for the medical model (e.g. ``ii-medical-q8``).

    Returns:
        A list of 0-_MAX_DIAGNOSES diagnosis strings (thinking blocks
        stripped, output parsed and validated).
    """
    messages = [
        {"role": "system", "content": _DIAGNOSE_SYSTEM},
        {"role": "user", "content": text},
    ]
    raw = proxy.chat_completion(model, messages)

    data = _extract_json(raw)
    if data is not None:
        try:
            return DiagnosisOutput(**data).diagnoses
        except (ValidationError, TypeError):
            pass

    return _parse_diagnoses(raw)


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

    vectors = proxy.embed(queries)

    seen: dict[str, ICD10Match] = {}
    with get_pool().connection() as conn:
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

    return sorted(seen.values(), key=lambda m: m.similarity, reverse=True)
