"""Shared medical pipeline functions used by ``/code`` and ``/diagnose``.

- :func:`diagnose` — forwards clinical text to a medical model and returns
  1-10 independent diagnoses (one per line).
- :func:`search_icd10` — batch-embeds multiple diagnosis queries and runs
  pgvector cosine-similarity searches over billable ICD-10-CM codes.
"""

from __future__ import annotations

import json
import re

from medicoder import proxy
from medicoder.db.pool import get_pool
from medicoder.schemas import ICD10Match

_MAX_DIAGNOSES = 10

_DIAGNOSE_SYSTEM = (
    "Produce concise, ICD-10 type diagnostic sentences describing the patient's conditions. "
    "Output one diagnosis per line, up to 10 lines. Do NOT include ICD-10 "
    "codes, code numbers, explanations, markdown, or 'patient has'.\n\n"
    "Correct:\n"
    "Cholera due to Vibrio cholerae 01, biovar cholerae\n"
    "Poisoning by unspecified agents primarily affecting the gastrointestinal "
    "system, accidental (unintentional), initial encounter\n\n"
    "Incorrect:\n"
    "Patient has cholera due to Vibrio cholerae 01, biovar cholerae"
)

_BOXED_RE = re.compile(r"\\boxed\{(.*)\}", re.DOTALL)
_NUM_RE = re.compile(r"^[\d]+[.)]?\s*")
_BULLET_RE = re.compile(r"^[-*]\s*")


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

    Args:
        text: Clinical text or patient description.
        model: LiteLLM alias for the medical model (e.g. ``ii-medical-q8``).

    Returns:
        A list of 0-_MAX_DIAGNOSES diagnosis sentences (thinking blocks
        stripped and output parsed into individual lines).
    """
    messages = [
        {"role": "system", "content": _DIAGNOSE_SYSTEM},
        {"role": "user", "content": text},
    ]
    raw = proxy.chat_completion(model, messages)
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
