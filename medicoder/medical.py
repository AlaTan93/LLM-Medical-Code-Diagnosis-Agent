"""Shared medical pipeline functions used by ``/code`` and ``/diagnose``.

- :func:`diagnose` — forwards clinical text to a medical model for a
  one-sentence diagnosis.
- :func:`search_icd10` — embeds a query and runs a pgvector cosine-similarity
  search over billable ICD-10-CM codes.
"""

from __future__ import annotations

import json

from medicoder import proxy
from medicoder.db.pool import get_pool
from medicoder.schemas import ICD10Match

_DIAGNOSE_SYSTEM = (
    "Produce a single concise diagnostic sentence describing the patient's "
    "condition. Do NOT include ICD-10 codes, code numbers, explanations, or "
    "markdown. Just one sentence."
)


def diagnose(text: str, model: str) -> str:
    """Forward clinical text to a medical model for a one-sentence diagnosis.

    Args:
        text: Clinical text or patient description.
        model: LiteLLM alias for the medical model (e.g. ``ii-medical-q8``).

    Returns:
        A one-sentence diagnosis (thinking blocks stripped).
    """
    messages = [
        {"role": "system", "content": _DIAGNOSE_SYSTEM},
        {"role": "user", "content": text},
    ]
    return proxy.chat_completion(model, messages)


def search_icd10(query: str, k: int = 3) -> list[ICD10Match]:
    """Find the top-k billable ICD-10-CM codes matching a description.

    Embeds the query with bge-m3 and runs a pgvector cosine-similarity search
    over billable codes that have embeddings.

    Args:
        query: A diagnosis description to match against ICD-10 codes.
        k: Maximum number of codes to return (default 3).

    Returns:
        Matching codes ranked by similarity (highest first).
    """
    vectors = proxy.embed([query])
    query_vec = json.dumps(vectors[0])

    with get_pool().connection() as conn:
        conn.execute("SET LOCAL hnsw.ef_search = 200")
        rows = conn.execute(
            """SELECT code, short_desc, long_desc,
                      1 - (embedding <=> %s::vector) AS similarity
               FROM icd10_codes
               WHERE is_billable AND embedding IS NOT NULL
               ORDER BY embedding <=> %s::vector
               LIMIT %s""",
            (query_vec, query_vec, k),
        ).fetchall()  # type: ignore

    return [
        ICD10Match(
            code=r["code"],
            short_desc=r["short_desc"],
            long_desc=r["long_desc"],
            similarity=round(float(r["similarity"]), 4),
        )
        for r in rows
    ]
