"""Idempotent bulk embedder for ICD-10-CM codes.

Embeds the ``long_desc`` of every ``icd10_codes`` row (billable and
non-billable) into the ``embedding`` column (pgvector ``vector(1024)``) using
the bge-m3 model via LiteLLM's ``/v1/embeddings`` endpoint. Non-billable codes
are pre-embedded so they are search-ready if they become billable in a future
release. Skips rows that already have an embedding, so re-runs only fill gaps.

Prerequisites:
    - ``icd10_codes`` loaded (``load_icd10.py``).
    - ``embedding`` column + HNSW index created (``02-embedding.sql``).
    - ``bge-m3`` pulled and the ``embed`` alias configured in LiteLLM.

Usage::

    python -m medicoder.db.embed_icd10
"""

from __future__ import annotations

import json
import sys

from medicoder import proxy
from medicoder.db.connect import connect

BATCH_SIZE = 128

COUNT_PENDING_SQL = "SELECT count(*) FROM icd10_codes WHERE embedding IS NULL"
SELECT_BATCH_SQL = (
    "SELECT order_number, long_desc FROM icd10_codes "
    "WHERE embedding IS NULL "
    "ORDER BY order_number LIMIT %s"
)
UPDATE_SQL = "UPDATE icd10_codes SET embedding = %s::vector WHERE order_number = %s"


def main() -> int:
    """Embed all ICD-10 codes that lack an embedding.

    Returns:
        Process exit code (0 on success, 1 on fatal error).
    """
    conn = connect()
    with conn, conn.cursor() as cur:
        cur.execute(COUNT_PENDING_SQL)
        total_pending = cur.fetchone()[0]
        if total_pending == 0:
            print("[embed_icd10] all codes already embedded; nothing to do.")
            return 0
        print(f"[embed_icd10] {total_pending} codes to embed")

        done = 0
        while True:
            cur.execute(SELECT_BATCH_SQL, (BATCH_SIZE,))
            rows = cur.fetchall()
            if not rows:
                break
            texts = [row[1] for row in rows]
            try:
                vectors = proxy.embed(texts)
            except Exception as e:
                print(f"[embed_icd10] embedding failed: {e}", file=sys.stderr)
                return 1
            for (order_number, _), vec in zip(rows, vectors):
                cur.execute(UPDATE_SQL, (json.dumps(vec), order_number))
            conn.commit()
            done += len(rows)
            pct = done * 100 // total_pending
            print(f"[embed_icd10] {done}/{total_pending} ({pct}%)")

    print(f"[embed_icd10] done — embedded {done} codes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
