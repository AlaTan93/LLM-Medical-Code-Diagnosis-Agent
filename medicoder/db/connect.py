# Written by AI
"""Shared Postgres connection for CLI scripts.

``load_icd10.py`` and ``embed_icd10.py`` both need a direct ``psycopg``
connection (outside the app's connection pool) that retries until the database
is reachable.  Previously each script carried its own copy of this logic.
"""

from __future__ import annotations

import os
import sys
import time

import psycopg


def connect(
    *,
    retries: int = 30,
    interval: float = 2.0,
) -> psycopg.Connection:
    """Open a Postgres connection using the standard libpq env vars.

    Retries until the database is reachable (tolerates container startup
    latency).  Connection parameters come from ``PGHOST``, ``PGPORT``,
    ``PGDATABASE``, ``PGUSER``, ``PGPASSWORD``.

    Args:
        retries: Maximum connection attempts.
        interval: Seconds between attempts.

    Returns:
        An open ``psycopg`` connection as the read/write ``medicoder`` role.

    Raises:
        RuntimeError: If no connection succeeds within the retry budget.
    """
    kwargs = {
        "host": os.environ.get("PGHOST", "postgres"),
        "port": int(os.environ.get("PGPORT", "5432")),
        "dbname": os.environ.get("PGDATABASE", "medicoder"),
        "user": os.environ.get("PGUSER", "medicoder"),
        "password": os.environ.get("PGPASSWORD", ""),
    }
    last_err: Exception | None = None
    for _ in range(retries):
        try:
            return psycopg.connect(connect_timeout=5, **kwargs)
        except psycopg.OperationalError as exc:
            last_err = exc
            print(f"[db] waiting for postgres: {exc}", file=sys.stderr)
            time.sleep(interval)
    raise RuntimeError(f"could not connect to postgres: {last_err}")
