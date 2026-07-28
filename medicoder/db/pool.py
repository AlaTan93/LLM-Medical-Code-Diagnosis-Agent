"""Postgres connection pool for the medicoder app.

A single process-wide ``ConnectionPool`` created by the FastAPI lifespan in
``main.py`` (``init_pool`` on startup, ``close_pool`` on shutdown). Route
modules obtain it via :func:`get_pool`.
"""

from __future__ import annotations

import os

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

_pool: ConnectionPool | None = None


def _configure(conn) -> None:
    """Configure a freshly checked-out connection to return dict rows.

    Args:
        conn: A ``psycopg`` connection from the pool.
    """
    conn.row_factory = dict_row


def init_pool() -> None:
    """Open the pool. Called once from the app lifespan on startup."""
    global _pool
    _pool = ConnectionPool(
        min_size=1,
        max_size=8,
        open=True,
        configure=_configure,
        kwargs={
            "host": os.environ["PGHOST"],
            "port": int(os.environ["PGPORT"]),
            "dbname": os.environ["PGDATABASE"],
            "user": os.environ["PGUSER"],
            "password": os.environ["PGPASSWORD"],
        },
    )


def close_pool() -> None:
    """Close the pool. Called once from the app lifespan on shutdown."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def get_pool() -> ConnectionPool:
    """The initialized pool. Raises if called before lifespan startup."""
    assert _pool is not None, "connection pool not initialized (lifespan didn't run)"
    return _pool
