"""medicoder-technical FastAPI app.

Serves the ICD-10-CM codes loaded into Postgres by the entrypoint's
``medicoder.db.load_icd10`` step. The container's CMD is ``python main.py``,
which starts a uvicorn server here (keeping the container alive).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


class ICD10Code(BaseModel):
    order_number: int
    code: str
    code_type: int
    is_billable: bool
    short_desc: str
    long_desc: str


_pool: ConnectionPool


def _configure(conn) -> None:
    conn.row_factory = dict_row


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
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
    try:
        yield
    finally:
        _pool.close()


app = FastAPI(title="medicoder-technical", lifespan=lifespan)

_COLS = (
    "order_number, code, code_type, is_billable, short_desc, long_desc"
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/codes", response_model=list[ICD10Code])
def list_codes(
    q: str = Query("", description="Filter codes by prefix (case-sensitive)."),
    limit: int = Query(50, ge=1, le=500),
) -> list[dict]:
    with _pool.connection() as conn:
        return conn.execute(
            f"SELECT {_COLS} FROM icd10_codes "
            "WHERE starts_with(code, %s) ORDER BY code LIMIT %s",
            (q, limit),
        ).fetchall()


@app.get("/codes/{order_number}", response_model=ICD10Code)
def get_code(order_number: int) -> dict:
    with _pool.connection() as conn:
        row = conn.execute(
            f"SELECT {_COLS} FROM icd10_codes WHERE order_number = %s",
            (order_number,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="code not found")
    return row


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
