"""ICD-10-CM code lookup routes (Postgres-backed)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from medicoder.db.pool import get_pool
from medicoder.schemas import ICD10Code

router = APIRouter()

_COLS = "order_number, code, code_type, is_billable, short_desc, long_desc"


@router.get("/codes", response_model=list[ICD10Code])
def list_codes(
    q: str = Query("", description="Filter codes by prefix (case-sensitive)."),
    limit: int = Query(50, ge=1, le=500),
) -> list[dict]:
    with get_pool().connection() as conn:
        return conn.execute(
            f"SELECT {_COLS} FROM icd10_codes "
            "WHERE starts_with(code, %s) ORDER BY code LIMIT %s",
            (q, limit),
        ).fetchall()  # type: ignore
    # reason: code still works despite type mismatch


@router.get("/codes/{order_number}", response_model=ICD10Code)
def get_code(order_number: int) -> dict:
    with get_pool().connection() as conn:
        row = conn.execute(
            f"SELECT {_COLS} FROM icd10_codes WHERE order_number = %s",
            (order_number,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="code not found")
    return row  # type: ignore
    # reason: code still works despite type mismatch
