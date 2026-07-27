"""medicoder-technical FastAPI app.

Serves the ICD-10-CM codes loaded into Postgres by the entrypoint's
``medicoder.db.load_icd10`` step. The container's CMD is ``python main.py``,
which starts a uvicorn server here (keeping the container alive).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
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
        ).fetchall() # type: ignore
    # reason: code still works despite type mismatch


@app.get("/codes/{order_number}", response_model=ICD10Code)
def get_code(order_number: int) -> dict:
    with _pool.connection() as conn:
        row = conn.execute(
            f"SELECT {_COLS} FROM icd10_codes WHERE order_number = %s",
            (order_number,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="code not found")
    return row # type: ignore
    # reason: code still works despite type mismatch


class TestRequest(BaseModel):
    prompt: str | None = None


class TestResponse(BaseModel):
    model: str
    prompt: str
    response: str
    elapsed_s: float


# Default prompt used when the POST body omits one.
DEFAULT_PROMPT = "What is the ICD-10-CM code for Type 2 diabetes mellitus without complications?"
# First call to a model loads it into VRAM (can take 10-60s); allow generous time.
_LLM_TIMEOUT = 120.0


# Test any LiteLLM alias with a prompt. LiteLLM is in-network only (no host
# port), so this endpoint on the app (port 8000) is the sole way to exercise the
# models from the host. `model` is a LiteLLM alias from docker/litellm/config.yaml
# (e.g. "ii-medical-q8", "medical-grpo", "qwen35-medical", "A").
#
# curl -X POST 'http://localhost:8000/test/ii-medical-q8' \
#   -H 'Content-Type: application/json' \
#   -d '{"prompt":"What is the ICD-10-CM code for essential hypertension?"}'
# (omit the -d body to use the default medical prompt)
@app.post("/test/{model}", response_model=TestResponse)
def test_model(model: str, body: TestRequest | None = None) -> TestResponse:
    prompt = body.prompt if body and body.prompt else DEFAULT_PROMPT
    base = os.environ.get("LLM_BASE_URL", "http://litellm:4000/v1").rstrip("/")
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=_LLM_TIMEOUT) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        # LiteLLM returns 400/404 for an unknown model alias, 429 rate-limit, etc.
        detail = e.read().decode(errors="replace")[:500]
        status = 404 if e.code in (400, 404) else e.code
        raise HTTPException(
            status_code=status,
            detail=f"model '{model}' not available via LiteLLM ({e.code}): {detail}",
        )
    except urllib.error.URLError as e:
        reason = str(e.reason)
        if "timed out" in reason.lower():
            raise HTTPException(status_code=504, detail=f"model '{model}' timed out")
        raise HTTPException(status_code=502, detail=f"LLM proxy unreachable: {reason}")
    elapsed = time.time() - started
    choices = data.get("choices") or []
    if not choices:
        raise HTTPException(status_code=502, detail=f"no choices in LiteLLM response: {data}")
    content = choices[0].get("message", {}).get("content", "")
    return TestResponse(
        model=model, prompt=prompt, response=content, elapsed_s=round(elapsed, 2)
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
