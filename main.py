"""medicoder-technical FastAPI app.

Thin assembly: creates the app, wires the lifespan (Postgres pool) and includes
the route modules under ``medicoder/routes/``. The container's CMD is
``python main.py``, which starts a uvicorn server here (keeping the container
alive).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from medicoder.db.pool import close_pool, init_pool
from medicoder.routes import agent, code, icd10, llm


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Open the DB pool on startup and close it on shutdown.

    Args:
        app: The FastAPI application the lifespan is attached to (unused beyond
            satisfying the lifespan protocol).
    """
    init_pool()
    try:
        yield
    finally:
        close_pool()


app = FastAPI(title="medicoder-technical", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe.

    Returns:
        A small JSON status map indicating the app is up.
    """
    return {"status": "ok"}


app.include_router(icd10.router)
app.include_router(llm.router)
app.include_router(agent.router)
app.include_router(code.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
