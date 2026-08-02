# medicoder-technical: Techniques & Implementation Guide

A study reference for every technique used in this project. Each section is
self-contained — jump to any topic without reading sequentially. Code snippets
show the essential pattern; cross-references point to the real implementation
in the repo for deeper study.

**How to use this document:**

1. Read **Part 1** for the architectural bird's-eye view.
2. Pick any part that interests you — each starts from first principles.
3. Follow the `file:line` references to study the full implementation.
4. Use the official docs links to go deeper on any technology.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Docker Compose Multi-Service Orchestration](#2-docker-compose-multi-service-orchestration)
3. [GPU Overlay Pattern](#3-gpu-overlay-pattern)
4. [Ollama Auto-Pull Sidecar](#4-ollama-auto-pull-sidecar)
5. [LiteLLM Proxy Layer](#5-litellm-proxy-layer)
6. [Postgres + pgvector + Hybrid Search](#6-postgres--pgvector--hybrid-search)
7. [Postgres Init Scripts & Idempotent Roles](#7-postgres-init-scripts--idempotent-roles)
8. [FastAPI Application Design](#8-fastapi-application-design)
9. [Connection Pooling with psycopg](#9-connection-pooling-with-psycopg)
10. [Bulk Data Loading](#10-bulk-data-loading)
11. [Idempotent Embedding Pipeline](#11-idempotent-embedding-pipeline)
12. [LLM Pipeline — Diagnose Step](#12-llm-pipeline--diagnose-step)
13. [LLM Pipeline — Hybrid Search Step](#13-llm-pipeline--hybrid-search-step)
14. [LangGraph Multi-Model Orchestration](#14-langgraph-multi-model-orchestration)
15. [Debate-Critic Reconciliation Loop](#15-debate-critic-reconciliation-loop)
16. [Evaluation Methodology](#16-evaluation-methodology)
17. [Docker Build Best Practices](#17-docker-build-best-practices)
18. [Environment-Driven Configuration](#18-environment-driven-configuration)
19. [Container Debugging with VSCode](#19-container-debugging-with-vscode)

---

## 1. Architecture Overview

### What we're building

A medical coding system that takes free-text clinical notes and returns
billable ICD-10-CM codes. The system uses LLMs for diagnosis generation
and vector similarity search for code matching — no tool-calling required.

### Service topology

```
                    Host machine
┌─────────────────────────────────────────────────────┐
│                                                     │
│  ┌─────────┐   ┌──────────┐   ┌─────────────────┐  │
│  │ postgres │   │ litellm  │   │   medicoder     │  │
│  │ (pgvec)  │   │ (proxy)  │   │   (FastAPI)     │  │
│  │          │◄──│          │◄──│ /code /diagnose │  │
│  │ icd10    │   │ routes   │   │ /search /test   │  │
│  │ codes    │   │ aliases  │   │                 │  │
│  │ +embed   │   │ to LLM   │   │  ┌───────────┐  │  │
│  │ +FTS     │   │          │   │  │ LangGraph │  │  │
│  │          │   │  ┌───────┤   │  │ StateGraph│  │  │
│  │          │   │  │ audit │   │  └───────────┘  │  │
│  │          │   │  │ log   │   │       │         │  │
│  └─────────┘   └──┤───────┤   │       ▼         │  │
│       ▲           │       │   │  ┌───────────┐  │  │
│       │           └───────┘   │  │  search   │  │  │
│       │              │        │  │ (vector + │  │  │
│       │              │        │  │  FTS)     │  │  │
│       │              ▼        │  └─────┬─────┘  │  │
│       │         ┌─────────┐   │        │        │  │
│       └─────────│ ollama  │◄──│────────┘        │  │
│            (GPU,│ (LLM +  │   │                 │  │
│         optional)│ embed)  │   │ :8000           │  │
│                 └─────────┘   └─────────────────┘  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Data flow

```
Clinical note
    │
    ▼
┌──────────────┐     ┌──────────────────┐
│  Diagnose    │────▶│  Search          │
│  (LLM model) │     │  (vector + FTS)  │
│              │     │                  │
│  1-10 dx     │     │  embed dx        │
│  + reasoning │     │  pgvector search │
│              │     │  + FTS candidates│
└──────────────┘     │  merge & rank    │
                     └──────┬───────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  ICD-10-CM   │
                     │  codes       │
                     │  (billable)  │
                     └──────────────┘
```

### Design philosophy

1. **No tool-calling** — medical LLMs are fine-tunes that generate text well
   but can't reliably call tools. The pipeline uses a fixed two-step order
   (diagnose → search) that doesn't require the model to orchestrate anything.
2. **Deterministic search** — vector similarity + FTS is deterministic; the
   same diagnoses always produce the same codes. Variability comes only from
   the LLM diagnosis step.
3. **Separation of concerns** — LiteLLM handles LLM routing/auditing, Postgres
   handles storage + vector search, FastAPI handles HTTP. No service does
   another's job.
4. **Idempotent everything** — data loading, embedding, and model pulling are
   all safe to re-run.

**Files to study:**
- `docker-compose.yml` — service definitions
- `main.py` — app wiring and router includes

---

## 2. Docker Compose Multi-Service Orchestration

### Concept

Docker Compose defines multiple containers that share a network, volumes,
and environment. Each service is a container; Compose handles networking,
dependency ordering, and restart policies.

### Key patterns used

#### Health-check dependencies

```yaml
services:
  medicoder:
    depends_on:
      postgres:
        condition: service_healthy   # wait for pg_isready
      litellm:
        condition: service_started   # just needs the container up
```

Without `condition: service_healthy`, the app would start before Postgres is
ready and fail on the first DB connection. The health check polls
`pg_isready` every 5s:

```yaml
postgres:
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U postgres -d medicoder"]
    interval: 5s
    timeout: 5s
    retries: 30
```

> **Docs:** [Docker Compose — depends_on with
> condition](https://docs.docker.com/compose/compose-file/05-services/#depends_on),
> [Healthcheck](https://docs.docker.com/compose/compose-file/05-services/#healthcheck)

#### Container isolation (no host port)

LiteLLM and Ollama publish **no host ports**. They're reachable only inside
the Compose network as `http://litellm:4000` and `http://ollama:11434`.
This prevents conflicts with host-level services and forces all access
through the app:

```yaml
litellm:
  # No "ports:" section — in-network only
  ...
```

Only the app publishes a port:

```yaml
medicoder:
  ports:
    - "8000:8000"
```

#### Named volumes for persistence

```yaml
postgres:
  volumes:
    - pgdata:/var/lib/postgresql/data
    - ./docker/postgres:/docker-entrypoint-initdb.d:ro

volumes:
  pgdata:
```

The `pgdata` volume survives container recreation. The init scripts mount
read-only so they run automatically on first boot.

> **Docs:** [Docker Compose — volumes](https://docs.docker.com/compose/compose-file/07-volumes/)

#### Entrypoint pattern

The app container runs the ICD-10 loader before starting the server:

```sh
#!/bin/sh
set -e
echo "[medicoder] ensuring ICD-10 data is loaded ..."
python -m medicoder.db.load_icd10
echo "[medicoder] starting: $*"
exec "$@"
```

The loader is idempotent — if data already exists, it exits immediately.
`exec "$@"` hands off to the CMD (`python main.py`) with PID 1.

**Files to study:**
- `docker-compose.yml` — full 3-service definition
- `docker/program/entrypoint.sh` — the entrypoint script

---

## 3. GPU Overlay Pattern

### Concept

The default stack runs without a GPU (cloud LLM or host Ollama). A GPU
overlay adds in-container Ollama services when you have AMD/NVIDIA hardware.

### Pattern: profile-gated overlay file

```yaml
# docker/docker-compose.gpu.yml
services:
  ollama-amd:
    profiles: ["gpu-amd"]
    image: ollama/ollama:rocm
    ...
  ollama-nvidia:
    profiles: ["gpu-nvidia"]
    image: ollama/ollama:latest
    ...
```

Activate with:

```bash
# AMD/ROCm
docker compose -f docker-compose.yml -f docker/docker-compose.gpu.yml \
   --profile gpu-amd up

# NVIDIA/CUDA
docker compose -f docker-compose.yml -f docker/docker-compose.gpu.yml \
   --profile gpu-nvidia up
```

### Pattern: shared network alias

Both GPU services share the alias `ollama`:

```yaml
ollama-amd:
  networks:
    default:
      aliases:
        - ollama
```

LiteLLM's config points at `http://ollama:11434` regardless of which vendor
is active. You must down one profile before up-ing the other.

### Pattern: environment override

The overlay **overrides** the upstream LLM config to point at the
in-container Ollama instead of the `.env` upstream:

```yaml
# In the overlay:
litellm:
  environment:
    LLM_UPSTREAM_MODEL: ollama/medgemma-27b-q4_k_s
    LLM_UPSTREAM_API_BASE: http://ollama:11434
    LLM_UPSTREAM_API_KEY: dummy
```

This means your `.env` upstream values are **ignored** in GPU mode.

> **Docs:** [Docker Compose — profiles](https://docs.docker.com/compose/profiles/),
> [Merge compose files](https://docs.docker.com/compose/multiple-compose-files/merge/)

**Files to study:**
- `docker/docker-compose.gpu.yml` — GPU overlay with AMD/NVIDIA profiles

---

## 4. Ollama Auto-Pull Sidecar

### Concept

When the GPU stack comes up, the models need to be pulled into Ollama
before they can be used. A sidecar container handles this automatically —
it polls Ollama until it's ready, then pulls every model listed in a TOML
config file.

### Pattern: fire-and-forget sidecar

```yaml
ollama-init:
  image: python:3.13-slim
  profiles: ["gpu-amd", "gpu-nvidia"]
  entrypoint: ["python", "/pull_models.py"]
  volumes:
    - ./docker/program/pull_models.py:/pull_models.py:ro
    - ./models.toml:/models.toml:ro
  depends_on:
    ollama-amd:
      condition: service_started
```

Nothing depends on this container — the app starts in parallel. The sidecar
exits when done.

### Pattern: polling for readiness

```python
def wait_for_ollama(base: str, deadline: float) -> None:
    while time.time() < deadline:
        try:
            get_existing(base)  # GET /api/tags
            return
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    raise SystemExit("Ollama not reachable; giving up")
```

### Pattern: idempotent pulls

```python
existing = {e.lower() for e in get_existing(base)}
for entry in models:
    name = entry["name"]
    if name.lower() in existing:
        print(f"[skip] {name}  (already present)")
        continue
    pull_one(base, name)
```

Models already in `/api/tags` are skipped — safe to re-run.

### Pattern: defense-in-depth verification

After all pulls report success, the sidecar re-checks `/api/tags` to confirm
each model actually landed:

```python
landed = {e.lower() for e in get_existing(base)}
for name in attempted:
    if name.lower() not in landed:
        failures.append(name)
```

This catches silent failures where Ollama streams a "success" status but the
model doesn't actually register.

### Configuration: `models.toml`

```toml
[[model]]
name = "hf.co/mradermacher/DeepSeek-R1-Medical-COT-GGUF:Q8_0"

[[model]]
name = "hf.co/unsloth/medgemma-27b-text-it-GGUF:Q4_K_S"

[[model]]
name = "hf.co/Abiray/zembed-1-Q4_K_M-GGUF:Q4_K_M"
```

Each entry is an Ollama registry tag. The sidecar uses Python 3.11+'s
`tomllib` to parse it — no third-party dependencies needed.

> **Docs:** [Ollama API — /api/pull](https://github.com/ollama/ollama/blob/main/docs/api.md#pull-a-model),
> [Python tomllib](https://docs.python.org/3/library/tomllib.html)

**Files to study:**
- `docker/program/pull_models.py` — full sidecar implementation (242 lines)
- `models.toml` — model registry

---

## 5. LiteLLM Proxy Layer

### Concept

[LiteLLM](https://docs.litellm.ai/docs/proxy/quick_start) is a proxy server
that exposes a unified OpenAI-compatible API. Your app always calls the same
endpoint (`http://litellm:4000/v1/chat/completions`) with a model alias;
LiteLLM routes the call to the actual upstream (cloud, self-deployed, or
local Ollama).

### Pattern: model aliasing

```yaml
# docker/litellm/config.yaml
model_list:
  - model_name: medgemma-27b-q4_k_s        # alias your app calls
    litellm_params:
      model: ollama/hf.co/unsloth/medgemma-27b-text-it-GGUF:Q4_K_S
      api_base: http://ollama:11434
      api_key: dummy

  - model_name: A                           # env-driven alias
    litellm_params:
      model: os.environ/LLM_UPSTREAM_MODEL
      api_base: os.environ/LLM_UPSTREAM_API_BASE
      api_key: os.environ/LLM_UPSTREAM_API_KEY
```

The app code never knows which actual model is behind the alias. Swap
models by editing config or env vars — zero code changes.

### Pattern: `os.environ/` in config

LiteLLM supports reading env vars directly in YAML via the `os.environ/`
prefix. This lets you keep secrets in `.env` rather than the config file:

```yaml
model: os.environ/LLM_UPSTREAM_MODEL
```

### Pattern: custom success callback for audit logging

LiteLLM's built-in `LiteLLM_SpendLogs` only tracks tokens and cost ($0 for
local models). To capture full prompts and responses, write a custom
callback:

```python
# docker/litellm/log_callback.py
from litellm.integrations.custom_logger import CustomLogger

class LLMCallLogger(CustomLogger):
    async def async_log_success_event(
        self, kwargs, response_obj, start_time, end_time
    ) -> None:
        # Extract prompt, thinking, output, tool_calls from kwargs + response_obj
        # INSERT INTO llm_call_log ...
```

Wire it in config:

```yaml
litellm_settings:
  callbacks: log_callback.llm_call_logger
```

The callback writes to a `llm_call_log` table in the `litellm` database via
LiteLLM's own Prisma client — no extra Postgres driver needed in the image.

> **Docs:** [LiteLLM Proxy — config.yaml
> reference](https://docs.litellm.ai/docs/proxy/configs),
> [Custom callbacks](https://docs.litellm.ai/docs/observability/custom_callback)

**Files to study:**
- `docker/litellm/config.yaml` — alias routing + callback registration
- `docker/litellm/log_callback.py` — audit logging callback (141 lines)

---

## 6. Postgres + pgvector + Hybrid Search

### Concept

[pgvector](https://github.com/pgvector/pgvector) adds vector similarity
search to PostgreSQL. Combined with PostgreSQL's built-in full-text search,
you get hybrid search: semantic matching from vectors + lexical matching
from FTS.

### Pattern: `halfvec` vs `vector`

pgvector's HNSW index caps `vector` at 2000 dimensions. For higher-dimension
embeddings (like zembed-1's 2560), use `halfvec` which supports up to 4000:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE icd10_codes
    ADD COLUMN embedding halfvec(2560);
```

`halfvec` stores float16 (half precision), halving storage with negligible
accuracy loss for similarity search.

> **Docs:** [pgvector — half precision
> vectors](https://github.com/pgvector/pgvector#half-precision-vectors)

### Pattern: HNSW index tuning

HNSW (Hierarchical Navigable Small World) is an approximate nearest neighbor
index. Two build-time parameters control the quality/speed tradeoff:

```sql
CREATE INDEX idx_icd10_codes_embedding
    ON icd10_codes USING hnsw (embedding halfvec_cosine_ops)
    WITH (m = 32, ef_construction = 128);
```

| Parameter | Default | Here | Effect |
|---|---|---|---|
| `m` | 16 | 32 | Max connections per node. Higher = better recall, more memory. |
| `ef_construction` | 64 | 128 | Search effort during index build. Higher = better index quality. |

At query time, set `ef_search` per-session for additional recall:

```python
conn.execute("SET LOCAL hnsw.ef_search = 200")
```

The default `ef_search=40` misses obvious matches on 98k codes. Setting 200
trades a few ms of latency for much better recall.

> **Docs:** [pgvector — HNSW
> indexing](https://github.com/pgvector/pgvector#hnsw)

### Pattern: cosine distance operator

pgvector uses the `<=>` operator for cosine distance (1 - cosine
similarity):

```sql
SELECT code, 1 - (embedding <=> '%s'::halfvec) AS vec_sim
FROM icd10_codes
WHERE is_billable
ORDER BY embedding <=> '%s'::halfvec
LIMIT 10;
```

Order by `<=>` ascending (closest = most similar). Compute `1 - distance`
to get a 0-1 similarity score.

### Pattern: generated columns

Postgres can compute columns automatically from other columns:

```sql
is_billable boolean GENERATED ALWAYS AS (code_type = 1) STORED,
```

This avoids a separate UPDATE step after loading — the column is populated
on INSERT. Use it for any derivable value.

Similarly, the FTS column:

```sql
search_tsv tsvector GENERATED ALWAYS AS (
    to_tsvector('english',
        coalesce(short_desc, '') || ' ' || coalesce(long_desc, ''))
) STORED;
```

> **Docs:** [PostgreSQL — generated
> columns](https://www.postgresql.org/docs/current/ddl-generated-columns.html)

### Pattern: partial index

Index only billable rows (the ones that matter for search):

```sql
CREATE INDEX idx_icd10_codes_fts
    ON icd10_codes USING gin (search_tsv)
    WHERE is_billable;
```

This halves the index size and speeds up queries that filter to billable.

> **Docs:** [PostgreSQL — partial
> indexes](https://www.postgresql.org/docs/current/indexes-partial.html)

### Pattern: hybrid vector-primary + FTS-boost search

The core search algorithm fetches candidates from two sources and merges
them:

```
┌──────────────────────────────────────────────┐
│              search_icd10(query)              │
│                                              │
│  ┌──────────────┐    ┌──────────────────┐   │
│  │ Vector search│    │  FTS search      │   │
│  │ (HNSW)       │    │  (AND → OR)      │   │
│  │              │    │                  │   │
│  │ k * 5 cands  │    │  k * 5 cands     │   │
│  └──────┬───────┘    └────────┬─────────┘   │
│         │                     │              │
│         └────────┬────────────┘              │
│                  ▼                           │
│         ┌────────────────┐                   │
│         │ _merge_and_    │                   │
│         │  score()       │                   │
│         │                │                   │
│         │ vec + FTS →    │                   │
│         │ boost          │                   │
│         │ vec only →     │                   │
│         │ raw vec_sim    │                   │
│         │ FTS only →     │                   │
│         │ floor filter   │                   │
│         └───────┬────────┘                   │
│                 ▼                            │
│          top-k results                       │
└──────────────────────────────────────────────┘
```

Scoring rules:

| Source | Score |
|---|---|
| In both vector + FTS | `vec_sim + FTS_BOOST * norm_fts` |
| Vector only | `vec_sim` (unchanged) |
| FTS only, AND mode (exact terms) | `vec_sim + FTS_BOOST * norm_fts` (no floor) |
| FTS only, OR mode (partial match) | Only if `vec_sim >= FTS_FLOOR` |

When `FTS_BOOST = 0` (current default), results are identical to pure vector
search. The FTS infrastructure is preserved for future tuning.

**Files to study:**
- `medicoder/medical.py:330` — `_fetch_vector()` (HNSW query)
- `medicoder/medical.py:432` — `_merge_and_score()` (the merge logic)
- `medicoder/medical.py:518` — `search_icd10()` (orchestrator)
- `docker/postgres/02-embedding.sql` — HNSW index definition
- `docker/postgres/03-fts.sql` — generated tsvector + GIN index

> **Docs:** [pgvector — querying](https://github.com/pgvector/pgvector#querying),
> [PostgreSQL — full text
> search](https://www.postgresql.org/docs/current/textsearch.html)

---

## 7. Postgres Init Scripts & Idempotent Roles

### Concept

PostgreSQL's Docker image runs scripts in `/docker-entrypoint-initdb.d/`
on first boot (empty data volume). Scripts run alphabetically. SQL files
run via `psql`; shell scripts run directly. This is where you create
extensions, tables, roles, and indexes.

### Pattern: ordered init scripts

```
docker/postgres/
├── 00-schema.sql        # extension + table + basic indexes
├── 01-roles.sh          # application role (medicoder)
├── 02-litellm.sh        # litellm role + database
├── 02-embedding.sql     # embedding column + HNSW index
└── 03-fts.sql           # tsvector column + GIN index
```

The numbering enforces order: schema before roles, roles before embedding,
embedding before FTS.

### Pattern: idempotent role creation

Postgres doesn't have `CREATE ROLE IF NOT EXISTS`. Use a `DO` block:

```sql
DO $do$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'medicoder') THEN
        CREATE ROLE medicoder LOGIN PASSWORD '${MEDICODER_PASSWORD}';
    ELSE
        ALTER ROLE medicoder LOGIN PASSWORD '${MEDICODER_PASSWORD}';
    END IF;
END
$do$;
```

### Pattern: conditional database creation with `\gexec`

`\gexec` runs the result of the preceding query as SQL:

```sql
SELECT 'CREATE DATABASE litellm OWNER litellm'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'litellm')
\gexec
```

This is the idiomatic way to "create database if not exists" in Postgres.

### Pattern: shell init script with env vars

```sh
#!/bin/sh
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<EOSQL
  -- SQL here, with ${ENV_VAR} expansion from the shell
EOSQL
```

The heredoc (`<<EOSQL`) passes the SQL to `psql`. Shell variables like
`${MEDICODER_PASSWORD}` are expanded before `psql` sees them.

> **Docs:** [PostgreSQL Docker — init
> scripts](https://hub.docker.com/_/postgres), [psql
> \gexec](https://www.postgresql.org/docs/current/app-psql.html)

**Files to study:**
- `docker/postgres/00-schema.sql` — table schema with generated column
- `docker/postgres/01-roles.sh` — idempotent role creation
- `docker/postgres/02-litellm.sh` — `\gexec` database creation

---

## 8. FastAPI Application Design

### Concept

[FastAPI](https://fastapi.tiangolo.com/) is an async Python web framework.
Key features used here: lifespan context (startup/shutdown), APIRouter for
modular routes, and Pydantic models for request/response validation.

### Pattern: lifespan context for resource management

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    init_pool()           # startup: open DB pool
    try:
        yield             # app runs here
    finally:
        close_pool()      # shutdown: close pool

app = FastAPI(lifespan=lifespan)
```

The `yield` separates startup from shutdown. The pool is open for all
requests and closed cleanly on exit. This replaced the older
`@app.on_event("startup")` pattern.

> **Docs:** [FastAPI — lifespan
> events](https://fastapi.tiangolo.com/advanced/events/)

### Pattern: modular routers

```python
# main.py
from medicoder.routes import code, diagnose, icd10, llm

app.include_router(icd10.router)
app.include_router(llm.router)
app.include_router(code.router)
app.include_router(diagnose.router)
```

Each route module defines its own `APIRouter`:

```python
# medicoder/routes/icd10.py
router = APIRouter()

@router.get("/codes")
def list_codes(...): ...

@router.post("/search")
def search_codes(...): ...
```

This keeps `main.py` thin (just wiring) and route logic isolated.

> **Docs:** [FastAPI — Bigger
> applications](https://fastapi.tiangolo.com/tutorial/bigger-applications/)

### Pattern: Pydantic models for validation + docs

```python
class CodeRequest(BaseModel):
    text: str
    medical_model: str = "medgemma-27b-q4_k_s"
    k: int = 3

class ICD10Match(BaseModel):
    code: str
    short_desc: str
    long_desc: str
    similarity: float
```

FastAPI uses these for:
- **Request validation** — invalid types/types return 422 automatically
- **Response serialization** — `response_model=list[ICD10Match]` ensures
  consistent JSON shape
- **OpenAPI docs** — `/docs` page is generated from these models

> **Docs:** [Pydantic v2 docs](https://docs.pydantic.dev/latest/)

### Pattern: TypedDict for internal state

When sharing mutable state between graph nodes (LangGraph), use a
`TypedDict` instead of a dataclass:

```python
class DiagnoseState(TypedDict):
    text: str
    k: int
    diagnoses_a: list[str]
    diagnoses_b: list[str]
    codes_a: list[ICD10Match]
    codes_b: list[ICD10Match]
    critic_rounds: list[CriticRound]
```

LangGraph requires a TypedDict for its state schema. Defined in
`schemas.py` to avoid circular imports between `routes/diagnose.py` and
`critic.py`.

**Files to study:**
- `main.py` — lifespan + router wiring
- `medicoder/schemas.py` — all Pydantic models + TypedDict
- `medicoder/routes/code.py` — a simple two-step pipeline route

---

## 9. Connection Pooling with psycopg

### Concept

Opening a new DB connection per request is expensive (TCP handshake, auth,
SSL). A connection pool keeps a set of warm connections and lends them out
per request.

### Pattern: psycopg ConnectionPool with lifespan

```python
# medicoder/db/pool.py
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

_pool: ConnectionPool | None = None

def init_pool() -> None:
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
```

`open=True` opens connections immediately on creation. `configure` sets the
row factory on every checked-out connection.

### Pattern: dict_row for ergonomic results

```python
def _configure(conn) -> None:
    conn.row_factory = dict_row
```

With `dict_row`, queries return Python dicts instead of tuples:

```python
# Without dict_row:
row = cur.fetchone()
code = row[0]      # positional — fragile

# With dict_row:
row = cur.fetchone()
code = row["code"]  # named — readable
```

### Pattern: context-managed checkout

```python
with get_pool().connection() as conn:
    rows = conn.execute("SELECT ...").fetchall()
# Connection returned to pool automatically
```

The `with` block borrows a connection and returns it on exit, even on
exception.

> **Docs:** [psycopg3 — connection
> pool](https://www.psycopg.org/psycopg3/docs/api/pool.html),
> [Row factories](https://www.psycopg.org/psycopg3/docs/api/rows.html)

**Files to study:**
- `medicoder/db/pool.py` — full pool implementation (56 lines)
- `medicoder/db/connect.py` — CLI connection (for scripts, not the app)

---

## 10. Bulk Data Loading

### Concept

The ICD-10-CM order file is a fixed-width ASCII file with ~98k rows. Loading
it into Postgres via individual INSERTs would be slow. The binary COPY
protocol is 10-100x faster.

### Pattern: fixed-width parsing with pandas

```python
import pandas as pd

COLSPECS = [(0, 5), (6, 13), (14, 15), (16, 76), (77, None)]
COLUMNS = ["order_number", "code", "code_type", "short_desc", "long_desc"]

df = pd.read_fwf(
    path,
    colspecs=COLSPECS,
    names=COLUMNS,
    dtype={"order_number": "int64", "code": "string", "code_type": "int8"},
)
```

`pd.read_fwf` parses fixed-width files given column boundaries. Each tuple
is `(start, end)` with `None` meaning "to end of line".

> **Docs:** [pandas —
> read_fwf](https://pandas.pydata.org/docs/reference/api/pandas.read_fwf.html)

### Pattern: binary COPY

```python
with conn.cursor() as cur:
    with cur.copy("COPY icd10_codes (...) FROM STDIN") as copy:
        for row in df.itertuples(index=False):
            copy.write_row(row)
    conn.commit()
```

`COPY ... FROM STDIN` streams rows directly into Postgres using the binary
protocol. `cur.copy()` is a context manager that manages the COPY stream.
`write_row()` sends a single row.

For 98k rows, this takes ~1 second vs ~60 seconds for individual INSERTs.

> **Docs:** [psycopg3 — COPY](https://www.psycopg.org/psycopg3/docs/cursor.html#copy),
> [PostgreSQL — COPY](https://www.postgresql.org/docs/current/sql-copy.html)

### Pattern: idempotent loading

```python
cur.execute("SELECT count(*) FROM icd10_codes")
existing = cur.fetchone()[0]
if existing and not force:
    print(f"already has {existing} rows; skipping.")
    return 0
```

Check row count before loading. If rows exist, skip unless `LOAD_FORCE=1`.
This makes the loader safe to run on every container boot.

**Files to study:**
- `medicoder/db/load_icd10.py` — full loader (130 lines)
- `docker/program/entrypoint.sh` — runs loader before app start

---

## 11. Idempotent Embedding Pipeline

### Concept

Each ICD-10 code needs a vector embedding for similarity search. The
embedder processes codes in batches via the embedding model API and stores
results in the `embedding` column. It's idempotent: only codes without an
embedding are processed.

### Pattern: batch embedding

```python
BATCH_SIZE = 128  # rows per API call

SELECT_BATCH_SQL = (
    "SELECT order_number, long_desc FROM icd10_codes "
    "WHERE embedding IS NULL ORDER BY order_number LIMIT %s"
)

while True:
    cur.execute(SELECT_BATCH_SQL, (BATCH_SIZE,))
    rows = cur.fetchall()
    if not rows:
        break
    texts = [_clean_embed_text(row[1]) for row in rows]
    vectors = proxy.embed(texts)  # one API call for BATCH_SIZE texts
    for (order_number, _), vec in zip(rows, vectors):
        cur.execute(
            "UPDATE icd10_codes SET embedding = %s::halfvec WHERE order_number = %s",
            (json.dumps(vec), order_number),
        )
    conn.commit()
```

Key decisions:
- **Batch API call** — one HTTP request embeds 128 texts, not 128 requests
- **Commit per batch** — if the process crashes, completed batches persist
- **`ORDER BY order_number`** — deterministic processing order

### Pattern: embedding text cleaning

The word "unspecified" in code descriptions tanks embedding similarity.
For example, C539 ("Malignant neoplasm of cervix uteri, unspecified")
scores only 0.48 similarity to the query "Malignant neoplasm of cervix
uteri". Stripping the qualifier before embedding raises it to 0.99:

```python
import re

_UNSPECIFIED_RE = re.compile(r",?\s*unspecified", re.IGNORECASE)

def _clean_embed_text(desc: str) -> str:
    return _UNSPECIFIED_RE.sub("", desc).strip()
```

This only affects the embedding input — the stored `long_desc` is
unchanged. To re-embed after changing the cleaning function:

```sql
UPDATE icd10_codes SET embedding = NULL WHERE long_desc ILIKE '%unspecified%';
```

Then re-run the embedder — it only processes rows where `embedding IS NULL`.

### Pattern: vector serialization

Embeddings are passed as JSON strings cast to `halfvec`:

```python
cur.execute(
    "UPDATE icd10_codes SET embedding = %s::halfvec WHERE order_number = %s",
    (json.dumps(vec), order_number),
)
```

`json.dumps(vec)` produces `[0.123, -0.456, ...]` which Postgres casts to
`halfvec(2560)`.

> **Docs:** [OpenAI embeddings API (compatibility
> format)](https://platform.openai.com/docs/guides/embeddings),
> [pgvector — insertion](https://github.com/pgvector/pgvector#insertion)

**Files to study:**
- `medicoder/db/embed_icd10.py` — full embedder (100 lines)
- `medicoder/proxy.py:85` — `embed()` function

---

## 12. LLM Pipeline — Diagnose Step

### Concept

The diagnose step sends clinical text to a medical LLM and gets back 1-10
diagnosis phrases. No tool-calling is involved — the model just returns
structured text that we parse.

### Pattern: no tool-calling design

Medical LLMs are text-generation fine-tunes. Most can't reliably call tools
(function calling). Instead of fighting this, the pipeline uses a fixed
two-step process:

1. Model generates diagnoses (plain text / JSON)
2. App code searches for codes (vector + FTS)

The model never needs to call tools or know about the code database.

### Pattern: thinking block separation

Reasoning models (like DeepSeek-R1) emit `<think>...</think>` blocks
inline with their output. The proxy separates these before parsing:

```python
_THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)

def extract_thinking(text: str) -> tuple[str, str]:
    """Returns (thinking, output) — thinking is the <think> content."""
    thinking_parts: list[str] = []

    def _capture(m):
        thinking_parts.append(m.group(1).strip())
        return ""

    cleaned = _THINK_RE.sub(_capture, text)

    # Handle unclosed <think> (model ran out of tokens mid-thought)
    if "<think>" in cleaned:
        before, after = cleaned.split("<think>", 1)
        thinking_parts.append(after.strip())
        cleaned = before

    # Handle orphaned </think> (Ollama stripped opening tag)
    if "</think>" in cleaned:
        before, after = cleaned.rsplit("</think>", 1)
        thinking_parts.append(before.strip())
        cleaned = after

    return "\n".join(t for t in thinking_parts if t), cleaned.strip()
```

Three cases handled: complete blocks, unclosed blocks (truncation), and
orphaned closers (Ollama bug).

### Pattern: structured JSON output parsing

Models don't always return clean JSON. Use a 3-strategy fallback:

```python
def _extract_json(raw: str) -> dict | None:
    # Strategy 1: direct json.loads (best case)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract from ```json ... ``` markdown fences
    if "```" in raw:
        for part in raw.split("```"):
            if part.strip().startswith("json"):
                try:
                    return json.loads(part.strip()[4:])
                except json.JSONDecodeError:
                    continue

    # Strategy 3: grab outermost { ... } substring
    first = raw.find("{")
    last = raw.rfind("}")
    if first != -1 and last > first:
        try:
            return json.loads(raw[first:last + 1])
        except json.JSONDecodeError:
            pass

    return None
```

### Pattern: incremental retry with temperature escalation

```python
for attempt in range(_MAX_RETRIES):
    temp = min(0.1 + attempt * 0.05, 0.3)  # 0.1 → 0.15 → 0.2 → 0.25

    if attempt == 0:
        system_prompt = _DIAGNOSE_SYSTEM         # full prompt with reasoning
        max_tokens = 4096
    else:
        system_prompt = _DIAGNOSE_SYSTEM_CONCISE  # concise, no reasoning field
        max_tokens = 2048

    raw, thinking = proxy.chat_completion(model, messages, temperature=temp)
    parsed = _try_parse_output(raw)
    if parsed is not None:
        return parsed
```

The first attempt uses a generous token budget and a full prompt that
includes a `reasoning` field. If the model's output is unparseable (e.g.,
thinking loop consumed all tokens), retries switch to a concise prompt
with a smaller budget and slightly higher temperature to break the loop.

### Pattern: system prompt with domain-specific terminology

```
ICD-10-CM naming conventions:
- Use "Malignant neoplasm of [site]" — not "cancer" or "carcinoma"
- Use "Unspecified" when the documentation does not specify the anatomical site
- Include clinical qualifiers where documented (e.g., "acute", "in remission")
```

Teaching the model ICD-10-CM terminology improves search quality — the
diagnoses are already close to the code descriptions in embedding space.

**Files to study:**
- `medicoder/proxy.py:26` — `chat_completion()` with thinking separation
- `medicoder/proxy.py:118` — `extract_thinking()` (3-case handler)
- `medicoder/medical.py:114` — `_extract_json()` (3-strategy parser)
- `medicoder/medical.py:228` — `diagnose()` with retry logic
- `medicoder/medical.py:34` — `_DIAGNOSE_SYSTEM` prompt

---

## 13. LLM Pipeline — Hybrid Search Step

### Concept

Given a list of diagnosis phrases, find the best-matching billable ICD-10
codes. The search combines vector similarity (semantic) with full-text
search (lexical) and merges the results.

### Pattern: batch embedding all queries

```python
def search_icd10(queries: list[str], k: int = 3) -> list[ICD10Match]:
    vectors = proxy.embed(queries)  # ONE API call for all diagnoses
    cand_k = k * _CANDIDATE_MULT    # over-fetch: k=3 → 15 candidates

    with get_pool().connection() as conn:
        conn.execute("SET LOCAL hnsw.ef_search = 200")
        for query_text, vec in zip(queries, vectors):
            vec_rows = _fetch_vector(conn, json.dumps(vec), cand_k)
            fts_rows, fts_is_and = _fetch_fts(conn, query_text, json.dumps(vec), cand_k)
            for m in _merge_and_score(vec_rows, fts_rows, k, fts_is_and):
                ...
```

All diagnoses are embedded in a single API call (not one per diagnosis).
`SET LOCAL` sets `ef_search` for the current transaction only.

### Pattern: candidate over-fetching

Fetch `k * 5` candidates from each source, then take the top `k` after
merging. This gives the merge step more material to work with:

```python
cand_k = k * _CANDIDATE_MULT  # k=3 → 15 candidates from each source
```

### Pattern: FTS AND → OR fallback

AND mode requires all query lexemes to match. When AND returns nothing
(e.g., stemming mismatch — "uterine" stems to `uterin` but the description
uses `uteri`), fall back to OR mode:

```python
def _fetch_fts(conn, query_text, query_vec_json, limit):
    rows = _fetch_fts_and(conn, query_text, query_vec_json, limit)
    if rows:
        return rows, True   # AND mode succeeded
    rows = _fetch_fts_or(conn, query_text, query_vec_json, limit)
    return rows, False      # fell back to OR mode
```

The AND/OR flag is passed to `_merge_and_score` — AND-mode candidates
bypass the similarity floor (they're exact-term hits), while OR-mode
candidates must pass the floor to filter noise.

### Pattern: cross-query deduplication

When multiple diagnoses match the same code, keep the highest score:

```python
seen: dict[str, ICD10Match] = {}
for m in _merge_and_score(vec_rows, fts_rows, k, fts_is_and):
    if m.code not in seen or m.similarity > seen[m.code].similarity:
        seen[m.code] = m

return sorted(seen.values(), key=lambda m: m.similarity, reverse=True)
```

**Files to study:**
- `medicoder/medical.py:518` — `search_icd10()` (full orchestrator)
- `medicoder/medical.py:432` — `_merge_and_score()` (scoring rules)
- `medicoder/medical.py:413` — `_fetch_fts()` (AND → OR fallback)

---

## 14. LangGraph Multi-Model Orchestration

### Concept

[LangGraph](https://langchain-ai.github.io/langgraph/) is a library for
building stateful, multi-step LLM workflows as directed graphs. Nodes are
functions; edges define execution order. This project uses it to run two
models in parallel and optionally trigger a critic loop.

### Pattern: StateGraph with TypedDict state

```python
from langgraph.graph import END, START, StateGraph

class DiagnoseState(TypedDict):
    text: str
    k: int
    enable_critic: bool
    diagnoses_a: list[str]
    diagnoses_b: list[str]
    codes_a: list[ICD10Match]
    codes_b: list[ICD10Match]
    critic_rounds: list[CriticRound]

g = StateGraph(DiagnoseState)
```

Each node receives the state dict, reads what it needs, and returns a
partial dict of updates. LangGraph merges the updates into the state.

### Pattern: parallel fan-out

```python
g.add_node("diagnose_a", diagnose_a)
g.add_node("diagnose_b", diagnose_b)

# Both run from START — parallel execution
g.add_edge(START, "diagnose_a")
g.add_edge(START, "diagnose_b")
```

Both nodes start simultaneously. LangGraph runs them concurrently.

### Pattern: fan-in with multiple edges

```python
g.add_edge("diagnose_a", "search")
g.add_edge("diagnose_b", "search")
```

The `search` node waits until both `diagnose_a` and `diagnose_b` have
completed. LangGraph handles the synchronization.

### Pattern: conditional edges

```python
g.add_conditional_edges("search", should_critic, {
    "critic": "critic_think",
    "end": END,
})
```

The `should_critic` function inspects the state and returns `"critic"` or
`"end"`. LangGraph routes to the corresponding node.

### Pattern: loop via conditional edge

```python
g.add_edge("critic_think", "critic_search")
g.add_conditional_edges("critic_search", should_continue_critic, {
    "loop": "critic_think",   # go back for another round
    "end": END,
})
```

This creates a cycle: `critic_think → critic_search → (loop back to
critic_think or end)`. The `should_continue_critic` function checks the
round count and the critic's `done` flag.

### Pattern: module separation

Graph **wiring** lives in `routes/diagnose.py` (just the topology).
Node **logic** (what each node does) lives in `critic.py`. This keeps
the route file focused on structure and the critic file focused on
behavior.

```
routes/diagnose.py          critic.py
─────────────────           ─────────
_build_graph()              should_critic()
  add_node("critic_think")  critic_think()
  add_edge(...)             critic_search()
  add_conditional_edges()   should_continue_critic()
```

> **Docs:** [LangGraph — StateGraph](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.StateGraph),
> [Conditional edges](https://langchain-ai.github.io/langgraph/concepts/low_level/#conditional-edges)

**Files to study:**
- `medicoder/routes/diagnose.py:92` — `_build_graph()` (full wiring)
- `medicoder/critic.py` — node implementations

---

## 15. Debate-Critic Reconciliation Loop

### Concept

When two medical models disagree (different top-1 code or different
diagnosis count), a critic model reviews both outputs and produces
reconciled diagnoses. The critic can loop for multiple rounds until
confident.

### Pattern: trigger logic

```python
def should_critic(state: DiagnoseState) -> str:
    if not state.get("enable_critic", True):
        return "end"
    if _MAX_CRITIC_ROUNDS <= 0:
        return "end"

    # Top-1 disagreement?
    codes_a = [c.code for c in state.get("codes_a", [])]
    codes_b = [c.code for c in state.get("codes_b", [])]
    top1_diff = (codes_a[:1] != codes_b[:1])

    # Diagnosis count mismatch?
    dx_a = len(state.get("diagnoses_a", []))
    dx_b = len(state.get("diagnoses_b", []))
    count_diff = (dx_a != dx_b)

    return "critic" if (top1_diff or count_diff) else "end"
```

Two conditions trigger the critic: different top-1 code, or different
diagnosis count. Both are signals that the models disagree substantively.

### Pattern: critic context building

The critic receives a structured summary of both models' outputs:

```python
# Simplified — the real implementation builds a richer context
context = f"""
Model A ({model_a}): {diagnoses_a} → {codes_a}
Reasoning: {reasoning_a}

Model B ({model_b}): {diagnoses_b} → {codes_b}
Reasoning: {reasoning_b}

Previous rounds: {round_history}
"""
```

The critic uses this to reason about where the models agree and disagree.

### Pattern: `done` flag for loop control

```python
class CriticOutput(BaseModel):
    reasoning: str = ""
    diagnoses: list[str]
    queries: list[str] = []
    done: bool = False    # True = confident, end the loop
```

The critic signals `done: true` when confident. The loop also has a hard
cap (`MAX_CRITIC_ROUNDS`) as a safety net.

### Pattern: ModelOutput dataclass for clean signatures

Instead of passing 10 positional arguments, bundle per-model data:

```python
@dataclass
class ModelOutput:
    model: str
    diagnoses: list[str]
    codes: list[ICD10Match]
    reasoning: str
```

Functions accept `ModelOutput` instead of individual arguments:

```python
def diagnose_critic(a: ModelOutput, b: ModelOutput, rounds: list[CriticRound]) -> ...:
```

This makes the code self-documenting and resistant to argument-order bugs.

**Files to study:**
- `medicoder/critic.py:49` — configuration constants
- `medicoder/critic.py:87` — `CriticOutput` Pydantic model
- `medicoder/critic.py` (full file) — critic logic, context building, graph nodes

---

## 16. Evaluation Methodology

### Concept

Measuring ICD-10-CM coding accuracy requires multiple metrics. A single
accuracy number hides important distinctions: is the system getting the
right disease but wrong specificity? Or is it totally wrong?

### Pattern: multi-level metrics

| Metric | What it measures |
|---|---|
| **Top-1 accuracy** | Is the highest-ranked code correct? |
| **Top-3 accuracy** | Is the correct code anywhere in the top 3? |
| **Recall** | Fraction of ground-truth codes found anywhere in results |
| **Precision@GT** | Fraction of results that match a ground-truth code |
| **F1** | Harmonic mean of recall and precision |
| **Category recall** | Same as recall but using 3-char prefixes (right disease, wrong specificity) |
| **Category precision** | Same for precision |

```python
def _model_metrics(gt: set[str], codes: list[str], dx_cnt: int, gt_cnt: int) -> dict:
    tp = set(codes) & gt
    denom = max(dx_cnt, gt_cnt)
    return {
        "top1_hit": bool(codes) and codes[0] in gt,
        "top3_hit": any(c in gt for c in codes[:3]),
        "recall": round(len(tp) / len(gt), 4) if gt else 0.0,
        "precision": round(len(tp) / denom, 4) if denom else 0.0,
        "cat_recall": round(_cat_recall(gt, codes), 4),
        "cat_precision": round(_cat_precision(gt, codes), 4),
    }
```

Category-level metrics use 3-char prefixes to distinguish "right disease,
wrong specificity" (e.g., `I10` vs `I119` — both hypertension) from total
misses (e.g., `I10` vs `C539`).

```python
def _cat_recall(gt: set[str], codes: list[str]) -> float:
    pred_cats = {c[:3] for c in codes}
    return sum(1 for g in gt if g[:3] in pred_cats) / len(gt)
```

### Pattern: search replay for isolation

To measure **search-only** changes (without re-running expensive LLM
calls), replay stored diagnoses through the current search endpoint:

```python
# replay_search.py
for ef_path in sorted(glob.glob("eval_output*.json")):
    with open(ef_path) as f:
        ef = json.load(f)
    for detail in ef["details"]:
        for label in ("model_a", "model_b", "critic"):
            diagnoses = detail[label]["diagnoses"]
            # Send stored diagnoses to current /search endpoint
            new_codes = call_search(diagnoses)
            # Compare against original codes from the eval file
```

This isolates the search step from the diagnosis step. If you change the
embedding model or search algorithm, you can measure the impact without
re-running the LLMs.

> **Docs:** [Information retrieval evaluation — precision and
> recall](https://en.wikipedia.org/wiki/Precision_and_recall)

**Files to study:**
- `evaluate.py` — full evaluator (544 lines)
- `replay_search.py` — search-only replay tool
- `analyze_cases.py` — case difficulty ranking

---

## 17. Docker Build Best Practices

### Pattern: uv for fast dependency management

[uv](https://github.com/astral-sh/uv) is a fast Python package installer
written in Rust. It's ~10-100x faster than pip:

```dockerfile
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
```

`--frozen` requires `uv.lock` to match `pyproject.toml` exactly —
reproducible builds. `--no-dev` skips dev dependencies. `--no-install-project`
installs only dependencies (the project code is COPYed separately for
better layer caching).

### Pattern: layer caching

```dockerfile
# Dependencies first (changes rarely → cached)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Application code second (changes often → only this layer rebuilds)
COPY medicoder ./medicoder
COPY main.py data/icd10cm_order_2026.txt ./
```

When you change application code, Docker reuses the cached dependency layer
and only rebuilds the COPY layer. This makes `docker compose up --build`
fast (~5 seconds for code-only changes).

### Pattern: non-root runtime

```dockerfile
RUN chmod +x /entrypoint.sh \
 && groupadd --system app && useradd --system --gid app --home-dir /app app \
 && chown --recursive app:app /app /entrypoint.sh
USER app
```

Running as non-root is a security best practice. If an attacker escapes the
container, they don't get root on the host.

### Pattern: unbuffered output

```dockerfile
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
```

`PYTHONUNBUFFERED=1` makes `print()` flush immediately — you see logs in
real-time via `docker logs`. Without this, Python buffers output and you
see nothing until the buffer fills or the process exits.

> **Docs:** [uv — Docker integration](https://github.com/astral-sh/uv/blob/main/docs/guides/integration/docker.md),
> [Docker — best practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)

**Files to study:**
- `Dockerfile` — full build (38 lines)

---

## 18. Environment-Driven Configuration

### Concept

Every tunable parameter should be configurable without code changes. The
chain is:

```
.env.example (documentation)
    ↓ copy to
.env (your values)
    ↓ read by
docker-compose.yml (${VAR:-default})
    ↓ passed to
container environment
    ↓ read by
os.environ.get("VAR", "default")
```

### Pattern: the full chain

**Step 1 — `.env.example` (committed, documents all options):**

```bash
# Max number of diagnoses a medical model can output per case.
MAX_DIAGNOSES=10
```

**Step 2 — `docker-compose.yml` (passes to container with default):**

```yaml
medicoder:
  environment:
    MAX_DIAGNOSES: ${MAX_DIAGNOSES:-10}
```

`${MAX_DIAGNOSES:-10}` reads from `.env`; falls back to `10` if unset.

**Step 3 — Python code (reads with type coercion):**

```python
import os

_MAX_DIAGNOSES = int(os.environ.get("MAX_DIAGNOSES", "10"))
_FTS_BOOST = float(os.environ.get("FTS_BOOST", "0.0"))
_CRITIC_MAX_TOKENS = int(os.environ.get("CRITIC_MAX_TOKENS", "8192"))
```

`os.environ.get()` returns strings. `int()` / `float()` coerce to the
right type. The default in the code matches the default in compose matches
the default in `.env.example` — three layers of consistency.

### Pattern: when NOT to use env vars

| Category | Examples | Why not |
|---|---|---|
| **Prompts** | System prompts, few-shot examples | Multi-line strings are awkward in `.env` |
| **Regex patterns** | `_THINK_RE`, `_BOXED_RE` | Implementation details |
| **SQL strings** | COPY statements, COUNT queries | Implementation details |
| **Schema definitions** | `COLSPECS`, column names | Tied to the data format |

The rule of thumb: if a user would reasonably want to tune it without
redeploying code, make it an env var. If it's an implementation detail tied
to the code structure, keep it hardcoded.

### Pattern: section-organized `.env.example`

Group variables by feature area with comments:

```bash
# ---------------------------------------------------------------------------
# Diagnose pipeline (medicoder/medical.py)
# ---------------------------------------------------------------------------
MAX_DIAGNOSES=10
DIAGNOSE_MAX_RETRIES=4
...

# ---------------------------------------------------------------------------
# Debate-critic loop (/diagnose endpoint)
# ---------------------------------------------------------------------------
MODEL_A=medgemma-27b-q4_k_s
...
```

This makes the file scannable — you can find what you need to tune without
reading every line.

> **Docs:** [Docker Compose — variable
> substitution](https://docs.docker.com/compose/environment-variables/),
> [Python os.environ](https://docs.python.org/3/library/os.html#os.environ)

**Files to study:**
- `.env.example` — all 25+ variables with comments
- `docker-compose.yml:84` — medicoder environment block
- `medicoder/medical.py:21` — env-driven constants
- `medicoder/proxy.py:14` — env-driven timeouts

---

## 19. Container Debugging with VSCode

### Concept

Debug Python code running inside a Docker container from VSCode on the
host. The source is bind-mounted so edits are live. [debugpy](https://github.com/microsoft/debugpy)
listens for attach inside the container.

### Pattern: decoupled start/attach

Because the app is a long-running server (uvicorn), starting the stack and
attaching the debugger are separate steps:

1. **Start** the stack with a debug overlay (installs debugpy, opens port 5678)
2. **Attach** VSCode to the running container (F5)
3. Set breakpoints, send requests, debug
4. **Detach** (Shift+F5) — the stack keeps running
5. **Re-attach** (F5) as often as needed

### Pattern: debug overlay

```yaml
# docker/docker-compose.debug.yml
services:
  medicoder:
    command: ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678",
              "--wait-for-client", "none", "main.py"]
    ports:
      - "5678:5678"     # debugpy attach port
    volumes:
      - ./medicoder:/app/medicore:ro   # live source mount
      - ./main.py:/app/main.py:ro
```

The overlay changes the command to wrap `main.py` with debugpy and mounts
the source code read-only for live editing.

### Pattern: VSCode launch config

```json
// .vscode/launch.json
{
    "name": "Python: Attach to medicoder container",
    "type": "debugpy",
    "request": "attach",
    "connect": { "host": "localhost", "port": 5678 },
    "pathMappings": [
        { "localRoot": "${workspaceFolder}", "remoteRoot": "/app" }
    ]
}
```

`pathMappings` maps host paths to container paths so VSCode can resolve
breakpoints set in your editor to the corresponding lines in the container.

### Pattern: breakpoint coverage caveat

The app starts serving immediately (no `--wait-for-client`). This means
module-level code and lifespan startup runs **before** you attach:
breakpoints there won't hit. Only request-path breakpoints hit, after you
send a request.

If you need to debug startup code, add `--wait-for-client` to the command
(this pauses execution until VSCode attaches):

```yaml
command: ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678",
          "--wait-for-client", "main.py"]
```

> **Docs:** [VSCode — attach to running
> container](https://code.visualstudio.com/docs/python/docker),
> [debugpy](https://github.com/microsoft/debugpy/wiki)

**Files to study:**
- `docker/docker-compose.debug.yml` — debug overlay
- `.vscode/launch.json` — attach configuration
- `.vscode/tasks.json` — debug-up/debug-down tasks

---

## Appendix: Technology Reference

| Technology | What it does | Docs |
|---|---|---|
| **Docker Compose** | Multi-container orchestration | [docs.docker.com/compose](https://docs.docker.com/compose/) |
| **FastAPI** | Async Python web framework | [fastapi.tiangolo.com](https://fastapi.tiangolo.com/) |
| **Pydantic** | Data validation + serialization | [docs.pydantic.dev](https://docs.pydantic.dev/latest/) |
| **psycopg 3** | PostgreSQL driver for Python | [psycopg.org/psycopg3](https://www.psycopg.org/psycopg3/docs/) |
| **pgvector** | Vector similarity search for Postgres | [github.com/pgvector/pgvector](https://github.com/pgvector/pgvector) |
| **LangGraph** | Stateful LLM workflow graphs | [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/) |
| **LiteLLM** | OpenAI-compatible LLM proxy | [docs.litellm.ai](https://docs.litellm.ai/) |
| **Ollama** | Local LLM runtime | [ollama.com](https://ollama.com/) |
| **uv** | Fast Python package manager | [github.com/astral-sh/uv](https://github.com/astral-sh/uv) |
| **debugpy** | Python debugger for VSCode | [github.com/microsoft/debugpy](https://github.com/microsoft/debugpy) |
| **pandas** | Data analysis (used for file parsing) | [pandas.pydata.org](https://pandas.pydata.org/) |

---

## Appendix: File-to-Technique Cross-Reference

| File | Primary techniques |
|---|---|
| `docker-compose.yml` | Multi-service orchestration, health checks, env-var chain, container isolation |
| `docker/docker-compose.gpu.yml` | GPU overlay, profile-gated services, shared network alias |
| `docker/docker-compose.debug.yml` | debugpy attach, bind-mount live editing |
| `docker/program/entrypoint.sh` | Idempotent loading before serve |
| `docker/program/pull_models.py` | Auto-pull sidecar, polling, idempotent pulls, defense-in-depth |
| `docker/postgres/00-schema.sql` | Generated columns, partial indexes, CHECK constraints |
| `docker/postgres/01-roles.sh` | Idempotent role creation (DO blocks) |
| `docker/postgres/02-litellm.sh` | `\gexec` conditional database creation |
| `docker/postgres/02-embedding.sql` | `halfvec`, HNSW index tuning |
| `docker/postgres/03-fts.sql` | Generated tsvector, GIN index (removed — see appendix below) |
| `docker/litellm/config.yaml` | Model aliasing, env-driven upstream, callback registration |
| `docker/litellm/log_callback.py` | Custom LiteLLM callback, async DB logging |
| `Dockerfile` | uv in Docker, layer caching, non-root user, unbuffered output |
| `main.py` | FastAPI lifespan, router wiring |
| `medicoder/proxy.py` | LLM client, thinking separation, env-driven timeouts |
| `medicoder/medical.py` | Diagnose pipeline, hybrid search, JSON parsing, retry strategy |
| `medicoder/critic.py` | Debate-critic loop, critic trigger logic, graph nodes |
| `medicoder/schemas.py` | Pydantic models, TypedDict for graph state |
| `medicoder/db/pool.py` | Connection pooling, dict_row, lifespan management |
| `medicoder/db/load_icd10.py` | Fixed-width parsing, binary COPY, idempotent loading |
| `medicoder/db/embed_icd10.py` | Batch embedding, skip-existing, text cleaning |
| `medicoder/routes/diagnose.py` | LangGraph StateGraph wiring (fan-out, fan-in, conditional edges) |
| `medicoder/routes/code.py` | Deterministic two-step pipeline |
| `medicoder/routes/icd10.py` | Direct search endpoint, code lookup |
| `evaluate.py` | Multi-level metrics, category-level evaluation |
| `replay_search.py` | Search-only isolation testing |
| `analyze_cases.py` | Case difficulty analysis across eval runs |
| `.env.example` | All tunable parameters with comments |
| `models.toml` | Model registry for auto-pull |

---

## Appendix: Full-Text Search — Tried and Retired

### What FTS is

PostgreSQL has built-in [full-text search](https://www.postgresql.org/docs/current/textsearch.html)
(FTS) — a lexical search engine that tokenises text into lexemes, indexes them
with a GIN index, and ranks matches with `ts_rank_cd`. Unlike vector similarity
(which captures *semantic* meaning), FTS matches *exact words and phrases*.

In this project, FTS was integrated alongside pgvector to create a hybrid
search. A generated `tsvector` column combined `short_desc` and `long_desc`:

```sql
search_tsv tsvector GENERATED ALWAYS AS (
    to_tsvector('english',
        coalesce(short_desc, '') || ' ' || coalesce(long_desc, ''))
) STORED;
```

### Why it was considered

The embedding model penalises codes containing the word "unspecified". For
example, C539 (*Malignant neoplasm of cervix uteri, unspecified*) scored only
0.48 vector similarity to the query *Malignant neoplasm of cervix uteri* —
below semantically related but clinically wrong codes like Z8541 (*Personal
history of malignant neoplasm of cervix uteri*).

FTS was expected to fix this: an exact-term match on "malignant", "neoplasm",
"cervix", and "uteri" would surface C539 regardless of its embedding penalty.

### What it was doing

On every search, the pipeline ran three DB queries per diagnosis:

1. **Vector search** — HNSW cosine-similarity, fetching `k × 5` candidates
2. **FTS AND search** — `plainto_tsquery` requiring all lexemes to match
3. **FTS OR fallback** — (only if AND returned nothing) any lexume may match

Results were merged with a scoring formula:

```
score = vec_sim + FTS_BOOST × normalized_ts_rank
```

Codes in both result sets got a boost; FTS-only codes were included if their
vector similarity exceeded a floor (`FTS_FLOOR = 0.50`). The infrastructure
totalled ~170 lines: `_Candidate` dataclass, `_fetch_fts_and()`,
`_fetch_fts_or()`, `_fetch_fts()`, and `_merge_and_score()`.

### What happened

Extensive replay testing (787 stored diagnoses across 8 eval files) measured
the impact of different `FTS_BOOST` values:

| `FTS_BOOST` | Improved | Regressed | Net |
|---|---|---|---|
| 0.0 (baseline) | — | — | — |
| 0.15 | 20 | 30 | -10 |
| 0.40 | 18 | 32 | -14 |

FTS pushed semantically similar but clinically wrong codes above correct ones.
For example, D508 (*Other iron deficiency anemia*) was boosted above D509
(*Iron deficiency anemia, unspecified*) because both contain "iron deficiency
anemia" — the FTS boost rewarded the lexical match without regard for which
code is the correct billable target.

The root cause was then addressed directly: **embedding text cleaning**
(`_clean_embed_text` in `embed_icd10.py`) strips the word "unspecified" from
the embedding input. C539's similarity rose from 0.48 to 0.99 without any FTS.
This made the hybrid search unnecessary.

### Why it was removed

With `FTS_BOOST` set to 0.0, FTS was fetching candidates that had near-zero
impact on final rankings (their raw `vec_sim` placed them below vector
candidates). The infrastructure added:

- **~170 lines** of merge/scoring code (`_Candidate`, `_fetch_fts_and`,
  `_fetch_fts_or`, `_fetch_fts`, `_merge_and_score`)
- **2 extra DB round-trips** per search query (FTS AND → FTS OR)
- **2 env vars** (`FTS_BOOST`, `FTS_FLOOR`) with associated config in
  `.env.example`, `docker-compose.yml`, and the README

Removing it simplified `search_icd10` from a 57-line multi-source merge to a
~20-line pure vector search, and reduced per-query DB round-trips from 3 to 1.

The agentic critic's own tools (`_tool_lookup`, `_tool_get`) can compensate
for any edge cases where exact-term browsing is needed — and they do so on
demand rather than on every search.

### What remains

The `search_tsv` generated column and GIN index (defined in `docker/postgres/03-fts.sql`,
now removed from the active schema) were part of the database during development.
No application code references `search_tsv` or any FTS function.
