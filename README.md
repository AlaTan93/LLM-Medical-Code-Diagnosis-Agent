# medicoder-technical

ICD-10-CM medical coding prototype. The stack runs as 2 or 4 containers:

| Service       | Image                              | Role                                                        |
| ------------- | ---------------------------------- | ---------------------------------------------------------- |
| `postgres`    | `pgvector/pgvector:pg16`           | Stores `icd10_codes`; `medicoder` (rw) + `agent` (ro) roles; LiteLLM audit DB |
| `medicoder`   | built from `Dockerfile`            | App + idempotent ICD-10 loader                              |
| `ollama-*`    | `ollama/ollama:rocm` / `ollama/ollama` | Local LLM inference (auto VRAM-managed). Optional, profile-gated |
| `litellm-*`   | `ghcr.io/berriai/litellm:main-stable` | Model-alias router + Postgres call/response audit log. Optional, profile-gated |

`medicoder` always points its OpenAI-compatible client at `LLM_BASE_URL`. With a
profile active that is `http://litellm:4000/v1` (LiteLLM routes to Ollama);
without one, set it to an external endpoint (OpenAI / an existing host).

## Prerequisites

- Docker + Compose v2.
- For the **AMD** path: the host `amdgpu`/ROCm kernel driver must be installed
  (`rocminfo` should list the GPU). The RX 7900 XTX is `gfx1100`. Docker only
  passes the devices through; it does not supply the driver.
- For the **NVIDIA** path: the [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit).

## Setup

1. Copy `.env.example` to `.env` and set the passwords.
2. For local LLM runs, edit `docker/litellm/config.yaml` to map model aliases
   (A, B, C ...) to Ollama tags, then pull the tags (see *Model management*
   below). Skip for external-LLM mode.

## Run

```bash
# 2 containers: program + postgres, LLM is external (LLM_BASE_URL in .env)
docker compose up --build

# 4 containers, AMD Radeon (RX 7900 XTX) local inference
docker compose --profile amd up --build

# 4 containers, NVIDIA fallback
docker compose --profile nvidia up --build
```

## Model management (Ollama + LiteLLM)

Ollama auto-manages VRAM: `OLLAMA_MAX_LOADED_MODELS=1` keeps a single model
resident; `OLLAMA_KEEP_ALIVE=5m` unloads it 5 min after the last request.

LiteLLM sits between the app and Ollama. The app calls a model alias (e.g.
`A`); `docker/litellm/config.yaml` maps that alias to an Ollama tag.

**Pull a model once** (stored in the `ollama-models` volume):

```bash
docker compose --profile amd exec ollama ollama pull llama3.2
```

**Audit log:** every LLM call and its response are written to the `LiteLLM_SpendLogs`
table in the `litellm` database. Query it:

```bash
docker compose exec postgres psql -U postgres -d litellm \
  -c "SELECT request_id, model, prompt_tokens, completion_tokens, startTime, endTime FROM \"LiteLLM_SpendLogs\" ORDER BY startTime DESC LIMIT 10;"
```

## What the loader does

On every boot `medicoder` runs `medicoder.db.load_icd10`, which parses the
fixed-width `icd10cm_order_2026.txt` and bulk-copies it into `icd10_codes`. It
skips when the table already has rows; force a reload with `LOAD_FORCE=1`:

```bash
docker compose exec medicoder sh -c 'LOAD_FORCE=1 python -m medicoder.db.load_icd10'
```

## Roles / the scoped SQL tool

`docker/postgres/01-roles.sh` creates two roles:

- `medicoder` — owns `icd10_codes` (read/write).
- `agent` — `SELECT`-only, with `default_transaction_read_only = on`. Connect the
  LangChain read-only SQL tool as this role (see `human_plan.txt`, Option 1).

## Optional: vector similarity (Option 2)

The `vector` extension is installed automatically. When you pick an embedding
model, add the column + index (edit the dimension first):

```bash
psql -h localhost -U medicoder -d medicoder \
     -f docker/postgres/02-embedding.sql.example
```

## Debugging (VSCode, in-container)

Debug the `medicoder` app while it runs in the container. The source is
bind-mounted, so host edits are picked up with **no rebuild**.

**One-time:** install the **Python Debugger** extension (`ms-python.debugpy`;
VSCode offers this via `.vscode/extensions.json`). The Python extension
(`ms-python.python`) is also recommended.

**Debug:**

1. Copy `.env.example` to `.env` (the debug task reads it).
2. Set a breakpoint in `main.py` or anywhere under `medicoder/`.
3. Run & Debug -> **Python: Attach to medicoder container** (F5).

F5 runs the `debug-up` task, which starts the `medicoder` + `postgres` stack
under `docker/docker-compose.debug.yml` (the app is launched under `debugpy`
with `--wait-for-client`, so it pauses until VSCode attaches), waits for the
debug port, then attaches. Stop with the **debug-down** task.

Equivalent shell commands:

```bash
docker compose --env-file .env -f docker-compose.yml \
               -f docker/docker-compose.debug.yml up medicoder   # then F5
docker compose --env-file .env -f docker-compose.yml \
               -f docker/docker-compose.debug.yml down           # stop
```

Notes:
- Breakpoints in `medicoder/db/load_icd10.py` won't hit — the loader runs
  before `debugpy` attaches. To debug it, change the `command` in
  `docker/docker-compose.debug.yml` to run debugpy against the module:
  `python -m debugpy ... -m medicoder.db.load_icd10`.
- Dependency changes still need `uv sync` (the debug command runs it) or a
  rebuild: `docker compose build medicoder`.
- Debug runs as the non-root `app` user, matching production.

## Layout

```
Dockerfile                     program image
docker-compose.yml             4/2 service orchestration
docker/docker-compose.debug.yml  VSCode debugpy attach override
docker/program/entrypoint.sh   runs loader, then the app command
docker/postgres/00-schema.sql  extension + icd10_codes table + indexes
docker/postgres/01-roles.sh    medicoder (rw) + agent (ro) roles
docker/postgres/02-litellm.sh  litellm role + audit database
docker/postgres/02-embedding.sql.example  optional pgvector column/index
docker/litellm/config.yaml     LiteLLM model-alias routing + DB logging
medicoder/db/load_icd10.py     fixed-width -> COPY loader
.vscode/{launch,tasks,extensions}.json  VSCode container debugging
.env.example                   all configuration
models/                        bind-mounted model files (gitignored)
```
