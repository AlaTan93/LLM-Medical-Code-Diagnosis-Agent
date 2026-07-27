# medicoder-technical

ICD-10-CM medical coding prototype. The default stack runs as 3 containers
(an optional 4th, in-container Ollama, is added under a GPU profile — see
[Local GPU LLM](#optional-local-gpu-llm-in-container-ollama)):

| Service    | Image                                 | Role                                                                  |
| ---------- | ------------------------------------- | -------------------------------------------------------------------- |
| `postgres` | `pgvector/pgvector:pg16`              | Stores `icd10_codes`; `medicoder` (rw) + `agent` (ro) roles; LiteLLM audit DB |
| `medicoder`| built from `Dockerfile`               | App + idempotent ICD-10 loader                                       |
| `litellm`  | `ghcr.io/berriai/litellm:main-stable` | Local proxy: routes the model alias to your upstream LLM + audits every call/response to Postgres |

`medicoder` always points its OpenAI-compatible client at the local LiteLLM
proxy (`LLM_BASE_URL=http://litellm:4000/v1`). LiteLLM forwards the model alias
(e.g. `A`) to the upstream LLM you configure in `.env` — any OpenAI-compatible
endpoint works: a cloud provider (OpenAI, Anthropic, Azure), a self-deployed
server (vLLM, TGI), or an Ollama you run on the host.

## Prerequisites

- Docker + Compose v2.
- An upstream LLM endpoint + API key (cloud or self-deployed); see `.env`.
- `uv` (Astral) — only needed for the host-side pre-pull script
  (`uv run --group bootstrap ...`); see [Pre-pull models](#pre-pull-models-one-time-bootstrap).

## Setup

1. Copy `.env.example` to `.env` and set the passwords.
2. Point LiteLLM at your LLM by setting `LLM_UPSTREAM_MODEL` / `_API_BASE` /
   `_API_KEY` in `.env` (see *Configure the upstream LLM* below).

## Run

```bash
# 3 containers: postgres + medicoder + litellm (routes to your LLM via .env)
docker compose up --build
```

## Configure the upstream LLM

LiteLLM routes the alias your app calls (`LLM_MODEL`, default `A`) to the
upstream LLM configured by three `.env` variables on the `litellm` container:

| Variable                | Example                       | Notes                                                |
| ----------------------- | ----------------------------- | ---------------------------------------------------- |
| `LLM_UPSTREAM_MODEL`    | `openai/gpt-4o-mini`          | LiteLLM provider prefix required for cloud providers |
| `LLM_UPSTREAM_API_BASE` | `https://api.openai.com/v1`   | Any OpenAI-compatible base URL                       |
| `LLM_UPSTREAM_API_KEY`  | `sk-...`                      | The upstream's key (dummy for a local Ollama)        |

Common choices:

- **Cloud** — `openai/gpt-4o-mini` + `https://api.openai.com/v1`;
  `anthropic/claude-3-5-sonnet` + `https://api.anthropic.com/v1`.
- **Self-deployed** — `openai/<model>` + your server's URL (vLLM, TGI, ...).
- **Ollama on the host** — `ollama/llama3.2` + `http://host.docker.internal:11434`
  (`host-gateway` is wired into the `litellm` service).

Add more aliases (B, C, ...) in `docker/litellm/config.yaml` to rotate models.

**Heads-up — GPU overlay overrides the upstream.** Running a `gpu-amd` /
`gpu-nvidia` profile loads `docker/docker-compose.gpu.yml`, which sets
`LLM_UPSTREAM_MODEL` / `_API_BASE` / `_API_KEY` on the `litellm` container to
point at the in-container Ollama. Your `.env` upstream values are **ignored**
in that mode; they apply only in the default stack (and the `debug-up-cloud`
task).

**Audit log:** every LLM call and its response are written to the `LiteLLM_SpendLogs`
table in the `litellm` database. Query it:

```bash
docker compose exec postgres psql -U postgres -d litellm \
  -c "SELECT request_id, model, prompt_tokens, completion_tokens, startTime, endTime FROM \"LiteLLM_SpendLogs\" ORDER BY startTime DESC LIMIT 10;"
```

## Optional: local GPU LLM (in-container Ollama)

Instead of a cloud/host endpoint, run Ollama in the stack on a GPU. The overlay
`docker/docker-compose.gpu.yml` adds two profile-gated Ollama services (each
answering to the network alias `ollama`) and repoints LiteLLM at it. The default
stack stays env-only; the overlay is opt-in.

| Mode | Command |
| ---- | ------- |
| AMD (ROCm) | `docker compose -f docker-compose.yml -f docker/docker-compose.gpu.yml --profile gpu-amd up` |
| NVIDIA (CUDA) | `docker compose -f docker-compose.yml -f docker/docker-compose.gpu.yml --profile gpu-nvidia up` |

Models listed as `kind = "ollama"` in `models.toml` **auto-pull** when the stack
comes up: an `ollama-init` sidecar (profile-gated, stdlib-only) polls
`http://ollama:11434` until the GPU Ollama is ready, then `POST /api/pull`s each
entry, skipping any already present. It's fire-and-forget — nothing depends on
it, so models arrive in parallel with the app. (`kind = "hf"` GGUF imports are
not handled by the sidecar; use `pull_models_before_build.py` for those.)

For a one-off manual pull (into the shared `ollama-models` volume), match the
service name to the profile:

```bash
docker compose -f docker-compose.yml -f docker/docker-compose.gpu.yml \
   --profile gpu-amd exec ollama-amd ollama pull llama3.2     # ollama-nvidia / --profile gpu-nvidia on CUDA
```

The served model is `OLLAMA_MODEL` (default `llama3.2`; see `.env.example`).
LiteLLM's alias then resolves to `ollama/${OLLAMA_MODEL}` at `http://ollama:11434`.
The two GPU services are mutually exclusive (host port `11434` + shared alias),
so down one profile before up-ing the other.

## Pre-pull models (one-time bootstrap)

`pull_models_before_build.py` populates the `ollama-models` volume with a
configurable list **before** the app serves — useful when you want specific HF
GGUF quants (not just registry tags) ready at first request. Run it on the host
after the GPU Ollama container is up:

```bash
docker compose --env-file .env -f docker-compose.yml \
   -f docker/docker-compose.gpu.yml --profile gpu-amd up -d ollama-amd
uv run --group bootstrap python pull_models_before_build.py   # --dry-run to preview
```

The list lives in `models.toml`. Two entry kinds:

```toml
[[model]]
kind = "ollama"            # registry pull (POST /api/pull)
name = "llama3.2"

[[model]]
kind = "hf"               # HuggingFace GGUF -> /api/blobs -> /api/create
repo = "bartowski/Llama-3.2-3B-Instruct-GGUF"
file = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
name = "llama3.2-3b-q4"   # local Ollama name after import
```

Re-runs are idempotent (existing models are skipped; HF downloads hit the local
`~/.cache/huggingface/hub` cache). `OLLAMA_BASE_URL` and `HF_TOKEN` are read
from `.env` (no need to pass them on the CLI); `HF_TOKEN` is only required for
gated repos. HF entries download **flat into `models/`** (set `MODELS_DIR` to
relocate; `*.gguf`/`*.bin` are gitignored); Ollama ingests its own copy into the
`ollama-models` volume. The `bootstrap` dep group is host-only — it is never
installed into the production image (the Dockerfile's `uv sync --no-dev` skips
named groups).

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

**Decoupled model.** Because the app is a long-running FastAPI/uvicorn server,
starting the stack and attaching the debugger are two separate steps:

1. Copy `.env.example` to `.env` (the debug task reads it).
2. **Start the stack** by running one of these tasks (Run Task, or the terminal
   panel's task runner). The task stays attached and streams logs:

   | Task | LLM backend |
   | ---- | ---------- |
   | `debug-up-cloud` | `LLM_UPSTREAM_*` in `.env` (cloud / host Ollama) |
   | `debug-up-amd` | in-container AMD/ROCm Ollama (`ollama/ollama:rocm`) |
   | `debug-up-nvidia` | in-container NVIDIA/CUDA Ollama (`ollama/ollama:latest`) |

3. Wait for the **`DEBUGPY_READY`** line in that task's terminal (debugpy is
   listening on `:5678`). The server is already serving.
4. Set a breakpoint in `main.py` or under `medicoder/`, then **F5** →
   *Python: Attach to medicoder container*. F5 only attaches — it does not start
   or stop the stack.
5. Trigger the breakpoint with a request, e.g. `curl localhost:8000/health`.
6. Detach with Shift+F5 and re-attach with F5 as often as you like — the stack
   keeps running. Stop it with the matching **`debug-down-{cloud,amd,nvidia}`**
   task when done.

Equivalent shell commands (cloud mode):

```bash
docker compose --env-file .env -f docker-compose.yml \
               -f docker/docker-compose.debug.yml up        # then F5
docker compose --env-file .env -f docker-compose.yml \
               -f docker/docker-compose.debug.yml down      # stop
```

Notes:
- **Breakpoint coverage.** The app starts serving immediately (no
  `--wait-for-client`), so module-level and `lifespan` startup code (e.g.
  `app = FastAPI(...)`, the DB-pool setup in `main.py`) runs *before* you attach
  — breakpoints there won't hit. Only request-path / handler breakpoints hit,
  after you send a request. That's the tradeoff of the decoupled model.
- Breakpoints in `medicoder/db/load_icd10.py` won't hit — the loader runs in the
  ENTRYPOINT before debugpy starts. To debug it, change the `command` in
  `docker/docker-compose.debug.yml` to run debugpy against the module:
  `python -m debugpy ... -m medicoder.db.load_icd10`.
- Dependency changes still need `uv sync` (the debug command runs it) or a
  rebuild: `docker compose build medicoder`.
- Debug runs as the non-root `app` user, matching production.

## Layout

```
Dockerfile                     program image
docker-compose.yml             3-service orchestration (postgres + medicoder + litellm)
docker/docker-compose.debug.yml  VSCode debugpy attach override (live source mounts)
docker/docker-compose.gpu.yml  optional in-container GPU Ollama (AMD/ROCm + NVIDIA/CUDA profiles)
docker/program/entrypoint.sh   runs loader, then the app command
docker/program/debug.sh        debugpy entrypoint used by the debug overlay
docker/program/pull_models.py  ollama-init sidecar: auto-pull models.toml on stack up
docker/postgres/00-schema.sql  extension + icd10_codes table + indexes
docker/postgres/01-roles.sh    medicoder (rw) + agent (ro) roles
docker/postgres/02-litellm.sh  litellm role + audit database
docker/postgres/02-embedding.sql.example  optional pgvector column/index
docker/litellm/config.yaml     LiteLLM alias -> upstream routing + DB logging
medicoder/db/load_icd10.py     fixed-width -> COPY loader
pull_models_before_build.py    host-side Ollama model pre-pull (registry + HF GGUF)
models.toml                    model list for pull_models_before_build.py
.vscode/{launch,tasks,extensions}.json  VSCode container debugging
.env.example                   all configuration
```
