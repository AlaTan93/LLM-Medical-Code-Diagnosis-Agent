# medicoder-technical

ICD-10-CM medical coding prototype. The stack runs as 3 containers:

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
docker-compose.yml             3-service orchestration (postgres + medicoder + litellm)
docker/docker-compose.debug.yml  VSCode debugpy attach override
docker/program/entrypoint.sh   runs loader, then the app command
docker/postgres/00-schema.sql  extension + icd10_codes table + indexes
docker/postgres/01-roles.sh    medicoder (rw) + agent (ro) roles
docker/postgres/02-litellm.sh  litellm role + audit database
docker/postgres/02-embedding.sql.example  optional pgvector column/index
docker/litellm/config.yaml     LiteLLM alias -> upstream routing + DB logging
medicoder/db/load_icd10.py     fixed-width -> COPY loader
.vscode/{launch,tasks,extensions}.json  VSCode container debugging
.env.example                   all configuration
```
