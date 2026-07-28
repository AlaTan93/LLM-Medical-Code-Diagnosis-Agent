# medicoder-technical

ICD-10-CM medical coding prototype. The default stack runs as 3 containers
(an optional 4th, in-container Ollama, is added under a GPU profile — see
[Local GPU LLM](#optional-local-gpu-llm-in-container-ollama)):

| Service    | Image                                 | Role                                                                  |
| ---------- | ------------------------------------- | -------------------------------------------------------------------- |
| `postgres` | `pgvector/pgvector:pg16`              | Stores `icd10_codes`; `medicoder` (rw) + `agent` (ro) roles; LiteLLM audit DB |
| `medicoder`| built from `Dockerfile`               | App + idempotent ICD-10 loader + `/test`, `/agent`, `/code` endpoints  |
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

LiteLLM routes model aliases your app calls to upstream LLMs. Alias `A` is
configured by three `.env` variables on the `litellm` container; the medical,
orchestrator, and embedding aliases are hardcoded to the in-container Ollama
(see `docker/litellm/config.yaml`):

| Alias | Model | Notes |
| ----- | ----- | ----- |
| `A` | `${LLM_UPSTREAM_MODEL}` | Env-driven; any OpenAI-compatible endpoint |
| `orchestrator` | `qwen2.5:7b` | Generalist tool-caller (reserved for future agent features) |
| `ii-medical-q8` | `II-Medical-8B-1706-GGUF:Q8_0` | Medical diagnosis generation |
| `deepseek-r1-medical-cot` | `DeepSeek-R1-Medical-COT:Q4_K_M` | Medical (thinking model) |
| `qwen35-medical` | `qwen35-9b-medical:Q4_K_M` | Medical |
| `embed` | `bge-m3` | 1024-dim embeddings for ICD-10 vector search |

Alias `A` is configured via `.env`:

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

**LLM call log:** a custom LiteLLM callback (`docker/litellm/log_callback.py`,
wired via `litellm_settings.callbacks`) writes every call to the `llm_call_log`
table in the `litellm` database — the prompt (`messages`), any reasoning
(`thinking`, i.e. `reasoning_content`/`<think>`), the reply (`output`), any
`tool_calls`, and token usage (`prompt_tokens`/`completion_tokens`/`total_tokens`).
(LiteLLM's built-in `LiteLLM_SpendLogs` only tracks tokens/cost, which is `$0`
for the local Ollama models, so it's not useful for inspecting prompts.) Recent
calls:

```bash
docker compose exec postgres psql -U postgres -d litellm \
  -c 'SELECT id, model, call_type, prompt_tokens, completion_tokens, substring(thinking,1,30) AS thinking, (tool_calls IS NOT NULL) AS tools, latency_ms FROM llm_call_log ORDER BY id DESC LIMIT 10;'
```

Inspect the prompt / thinking / output / tool calls of the most recent call:

```bash
docker compose exec postgres psql -U postgres -d litellm \
  -c 'SELECT model, substring(messages::text,1,200) AS prompt, substring(thinking,1,150) AS thinking, substring(output,1,150) AS output, substring(tool_calls::text,1,150) AS tools FROM llm_call_log ORDER BY id DESC LIMIT 1;'
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

Models listed in `models.toml` **auto-pull** when the stack comes up: an
`ollama-init` sidecar (profile-gated, stdlib-only) polls `http://ollama:11434`
until the GPU Ollama is ready, then `POST /api/pull`s each entry, skipping any
already present. It's fire-and-forget — nothing depends on it, so models arrive
in parallel with the app.

The container Ollama is **fully isolated**: it publishes no host port, so it
never clashes with an Ollama you run on the host (e.g. a native one on 11434).
It's reachable only inside the compose network as `http://ollama:11434`. The two
GPU services share that alias, so down one profile before up-ing the other.

## Pre-pull models (automatic)

Models for the in-container Ollama are pulled automatically by the `ollama-init`
sidecar when the GPU stack comes up — no manual step. Edit **`models.toml`** to
list the Ollama-registry tags you want available:

```toml
[[model]]
name = "hf.co/rwibawa/DeepSeek-R1-Medical-COT:Q4_K_M"

[[model]]
name = "hf.co/Intelligent-Internet/II-Medical-8B-1706-GGUF:Q8_0"

[[model]]
name = "hf.co/qaootkcx/qwen35-9b-medical:Q4_K_M"

[[model]]
name = "bge-m3"

[[model]]
name = "qwen2.5:7b"
```

Each entry is a registry tag (also the local Ollama name). Re-runs are
idempotent: anything already in `/api/tags` is skipped, so the sidecar only
pulls what's missing. For a one-off pull outside the sidecar:

```bash
docker compose -f docker-compose.yml -f docker/docker-compose.gpu.yml \
   --profile gpu-amd exec ollama-amd ollama pull <model>
```

## Testing models (POST /test/{model})

LiteLLM has **no host port** (in-network only), so the models can't be reached
directly from the host. The app exposes a temporary testing endpoint that calls
LiteLLM internally. `model` is a LiteLLM alias from `docker/litellm/config.yaml`
(e.g. `ii-medical-q8`, `deepseek-r1-medical-cot`, `qwen35-medical`, `A`):

```bash
# default medical prompt
curl -X POST http://localhost:8000/test/qwen35-medical

# custom prompt
curl -X POST http://localhost:8000/test/ii-medical-q8 \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"What is the ICD-10-CM code for essential hypertension?"}'
```

Returns `{"model","prompt","response","elapsed_s"}`. Unknown alias → `404`;
LiteLLM unreachable → `502`; model-load timeout → `504`. (First call per model
loads it into VRAM, ~10–60s.)

## Agentic LLM calls (POST /agent/{model})

Like `/test/{model}`, but runs the prompt through a langgraph
`create_react_agent` so the model can call tools. The model is reached via
LangChain's `openai:` provider (`ChatOpenAI` with `use_responses_api=False`),
which reads `OPENAI_BASE_URL` / `OPENAI_API_KEY` — both set automatically on the
`medicoder` container (derived from `LLM_BASE_URL` + a dummy key; LiteLLM
enforces no key).

Two test tools are wired up:

- **`echo(text)`** — echoes its argument back verbatim.
- **`get_flag(text)`** — returns a secret flag only when called with `"hello"`.
  This is an anti-confabulation probe: the flag can only be obtained by actually
  calling the tool, so if it appears in the response the model genuinely
  tool-called (rather than hallucinating).

The response includes a `tool_results` array — one entry per tool call the model
made, recording the tool name, the arguments the model supplied, and the value
the tool returned. This is the **ground-truth** tool output, independent of the
model's own textual summary (which may paraphrase, truncate, or omit results).

> Use a **tool-calling-capable** model — e.g. `ii-medical-q8` or
> `deepseek-r1-medical-cot`. Note that only `ii-medical-q8` and
> `deepseek-r1-medical-cot` are known to emit structured tool calls through the
> local Ollama; `qwen35-medical` ignores tools despite declaring the capability.

```bash
# default prompt (exercises echo)
curl -X POST http://localhost:8000/agent/ii-medical-q8

# custom prompt
curl -X POST http://localhost:8000/agent/ii-medical-q8 \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Call the get_flag tool with hello and tell me the flag"}'
```

Returns `{"model","prompt","response","elapsed_s","tool_results"}`. The agent
may make several LLM round-trips, so cold-start calls can be slower than
`/test`.

## Coding pipeline (POST /code)

The main endpoint: a deterministic two-step pipeline that produces billable
ICD-10-CM codes from clinical text. No LLM orchestrator is needed — the steps
always run in fixed order, which is more reliable than asking a generalist
model to chain tool calls.

1. **Diagnose** — the clinical text is forwarded to a medical model (default
   `ii-medical-q8`) via a plain chat completion (no tools required). The model
   returns a one-sentence diagnosis. Thinking blocks (`<think>…</think>` and
   orphaned `</think>` tags) are stripped automatically.
2. **Search** — the diagnosis is embedded with bge-m3 and a pgvector
   cosine-similarity search returns the top-k billable ICD-10 codes. The HNSW
   index uses `m=32, ef_construction=128` and the query sets
   `hnsw.ef_search=200` for high recall over 74k billable codes.

The medical models are used only as text generators — they never need to call
tools.

```bash
curl -X POST http://localhost:8000/code \
  -H 'Content-Type: application/json' \
  -d '{"text":"Patient has type 2 diabetes mellitus without complications"}'
```

Optional `medical_model` field overrides the diagnosis model (default
`ii-medical-q8`):

```bash
curl -X POST http://localhost:8000/code \
  -H 'Content-Type: application/json' \
  -d '{"text":"...","medical_model":"deepseek-r1-medical-cot"}'
```

Returns `{"diagnosis","codes","tool_results","elapsed_s"}` where `codes` is the
ranked list of billable ICD-10-CM matches (`code`, `short_desc`, `long_desc`,
`similarity`), and `tool_results` records each step's input and output.

## Data loading

### ICD-10 codes

On every boot `medicoder` runs `medicoder.db.load_icd10`, which parses the
fixed-width `icd10cm_order_2026.txt` and bulk-copies it into `icd10_codes`. It
skips when the table already has rows; force a reload with `LOAD_FORCE=1`:

```bash
docker compose exec medicoder sh -c 'LOAD_FORCE=1 python -m medicoder.db.load_icd10'
```

### Vector embeddings

The `vector` extension and a 1024-dim `embedding` column + HNSW index are
created by `docker/postgres/02-embedding.sql` (runs automatically on a fresh
data volume; for an existing volume, apply manually as the `postgres`
superuser).

To populate embeddings (bge-m3 via the `embed` LiteLLM alias), run the
idempotent bulk embedder — it fills in every code that lacks an embedding
(billable and non-billable), skipping rows already done:

```bash
docker compose exec medicoder python -m medicoder.db.embed_icd10
```

All 98,186 codes are embedded (74,719 billable + 23,467 non-billable). The
non-billable codes are pre-embedded so they're search-ready if CMS reclassifies
them. The `/code` search query filters to `is_billable` at query time.

## Roles / the scoped SQL tool

`docker/postgres/01-roles.sh` creates two roles:

- `medicoder` — owns `icd10_codes` (read/write).
- `agent` — `SELECT`-only, with `default_transaction_read_only = on`. Connect the
  LangChain read-only SQL tool as this role (see `human_plan.txt`, Option 1).

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
Dockerfile                       program image
docker-compose.yml               3-service orchestration (postgres + medicoder + litellm)
docker/docker-compose.debug.yml  VSCode debugpy attach override (live source mounts)
docker/docker-compose.gpu.yml    optional in-container GPU Ollama (AMD/ROCm + NVIDIA/CUDA profiles)
docker/program/entrypoint.sh     runs loader, then the app command
docker/program/debug.sh          debugpy entrypoint used by the debug overlay
docker/program/pull_models.py    ollama-init sidecar: auto-pull models.toml on stack up
docker/postgres/00-schema.sql    extension + icd10_codes table + indexes
docker/postgres/01-roles.sh      medicoder (rw) + agent (ro) roles
docker/postgres/02-litellm.sh    litellm role + audit database
docker/postgres/02-embedding.sql pgvector embedding column + HNSW index (1024-dim)
docker/litellm/config.yaml       LiteLLM alias -> upstream routing + DB logging
docker/litellm/log_callback.py   custom callback -> llm_call_log (prompts/thinking/output/tools)
medicoder/proxy.py               shared LiteLLM client: chat_completion(), embed(), strip_thinking()
medicoder/messages.py            shared message utils: extract_tool_results(), last_content()
medicoder/schemas.py             Pydantic models (ICD10Code, TestRequest/Response, CodeRequest/Response, ToolResult)
medicoder/db/connect.py          shared CLI Postgres connection (retried)
medicoder/db/load_icd10.py       fixed-width -> COPY loader
medicoder/db/embed_icd10.py      idempotent bulk embedder (bge-m3 via LiteLLM)
medicoder/db/pool.py             psycopg connection pool (lifespan-managed)
medicoder/routes/icd10.py        /codes endpoints
medicoder/routes/llm.py          POST /test/{model} — call a LiteLLM alias
medicoder/routes/agent.py        POST /agent/{model} — langgraph ReAct agent + echo/get_flag tools
medicoder/routes/code.py         POST /code — deterministic pipeline (diagnose + search_icd10)
main.py                          FastAPI app + lifespan + router wiring
models.toml                      registry models for the ollama-init sidecar
.vscode/{launch,tasks,extensions}.json  VSCode container debugging
.env.example                     all configuration
```
