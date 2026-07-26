# medicoder-technical

ICD-10-CM medical coding prototype. The stack runs as 2 or 3 containers:

| Service       | Image                              | Role                                                        |
| ------------- | ---------------------------------- | ---------------------------------------------------------- |
| `postgres`    | `pgvector/pgvector:pg16`           | Stores `icd10_codes`; `medicoder` (rw) + `agent` (ro) roles |
| `medicoder`   | built from `Dockerfile`            | App + idempotent ICD-10 loader                              |
| `llamacpp-*`  | `ghcr.io/ggml-org/llama.cpp:...`   | Local LLM (OpenAI-compatible API). Optional, profile-gated  |

`medicoder` always points its OpenAI-compatible client at `LLM_BASE_URL`. With a
profile active that is `http://llamacpp:8080/v1`; without one, set it to an
external endpoint (LiteLLM / OpenAI / an existing llama.cpp host).

## Prerequisites

- Docker + Compose v2.
- For the **AMD** path: the host `amdgpu`/ROCm kernel driver must be installed
  (`rocminfo` should list the GPU). The RX 7900 XTX is `gfx1100`. Docker only
  passes the devices through; it does not supply the driver.
- For the **NVIDIA** path: the [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit).

## Setup

1. Copy `.env.example` to `.env` and set the passwords.
2. Drop a GGUF model into `./models/` and set `LLM_MODEL_FILE` (e.g.
   `qwen2.5-7b-instruct-q4_k_m.gguf`). Leave empty for the external-LLM mode.

## Run

```bash
# 2 containers: program + postgres, LLM is external (LLM_BASE_URL in .env)
docker compose up --build

# 3 containers, AMD Radeon (RX 7900 XTX) local inference
docker compose --profile amd up --build

# 3 containers, NVIDIA fallback
docker compose --profile nvidia up --build
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

## Layout

```
Dockerfile                     program image
docker-compose.yml             3/2 service orchestration
docker/program/entrypoint.sh   runs loader, then the app command
docker/postgres/00-schema.sql  extension + icd10_codes table + indexes
docker/postgres/01-roles.sh    medicoder (rw) + agent (ro) roles
docker/postgres/02-embedding.sql.example  optional pgvector column/index
medicoder/db/load_icd10.py     fixed-width -> COPY loader
.env.example                   all configuration
models/                        bind-mounted GGUF models (gitignored)
```
