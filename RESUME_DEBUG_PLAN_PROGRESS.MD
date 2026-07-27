# Resume: one-click debugger + AMD/NVIDIA GPU Ollama profiles

## Goal (two independent parts)
- **Part A** — make the VSCode F5 debugger attach in one click (symptom was "task completes, but nothing happens").
- **Part B** — restore optional in-container GPU Ollama (AMD/ROCm + NVIDIA/CUDA) as compose profiles; default stack stays env-only.

---

## Part A — STATUS: IMPLEMENTED + MOSTLY VERIFIED (one client test pending)

### Three root-cause bugs found and fixed
1. **Foreground preLaunchTask handoff** — `tasks.json` `debug-up` was a *foreground* task; VSCode's exit→attach handoff is the fragile link. → Rewrote as the canonical **background task + `problemMatcher`** pattern (`isBackground: true`, blocking `up medicoder` so logs stream into the task's own stdout; `endsPattern: "DEBUGPY_READY"`). File: `.vscode/tasks.json` (done).
2. **Marker leak** — the entrypoint's `echo "[medicoder] starting: $*"` dumped the *entire* inline `sh -c` debug script into the logs, including the literal `DEBUGPY_READY`, so the matcher false-matched the source before debugpy was listening. → Moved the debug logic into a script file `docker/program/debug.sh` (mounted at `/debug.sh`); command is now `sh /debug.sh`, so the entrypoint only echoes `sh /debug.sh` (no leak). Files: `docker/program/debug.sh` (new), `docker/docker-compose.debug.yml` (rewritten).
3. **Wrong hex port** — the `/proc/net/tcp` watcher searched for `:1634` (= **5684**), not `:162E` (= **5678**). 5678 = 0x162E. This is why DEBUGPY_TIMEOUT always fired. → Fixed in `docker/program/debug.sh` (`:1634` → `:162E`).

### Verified
- `DEBUGPY_READY` now fires cleanly: ~3s post-start, single emission, no leak, no timeout. (`docker compose ... up -d medicoder` then `docker logs ... | grep DEBUGPY_READY`.)
- DAP attach works: `initialize`→`initialized`, `attach`, `configurationDone` all get responses; `main.py` runs after attach (process/thread events seen). (Raw DAP client at `/tmp/dap_test.py`.)
- Ports merge correctly across compose files: debug container exposes both `8000:8000` (from base) and `5678:5678` (from override) — no edit needed.

### NOT yet verified — the breakpoint-hit DAP test (test-harness timing bug, not a server bug)
The properly-sequenced DAP client (`/tmp/dap_test.py`) got `ConnectionResetError` because after `docker restart`, `docker logs | grep DEBUGPY_READY` matched the **stale** marker from the previous run (logs accumulate), so the client connected before the fresh debugpy was listening.
**To re-verify cleanly:**
```bash
docker compose --env-file .env -f docker-compose.yml -f docker/docker-compose.debug.yml down
docker compose --env-file .env -f docker-compose.yml -f docker/docker-compose.debug.yml up -d medicoder
# wait for a FRESH marker (use --since to ignore old logs):
for i in $(seq 1 60); do docker logs --since 2m medicoder_technical-medicoder-1 2>&1 | grep -aq DEBUGPY_READY && { echo ready; break; }; sleep 1; done
python3 /tmp/dap_test.py   # expect: breakpoint verified: [True]; PASS: breakpoint hit
```
If the breakpoint at `/app/main.py:58` (`app = FastAPI(...)`) is hit → fully verified end-to-end.

### Part A remaining steps
1. Re-run the DAP breakpoint test above (clean timing). (Server side already proven; this just confirms the breakpoint binds.)
2. Update **`README.md`** debug section — it's STALE ("waits for the debug port"). Reflect: background task streams output; matcher fires on `DEBUGPY_READY`; `main.py` is now a uvicorn server so **handler breakpoints need an HTTP request** (`curl localhost:8000/health`), module-level breakpoints hit on first run after attach.
3. **User does the final F5 one-click confirmation in VSCode** (I can't press F5 myself). Set a breakpoint in `main.py`, press F5, confirm it attaches and the breakpoint hits.

### Part A files (current state)
| File | State |
|---|---|
| `.vscode/tasks.json` | DONE — background task + problemMatcher |
| `.vscode/launch.json` | UNCHANGED (correct: preLaunchTask/postDebugTask/connect 127.0.0.1:5678/pathMappings) |
| `docker/docker-compose.debug.yml` | DONE — mounts `debug.sh`, command `["sh","/debug.sh"]` |
| `docker/program/debug.sh` | DONE (new) — hex fixed to `0x162E` |
| `README.md` | PENDING (stale debug section) |

---

## Part B — STATUS: NOT STARTED (plan locked, details confirmed)

### Design
Single overlay **`docker/docker-compose.gpu.yml`** (default stays env-only; loaded only for GPU mode). Two profile-gated Ollama services sharing network alias `ollama`; a `litellm` env override (applies whenever the file is loaded) repoints LiteLLM at the local Ollama.

- `ollama-amd` (profile `gpu-amd`): image `ollama/ollama:rocm` (CONFIRMED valid tag on Docker Hub), devices `/dev/kfd` + `/dev/dri`, alias `ollama`, volume `ollama-models`, port `11434`.
- `ollama-nvidia` (profile `gpu-nvidia`): image `ollama/ollama:latest`, `deploy.resources.reservations.devices` (`driver: nvidia, count: all, capabilities: [gpu]`), alias `ollama`, volume `ollama-models`, port `11434`.
- `litellm` env override: `LLM_UPSTREAM_MODEL=ollama/${OLLAMA_MODEL:-llama3.2}`, `LLM_UPSTREAM_API_BASE=http://ollama:11434`, `LLM_UPSTREAM_API_KEY=dummy`.
- top-level `volumes: ollama-models:`.

### Usage (to document)
- Default (cloud / host Ollama): `docker compose up`
- AMD: `docker compose -f docker-compose.yml -f docker/docker-compose.gpu.yml --profile gpu-amd up`
- NVIDIA: `docker compose -f docker-compose.yml -f docker/docker-compose.gpu.yml --profile gpu-nvidia up`
- Pull a model once: `docker compose exec ollama ollama pull llama3.2`

### Part B steps (in order)
1. Create `docker/docker-compose.gpu.yml` (design above).
2. Update `.env.example` — add `OLLAMA_MODEL=llama3.2` under a GPU section.
3. Update `README.md` — new "Local GPU LLM (optional)" section (the 3 modes + model pull).
4. Validate: `docker compose -f docker-compose.yml -f docker/docker-compose.gpu.yml --profile gpu-amd config -q` (and `gpu-nvidia`). (Can't runtime-test GPU here — no GPU — so config validation is the bar; user tests runtime on their GPU box.)

### Part B files (current state)
| File | State |
|---|---|
| `docker/docker-compose.gpu.yml` | PENDING (new) |
| `.env.example` | PENDING (add OLLAMA_MODEL) |
| `README.md` | PENDING (GPU section; also needs Part A debug update) |

---

## Notes for resumption
- All Part A code changes are **saved to disk** (uncommitted): `.vscode/tasks.json`, `docker/docker-compose.debug.yml`, `docker/program/debug.sh` (new), plus the earlier `main.py` (FastAPI app).
- The debug stack is **down** (clean state). Named volume `pgdata` preserved.
- `/tmp/dap_test.py` is the properly-sequenced DAP client (re-usable).
- Key gotcha to remember: `/proc/net/tcp` port hex for 5678 is `162E`, not `1634`.
- The entrypoint (`docker/program/entrypoint.sh`) echoes `$@` — never put a readiness marker string literally in an inline `sh -c` command; it leaks into logs. Use a script file (as done) or runtime string concatenation.
