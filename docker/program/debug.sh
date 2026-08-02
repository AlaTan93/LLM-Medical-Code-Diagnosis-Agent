#!/bin/sh
# Written by AI
# Debug entrypoint for the medicoder container.
#
# Run by docker/docker-compose.debug.yml (command: sh /debug.sh) AFTER the
# baked ENTRYPOINT has loaded ICD-10 data. It:
#   1. syncs deps into the baked venv (adds debugpy via the dev group),
#   2. reports readiness WITHOUT connecting to :5678 (see below),
#   3. runs the app under debugpy, serving immediately (no --wait-for-client).
#
# Decoupled debug model: the app is a long-running FastAPI/uvicorn server, so
# debugpy just listens on :5678 while the server serves. VSCode attaches with
# F5 whenever, and detaches/reattaches without restarting the stack. To hit
# handler breakpoints, send a request after attaching (curl localhost:8000/health).
# Module-level / lifespan-startup code runs before attach, so breakpoints there
# are NOT hit — that's the accepted tradeoff of this model.
#
# Kept as a file (not an inline `sh -c` script) so the ENTRYPOINT's
# `echo "[medicoder] starting: $*"` prints only `sh /debug.sh` — an inline
# script would be echoed verbatim and leak the DEBUGPY_READY marker into the
# logs, falsely tripping a VSCode task's readiness matcher.
set -e

uv sync --frozen --no-install-project

# Report readiness WITHOUT connecting to :5678 — we still watch /proc/net/tcp
# rather than probing the port so the marker reflects "debugpy is listening"
# (5678 == 0x162E, LISTEN state == 0A) without a stray connect. The start task
# prints this line so the terminal shows when it's safe to F5.
(
  for i in $(seq 1 120); do
    awk '$2 ~ /:162E$/ && $4 == "0A" { ok=1 } END { exit !ok }' /proc/net/tcp \
      && { echo DEBUGPY_READY; exit 0; }
    sleep 0.5
  done
  echo DEBUGPY_TIMEOUT
) &

# -Xfrozen_modules=off avoids missed breakpoints (the image compiles bytecode).
# No --wait-for-client: the server starts and serves; VSCode attaches on demand.
python -Xfrozen_modules=off -m debugpy --listen 0.0.0.0:5678 main.py
