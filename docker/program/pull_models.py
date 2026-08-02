#!/usr/bin/env python3
"""Written by AI
Sidecar: auto-pull every model in models.toml into the running Ollama when
the GPU compose stack comes up.

Run by the ``ollama-init`` service in docker/docker-compose.gpu.yml (profiles
gpu-amd / gpu-nvidia). It polls http://ollama:11434 until the active Ollama
service is reachable (whichever GPU vendor is up), then POSTs /api/pull for each
entry, skipping anything already present. Stdlib-only, so the sidecar runs in a
plain python:3.13-slim container with no pip installs.

Fire-and-forget: nothing depends on this container, so it pulls in parallel with
the medicoder app and exits when done.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_CONFIG = Path(os.environ.get("MODELS_CONFIG", "/models.toml"))
DEFAULT_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
POLL_TIMEOUT = float(os.environ.get("OLLAMA_INIT_TIMEOUT", "180"))


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_models(path: Path) -> list[dict]:
    """Read the ``[[model]]`` entries from a TOML config file.

    Args:
        path: Path to ``models.toml``. Need not exist (returns an empty list
            with a notice if missing).

    Returns:
        A list of model entries, each minimally containing a ``name`` key.

    Raises:
        SystemExit: If an entry is missing its ``name`` field.
    """
    if not path.is_file():
        print(f"pull_models: {path} not found; nothing to pull")
        return []
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    entries = list(data.get("model", []))
    for i, e in enumerate(entries):
        if "name" not in e:
            raise SystemExit(f"{path}: entry #{i} missing 'name'")
    return entries


# --------------------------------------------------------------------------- #
# Ollama client (stdlib urllib)
# --------------------------------------------------------------------------- #
def get_existing(base: str, timeout: float = 10.0) -> set[str]:
    """List model names already registered in the running Ollama.

    Args:
        base: Ollama base URL (e.g. "http://ollama:11434").
        timeout: Request timeout in seconds.

    Returns:
        The set of model names reported by ``GET /api/tags`` (lowercased by
        Ollama).
    """
    with urllib.request.urlopen(f"{base}/api/tags", timeout=timeout) as resp:
        return {m["name"] for m in json.load(resp).get("models", [])}


def wait_for_ollama(base: str, deadline: float) -> None:
    """Poll Ollama until it responds or the deadline passes.

    Args:
        base: Ollama base URL.
        deadline: Absolute ``time.time()`` deadline (epoch seconds).

    Raises:
        SystemExit: If Ollama stays unreachable past the deadline.
    """
    print(f"pull_models: waiting for Ollama at {base} ...")
    while time.time() < deadline:
        try:
            existing = get_existing(base)
            print(f"pull_models: Ollama up ({len(existing)} model(s) present)")
            return
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    raise SystemExit(
        f"pull_models: Ollama not reachable at {base} after {POLL_TIMEOUT:.0f}s; giving up"
    )


def pull_one(base: str, name: str) -> None:
    """Pull a single model into Ollama, streaming progress to stdout.

    Reads the newline-delimited JSON status stream from ``POST /api/pull`` and
    renders download percentages inline. Returns as soon as Ollama reports a
    ``success`` status.

    Args:
        base: Ollama base URL.
        name: The model tag to pull (e.g. "hf.co/.../model:Q8_0").

    Raises:
        RuntimeError: If Ollama reports an ``error`` in the stream, or the
            stream ends without a ``success`` status.
    """
    print(f"  pulling {name} ...")
    req = urllib.request.Request(
        f"{base}/api/pull",
        data=json.dumps({"model": name}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_status = None
    saw_success = False
    # timeout=None: a pull can run for many minutes (multi-GB); don't abort mid-stream.
    with urllib.request.urlopen(req, timeout=None) as resp:
        for raw in resp:  # iterating yields newline-delimited bytes
            if not raw.strip():
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            # Ollama reports failures as {"error": "..."} in the stream — detect
            # them so we don't log a false [ok] for a pull that downloaded nothing.
            if msg.get("error"):
                _progress_done()
                raise RuntimeError(f"ollama: {msg['error']}")
            status = msg.get("status", "")
            total = msg.get("total")
            completed = msg.get("completed")
            if total and completed is not None and status == last_status:
                _progress(status, completed, total)
            else:
                _progress_done()
                print(f"    {status}")
                last_status = status
                if status == "success":
                    saw_success = True
                    return
    _progress_done()
    if not saw_success:
        raise RuntimeError("pull stream ended without a success status")


def _progress(label: str, done: int, total: int) -> None:
    """Render an inline ``done/total (pct%)`` progress line for a pull.

    Args:
        label: Status label from Ollama (e.g. "pulling manifest").
        done: Bytes completed so far.
        total: Total bytes expected.
    """
    if total <= 0:
        return
    sys.stdout.write(f"\r    {label}: {done:,}/{total:,} ({done * 100 / total:5.1f}%)")
    sys.stdout.flush()


def _progress_done() -> None:
    """Clear the inline progress line written by :func:`_progress`."""
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    """CLI entry: pull every model listed in the config into Ollama.

    Waits for Ollama, skips models already present, pulls the rest, then
    re-checks ``/api/tags`` to confirm each reported success actually landed.

    Returns:
        Process exit code (0 if all pulls succeeded, 1 if any failed).
    """
    parser = argparse.ArgumentParser(description="Auto-pull ollama models listed in a TOML file.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="models.toml path")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Ollama base URL")
    args = parser.parse_args()

    models = load_models(args.config)

    if not models:
        print("pull_models: no entries; exiting.")
        return 0

    deadline = time.time() + POLL_TIMEOUT
    wait_for_ollama(args.base_url, deadline)
    # Case-insensitive: Ollama lowercases names in /api/tags (e.g. hf.co/...).
    existing = {e.lower() for e in get_existing(args.base_url)}

    attempted: list[str] = []  # entries that returned [ok] from pull_one
    failures: list[str] = []
    for entry in models:
        name = entry["name"]
        # Ollama stores models with an explicit tag (e.g. "foo:latest"); a
        # tagless name in models.toml resolves to :latest, so check both.
        aliases = {a.lower() for a in {name, name if ":" in name else f"{name}:latest"}}
        if existing & aliases:
            print(f"[skip] {name}  (already present)")
            continue
        try:
            pull_one(args.base_url, name)
            attempted.append(name)
            print(f"[ok]   {name}")
        except Exception as e:  # noqa: BLE001 - report and continue
            _progress_done()
            print(f"[fail] {name}: {e}")
            failures.append(name)

    # Defense in depth: re-fetch /api/tags and confirm each "ok" actually
    # registered. Catches silent failures where Ollama streams success but the
    # model doesn't land (e.g. wrong repo, empty manifest).
    if attempted:
        landed = {e.lower() for e in get_existing(args.base_url)}
        for name in attempted:
            aliases = {a.lower() for a in {name, name if ":" in name else f"{name}:latest"}}
            if not (landed & aliases):
                print(f"[fail] {name}: pull reported success but model is absent from /api/tags")
                failures.append(name)

    if failures:
        print(f"\npull_models: {len(failures)} failure(s): {', '.join(failures)}")
        return 1
    n = len(models)
    print(f"\npull_models: done ({n} entr{'y' if n == 1 else 'ies'}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
