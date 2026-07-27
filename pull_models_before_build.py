#!/usr/bin/env python3
"""Pre-populate a running Ollama with a configurable list of models.

Run this on the host AFTER starting the GPU Ollama stack (so the ollama-models
named volume it writes into exists), and BEFORE the medicoder app needs the
models:

    docker compose --env-file .env -f docker-compose.yml \\
        -f docker/docker-compose.gpu.yml --profile gpu-amd up -d ollama-amd
    uv run --group bootstrap python pull_models_before_build.py

Models land in the ``ollama-models`` Docker named volume. Only a running Ollama
container can write there, so this script drives it over HTTP at
``OLLAMA_BASE_URL`` (the gpu stack publishes ``11434:11434``). The volume
persists across ``down``/``up``, so this is a one-time bootstrap; re-runs are
idempotent (existing models are skipped, HF downloads hit the local cache).

The model list lives in ``models.toml``. Two entry kinds:

  kind = "ollama"   POST /api/pull <name>                 (Ollama registry)
  kind = "hf"       hf_hub_download -> sha256 ->          (HuggingFace GGUF)
                    POST /api/blobs/<digest> ->
                    POST /api/create {model, files}

Exit code is non-zero if any entry failed (unless --keep-going, which still
exits non-zero if anything failed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import tomllib
from pathlib import Path
from typing import Any, Iterator

import requests
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "models.toml"
DEFAULT_ENV = REPO_ROOT / ".env"
DEFAULT_MODELS_DIR = "models"  # HF GGUFs download flat here (repo-relative).
CHUNK = 1024 * 1024  # 1 MiB for hashing / uploading / progress.


# --------------------------------------------------------------------------- #
# Config / env
# --------------------------------------------------------------------------- #
def load_env(path: Path) -> None:
    """Load KEY=VALUE pairs from ``path`` into os.environ without overriding
    values already present in the real environment (shell wins)."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_models(path: Path) -> list[dict[str, Any]]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    entries = data.get("model", [])
    if not isinstance(entries, list):
        raise SystemExit(f"{path}: expected a [[model]] array, got {type(entries).__name__}")
    out: list[dict[str, Any]] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or "kind" not in entry:
            raise SystemExit(f"{path}: entry #{i} missing 'kind'")
        kind = entry["kind"]
        if kind not in ("ollama", "hf"):
            raise SystemExit(f"{path}: entry #{i} has kind={kind!r}; expected 'ollama' or 'hf'")
        if "name" not in entry:
            raise SystemExit(f"{path}: entry #{i} ({kind}) missing 'name'")
        if kind == "hf" and ("repo" not in entry or "file" not in entry):
            raise SystemExit(f"{path}: entry #{i} (hf) needs 'repo' and 'file'")
        out.append(entry)
    return out


# --------------------------------------------------------------------------- #
# Ollama client
# --------------------------------------------------------------------------- #
def existing_models(base: str, session: requests.Session) -> set[str]:
    """Names of models already in Ollama's local store (GET /api/tags)."""
    resp = session.get(f"{base}/api/tags", timeout=30)
    resp.raise_for_status()
    return {m["name"] for m in resp.json().get("models", [])}


def pull_ollama(base: str, name: str, session: requests.Session) -> None:
    """POST /api/pull, streaming progress lines to stdout."""
    print(f"  pulling from registry: {name}")
    with session.post(f"{base}/api/pull", json={"model": name}, stream=True, timeout=None) as resp:
        resp.raise_for_status()
        last_status = None
        for raw in resp.iter_lines():
            if not raw:
                continue
            msg = json.loads(raw)
            status = msg.get("status", "")
            total = msg.get("total")
            completed = msg.get("completed")
            if total and completed is not None and status == last_status:
                # Same phase as before -> update the percent in place.
                _progress(status, completed, total)
            else:
                _progress_done()
                print(f"    {status}")
                last_status = status
                if total and completed is not None:
                    _progress(status, completed, total)
        _progress_done()


def import_hf(
    base: str,
    repo: str,
    file: str,
    name: str,
    token: str | None,
    models_dir: Path,
    session: requests.Session,
) -> None:
    """Download a GGUF from HuggingFace into ``models_dir`` and import it into Ollama.

    ``local_dir`` makes hf_hub_download place a flat ``models_dir/<file>`` (no
    nested cache layout), so the GGUF is visible/inspectable and matches the
    ``models/`` convention (gitignored for *.gguf/*.bin).
    """
    print(f"  downloading from HF: {repo}/{file}")
    local = hf_hub_download(
        repo_id=repo, filename=file, token=token or None, local_dir=str(models_dir)
    )
    size = Path(local).stat().st_size
    print(f"    saved at {local} ({size:,} bytes)")

    print("    hashing (sha256) ...")
    digest = sha256_file(local)
    blob_url = f"{base}/api/blobs/{digest}"

    head = session.head(blob_url, timeout=30)
    if head.status_code == 200:
        print("    blob already on server, skipping upload")
    elif head.status_code == 404:
        print("    uploading blob ...")
        upload_blob(local, blob_url, session)
        _progress_done()
    else:
        head.raise_for_status()

    print(f"    creating model {name} ...")
    with session.post(
        f"{base}/api/create",
        json={"model": name, "files": {file: digest}},
        stream=True,
        timeout=None,
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw:
                continue
            msg = json.loads(raw)
            status = msg.get("status", "")
            if status:
                print(f"      {status}")
            if status == "success":
                return
    raise RuntimeError(f"create did not report success for {name}")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def upload_blob(path: str, url: str, session: requests.Session) -> None:
    """Stream a file to POST /api/blobs/<digest> with a progress generator."""
    size = os.path.getsize(path)
    sent = 0

    def gen() -> Iterator[bytes]:
        nonlocal sent
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(CHUNK), b""):
                sent += len(chunk)
                yield chunk
                _progress("upload", sent, size)

    resp = session.post(
        url,
        data=gen(),
        headers={"Content-Type": "application/octet-stream"},
        timeout=None,
    )
    resp.raise_for_status()


# --------------------------------------------------------------------------- #
# Progress helpers (carriage-return in-place line; no extra deps).
# --------------------------------------------------------------------------- #
def _progress(label: str, done: int, total: int) -> None:
    if total <= 0:
        return
    pct = done * 100.0 / total
    sys.stdout.write(f"\r    {label}: {done:,}/{total:,} bytes ({pct:5.1f}%)")
    sys.stdout.flush()


def _progress_done() -> None:
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-pull models into a running Ollama.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=f"model list (default: {DEFAULT_CONFIG.name})")
    parser.add_argument("--base-url", default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
                        help="Ollama base URL (default: $OLLAMA_BASE_URL or http://localhost:11434)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and what would be skipped, then exit")
    parser.add_argument("--keep-going", action="store_true", help="continue past failures (exit code still non-zero if any failed)")
    parser.add_argument("--no-env-file", action="store_true", help=f"do not load {DEFAULT_ENV.name} (use real env only)")
    parser.add_argument("--models-dir", default=None,
                        help=f"directory for HF GGUF downloads (default: $MODELS_DIR or {DEFAULT_MODELS_DIR!r})")
    args = parser.parse_args()

    if not args.no_env_file:
        load_env(DEFAULT_ENV)
    # Re-read after loading .env so --base-url default can reflect it if unset on CLI.
    base_url = args.base_url

    # Resolve the download dir: CLI > shell env > .env (loaded above) > default.
    # Resolved after load_env so MODELS_DIR set in .env is honored.
    models_dir_raw = args.models_dir or os.environ.get("MODELS_DIR", DEFAULT_MODELS_DIR)
    models_dir = Path(models_dir_raw)
    if not models_dir.is_absolute():
        models_dir = (REPO_ROOT / models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    models = load_models(args.config)
    token = os.environ.get("HF_TOKEN") or None

    session = requests.Session()

    # Health check up front: nothing else makes sense if Ollama isn't reachable.
    try:
        have = existing_models(base_url, session)
    except requests.RequestException as e:
        print(f"error: cannot reach Ollama at {base_url} ({e})")
        print("start the GPU stack first, e.g.:")
        print("  docker compose --env-file .env -f docker-compose.yml \\")
        print("      -f docker/docker-compose.gpu.yml --profile gpu-amd up -d ollama-amd")
        return 2

    print(f"ollama: {base_url}  ({len(have)} model(s) already present)")
    print(f"plan: {len(models)} entr{'y' if len(models) == 1 else 'ies'} in {args.config.name}")
    print(f"hf downloads -> {models_dir}")

    failures: list[str] = []
    for entry in models:
        kind = entry["kind"]
        name = entry["name"]
        tag = "ollama" if kind == "ollama" else f"hf:{entry['repo']}/{entry['file']}"
        if name in have:
            print(f"[skip] {name}  (already present; source: {tag})")
            continue
        if args.dry_run:
            print(f"[would] {name}  (source: {tag})")
            continue
        print(f"[pull] {name}  (source: {tag})")
        t0 = time.time()
        try:
            if kind == "ollama":
                pull_ollama(base_url, name, session)
            else:
                import_hf(base_url, entry["repo"], entry["file"], name, token, models_dir, session)
            print(f"  done in {time.time() - t0:.1f}s")
            have.add(name)
        except Exception as e:  # noqa: BLE001 - report and move on
            _progress_done()
            print(f"  FAILED: {e}")
            failures.append(name)
            if not args.keep_going:
                print("  (stopping; pass --keep-going to continue past failures)")
                break

    if args.dry_run:
        print("dry-run: no changes made.")
        return 0

    if failures:
        print(f"\n{len(failures)} failure(s): {', '.join(failures)}")
        return 1

    print(f"\nall {len(models)} model(s) present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
