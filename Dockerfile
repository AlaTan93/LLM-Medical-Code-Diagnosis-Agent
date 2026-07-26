# syntax=docker/dockerfile:1
#
# medicoder program container.
# Python 3.13 + uv (deps pinned by uv.lock). Runs the idempotent ICD-10 loader
# on boot, then the application command.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Pin uv to the same version used on the host to keep the lock reproducible.
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first (cacheable layer; --frozen requires uv.lock to
# match pyproject.toml exactly).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Application code + the ICD-10 source file used by the loader.
COPY medicoder ./medicoder
COPY main.py data/icd10cm_order_2026.txt ./
COPY docker/program/entrypoint.sh /entrypoint.sh

# Non-root runtime.
RUN chmod +x /entrypoint.sh \
 && groupadd --system app && useradd --system --gid app --home-dir /app app \
 && chown --recursive app:app /app /entrypoint.sh
USER app

ENTRYPOINT ["/entrypoint.sh"]
# Override per-environment; default runs the current demo entrypoint.
CMD ["python", "main.py"]
