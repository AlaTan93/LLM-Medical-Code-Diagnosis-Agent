#!/bin/sh
# Written by AI
# Entrypoint for the medicoder program container.
# Ensures the ICD-10 data is loaded (idempotent), then hands off to the command.
set -e

echo "[medicoder] ensuring ICD-10 data is loaded ..."
python -m medicoder.db.load_icd10

echo "[medicoder] starting: $*"
exec "$@"
