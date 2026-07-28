"""Idempotent loader for the ICD-10-CM order file into PostgreSQL.

The source file ``icd10cm_order_YYYY.txt`` is a fixed-width ASCII file. Column
offsets are documented in CMS' file layout and were verified against the 2026
release:

    (0, 5)   order number
    (6, 13)  ICD-10-CM code (7 chars, left-justified, space-padded)
    (14, 15) description type (0 = category header, 1 = billable code)
    (16, 76) short description (60 chars)
    (77, ..) long description (variable width)

The loader connects via :func:`medicoder.db.connect.connect` (shared with
``embed_icd10.py``) and uses the binary COPY protocol for speed. It is safe to
re-run: if the table already holds rows it exits early unless ``LOAD_FORCE=1``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

from medicoder.db.connect import connect

COLSPECS = [(0, 5), (6, 13), (14, 15), (16, 76), (77, None)]
COLUMNS = ["order_number", "code", "code_type", "short_desc", "long_desc"]

COUNT_SQL = "SELECT count(*) FROM icd10_codes"
TRUNCATE_SQL = "TRUNCATE TABLE icd10_codes"
COPY_SQL = (
    "COPY icd10_codes (order_number, code, code_type, short_desc, long_desc) "
    "FROM STDIN"
)


def _read_frame(path: Path) -> pd.DataFrame:
    """Parse the fixed-width ICD-10-CM order file into a clean DataFrame.

    Column offsets come from :data:`COLSPECS`; string columns are stripped of
    surrounding whitespace and rows missing an order number or code are dropped.

    Args:
        path: Path to ``icd10cm_order_YYYY.txt``.

    Returns:
        A DataFrame with columns ``order_number, code, code_type, short_desc,
        long_desc``.
    """
    df = pd.read_fwf(
        path,
        colspecs=COLSPECS,
        names=COLUMNS,
        dtype={"order_number": "int64", "code": "string", "code_type": "int8"},
    )
    df = df.dropna(subset=["order_number", "code"])
    for col in ("code", "short_desc", "long_desc"):
        df[col] = df[col].astype("string").str.strip()
    df["code_type"] = df["code_type"].astype("int8")
    return df[["order_number", "code", "code_type", "short_desc", "long_desc"]]


def load(path: Path, *, force: bool = False) -> int:
    """Idempotently load the ICD-10-CM order file into Postgres via COPY.

    If the target table already holds rows the load is skipped unless
    ``force`` is set, in which case the table is truncated first.

    Args:
        path: Path to the fixed-width ICD-10-CM order file.
        force: If True, truncate and reload even when rows already exist.

    Returns:
        The total row count of ``icd10_codes`` after loading (0 if skipped).
    """
    print(f"[load_icd10] parsing {path} ...")
    df = _read_frame(path)
    print(f"[load_icd10] parsed {len(df)} rows")

    with connect() as conn, conn.cursor() as cur:
        cur.execute(COUNT_SQL)
        existing = cur.fetchone()[0]
        if existing and not force:
            print(f"[load_icd10] icd10_codes already has {existing} rows; skipping.")
            return 0
        if existing and force:
            print(f"[load_icd10] force=True; truncating {existing} rows.")
            cur.execute(TRUNCATE_SQL)

        with cur.copy(COPY_SQL) as copy:
            for row in df.itertuples(index=False):
                copy.write_row(
                    (
                        int(row.order_number),
                        row.code,
                        int(row.code_type),
                        row.short_desc,
                        row.long_desc,
                    )
                )
        conn.commit()

        cur.execute(COUNT_SQL)
        total = cur.fetchone()[0]
        print(f"[load_icd10] loaded {total} rows into icd10_codes.")
        return total


def main() -> int:
    """CLI entry: load the ICD-10 file referenced by ``ICD10_FILE``.

    ``LOAD_FORCE=1`` forces a reload. The path defaults to
    ``icd10cm_order_2026.txt``.

    Returns:
        Process exit code (0 on success, 1 if the data file is missing).
    """
    path = Path(os.environ.get("ICD10_FILE", "icd10cm_order_2026.txt"))
    force = os.environ.get("LOAD_FORCE", "").lower() in ("1", "true", "yes")
    if not path.exists():
        print(f"[load_icd10] data file not found: {path}", file=sys.stderr)
        return 1
    load(path, force=force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
