-- Runs once on an empty data volume, as the bootstrap superuser.
-- See: docker/postgres/01-roles.sh for role creation + grants.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS public.icd10_codes (
    order_number integer  PRIMARY KEY,
    code         text     NOT NULL,
    code_type    smallint NOT NULL CHECK (code_type IN (0, 1)),
    -- 0 = category header, 1 = billable code (mirrors the source file).
    -- Generated boolean keeps the "only billable rows for embedding" rule cheap.
    is_billable  boolean  GENERATED ALWAYS AS (code_type = 1) STORED,
    short_desc   text     NOT NULL,
    long_desc    text     NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_icd10_codes_code
    ON public.icd10_codes (code);

-- Partial index: cheap filtering of billable rows for embedding/lookup.
CREATE INDEX IF NOT EXISTS idx_icd10_codes_billable
    ON public.icd10_codes (is_billable) WHERE is_billable;
