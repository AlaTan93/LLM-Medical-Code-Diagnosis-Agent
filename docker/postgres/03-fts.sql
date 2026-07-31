-- Full-text search index for hybrid search.
--
-- Vector similarity alone penalises "unspecified" codes: the embedding model
-- ranks C539 ("Malignant neoplasm of cervix uteri, unspecified") far below
-- Z8541 ("Personal history of malignant neoplasm of cervix uteri") even when
-- the query is the exact ICD-10 description.
--
-- This adds a generated tsvector column (combining short_desc + long_desc) and
-- a GIN index so search_icd10 can run a lexical query alongside the vector
-- query and merge the two rankings.
--
-- Runs automatically on a fresh data volume.  For an existing volume:
--   psql -h localhost -U medicoder -d medicoder -f docker/postgres/03-fts.sql

ALTER TABLE public.icd10_codes
    ADD COLUMN IF NOT EXISTS search_tsv tsvector
    GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(short_desc, '') || ' ' || coalesce(long_desc, ''))
    ) STORED;

CREATE INDEX IF NOT EXISTS idx_icd10_codes_fts
    ON public.icd10_codes USING gin (search_tsv)
    WHERE is_billable;
