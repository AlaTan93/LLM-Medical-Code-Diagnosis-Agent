-- Written by AI
-- Vector similarity search for ICD-10-CM codes.
-- The pgvector extension is already installed by 00-schema.sql; this file adds
-- the embedding column + HNSW index over billable rows.
--
-- Runs automatically on a fresh data volume (docker-entrypoint-initdb.d). For
-- an existing volume, apply manually:
--   psql -h localhost -U medicoder -d medicoder -f docker/postgres/02-embedding.sql
--
-- Dimension: 2560 (zembed-1).  Uses halfvec (float16) because pgvector HNSW
-- caps vector at 2000 dims; halfvec supports up to 4000.

CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE public.icd10_codes
    ADD COLUMN IF NOT EXISTS embedding halfvec(2560);

-- HNSW over all rows (billable + non-billable). Non-billable codes are
-- pre-embedded so they are search-ready if they become billable in a future
-- release. The /code search query filters to is_billable at query time.
-- m=32 / ef_construction=128 for higher recall on 98k codes (defaults of 16/64
-- miss obvious matches at the default ef_search=40). The /code route also sets
-- hnsw.ef_search=200 per-query for additional safety.
CREATE INDEX IF NOT EXISTS idx_icd10_codes_embedding
    ON public.icd10_codes USING hnsw (embedding halfvec_cosine_ops)
    WITH (m = 32, ef_construction = 128);
