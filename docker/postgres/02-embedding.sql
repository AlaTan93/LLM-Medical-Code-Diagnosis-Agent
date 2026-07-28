-- Vector similarity search for ICD-10-CM codes.
-- The pgvector extension is already installed by 00-schema.sql; this file adds
-- the embedding column + HNSW index over billable rows.
--
-- Runs automatically on a fresh data volume (docker-entrypoint-initdb.d). For
-- an existing volume, apply manually:
--   psql -h localhost -U medicoder -d medicoder -f docker/postgres/02-embedding.sql
--
-- Dimension: 1024 (bge-m3).

CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE public.icd10_codes
    ADD COLUMN IF NOT EXISTS embedding vector(1024);

-- HNSW over all rows (billable + non-billable). Non-billable codes are
-- pre-embedded so they are search-ready if they become billable in a future
-- release. The /code search query filters to is_billable at query time.
CREATE INDEX IF NOT EXISTS idx_icd10_codes_embedding
    ON public.icd10_codes USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
