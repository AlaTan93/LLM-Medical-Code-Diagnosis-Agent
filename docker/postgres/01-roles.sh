#!/bin/sh
# Runs once on an empty data volume, after 00-schema.sql.
# Creates the application roles used by the program container:
#   - medicoder : read/write, owns icd10_codes (used for ingestion + the app)
#   - agent     : SELECT-only, plus default_transaction_read_only = on.
#                 Connect the LangChain scoped SQL tool as this role.
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<EOSQL
DO \$do\$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'medicoder') THEN
        CREATE ROLE medicoder LOGIN PASSWORD '${MEDICODER_PASSWORD}';
    ELSE
        ALTER ROLE medicoder LOGIN PASSWORD '${MEDICODER_PASSWORD}';
    END IF;

    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'agent') THEN
        CREATE ROLE agent LOGIN PASSWORD '${AGENT_PASSWORD}';
    ELSE
        ALTER ROLE agent LOGIN PASSWORD '${AGENT_PASSWORD}';
    END IF;
END
\$do\$;

-- Ownership + grants
ALTER TABLE public.icd10_codes OWNER TO medicoder;
GRANT USAGE ON SCHEMA public TO medicoder, agent;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.icd10_codes TO medicoder;
GRANT SELECT ON public.icd10_codes TO agent;

-- Defense in depth for the read-only agent role.
ALTER ROLE agent SET default_transaction_read_only = on;
EOSQL
