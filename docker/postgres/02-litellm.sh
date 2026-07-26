#!/bin/sh
# Runs once on an empty data volume, after 01-roles.sh.
# Creates the role + database that LiteLLM uses for call/response auditing.
# LiteLLM runs Prisma migrations on first boot to create its tables
# (LiteLLM_SpendLogs, LiteLLM_ProxyModelTable, ...).
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<EOSQL
DO \$do\$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'litellm') THEN
        CREATE ROLE litellm LOGIN PASSWORD '${LITELLM_DB_PASSWORD}';
    ELSE
        ALTER ROLE litellm LOGIN PASSWORD '${LITELLM_DB_PASSWORD}';
    END IF;
END
\$do\$;

-- \gexec runs the preceding SELECT's result string as SQL, so the
-- CREATE DATABASE only fires when the DB does not yet exist.
SELECT 'CREATE DATABASE litellm OWNER litellm'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'litellm')\gexec
EOSQL
