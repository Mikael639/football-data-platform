#!/bin/sh
set -eu

if [ -z "${DB_WRITER_USER:-}" ] || [ -z "${DB_READER_USER:-}" ]; then
  echo "DB_WRITER_USER and DB_READER_USER are required." >&2
  exit 1
fi

DB_WRITER_PASSWORD="${DB_WRITER_PASSWORD:-}"
DB_READER_PASSWORD="${DB_READER_PASSWORD:-}"

if [ -z "${DB_WRITER_PASSWORD}" ] && [ -n "${DB_WRITER_PASSWORD_FILE:-}" ]; then
  if [ ! -f "${DB_WRITER_PASSWORD_FILE}" ]; then
    echo "Database writer password secret file missing." >&2
    exit 1
  fi
  DB_WRITER_PASSWORD="$(tr -d '\r\n' < "${DB_WRITER_PASSWORD_FILE}")"
fi

if [ -z "${DB_READER_PASSWORD}" ] && [ -n "${DB_READER_PASSWORD_FILE:-}" ]; then
  if [ ! -f "${DB_READER_PASSWORD_FILE}" ]; then
    echo "Database reader password secret file missing." >&2
    exit 1
  fi
  DB_READER_PASSWORD="$(tr -d '\r\n' < "${DB_READER_PASSWORD_FILE}")"
fi

if [ -z "${DB_WRITER_PASSWORD}" ] || [ -z "${DB_READER_PASSWORD}" ]; then
  echo "Database writer and reader passwords are required." >&2
  exit 1
fi

psql -v ON_ERROR_STOP=1 --username "${POSTGRES_USER}" --dbname "${POSTGRES_DB}" \
  -v db_writer_user="${DB_WRITER_USER}" \
  -v db_reader_user="${DB_READER_USER}" \
  -v db_writer_password="${DB_WRITER_PASSWORD}" \
  -v db_reader_password="${DB_READER_PASSWORD}" <<'SQL'
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'db_writer_user', :'db_writer_password')
WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = :'db_writer_user')\gexec

SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'db_reader_user', :'db_reader_password')
WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = :'db_reader_user')\gexec
SQL

for sql_file in \
  /app-sql/01_schema.sql \
  /app-sql/02_migrations.sql \
  /app-sql/03_migrations.sql \
  /app-sql/04_migrations.sql \
  /app-sql/06_migrations.sql \
  /app-sql/02_indexes.sql
do
  if [ -f "${sql_file}" ]; then
    psql -v ON_ERROR_STOP=1 --username "${POSTGRES_USER}" --dbname "${POSTGRES_DB}" -f "${sql_file}"
  fi
done

psql -v ON_ERROR_STOP=1 --username "${POSTGRES_USER}" --dbname "${POSTGRES_DB}" \
  -v db_writer_user="${DB_WRITER_USER}" \
  -v db_reader_user="${DB_READER_USER}" <<'SQL'
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON SCHEMA public FROM PUBLIC;

GRANT USAGE ON SCHEMA public TO :"db_writer_user";
GRANT USAGE ON SCHEMA public TO :"db_reader_user";

GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE ON ALL TABLES IN SCHEMA public TO :"db_writer_user";
GRANT SELECT ON ALL TABLES IN SCHEMA public TO :"db_reader_user";
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO :"db_writer_user";

ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE ON TABLES TO :"db_writer_user";

ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT SELECT ON TABLES TO :"db_reader_user";

ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT USAGE, SELECT ON SEQUENCES TO :"db_writer_user";

SELECT format('REVOKE ALL ON DATABASE %I FROM PUBLIC', current_database())\gexec
SELECT format('GRANT CONNECT ON DATABASE %I TO %I', current_database(), :'db_writer_user')\gexec
SELECT format('GRANT CONNECT ON DATABASE %I TO %I', current_database(), :'db_reader_user')\gexec
SQL
