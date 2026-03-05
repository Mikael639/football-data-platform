#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${BACKUP_PASSPHRASE:-}" ]]; then
  echo "BACKUP_PASSPHRASE is required." >&2
  exit 1
fi

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.prod.yml}"
ENV_FILE="${ENV_FILE:-.env.prod}"
BACKUP_DIR="${BACKUP_DIR:-./backups}"
DB_NAME="${POSTGRES_DB:-football_dw}"
DB_USER="${POSTGRES_SUPERUSER:-postgres}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "${BACKUP_DIR}"
OUTFILE="${BACKUP_DIR}/${DB_NAME}_${TIMESTAMP}.sql.gz.enc"

docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" exec -T postgres \
  pg_dump -U "${DB_USER}" -d "${DB_NAME}" \
  | gzip \
  | openssl enc -aes-256-cbc -pbkdf2 -salt -pass env:BACKUP_PASSPHRASE -out "${OUTFILE}"

echo "Encrypted backup written to ${OUTFILE}"
