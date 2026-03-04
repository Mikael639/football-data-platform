#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <encrypted-backup-file>" >&2
  exit 1
fi

if [[ -z "${BACKUP_PASSPHRASE:-}" ]]; then
  echo "BACKUP_PASSPHRASE is required." >&2
  exit 1
fi

INPUT_FILE="$1"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.prod.yml}"
ENV_FILE="${ENV_FILE:-.env.prod}"
DB_NAME="${POSTGRES_DB:-football_dw}"
DB_USER="${POSTGRES_SUPERUSER:-postgres}"

openssl enc -d -aes-256-cbc -pbkdf2 -pass env:BACKUP_PASSPHRASE -in "${INPUT_FILE}" \
  | gunzip \
  | docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" exec -T postgres \
      psql -U "${DB_USER}" -d "${DB_NAME}"

echo "Restore completed from ${INPUT_FILE}"
