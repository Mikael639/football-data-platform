#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SECRETS_DIR="${ROOT_DIR}/secrets"
ENV_FILE="${ROOT_DIR}/.env.prod"

FOOTBALL_DATA_TOKEN_SECRET_OCID="${FOOTBALL_DATA_TOKEN_SECRET_OCID:-}"
SUPABASE_DB_URL_SECRET_OCID="${SUPABASE_DB_URL_SECRET_OCID:-}"
STUDY_SUPABASE_DB_URL_SECRET_OCID="${STUDY_SUPABASE_DB_URL_SECRET_OCID:-}"

if ! command -v oci >/dev/null 2>&1; then
    echo "OCI CLI is required. Install it on the VM before running this script." >&2
    exit 1
fi

if [[ -z "${FOOTBALL_DATA_TOKEN_SECRET_OCID}" ]]; then
    echo "FOOTBALL_DATA_TOKEN_SECRET_OCID is required." >&2
    exit 1
fi

if [[ -z "${SUPABASE_DB_URL_SECRET_OCID}" ]]; then
    echo "SUPABASE_DB_URL_SECRET_OCID is required." >&2
    exit 1
fi

mkdir -p "${SECRETS_DIR}"
umask 077

fetch_secret_value() {
    local secret_ocid="$1"
    OCI_CLI_AUTH=instance_principal \
        oci secrets secret-bundle get \
        --secret-id "${secret_ocid}" \
        --query 'data."secret-bundle-content".content' \
        --raw-output \
        | base64 --decode
}

write_secret_file() {
    local secret_ocid="$1"
    local target_path="$2"
    local tmp_path
    tmp_path="$(mktemp)"
    fetch_secret_value "${secret_ocid}" > "${tmp_path}"
    mv "${tmp_path}" "${target_path}"
    chmod 600 "${target_path}"
}

write_secret_file "${FOOTBALL_DATA_TOKEN_SECRET_OCID}" "${SECRETS_DIR}/football_data_token.txt"
write_secret_file "${SUPABASE_DB_URL_SECRET_OCID}" "${SECRETS_DIR}/supabase_db_url.txt"

if [[ -n "${STUDY_SUPABASE_DB_URL_SECRET_OCID}" ]]; then
    write_secret_file "${STUDY_SUPABASE_DB_URL_SECRET_OCID}" "${SECRETS_DIR}/study_supabase_db_url.txt"
else
    cp "${SECRETS_DIR}/supabase_db_url.txt" "${SECRETS_DIR}/study_supabase_db_url.txt"
    chmod 600 "${SECRETS_DIR}/study_supabase_db_url.txt"
fi

if [[ -f "${ENV_FILE}" ]]; then
    python3 - "${ENV_FILE}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
raw_lines = path.read_text(encoding="utf-8").splitlines()
updates = {
    "FOOTBALL_DATA_TOKEN": "",
    "SUPABASE_DB_URL": "",
    "STUDY_SUPABASE_DB_URL": "",
    "FOOTBALL_DATA_TOKEN_FILE": "/run/secrets/football_data_token",
    "SUPABASE_DB_URL_FILE": "/run/secrets/supabase_db_url",
    "STUDY_SUPABASE_DB_URL_FILE": "/run/secrets/study_supabase_db_url",
}
remaining = set(updates)
result: list[str] = []

for line in raw_lines:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in line:
        result.append(line)
        continue
    key, _ = line.split("=", 1)
    key = key.strip()
    if key in updates:
        result.append(f"{key}={updates[key]}")
        remaining.discard(key)
    else:
        result.append(line)

for key in (
    "FOOTBALL_DATA_TOKEN",
    "FOOTBALL_DATA_TOKEN_FILE",
    "SUPABASE_DB_URL",
    "SUPABASE_DB_URL_FILE",
    "STUDY_SUPABASE_DB_URL",
    "STUDY_SUPABASE_DB_URL_FILE",
):
    if key in remaining:
        result.append(f"{key}={updates[key]}")

path.write_text("\n".join(result) + "\n", encoding="utf-8")
PY
fi

echo "OCI Vault secrets synchronized into ${SECRETS_DIR}."
echo "Restart the stack to apply them:"
echo "  docker compose --env-file .env.prod -f docker-compose.prod.yml up -d --build"
