#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

COMPOSE_FILES=(-f docker-compose.prod.yml -f docker-compose.prod.free.yml)
ENV_FILE=".env.prod"
VAULT_ENV_FILE="secrets/oci_vault_ids.env"
RUN_PIPELINE=true

for arg in "$@"; do
    case "${arg}" in
        --skip-pipeline)
            RUN_PIPELINE=false
            ;;
        *)
            echo "Unknown option: ${arg}" >&2
            echo "Usage: bash scripts/prod_update_free.sh [--skip-pipeline]" >&2
            exit 1
            ;;
    esac
done

export PATH="${HOME}/bin:${PATH}"

if [[ -f "${VAULT_ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${VAULT_ENV_FILE}"
fi

echo "[1/6] Update repository"
git fetch --all
git checkout improve-platform
git pull origin improve-platform

if [[ -n "${FOOTBALL_DATA_TOKEN_SECRET_OCID:-}" && -n "${SUPABASE_DB_URL_SECRET_OCID:-}" ]]; then
    echo "[2/6] Sync OCI Vault secrets"
    bash scripts/oci_vault_sync.sh
else
    echo "[2/6] Skip OCI Vault sync (missing FOOTBALL_DATA_TOKEN_SECRET_OCID or SUPABASE_DB_URL_SECRET_OCID)"
fi

echo "[3/6] Stop heavy services"
docker compose --env-file "${ENV_FILE}" "${COMPOSE_FILES[@]}" stop pipeline pipeline_scheduler || true
docker compose --env-file "${ENV_FILE}" "${COMPOSE_FILES[@]}" rm -f pipeline pipeline_scheduler || true

echo "[4/6] Start lightweight stack"
docker compose --env-file "${ENV_FILE}" "${COMPOSE_FILES[@]}" up -d --build postgres dashboard proxy

if [[ "${RUN_PIPELINE}" == "true" ]]; then
    echo "[5/6] Run manual pipeline"
    docker compose --env-file "${ENV_FILE}" "${COMPOSE_FILES[@]}" run --rm pipeline
else
    echo "[5/6] Skip manual pipeline"
fi

echo "[6/6] Status"
docker compose --env-file "${ENV_FILE}" "${COMPOSE_FILES[@]}" ps

APP_DOMAIN="$(python3 - <<'PY'
from pathlib import Path

domain = ""
for line in Path(".env.prod").read_text(encoding="utf-8").splitlines():
    if line.startswith("APP_DOMAIN="):
        domain = line.split("=", 1)[1].strip().strip('"').strip("'")
        break
print(domain)
PY
)"

if [[ -n "${APP_DOMAIN}" ]]; then
    echo "Quick HTTP check: https://${APP_DOMAIN}"
    curl -I --max-time 20 "https://${APP_DOMAIN}" || true
fi

echo "Done."
