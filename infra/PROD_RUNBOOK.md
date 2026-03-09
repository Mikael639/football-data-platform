# Production Security Runbook

## Go-live checklist

1. Run `make prep-prod` to generate/update `.env.prod`, secrets files and auth hash.
2. Review `.env.prod` and set real values (`APP_DOMAIN`, `TLS_EMAIL`).
3. Start stack with `make up-prod`.
4. Run ETL with `make run-prod-pipeline`.
5. Validate security controls with `make smoke-prod`.
6. Configure scheduled encrypted backups.

## 1) Secrets

- Do not use plain `.env` secrets in production.
- Preferred: inject secrets from a managed store (AWS Secrets Manager, GCP Secret Manager, Azure Key Vault, Vault).
- Supported file-based secret variables in app config:
  - `DB_PASSWORD_FILE`
  - `FOOTBALL_DATA_TOKEN_FILE`
  - `SUPABASE_DB_URL_FILE`
  - `STUDY_SUPABASE_DB_URL_FILE`

OCI Vault workflow:

1. Create three secrets in OCI Vault:
   - `football-data-token`
   - `supabase-db-url`
   - `study-supabase-db-url` (optional)
2. Create a Dynamic Group for the production instance.
3. Add a policy that allows this Dynamic Group to `read secret-family` in the compartment that contains the secrets.
4. Install OCI CLI on the VM.
5. Export the secret OCIDs on the VM and run `scripts/oci_vault_sync.sh`.
6. Restart the stack with `docker compose --env-file .env.prod -f docker-compose.prod.yml up -d --build`.

After sync, `.env.prod` is rewritten to blank direct values and prefer `/run/secrets/...` paths.

## 2) Network model

- Use `docker-compose.prod.yml`.
- PostgreSQL is only on the `internal` Docker network.
- Public exposure is only through the Caddy proxy (`80/443`).

## 3) Auth and TLS

- Caddy handles HTTPS certificates and Basic Auth.
- Configure:
  - `APP_DOMAIN`
  - `TLS_EMAIL`
  - `BASIC_AUTH_USER`
  - `BASIC_AUTH_HASH` (bcrypt hash)

## 4) Database least privilege

Roles created at DB bootstrap:

- `pipeline_rw`: read/write for ETL.
- `dashboard_ro`: read-only for dashboard.

Script: `infra/postgres/init/10_init_schema_and_roles.sh`

## 5) Container hardening

- Pipeline and dashboard images run as non-root user (`uid 10001`).
- Runtime capabilities dropped in production compose (`cap_drop: ALL`).
- `no-new-privileges` enabled.

## 6) Dependency and image scanning

- Dependabot config: `.github/dependabot.yml`
- Security workflow: `.github/workflows/security.yml`
  - `pip-audit` for Python dependency CVEs
  - `trivy` filesystem scan

## 7) Backups and restore drills

Scripts:

- `scripts/backup_postgres_encrypted.sh`
- `scripts/restore_postgres_encrypted.sh`

Run restore drills on a non-production environment on a schedule.

## 8) Logging and alerting

Recommended next step:

- Forward container logs to centralized storage (ELK/Loki/Datadog/CloudWatch).
- Set alerts on:
  - repeated pipeline failures
  - DQ check failures
  - backup job failures
  - certificate expiration and proxy errors
