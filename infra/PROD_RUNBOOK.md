# Production Security Runbook

## 1) Secrets

- Do not use plain `.env` secrets in production.
- Preferred: inject secrets from a managed store (AWS Secrets Manager, GCP Secret Manager, Azure Key Vault, Vault).
- Supported file-based secret variables in app config:
  - `DB_PASSWORD_FILE`
  - `FOOTBALL_DATA_TOKEN_FILE`
  - `SUPABASE_DB_URL_FILE`
  - `STUDY_SUPABASE_DB_URL_FILE`

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
