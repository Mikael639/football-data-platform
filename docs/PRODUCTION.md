# Guide Production

Ce guide decrit le lancement de la stack production basee sur `docker-compose.prod.yml`.

## Prerequis

- Docker Desktop (Docker Compose)
- Fichier `.env.prod` configure
- Secrets presents dans `secrets/`

## 1) Copiez le fichier d environnement production

Linux/macOS (bash/zsh) :

```bash
cp .env.prod.example .env.prod
```

Windows PowerShell :

```powershell
Copy-Item .env.prod.example .env.prod
```

## 2) Generez le hash Basic Auth

Linux/macOS (bash/zsh) :

```bash
docker run --rm caddy:2.8-alpine caddy hash-password --plaintext "change-me"
```

Windows PowerShell :

```powershell
docker run --rm caddy:2.8-alpine caddy hash-password --plaintext "change-me"
```

Renseignez ensuite la valeur dans `BASIC_AUTH_HASH` dans `.env.prod`.

## 3) Creez les secrets (ne les committez jamais)

- `secrets/postgres_superuser_password.txt`
- `secrets/db_writer_password.txt`
- `secrets/db_reader_password.txt`

## 4) Demarrez la stack production

Linux/macOS (bash/zsh) :

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d postgres dashboard proxy
```

Windows PowerShell :

```powershell
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d postgres dashboard proxy
```

## 5) Demarrez le scheduler pipeline

Linux/macOS (bash/zsh) :

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d pipeline_scheduler
```

Windows PowerShell :

```powershell
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d pipeline_scheduler
```

## Workflow Automatise (PowerShell)

Ces scripts sont PowerShell :

```powershell
powershell -ExecutionPolicy Bypass -File scripts/prod_prepare.ps1
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d postgres dashboard proxy
docker compose --env-file .env.prod -f docker-compose.prod.yml run --rm pipeline
powershell -ExecutionPolicy Bypass -File scripts/prod_smoke_test.ps1
```

## Liens

- Runbook detaille : [../infra/PROD_RUNBOOK.md](../infra/PROD_RUNBOOK.md)
- Guide backup : [BACKUP.md](BACKUP.md)
