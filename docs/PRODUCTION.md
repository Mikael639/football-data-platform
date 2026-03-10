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
- `secrets/football_data_token.txt`
- `secrets/supabase_db_url.txt`
- `secrets/study_supabase_db_url.txt`

Les trois fichiers base de donnees sont obligatoires.
Les fichiers `football_data_token.txt` et `supabase_db_url.txt` doivent exister si vous utilisez `docker-compose.prod.yml`.
Vous pouvez laisser `study_supabase_db_url.txt` vide si vous reutilisez la meme URL Supabase.

## 4) Utilisez OCI Vault pour les secrets applicatifs

Sur Oracle Cloud Infrastructure :

1. Creez un Vault puis des secrets pour :
   - `football-data-token`
   - `supabase-db-url`
   - `study-supabase-db-url` (optionnel)
2. Creez un Dynamic Group qui cible votre instance de production.
3. Ajoutez une policy autorisant ce Dynamic Group a lire `secret-family` dans le compartment des secrets.
4. Installez OCI CLI sur la VM.
5. Sur la VM, exportez les OCID des secrets puis lancez :

```bash
export FOOTBALL_DATA_TOKEN_SECRET_OCID="ocid1.vaultsecret.oc1..."
export SUPABASE_DB_URL_SECRET_OCID="ocid1.vaultsecret.oc1..."
export STUDY_SUPABASE_DB_URL_SECRET_OCID="ocid1.vaultsecret.oc1..."
bash scripts/oci_vault_sync.sh
```

Le script :

- telecharge les secrets depuis OCI Vault avec `OCI_CLI_AUTH=instance_principal`
- ecrit les valeurs dans `secrets/*.txt`
- vide les valeurs directes dans `.env.prod`
- force l usage de `/run/secrets/...` dans les conteneurs

## 5) Demarrez la stack production

Linux/macOS (bash/zsh) :

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d postgres dashboard proxy
```

Windows PowerShell :

```powershell
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d postgres dashboard proxy
```

## 6) Mode OCI Always Free (E2.1.Micro)

Si vous etes limites a `VM.Standard.E2.1.Micro`, utilisez l override :

- `docker-compose.prod.free.yml`

Objectif :

- garder `postgres + dashboard + proxy`
- desactiver le scheduler par defaut
- lancer le pipeline uniquement a la demande
- reduire la pression memoire sur PostgreSQL et Streamlit

Demarrage minimal :

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml -f docker-compose.prod.free.yml up -d postgres dashboard proxy
```

Pipeline manuel :

```bash
docker compose --env-file .env.prod -f docker-compose.prod.yml -f docker-compose.prod.free.yml run --rm pipeline
```

Scheduler uniquement si vous acceptez le risque de saturation :

```bash
docker compose --profile scheduler --env-file .env.prod -f docker-compose.prod.yml -f docker-compose.prod.free.yml up -d pipeline_scheduler
```

Commande unique sur la VM :

1. Creez une fois `secrets/oci_vault_ids.env` a partir de `scripts/oci_vault_ids.env.example`
2. Renseignez les `OCID` OCI Vault
3. Lancez :

```bash
bash scripts/prod_update_free.sh
```

Options :

```bash
bash scripts/prod_update_free.sh --skip-pipeline
```

Routine hebdomadaire recommandee sur `E2.1.Micro` :

1. `bash scripts/prod_update_free.sh`
2. Verifiez le dashboard sur `https://votre-domaine`
3. Ne lancez pas `pipeline_scheduler` en continu sur cette shape
4. N utilisez le scheduler que ponctuellement, ou pas du tout

## 7) Demarrez le scheduler pipeline

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
