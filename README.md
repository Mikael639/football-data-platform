# football-data-platform

Plateforme data football avec ingestion API, pipeline ETL, warehouse PostgreSQL, controles de qualite et dashboard Streamlit.

## Prerequisites

- Python 3.11
- Docker + Docker Compose
- `pip` (pour les outils locaux)
- Token `football-data.org` uniquement si tu utilises les modes live (`api`/`hybrid`)

## Quickstart (pour un ami)

1. Cloner le repo puis entrer dans le dossier.
2. Copier le fichier d environnement.
3. Choisir le mode de data (`csv`, `mock`, `api`, `hybrid`).
4. Lancer la stack.

```powershell
Copy-Item .env.example .env
pip install -r requirements.txt
.\start.ps1
```

URL dashboard: `http://localhost:9001`

## Choix du mode de donnees

- `DATA_MODE=mock`: demo locale sans API.
- `DATA_MODE=csv`: historique depuis `data/raw/*_cleaned.csv`.
- `DATA_MODE=api`: live uniquement via `football-data.org`.
- `DATA_MODE=hybrid`: historique CSV + live API (recommande si token dispo).

Variables importantes:

- `FOOTBALL_DATA_TOKEN`: requis pour `api` et `hybrid`.
- `LIVE_COMPETITION_CODES`: competitions live chargees.
- `INCREMENTAL=true`: limite la fenetre API via `INCREMENTAL_DAYS`.
- `PIPELINE_INTERVAL_SECONDS`: frequence auto du scheduler pipeline (defaut `1800` = 30 min).
- `DB_*` / `DATABASE_URL`: connexion PostgreSQL.
- `DASHBOARD_ADMIN_USERNAME` et `DASHBOARD_ADMIN_PASSWORD`: identifiants admin pour acceder a `Monitoring` et aux controles pipeline.
- `ENRICH_PLAYER_STATS=true`: active l enrichissement des stats joueur-match.
- `PLAYER_STATS_PROVIDER=fbref|custom_http`: provider utilise pour `fact_player_match_stats` en mode `api`/`hybrid`.
- `PLAYER_STATS_BASE_URL`: requis si `PLAYER_STATS_PROVIDER=custom_http` (endpoint attendu: `/player-match-stats`).
- `PLAYER_STATS_TOKEN` ou `PLAYER_STATS_TOKEN_FILE`: token optionnel transmis au provider `custom_http`.
- `PLAYER_STATS_TIMEOUT_SEC`: timeout HTTP du provider (defaut `30`).

## Token football-data.org

Chaque personne doit utiliser son propre token.

1. Creer un compte sur `https://www.football-data.org`.
2. Recuperer le token API depuis le dashboard/profile du compte.
3. Le mettre dans `.env`:

```env
FOOTBALL_DATA_TOKEN=ton_token_ici
DATA_MODE=hybrid
```

Sans token:

- utiliser `DATA_MODE=csv` ou `DATA_MODE=mock`.

## Lancement manuel Docker (alternative a `start.ps1`)

```powershell
docker compose up -d postgres
make init
docker compose up -d dashboard pipeline_scheduler
```

## Pages dashboard

- `Overview`: KPI globaux, classement, courbe de position, calendrier.
- `Team`: forme 5/10, split domicile/exterieur, calendrier, courbe.
- `Players`: effectif du club filtre.
- `Live Leagues`: lecture multi-ligues.
- `Europe`: suivi UEFA (classement, calendrier, phases).
- `Monitoring`: runs pipeline, volumes, qualite.
- `History`: historique des classements de fin de saison.
- `Prediction`: baseline Poisson (1N2 + score probable).

`Monitoring` est reserve a l admin via `DASHBOARD_ADMIN_USERNAME` + `DASHBOARD_ADMIN_PASSWORD`.

## Alertes automatiques (pipeline)

Config `.env`:

- `ALERTS_ENABLED=true|false`
- `ALERT_WEBHOOK_URL` (optionnel)
- `ALERT_SMTP_*` (optionnel)

Les alertes sont emises en cas de run failed, DQ fail/warn, anomalie de volume charge, ou fraicheur stale.

## Securite (important)

Projet adapte a un usage local/dev. Tel quel, ce n est pas un setup production internet.

Points deja en place dans ce repo:

- `.env` est ignore par git.
- Le token est lu via variables d environnement/fichier secret (pas hardcode dans le code).
- Un stack prod dedie existe: `docker-compose.prod.yml` (DB non exposee, proxy HTTPS, auth, users DB separes).

## Production (mode secure)

Runbook detaille: `infra/PROD_RUNBOOK.md`.

1. Copier l env prod:

```powershell
Copy-Item .env.prod.example .env.prod
```

2. Generer le hash Basic Auth:

```powershell
docker run --rm caddy:2.8-alpine caddy hash-password --plaintext "change-me"
```

Mettre le resultat dans `BASIC_AUTH_HASH` dans `.env.prod`.

3. Creer les secrets (jamais commits):

- `secrets/postgres_superuser_password.txt`
- `secrets/db_writer_password.txt`
- `secrets/db_reader_password.txt`

4. Demarrer la prod:

```powershell
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d postgres dashboard proxy
```

5. Lancer le scheduler pipeline (run auto toutes les 30 minutes):

```powershell
docker compose --env-file .env.prod -f docker-compose.prod.yml up -d pipeline_scheduler
```

6. Optionnel: lancer une pipeline one-shot:

```powershell
docker compose --env-file .env.prod -f docker-compose.prod.yml run --rm pipeline
```

### Workflow automatise (recommande)

```powershell
make prep-prod
make up-prod
make run-prod-pipeline
make smoke-prod
```

Scripts utilises:

- `scripts/prod_prepare.ps1`
- `scripts/prod_smoke_test.ps1`

### Ce que fait la stack prod

- PostgreSQL n expose pas le port `5432` vers l exterieur.
- Reverse proxy Caddy devant Streamlit avec HTTPS + Basic Auth.
- Deux users DB applicatifs:
  - `pipeline_rw`: read/write ETL
  - `dashboard_ro`: read-only dashboard
- Conteneurs applicatifs en non-root + `cap_drop: ALL`.

### Token football-data en prod

- Chaque personne/service doit utiliser son propre token.
- Ne jamais partager ton token personnel.
- Recommande: injection via secret manager.
- Supporte aussi un fichier secret avec `FOOTBALL_DATA_TOKEN_FILE`.

### Backup chiffre (prod)

Scripts fournis:

- `scripts/backup_postgres_encrypted.sh`
- `scripts/restore_postgres_encrypted.sh`

Exemple:

```bash
export BACKUP_PASSPHRASE='change-me'
./scripts/backup_postgres_encrypted.sh
```

## Tests et qualite

```powershell
pytest -q
pre-commit run --all-files
```

Scans securite CI:

- `pip-audit` (vulnerabilites Python)
- `trivy` (filesystem scan)
- `dependabot` (maj automatiques)
- `gitleaks` (secret scanning)
## Commandes utiles

```powershell
docker compose up -d
docker compose down
make migrate
make study-fbref
make study-fbref-docker
```
