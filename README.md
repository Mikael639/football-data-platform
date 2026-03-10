# football-data-platform

Plateforme data football prete a lancer en local : ingestion API, pipeline ETL, entrepot PostgreSQL, controles de qualite et dashboard Streamlit.

## Demarrage Rapide (5 min)

Objectif : lancer la plateforme et ouvrir le dashboard en quelques minutes.

Prerequis :

- Docker Desktop (inclut Docker Compose)
- Token `football-data.org` uniquement pour `DATA_MODE=api` ou `DATA_MODE=hybrid`

### Linux/macOS (bash/zsh)

```bash
git clone <URL_DU_REPO>
cd football-data-platform
cp .env.example .env
# Editez .env puis choisissez DATA_MODE=mock|csv|api|hybrid
docker compose up -d postgres
docker compose run --rm pipeline python -m src.run_pipeline
docker compose up -d --build dashboard
```

### Windows PowerShell

```powershell
git clone <URL_DU_REPO>
Set-Location football-data-platform
Copy-Item .env.example .env
# Editez .env puis choisissez DATA_MODE=mock|csv|api|hybrid
docker compose up -d postgres
docker compose run --rm pipeline python -m src.run_pipeline
docker compose up -d --build dashboard
```

Accedez au dashboard : `http://localhost:9001`.

Arretez la plateforme :

- Linux/macOS : `docker compose down`
- Windows PowerShell : `.\stop.ps1` (ou `docker compose down`)

Option rapide Windows PowerShell : `.\start.ps1` (force `DATA_MODE=hybrid`).

## Sommaire

- [Modes De Donnees](#modes-de-donnees)
- [Commandes Utiles](#commandes-utiles)
- [Variables Essentielles](#variables-essentielles)
- [Pages Du Dashboard](#pages-du-dashboard)
- [Tests Et Qualite](#tests-et-qualite)
- [Production Et Securite](#production-et-securite)

## Modes De Donnees

- `DATA_MODE=mock` : demo locale sans API
- `DATA_MODE=csv` : historique depuis `data/raw/*_cleaned.csv`
- `DATA_MODE=api` : donnees live via `football-data.org`
- `DATA_MODE=hybrid` : historique CSV + donnees live API

Si vous utilisez `api` ou `hybrid`, renseignez `FOOTBALL_DATA_TOKEN` dans `.env`.

## Commandes Utiles

Les commandes suivantes sont identiques sous Linux/macOS et Windows PowerShell.

Demarrez toute la stack :

```bash
docker compose up -d
```

Arretez toute la stack :

```bash
docker compose down
```

Demarrez le scheduler automatique :

```bash
docker compose up -d pipeline_scheduler
```

Lancez les migrations :

```bash
make migrate
```

Lancez l etude FBRef :

```bash
make study-fbref
```

## Variables Essentielles

- `FOOTBALL_DATA_TOKEN` : requis pour `api` et `hybrid`
- `LIVE_COMPETITION_CODES` : competitions live chargees
- `INCREMENTAL=true` : limite la fenetre API avec `INCREMENTAL_DAYS`
- `PIPELINE_INTERVAL_SECONDS` : frequence du scheduler (defaut `1800`, soit 30 min)
- `DB_*` / `DATABASE_URL` : connexion PostgreSQL
- `DASHBOARD_ADMIN_USERNAME` et `DASHBOARD_ADMIN_PASSWORD` : acces admin a `Monitoring`
- `ENRICH_PLAYER_STATS=true` : active l enrichissement joueur-match
- `PLAYER_STATS_PROVIDER=fbref|custom_http` : provider en mode `api`/`hybrid`
- `PLAYER_STATS_BASE_URL` : requis si `PLAYER_STATS_PROVIDER=custom_http` (endpoint attendu : `/player-match-stats`)
- `PLAYER_STATS_TOKEN` ou `PLAYER_STATS_TOKEN_FILE` : token optionnel pour `custom_http`
- `PLAYER_STATS_TIMEOUT_SEC` : timeout HTTP du provider (defaut `30`)

## Pages Du Dashboard

- `Overview` : KPI globaux, classement, courbe de position, calendrier
- `Team` : forme 5/10, split domicile/exterieur, calendrier, courbe
- `Players` : effectif du club filtre
- `Live Leagues` : lecture multi-ligues
- `Europe` : suivi UEFA (classement, calendrier, phases)
- `Monitoring` : runs pipeline, volumes, qualite
- `History` : historique des classements de fin de saison
- `Prediction` : baseline Poisson (1N2 + score probable)

`Monitoring` est reserve a l admin via `DASHBOARD_ADMIN_USERNAME` et `DASHBOARD_ADMIN_PASSWORD`.

## Tests Et Qualite

```bash
pytest -q
pre-commit run --all-files
```

Scans CI :

- `pip-audit` (vulnerabilites Python)
- `trivy` (filesystem scan)
- `dependabot` (mises a jour automatiques)
- `gitleaks` (secret scanning)

## Production Et Securite

Le projet est adapte a un usage local/dev. Pour un deploiement public, utilisez la stack de production dediee.

- Si vous etes bloques sur `OCI VM.Standard.E2.1.Micro`, utilisez le mode leger documente dans `docs/PRODUCTION.md` avec `docker-compose.prod.free.yml`.
- Guide production : [docs/PRODUCTION.md](docs/PRODUCTION.md)
- Guide backup chiffre : [docs/BACKUP.md](docs/BACKUP.md)
- Runbook detaille : [infra/PROD_RUNBOOK.md](infra/PROD_RUNBOOK.md)
