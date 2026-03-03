# football-data-platform

End-to-end football data platform with API ingestion, ETL pipeline, PostgreSQL warehouse, data quality checks, and a Streamlit dashboard.

## Prerequisites

- Python 3.11
- Docker and Docker Compose
- `pip` for local Python tooling
- A `football-data.org` API token if you want to run the live API mode

## Setup

1. Copy the example environment file.
2. Fill in the variables you need.
3. Install local dependencies.

```powershell
Copy-Item .env.example .env
pip install -r requirements.txt
pip install pre-commit
pre-commit install
```

Important environment variables:

- `DATA_MODE` is the new canonical runtime mode. `PIPELINE_MODE` is still supported as a legacy alias.
- `PIPELINE_MODE=mock` runs the local mock payload.
- `PIPELINE_MODE=api` uses `football-data.org`.
- `DATA_MODE=csv` loads historical La Liga matches from `data/raw/*_cleaned.csv`.
- `DATA_MODE=hybrid` loads CSV history first, then merges the current live season from `football-data.org`.
- `LIVE_COMPETITION_CODES=PD,PL,SA,BL1,FL1` controls which live competitions are ingested for `api` and `hybrid`.
- `INCREMENTAL=true` limits API match extraction to the rolling `INCREMENTAL_DAYS` window.
- `DB_*` controls the local PostgreSQL connection for pipeline and dashboard.
- `DATABASE_URL` can override the computed PostgreSQL URL when needed.
- `ENRICH_LIVE=false` keeps the dashboard in DB-first mode with no mandatory external calls.
- `SUPABASE_DB_URL` and `STUDY_SUPABASE_DB_URL` are optional and only used by the FBref study tooling.

## Run With Docker

Start PostgreSQL first:

```powershell
docker compose up -d postgres
```

Initialize the schema:

```powershell
make init
```

Apply additive migrations on an existing database:

```powershell
make migrate
```

Run the pipeline container:

```powershell
docker compose run --rm pipeline python -m src.run_pipeline
```

The pipeline persists runtime timings and row volumes into `pipeline_run_log.metrics_jsonb` and `pipeline_run_log.volumes_jsonb`.

## CSV Historical Mode

`DATA_MODE=csv` expects player match-log files under `data/raw/*_cleaned.csv`.
The parser validates these minimum columns:

- `Date`
- `Comp`
- `Round`
- `Venue`
- `Result`
- `Squad`
- `Opponent`

Only `La Liga` rows are loaded in this mode. Match rows are deduplicated from player-level logs using:

`(season, comp, date, squad, opponent, venue)`

When the CSV does not contain a kickoff time, the pipeline stores `kickoff_utc` at `12:00:00Z` for that match date to keep the warehouse and dashboard usable.
After `fact_match` is loaded, the pipeline computes `fact_standings_snapshot` from finished matches and matchdays so the dashboard can render classement and position curves without calling the API.

## Hybrid Mode

`DATA_MODE=hybrid` is the recommended production mode when you want:

- historical seasons from `data/raw/*_cleaned.csv`
- the current top-league seasons from `football-data.org`

The pipeline transforms both sources, merges the warehouse tables, keeps the richer team metadata when the same club exists in both sources, then recomputes `fact_standings_snapshot` from the final `fact_match` table.

Start the dashboard:

```powershell
docker compose up -d dashboard
```

The dashboard is exposed on `http://localhost:9001` by default.
Pages:

- `Overview`: KPI globaux, classement courant, courbe de position, calendrier recent/prochain.
- `Team`: header equipe, forme 5/10 matchs, split domicile/exterieur, calendrier et courbe de classement.
- `Monitoring`: runs pipeline, durees/volumes, checks DQ par `run_id`.
- `Joueurs`: effectif en base pour le club filtre.
- `Live Leagues`: dernier classement disponible en base par competition chargee.

## Run Locally

Run the pipeline:

```powershell
python -m src.run_pipeline
```

Run the dashboard:

```powershell
streamlit run dashboard/app.py
```

## Tests And Code Quality

Run the local test suite:

```powershell
pytest -q
```

Run formatting and linting on the baseline scoped files:

```powershell
pre-commit run --all-files
```

## Useful Commands

```powershell
docker compose up -d
docker compose down
make study-fbref
make study-fbref-docker
```
