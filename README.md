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
- `INCREMENTAL=true` limits API match extraction to the rolling `INCREMENTAL_DAYS` window.
- `DB_*` controls the local PostgreSQL connection for pipeline and dashboard.
- `DATABASE_URL` can override the computed PostgreSQL URL when needed.
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

Start the dashboard:

```powershell
docker compose up -d dashboard
```

The dashboard is exposed on `http://localhost:9001` by default.

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
