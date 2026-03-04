$ErrorActionPreference = "Stop"

Write-Host "Starting PostgreSQL..." -ForegroundColor Cyan
docker compose up -d postgres

Write-Host "Running pipeline in hybrid mode..." -ForegroundColor Cyan
docker compose run --rm -e DATA_MODE=hybrid pipeline python -m src.run_pipeline

Write-Host "Starting dashboard..." -ForegroundColor Cyan
docker compose up -d --build dashboard

Write-Host ""
Write-Host "Football Data Platform is available at http://localhost:9001" -ForegroundColor Green
