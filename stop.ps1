$ErrorActionPreference = "Stop"

Write-Host "Stopping Football Data Platform..." -ForegroundColor Cyan
docker compose down

Write-Host "All services stopped." -ForegroundColor Green
