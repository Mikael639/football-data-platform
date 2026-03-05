$ErrorActionPreference = "Stop"

function Get-EnvValue {
    param(
        [string]$Path,
        [string]$Key
    )
    $line = Get-Content $Path | Where-Object { $_ -match "^$Key=" } | Select-Object -First 1
    if (-not $line) {
        return ""
    }
    return ($line -split "=", 2)[1]
}

if (-not (Test-Path ".env.prod")) {
    throw ".env.prod not found."
}

$user = Get-EnvValue ".env.prod" "BASIC_AUTH_USER"
$domain = Get-EnvValue ".env.prod" "APP_DOMAIN"
if ([string]::IsNullOrWhiteSpace($domain)) {
    $domain = "localhost"
}

$readerPasswordPath = "secrets/db_reader_password.txt"
if (-not (Test-Path $readerPasswordPath)) {
    throw "Missing $readerPasswordPath"
}

$readerPassword = (Get-Content $readerPasswordPath -Raw).Trim()

Write-Host "Checking stack status..." -ForegroundColor Cyan
docker compose --env-file .env.prod -f docker-compose.prod.yml ps

Write-Host "Checking unauthenticated access returns 401..." -ForegroundColor Cyan
$unauth = curl.exe -k -I "https://$domain" 2>$null
if ($unauth -notmatch "401 Unauthorized") {
    throw "Expected 401 on unauthenticated request."
}

if ([string]::IsNullOrWhiteSpace($user)) {
    throw "BASIC_AUTH_USER is empty in .env.prod."
}

$pwd = Read-Host "Enter Basic Auth password for smoke test"
Write-Host "Checking authenticated access returns 200..." -ForegroundColor Cyan
$auth = curl.exe -k -I -u "${user}:$pwd" "https://$domain" 2>$null
if ($auth -notmatch "200 OK") {
    throw "Expected 200 on authenticated request."
}

Write-Host "Checking read-only DB user cannot write..." -ForegroundColor Cyan
$insertAttempt = docker exec -e "PGPASSWORD=$readerPassword" football_postgres_prod psql -U dashboard_ro -d football_dw -c "INSERT INTO dim_date (date_id, year, month, day) VALUES ('2099-01-01', 2099, 1, 1);" 2>&1
if ($insertAttempt -notmatch "permission denied") {
    throw "Expected permission denied for dashboard_ro write attempt."
}

Write-Host "Smoke test passed." -ForegroundColor Green
