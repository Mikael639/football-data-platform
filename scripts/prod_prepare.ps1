$ErrorActionPreference = "Stop"

param(
    [string]$Domain = "localhost",
    [string]$TlsEmail = "admin@example.com",
    [string]$BasicAuthUser = "admin",
    [string]$BasicAuthPassword = ""
)

function New-RandomSecret {
    param([int]$Length = 32)
    $chars = "abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789!@#$%*-_=+"
    $bytes = New-Object 'System.Byte[]' ($Length)
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $secret = -join ($bytes | ForEach-Object { $chars[$_ % $chars.Length] })
    return $secret
}

if (-not (Test-Path ".env.prod")) {
    Copy-Item ".env.prod.example" ".env.prod"
    Write-Host "Created .env.prod from .env.prod.example" -ForegroundColor Cyan
}

if (-not (Test-Path "secrets")) {
    New-Item -ItemType Directory -Path "secrets" | Out-Null
}

$superuserPath = "secrets/postgres_superuser_password.txt"
$writerPath = "secrets/db_writer_password.txt"
$readerPath = "secrets/db_reader_password.txt"
$footballDataTokenPath = "secrets/football_data_token.txt"
$supabaseDbUrlPath = "secrets/supabase_db_url.txt"
$studySupabaseDbUrlPath = "secrets/study_supabase_db_url.txt"

if (-not (Test-Path $superuserPath)) {
    Set-Content -Path $superuserPath -NoNewline -Value (New-RandomSecret)
}
if (-not (Test-Path $writerPath)) {
    Set-Content -Path $writerPath -NoNewline -Value (New-RandomSecret)
}
if (-not (Test-Path $readerPath)) {
    Set-Content -Path $readerPath -NoNewline -Value (New-RandomSecret)
}
if (-not (Test-Path $footballDataTokenPath)) {
    Set-Content -Path $footballDataTokenPath -NoNewline -Value ""
}
if (-not (Test-Path $supabaseDbUrlPath)) {
    Set-Content -Path $supabaseDbUrlPath -NoNewline -Value ""
}
if (-not (Test-Path $studySupabaseDbUrlPath)) {
    Set-Content -Path $studySupabaseDbUrlPath -NoNewline -Value ""
}

if ([string]::IsNullOrWhiteSpace($BasicAuthPassword)) {
    $BasicAuthPassword = New-RandomSecret -Length 20
}

$basicAuthHash = docker run --rm caddy:2.8-alpine caddy hash-password --plaintext "$BasicAuthPassword"
if ([string]::IsNullOrWhiteSpace($basicAuthHash)) {
    throw "Failed to generate Caddy basic auth hash."
}

$envLines = Get-Content ".env.prod"
$envLines = $envLines -replace "^APP_DOMAIN=.*", "APP_DOMAIN=$Domain"
$envLines = $envLines -replace "^TLS_EMAIL=.*", "TLS_EMAIL=$TlsEmail"
$envLines = $envLines -replace "^BASIC_AUTH_USER=.*", "BASIC_AUTH_USER=$BasicAuthUser"
$envLines = $envLines -replace "^BASIC_AUTH_HASH=.*", "BASIC_AUTH_HASH=$basicAuthHash"
Set-Content ".env.prod" $envLines

Write-Host ""
Write-Host "Production prep completed." -ForegroundColor Green
Write-Host "Domain: $Domain"
Write-Host "Basic auth user: $BasicAuthUser"
Write-Host "Basic auth password: $BasicAuthPassword" -ForegroundColor Yellow
Write-Host ""
Write-Host "Next steps:"
Write-Host "1) docker compose --env-file .env.prod -f docker-compose.prod.yml up -d postgres dashboard proxy"
Write-Host "2) docker compose --env-file .env.prod -f docker-compose.prod.yml run --rm pipeline"
