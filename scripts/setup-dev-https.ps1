param(
    [string]$LanIp
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = (Resolve-Path (Join-Path $scriptDir "..")).Path
$certDir = Join-Path $rootDir "web/certs"
$keyFile = Join-Path $certDir "dev-key.pem"
$certFile = Join-Path $certDir "dev-cert.pem"
$envFile = Join-Path $rootDir ".env"

function Get-LanIpAddress {
    $ips = @(
        [System.Net.Dns]::GetHostAddresses([System.Net.Dns]::GetHostName()) |
            Where-Object {
                $_.AddressFamily -eq [System.Net.Sockets.AddressFamily]::InterNetwork -and
                $_.IPAddressToString -ne "127.0.0.1" -and
                -not $_.IPAddressToString.StartsWith("169.254.")
            } |
            ForEach-Object { $_.IPAddressToString } |
            Select-Object -Unique
    )

    if ($ips.Count -gt 0) {
        return $ips[0]
    }

    return $null
}

if (-not (Get-Command mkcert -ErrorAction SilentlyContinue)) {
    Write-Host "mkcert is required but not installed."
    Write-Host "Install on Windows with: choco install mkcert"
    exit 1
}

if (-not $LanIp) {
    $LanIp = Get-LanIpAddress
}

if (-not $LanIp) {
    Write-Host "Could not auto-detect LAN IP."
    Write-Host "Run again with your LAN IP: powershell -ExecutionPolicy Bypass -File scripts/setup-dev-https.ps1 -LanIp 192.168.1.42"
    exit 1
}

New-Item -ItemType Directory -Path $certDir -Force | Out-Null

Write-Host "Installing local CA (mkcert -install)..."
& mkcert -install
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host ("Generating cert for localhost + {0}..." -f $LanIp)
& mkcert -key-file $keyFile -cert-file $certFile localhost 127.0.0.1 ::1 $LanIp
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$existing = @{}

if (Test-Path -Path $envFile) {
    foreach ($line in [System.IO.File]::ReadAllLines($envFile)) {
        if (-not $line) {
            continue
        }

        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#") -or -not $line.Contains("=")) {
            continue
        }

        $parts = $line.Split("=", 2)
        $existing[$parts[0].Trim()] = $parts[1].Trim()
    }
}

$existing["DEV_HTTPS"] = "true"
$existing["DEV_HTTPS_KEY_FILE"] = "/app/certs/dev-key.pem"
$existing["DEV_HTTPS_CERT_FILE"] = "/app/certs/dev-cert.pem"
$existing["VITE_SIGNALING_BASE_URL"] = "/api"

if (-not $existing.ContainsKey("VITE_API_PROXY_TARGET")) {
    $existing["VITE_API_PROXY_TARGET"] = "http://api:8000"
}

$orderedKeys = @(
    "VITE_SIGNALING_BASE_URL",
    "VITE_API_PROXY_TARGET",
    "DEV_HTTPS",
    "DEV_HTTPS_KEY_FILE",
    "DEV_HTTPS_CERT_FILE"
)

$outputLines = New-Object System.Collections.Generic.List[string]

foreach ($key in $orderedKeys) {
    $outputLines.Add("$key=$($existing[$key])")
}

$remainingKeys = $existing.Keys |
    Where-Object { $orderedKeys -notcontains $_ } |
    Sort-Object

foreach ($key in $remainingKeys) {
    $outputLines.Add("$key=$($existing[$key])")
}

$content = ($outputLines -join [Environment]::NewLine) + [Environment]::NewLine
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($envFile, $content, $utf8NoBom)

Write-Host ""
Write-Host "HTTPS dev setup complete."
Write-Host ("- Cert: {0}" -f $certFile)
Write-Host ("- Key:  {0}" -f $keyFile)
Write-Host ("- LAN URL: https://{0}:5173" -f $LanIp)
Write-Host ""
Write-Host "Next steps:"
Write-Host "1) docker compose down"
Write-Host "2) docker compose up --build"
Write-Host "3) Confirm Vite logs show https:// URLs"
