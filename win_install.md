Install `mkcert` on Windows (for example with Chocolatey):

```powershell
choco install mkcert
```

Run the repo helper script from the project root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup-dev-https.ps1
```

If IP auto-detect fails:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup-dev-https.ps1 -LanIp 192.168.1.42
```

This script follows the same paths as the macOS helper:

- writes cert files to `web/certs/dev-cert.pem` and `web/certs/dev-key.pem`
- updates root `.env` (not `web/.env`) with:
  - `VITE_SIGNALING_BASE_URL=/api`
  - `VITE_API_PROXY_TARGET=http://api:8000`
  - `DEV_HTTPS=true`
  - `DEV_HTTPS_KEY_FILE=/app/certs/dev-key.pem`
  - `DEV_HTTPS_CERT_FILE=/app/certs/dev-cert.pem`

Then restart services:

```powershell
docker compose down
docker compose up --build
```

Open from phone:

```text
https://<your-ip>:5173
```
