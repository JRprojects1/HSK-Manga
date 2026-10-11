# PaddleOCR Startup Script
# Run this on the machine that will host the server before starting PaddleOCR servers
#
# Launches 4 PaddleOCR instances on ports 5000-5003 (Tier 1 scale).
# Uses 4GB / 12GB VRAM with 4 instances.
#
# Usage:
#   .\start-paddleocr.ps1                    # uses current directory for ocr_server.py
#   .\start-paddleocr.ps1 -OcrDir "C:\path"  # explicit directory containing ocr_server.py

param(
    [string]$OcrDir = (Get-Location).Path
)

Write-Host "Starting PaddleOCR servers from: $OcrDir"
Write-Host ""

if (-not (Test-Path (Join-Path $OcrDir "ocr_server.py"))) {
    Write-Host "ERROR: ocr_server.py not found in $OcrDir"
    Write-Host "Run from the directory containing ocr_server.py, or pass -OcrDir <path>"
    exit 1
}

$ports = @(5000, 5001, 5002, 5003)

foreach ($port in $ports) {
    Start-Process -FilePath "python" `
                  -ArgumentList "ocr_server.py", "--port", $port `
                  -WorkingDirectory $OcrDir `
                  -NoNewWindow
    Write-Host "  Started PaddleOCR on port $port"
}

Write-Host ""
Write-Host "All 4 servers launched. Waiting 15 seconds for model loading..."
Start-Sleep -Seconds 15

# Verify each is responsive (404 at root is expected — means server is up, just no handler at /)
Write-Host ""
Write-Host "Verifying..."
foreach ($port in $ports) {
    try {
        Invoke-WebRequest -Uri "http://localhost:$port" -UseBasicParsing -TimeoutSec 5 | Out-Null
        Write-Host "  Port $port: OK (responded 200)"
    } catch {
        if ($_.Exception.Message -like "*(404)*") {
            Write-Host "  Port $port: OK (404 at root, server up)"
        } else {
            Write-Host "  Port $port: FAILED - $($_.Exception.Message)"
        }
    }
}

Write-Host ""
Write-Host "PaddleOCR servers ready on ports: $($ports -join ', ')"
Write-Host "Don't close this window — closing kills all 4 server processes."
