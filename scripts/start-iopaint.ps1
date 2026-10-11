# IOPaint Startup Script with CUDA Optimizations
# Run this on the machine that will host the server before starting IOPaint

# CUDA Environment Variables for better performance
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512"
$env:OMP_NUM_THREADS = "16"
$env:KMP_BLOCKTIME = "1"

Write-Host "CUDA environment variables set:"
Write-Host "  PYTORCH_CUDA_ALLOC_CONF = $env:PYTORCH_CUDA_ALLOC_CONF"
Write-Host "  OMP_NUM_THREADS = $env:OMP_NUM_THREADS"
Write-Host "  KMP_BLOCKTIME = $env:KMP_BLOCKTIME"
Write-Host ""

# Start IOPaint instances
Write-Host "Starting IOPaint servers..."

# Server 1 on port 8081
Start-Process -FilePath "iopaint" -ArgumentList "start", "--model=lama", "--device=cuda", "--port=8081", "--host=0.0.0.0" -NoNewWindow
Write-Host "Started IOPaint on port 8081"

# Server 2 on port 8082
Start-Process -FilePath "iopaint" -ArgumentList "start", "--model=lama", "--device=cuda", "--port=8082", "--host=0.0.0.0" -NoNewWindow
Write-Host "Started IOPaint on port 8082"

Write-Host ""
Write-Host "Both IOPaint servers started. Waiting 10 seconds for model loading..."
Start-Sleep -Seconds 10

# Warmup requests (A2)
Write-Host ""
Write-Host "Sending warmup requests..."

# Small test image (1x1 white pixel as base64)
$testImage = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
$testMask = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIA9n1wPAAAAABJRU5ErkJggg=="

$body = @{
    image = "data:image/png;base64,$testImage"
    mask = "data:image/png;base64,$testMask"
} | ConvertTo-Json

try {
    Invoke-RestMethod -Uri "http://localhost:8081/api/v1/inpaint" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 60 | Out-Null
    Write-Host "Warmup complete for port 8081"
} catch {
    Write-Host "Warmup failed for port 8081 (server may still be loading): $_"
}

try {
    Invoke-RestMethod -Uri "http://localhost:8082/api/v1/inpaint" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 60 | Out-Null
    Write-Host "Warmup complete for port 8082"
} catch {
    Write-Host "Warmup failed for port 8082 (server may still be loading): $_"
}

Write-Host ""
Write-Host "IOPaint servers ready!"
Write-Host "  - http://localhost:8081"
Write-Host "  - http://localhost:8082"
