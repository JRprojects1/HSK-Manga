#!/bin/bash
# IOPaint Startup Script with CUDA Optimizations
# Run this on the machine that will host the server before starting IOPaint

# CUDA Environment Variables for better performance
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export OMP_NUM_THREADS="16"
export KMP_BLOCKTIME="1"

echo "CUDA environment variables set:"
echo "  PYTORCH_CUDA_ALLOC_CONF = $PYTORCH_CUDA_ALLOC_CONF"
echo "  OMP_NUM_THREADS = $OMP_NUM_THREADS"
echo "  KMP_BLOCKTIME = $KMP_BLOCKTIME"
echo ""

# Start IOPaint instances in background
echo "Starting IOPaint servers..."

# Server 1 on port 8081
iopaint start --model=lama --device=cuda --port=8081 --host=0.0.0.0 &
echo "Started IOPaint on port 8081"

# Server 2 on port 8082
iopaint start --model=lama --device=cuda --port=8082 --host=0.0.0.0 &
echo "Started IOPaint on port 8082"

echo ""
echo "Both IOPaint servers started. Waiting 15 seconds for model loading..."
sleep 15

# Warmup requests (A2)
echo ""
echo "Sending warmup requests..."

# Small test image (1x1 white pixel as base64)
TEST_IMAGE="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
TEST_MASK="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIA9n1wPAAAAABJRU5ErkJggg=="

curl -s -X POST http://localhost:8081/api/v1/inpaint \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"data:image/png;base64,$TEST_IMAGE\", \"mask\": \"data:image/png;base64,$TEST_MASK\"}" \
  --max-time 60 > /dev/null && echo "Warmup complete for port 8081" || echo "Warmup failed for port 8081"

curl -s -X POST http://localhost:8082/api/v1/inpaint \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"data:image/png;base64,$TEST_IMAGE\", \"mask\": \"data:image/png;base64,$TEST_MASK\"}" \
  --max-time 60 > /dev/null && echo "Warmup complete for port 8082" || echo "Warmup failed for port 8082"

echo ""
echo "IOPaint servers ready!"
echo "  - http://localhost:8081"
echo "  - http://localhost:8082"
echo ""
echo "Press Ctrl+C to stop servers"

# Keep script running
wait
