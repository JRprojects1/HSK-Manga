"""
Simple LaMa Inpainting Server

A lightweight Flask server wrapping simple-lama-inpainting for faster inpainting
compared to IOPaint (eliminates HTTP overhead by being a simpler wrapper).

Usage:
    pip install simple-lama-inpainting flask pillow
    python simple-lama-server.py --port 8090 --device cuda

The API is compatible with the existing TypeScript inpainting client.
"""

import argparse
import base64
import io
import threading
import time
from flask import Flask, request, jsonify, Response
from PIL import Image

app = Flask(__name__)

# Global model instance (loaded once at startup)
inpainter = None
device = "cuda"

# The model is shared across Flask's request threads. Serialize the GPU
# inference so concurrent requests can't race on it (the GPU runs kernels
# serially anyway, so this costs ~no throughput).
_inpaint_lock = threading.Lock()


def load_model(device_name: str):
    """Load the LaMa model"""
    global inpainter, device
    device = device_name

    print(f"Loading LaMa model on {device}...")
    start_time = time.time()

    from simple_lama_inpainting import SimpleLama
    inpainter = SimpleLama(device=device)

    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s")


def decode_base64_image(base64_str: str) -> Image.Image:
    """Decode a base64 string to PIL Image"""
    # Handle data URL format (data:image/png;base64,...)
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data))


@app.route("/api/v1/inpaint", methods=["POST"])
def inpaint():
    """
    Inpaint endpoint - compatible with IOPaint API.

    Expects JSON body with:
    - image: base64-encoded image
    - mask: base64-encoded mask (white = areas to inpaint)

    Returns the inpainted image as raw PNG bytes (Content-Type: image/png).
    The TS client reads the response body directly as image bytes, so this
    MUST NOT be wrapped in JSON.
    """
    try:
        start_time = time.time()

        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        if "image" not in data or "mask" not in data:
            return jsonify({"error": "Missing 'image' or 'mask' field"}), 400

        # Decode inputs
        image = decode_base64_image(data["image"])
        mask = decode_base64_image(data["mask"])

        # Ensure mask is grayscale
        if mask.mode != "L":
            mask = mask.convert("L")

        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        decode_time = time.time() - start_time

        # Run inpainting. Only the GPU inference is locked — decode (above) and
        # encode (below) stay outside it, so they still overlap across the
        # concurrent requests the pipeline sends.
        inpaint_start = time.time()
        with _inpaint_lock:
            result = inpainter(image, mask)
        inpaint_time = time.time() - inpaint_start

        # Encode result as raw PNG bytes (NOT JSON) — the TS client reads the
        # response body directly via response.arrayBuffer(), matching IOPaint.
        encode_start = time.time()
        out = io.BytesIO()
        result.save(out, format="PNG")
        png_bytes = out.getvalue()
        encode_time = time.time() - encode_start

        total_time = time.time() - start_time

        print(f"Inpaint: decode={decode_time*1000:.0f}ms, "
              f"inpaint={inpaint_time*1000:.0f}ms, "
              f"encode={encode_time*1000:.0f}ms, "
              f"total={total_time*1000:.0f}ms")

        # Per-request timing also exposed as headers for optional inspection.
        return Response(
            png_bytes,
            mimetype="image/png",
            headers={
                "X-Timing-Decode-Ms": str(round(decode_time * 1000)),
                "X-Timing-Inpaint-Ms": str(round(inpaint_time * 1000)),
                "X-Timing-Encode-Ms": str(round(encode_time * 1000)),
                "X-Timing-Total-Ms": str(round(total_time * 1000)),
            },
        )

    except Exception as e:
        print(f"Error during inpainting: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model": "lama",
        "device": device,
        "model_loaded": inpainter is not None
    })


@app.route("/", methods=["GET"])
def index():
    """Root endpoint with API info"""
    return jsonify({
        "name": "Simple LaMa Inpainting Server",
        "version": "1.0.0",
        "endpoints": {
            "POST /api/v1/inpaint": "Inpaint an image (JSON with base64 image and mask)",
            "GET /health": "Health check",
        }
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple LaMa Inpainting Server")
    parser.add_argument("--port", type=int, default=8090, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="Device to run model on")

    args = parser.parse_args()

    # Load model at startup
    load_model(args.device)

    print(f"\nStarting server on http://{args.host}:{args.port}")
    print(f"Device: {args.device}")
    print(f"API endpoint: POST /api/v1/inpaint")
    print()

    # Run Flask app
    app.run(host=args.host, port=args.port, threaded=True)
