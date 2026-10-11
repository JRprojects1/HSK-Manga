# Unified PaddleOCR server for HSK Manga.
#
# One process can host multiple recognition heads ("langs"), selected per
# request via an optional "lang" field on POST /ocr — or run one head per
# port exactly like the legacy deployment. The HTTP contract is byte-
# compatible with the legacy servers: {success, results:[{box,text,
# confidence}], image_width, image_height}.
#
# Heads:
#   cn  — Chinese/Japanese/(English) . v5: PP-OCRv5_server_det/rec (current prod)
#                                      v6: PP-OCRv6_medium_det/rec (also covers EN
#                                          + 46 Latin langs; needs paddleocr>=3.7.0)
#   en  — dedicated English: en_PP-OCRv5_mobile_rec (redundant once cn runs v6)
#   ko  — Korean: korean_PP-OCRv5_mobile_rec (needs paddleocr>=3.1.0; Korean is
#         NOT covered by base v5 or by v6 — this head is the only Korean path)
#
# Japanese has no head of its own: the cn head's models read Japanese natively.
#
# Examples:
#   python ocr_server.py --port 5000                        # cn on v5 == legacy prod
#   python ocr_server.py --port 5010 --ocr-version v6       # cn on v6, for A/B
#   python ocr_server.py --port 5002 --langs ko             # Korean head
#   python ocr_server.py --port 5000 --langs cn,ko --ocr-version v6   # consolidated
#
# Keep paddlepaddle-gpu at 3.2.2 — paddle 3.3.x is broken with paddleocr 3.7.0
# (ConvertPirAttribute2RuntimeAttribute crash, PaddleOCR issue #18162).

import os
os.environ['FLAGS_allocator_strategy'] = 'auto_growth'
os.environ['FLAGS_eager_delete_tensor_gb'] = '0.0'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

# Fix DLL loading for PaddlePaddle GPU on Windows
import sys
import importlib.util
nvidia_path = os.path.join(sys.prefix, 'Lib', 'site-packages', 'nvidia')
if os.path.exists(nvidia_path):
    for lib_dir in os.listdir(nvidia_path):
        bin_path = os.path.join(nvidia_path, lib_dir, 'bin')
        if os.path.exists(bin_path):
            os.add_dll_directory(bin_path)

import paddle
print(f"Paddle {paddle.__version__} on {paddle.device.get_device()}")

# Hide torch from modelscope (PaddleOCR doesn't need PyTorch)
_original_find_spec = importlib.util.find_spec
def _patched_find_spec(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        return None
    return _original_find_spec(name, *args, **kwargs)
importlib.util.find_spec = _patched_find_spec

from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
import paddleocr as paddleocr_pkg
import base64
import io
from PIL import Image
import traceback
import numpy as np
import argparse
import threading

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=5000)
parser.add_argument('--langs', type=str, default='cn',
                    help='Comma-separated heads to load: cn,en,ko (default: cn)')
parser.add_argument('--ocr-version', type=str, default='v5', choices=['v5', 'v6'],
                    help='Model generation for the cn head (en/ko are always v5 — no v6 variants exist)')
args, _ = parser.parse_known_args()

# Explicit model names everywhere (pinning): the library default changed to
# PP-OCRv6_medium in paddleocr 3.7.0, so never construct PaddleOCR() bare.
MODEL_CONFIGS = {
    'cn': {
        'v5': {'det': 'PP-OCRv5_server_det', 'rec': 'PP-OCRv5_server_rec'},
        'v6': {'det': 'PP-OCRv6_medium_det', 'rec': 'PP-OCRv6_medium_rec'},
    },
    'en': {
        'v5': {'det': 'PP-OCRv5_server_det', 'rec': 'en_PP-OCRv5_mobile_rec'},
    },
    'ko': {
        'v5': {'det': 'PP-OCRv5_server_det', 'rec': 'korean_PP-OCRv5_mobile_rec'},
    },
}

requested_langs = [l.strip() for l in args.langs.split(',') if l.strip()]
for lang in requested_langs:
    if lang not in MODEL_CONFIGS:
        print(f"ERROR: unknown lang '{lang}' (valid: {', '.join(MODEL_CONFIGS)})")
        sys.exit(1)

pipelines = {}
models_loaded = {}
for lang in requested_langs:
    version = args.ocr_version if args.ocr_version in MODEL_CONFIGS[lang] else 'v5'
    cfg = MODEL_CONFIGS[lang][version]
    print(f"Loading {lang} head ({version}: det={cfg['det']}, rec={cfg['rec']}) for port {args.port}...")
    pipelines[lang] = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device='gpu:0',
        text_detection_model_name=cfg['det'],
        text_recognition_model_name=cfg['rec'],
    )
    models_loaded[lang] = {'version': version, **cfg}

DEFAULT_LANG = requested_langs[0]

# One global lock: inference is GPU-serialized across all heads, matching the
# legacy per-process behavior.
inference_lock = threading.Lock()
print(f"OCR server ready on port {args.port}: heads={', '.join(requested_langs)} (default: {DEFAULT_LANG})")


@app.route('/ocr', methods=['POST'])
def run_ocr():
    try:
        data = request.json
        lang = data.get('lang', DEFAULT_LANG)
        ocr = pipelines.get(lang)
        if ocr is None:
            # Fall back to the default head instead of erroring: the Node client
            # historically sends EN to the CN pool when no EN pool is configured
            # (CN models read English passably; on v6 they read it excellently).
            # For ko this yields empty results — the Node side logs a loud
            # warning about the missing PADDLEOCR_URLS_KO in that case.
            print(f"WARNING: lang '{lang}' not loaded (loaded: {', '.join(pipelines)}) — falling back to '{DEFAULT_LANG}'")
            ocr = pipelines[DEFAULT_LANG]

        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_array = np.array(image)
        height, width = img_array.shape[:2]
        print(f"Processing image ({lang}): {width}x{height}")

        with inference_lock:
            result = ocr.predict(img_array)

        output = []
        if result and len(result) > 0:
            for page_result in result:
                if isinstance(page_result, dict):
                    texts = page_result.get('rec_texts', [])
                    scores = page_result.get('rec_scores', [])
                    # dt_polys pairing kept for byte-compatibility with the
                    # legacy servers. It aligns with rec_texts only because
                    # text_rec_score_thresh defaults to 0 (no filtering) —
                    # if a score threshold is ever set, switch to rec_polys.
                    polys = page_result.get('dt_polys', [])
                    print(f"Found {len(texts)} text regions")
                    for text, confidence, poly in zip(texts, scores, polys):
                        if not text or not text.strip():
                            continue
                        box = poly.tolist() if hasattr(poly, 'tolist') else poly
                        output.append({
                            'box': box,
                            'text': text,
                            'confidence': float(confidence)
                        })

        return jsonify({
            'success': True,
            'results': output,
            'image_width': width,
            'image_height': height
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'port': args.port,
        'langs': requested_langs,
        'default_lang': DEFAULT_LANG,
        'models': models_loaded,
        'paddleocr': getattr(paddleocr_pkg, '__version__', 'unknown'),
        'paddle': paddle.__version__,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=args.port)
