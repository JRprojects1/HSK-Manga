# HSK Manga

Turn comics, webtoons, PDFs, and text into **graded Chinese reading practice (HSK 1–6)**.

Give it a manga/manhua/webtoon chapter (a URL, uploaded images, a PDF, or an HTML file) and a target HSK level. It reads the text off each page, rewrites it using only vocabulary at or below that level, paints the new text back onto the artwork, and gives you an interactive reader with pinyin tooltips and an original ↔ simplified toggle.

Source pages don't have to be Chinese. Chinese is **simplified in place**. **English, Japanese, and Korean** comics are **translated into leveled Chinese**, so learners can read series they already know.

> Built for learners who can read *some* Chinese but get stuck on native material. Instead of looking up every third character, they get the same story at a level they can actually read.

It runs two ways:

- **Web app**: paste a chapter URL (or upload pages), pick a level, and read the result in the browser.
- **CLI**: batch-process local images, PDFs, HTML, or plain text from the terminal.

---

## How it works

```
 input (URL / images / PDF / HTML / text)
        │
        ▼
 1. Download / extract pages             downloaders/, image-processor, html-processor
 2. OCR: find text + bounding boxes      text-extractor  (paddleocr | manga-ocr | gemini | claude)
 3. Detect source language               language  (chinese | english | japanese | korean)
 4. Traditional → Simplified             opencc-js
 5. Group regions into speech bubbles    manga-pipeline
 6. HSK analysis                         hsk-checker + tokenizer (segmentit / jieba)
 7. LLM simplify / translate to level    simplifier  (HSK-biased decoding)
 8. Inpaint original text away           inpainting  (optional server)
 9. Redraw new text on the page          text-replacer
10. Render interactive HTML reader       manga-pipeline
```

Plain-text input skips the vision steps and goes straight to analysis and simplification.

**Simplification** is the core of the project. The rewriter is an LLM steered toward the target HSK vocabulary at decode time. On providers that support it (Fireworks and DeepSeek), a **logit-bias map** raises the probability of in-level tokens and lowers out-of-level ones, so the model is held inside the learner's vocabulary instead of only being *asked* to stay there. A chapter-analysis pass first collects names and key terms to keep, and the bias on those tokens is lifted so proper nouns survive.

The bias maps in `data/` are built for the DeepSeek V4 tokenizer, and `scripts/build-bias-map.js` rebuilds them. The research behind the bias settings is written up in [docs/bias-variant-comparison.md](docs/bias-variant-comparison.md).

---

## Features

- **Many input types**: chapter URLs, uploaded images, image folders, PDFs, downloaded HTML, and raw text.
- **Four source languages**: Chinese (simplified in place) plus English, Japanese, and Korean (translated into leveled Chinese). Auto-detected or set explicitly.
- **Comic-site adapters**: WEBTOON, MangaDex, Tapas, ManhuaGui, Bilibili Manga, DMZJ, Kuaikan, and MANGA Plus, plus a generic fallback and federated search across sources (with AniList title resolution).
- **Level-aware rewriting**: target HSK 1–6, or several levels at once.
- **Traditional → Simplified** conversion built in.
- **Pluggable OCR**: self-hosted PaddleOCR / manga-ocr, or Gemini / Claude vision.
- **Inpainting**: the original text is removed and the new text is redrawn to fit the bubble.
- **Interactive reader**: pinyin tooltips, per-word dictionary info, tap-to-reveal sentence translations, flashcard export, and an original/simplified toggle.
- **Chapter audio** (optional): edge-tts narration synced to the reader.
- **Web app extras** (all optional): Google or email sign-in with a cross-device chapter library, per-IP daily quotas, a daily LLM spend cap, Cloudflare Turnstile bot protection, and an admin dashboard.

---

## Quick start (text only)

You need **Node.js 20+** and an LLM API key. [Fireworks](https://fireworks.ai) is what the project is tuned for.

```bash
git clone https://github.com/JRprojects1/<this-repo>.git
cd <this-repo>
npm install
npm run build
cp .env.example .env        # then put your key in FIREWORKS_API_KEY
node dist/index.js "随着全球化进程的不断加速，社会结构正在经历前所未有的重塑。" -l hsk2
```

Vocabulary statistics work without any API key:

```bash
node dist/index.js "今天天气很好，我们一起去公园散步吧。" -l hsk2 --stats
```

To run the web app, use `npm run start` and open http://localhost:3000.

## Full setup (comics and images)

Processing comic pages also needs **OCR**, and optionally **inpainting** and **audio**.

### 1. OCR

Pick one:

- **Gemini or Claude vision.** No extra server. Set `GEMINI_API_KEY` or `ANTHROPIC_API_KEY` and pass `-p gemini` or `-p claude`.
- **Self-hosted PaddleOCR.** Best quality, and what production uses. Run the bundled server with Python 3.10:

  ```bash
  pip install paddlepaddle-gpu==3.2.2 "paddleocr>=3.4" flask pillow numpy   # or paddlepaddle for CPU
  python scripts/ocr-server/ocr_server.py --port 5000                       # Chinese / Japanese / English
  python scripts/ocr-server/ocr_server.py --port 5002 --langs ko            # Korean
  ```

  Then set `PADDLEOCR_URLS=http://localhost:5000`, and `PADDLEOCR_URLS_KO=http://localhost:5002` if you want Korean input. List several URLs, comma-separated, to load-balance. On Windows, `scripts/start-paddleocr.ps1` starts four instances. This was developed on Windows with an NVIDIA GPU; the header of `ocr_server.py` has model and version notes.

### 2. Inpainting (optional)

Without inpainting, the original text is covered with a solid fill. For clean removal, run a LaMa server:

```bash
pip install simple-lama-inpainting flask pillow
python scripts/simple-lama-server.py --port 8090 --device cuda
```

Then set `ENABLE_INPAINTING=true` and `INPAINTING_SERVER_URLS=http://localhost:8090`. `scripts/start-iopaint.*` runs IOPaint instead.

### 3. Data files

Two large files aren't in the repo. The app runs without them, but some features get worse:

| File | What it's for | Where to get it |
| --- | --- | --- |
| `data/cedict_ts.u8` | Dictionary definitions in the reader | [CC-CEDICT](https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz). Unzip it and rename the file. |
| `data/dsv4-tokenizer.json` | Keeps names and key terms from being simplified away | `tokenizer.json` from the DeepSeek V4 model repository on Hugging Face |

### 4. Audio (optional)

Install Python and `pip install "edge-tts>=7"`. Set `CHAPTER_AUDIO=1`, then build the shared word-pronunciation library once:

```bash
node scripts/seed-word-library.js
```

### Other system notes

- PDF input uses `pdf-poppler`. On macOS or Linux you may need Poppler installed (`brew install poppler` or `apt install poppler-utils`).
- `puppeteer` downloads a copy of Chrome during `npm install`. It's only used for a few sites that build their pages with JavaScript.

---

## CLI usage

```bash
# Plain text → HSK level
node dist/index.js "今天天气很好，我们一起去公园散步吧。" -l hsk2
node dist/index.js input.txt -l hsk3 -o output.txt
echo "中文文本" | node dist/index.js -l hsk1

# Manga / manhua images (folder, single image, or PDF)
node dist/index.js manga ./pages/ -l hsk3 -p paddleocr
node dist/index.js manga page.png -a              # analyze only
node dist/index.js manga ./pages/ -l hsk2,hsk3 -e # multiple levels + English
node dist/index.js manga ./en-pages/ -l hsk3 --input-language english

# Downloaded HTML page
node dist/index.js html chapter.html -l hsk2 -p paddleocr

# Chapter from a URL
node dist/index.js webtoon "https://..." -l hsk3          # download + process
node dist/index.js webtoon "https://..." -d               # download only
```

During development, `npm run dev -- <args>` runs the CLI through `ts-node` without building.

| Option | Subcommands | Description |
| --- | --- | --- |
| `-l, --level` | all | Target HSK level, `hsk1`–`hsk6` (comma-separated list for `manga`) |
| `-o, --output` | all | Output file or directory |
| `-j, --json` | text | Output as JSON |
| `-s, --stats` | text | Show vocabulary statistics only |
| `-p, --provider` | manga, html, webtoon | OCR provider: `paddleocr`, `manga-ocr`, `gemini`, `claude` |
| `--no-convert` | manga, html, webtoon | Skip Traditional → Simplified conversion |
| `-e, --english` | manga | Include English translation |
| `-a, --analyze` | manga | Extract and show HSK stats without rewriting |
| `-s, --separate` | manga | One output file per page instead of a combined scroll |
| `--input-language` | manga | `auto`, `chinese`, `english`, `japanese`, `korean` |
| `-d, --download-only` | webtoon | Download images without processing |

---

## Web app

```bash
npm run start        # production (run npm run build first)
npm run dev:server   # development via ts-node
npm run test:smoke   # end-to-end smoke test against a running server (needs OCR + LLM)
```

The server listens on `http://localhost:3000`; set `PORT` to change it. Pages: `/` (landing), `/start` (process a chapter), `/about`, `/feedback`, `/blog/` and `/admin` (needs `ADMIN_TOKEN`). Finished readers are served from `/outputs/`.

Everything beyond an LLM key and OCR is optional and switches off cleanly when unset. Without Google or Resend keys the site works anonymously, and without Turnstile keys the bot check is skipped.

### API

| Method | Route | Purpose |
| --- | --- | --- |
| `POST` | `/api/process` | Queue chapter job(s): `{ urls, levels, inputLanguage }` |
| `POST` | `/api/upload-images` | Upload page images for processing |
| `GET` | `/api/status/:jobId` | Poll job progress |
| `GET` | `/api/search`, `/api/search/stream` | Federated search across comic sources |
| `GET` | `/api/chapter-list`, `/api/chapter-by-number`, `/api/webtoons-chapter-count` | Series and chapter lookup |
| `GET` | `/api/config`, `/api/eta`, `/api/health` | Runtime config, ETA estimate, health check |
| `GET`/`POST` | `/api/auth/*`, `/api/me`, `/api/library/*` | Optional sign-in and chapter library |
| `GET` | `/api/admin/stats`, `/api/admin/metrics` | Admin dashboard data (needs `ADMIN_TOKEN`) |

---

## Configuration

All settings are environment variables. [.env.example](.env.example) lists the useful ones with comments. The pipeline also reads many tuning settings from the environment (region grouping, filtering, text fitting, concurrency); the defaults are what production uses.

## Project structure

```
src/
  index.ts             CLI entry (commander)
  server.ts            Express server entry
  routes/              API + auth routes
  services/            jobs, quota, spend cap, cache, auth, library, email, metrics
  downloaders/         per-site adapters, federated search, AniList title lookup
  language.ts          source-language model + detection
  text-extractor.ts    OCR / vision providers
  hsk-checker.ts       HSK vocabulary analysis
  tokenizer.ts         Chinese segmentation
  simplifier.ts        LLM simplification/translation + HSK logit biasing
  manga-pipeline.ts    end-to-end image pipeline + reader rendering
  text-replacer.ts     redraw text onto pages
  inpainting.ts        text removal via inpainting server
  audio-generator.ts   edge-tts chapter audio
data/                  HSK word lists, logit-bias maps, thesaurus, English word lists
public/                web frontend
scripts/               OCR / inpainting servers, TTS helper, data builders, smoke test
docs/                  research write-ups (bias tuning, audio)
```

## License

[MIT](LICENSE) © 2026 Jerry Rao
