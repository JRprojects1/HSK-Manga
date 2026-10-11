# Audio — handoff

Everything about pronunciation audio: what ships, what was measured, what was
rejected, and what is left. Nothing is committed yet.

Last updated 2026-08-08, after the batched-synthesis rewrite.

---

## 1. State of the code

| File | Status | Role |
| --- | --- | --- |
| [scripts/tts_batch.py](scripts/tts_batch.py) | new | Does the real work. Batches text into few edge-tts calls, collects `WordBoundary` events, aligns them back to each phrase. |
| [src/audio-generator.ts](src/audio-generator.ts) | new | Drives the helper, owns the shared word library, writes the manifest. |
| [scripts/seed-word-library.js](scripts/seed-word-library.js) | new | One-shot HSK 1-4 (or `--hsk6`) library seed. |
| [src/manga-pipeline.ts](src/manga-pipeline.ts) | modified | Calls the generator after `data.json`. Gated on `CHAPTER_AUDIO=1`, try/catch so audio failure never fails a chapter. |
| [src/image-processor.ts](src/image-processor.ts) | modified | Reader playback: `playSeg`, the rAF loop that stops at segment end, karaoke off per-phrase spans. |
| [src/server.ts](src/server.ts) | modified | Mounts `/word-audio/` → `data/word-library`, 30d immutable. |
| [src/services/eventLogService.ts](src/services/eventLogService.ts) | modified | `audio_bubble` / `audio_word` in `ALLOWED_EVENTS`; props cap 8 → 12. |

### Architecture

Everything plays as a **span inside a batch mp3**. Nothing is one-clip-per-file.

```
bubble  p0b0 -> { file: "b_1bb2….mp3", s: 0.21, e: 2.51,
                  ph: [[0.21,0.348], [0.347,0.522], …, null] }
word    我   -> { file: "/word-audio/b_b4e1….mp3", s: 3.025, e: 3.623 }
```

`ph` holds per-phrase spans parallel to the block's `phrases` array; `null` means
unvoiced (punctuation). This is what drives karaoke — real timings now, not a
proportional estimate.

Bubble files are chapter-local (`audio/`). Word files come from the site-wide
shared library, referenced by absolute route so the CDN caches each clip once
across every chapter.

### Verified working (2026-08-08)

Sample chapter regenerated from scratch on Computer 1:

```
audio generated in 13.6s
  manifest format : 2
  bubbles         : 191
  words           : 380
  batch files     : 21 referenced, 21 on disk
  size            : 2.7 MB
  bubbles with no phrase spans : 0
```

Down from **580 files / 8.5 MB** under the old per-clip scheme.

Browser smoke test (headless Chrome, HTTP with Range support), targeting the
bubble furthest into its batch file:

```
target bubble : p94b0  span 23.37s -> 23.61s
playback ran  : 23.35s -> 23.84s
started at span : YES      stopped at span : YES
```

The 13.6s included synthesising 19 chapter-specific words. The word library was
already fully seeded (2,128 words across 181 batch files), so re-seeding is a
no-op at 0.1s.

### Environment

```bash
CHAPTER_AUDIO=1              # off by default; nothing generates without it
TTS_VOICE=zh-CN-XiaoyiNeural
TTS_BATCH_CHARS=120          # hanzi per call
TTS_CONCURRENCY=16
WORD_LIBRARY_DIR=            # defaults to data/word-library
```

Requires **`pip install "edge-tts>=7"`** — see §8.

---

## 2. Performance, and how it got there

Target was ≤10s. Landed at ~13.6s cold on a 3k-hanzi chapter, ~10s once the
word tail is cached.

The bottleneck was never synthesis, it was **per-call overhead**:

```
fixed cost per edge-tts call : ~2,638 ms
marginal cost                : ~34 ms per hanzi
```

A two-character word cost 2.6s, of which 2.57s was overhead. That is why an
early concurrency sweep looked like a plateau at ~1.8 clips/sec from concurrency
6 → 48 — it was measuring the wrong axis.

Two changes, both needed:

**(a) Words moved into a shared library.** They were 67% of the calls but only
21% of the hanzi, so removing them removes most of the call count.

**(b) Bubbles batched.** Measured sweep, bubbles only:

```
batch  conc  calls    wall
  350     6      7   13.5s
  200    10     12   10.7s
  120    16     20    5.5s   <- chosen
   80    24     31    5.3s
```

120/16 is the default. 24 and 32 barely improve on it while pushing much harder
on an unofficial endpoint.

For scale: one call per clip was **~319s** on the same chapter.

### Seeding the library

```
HSK 1-4:  1,987 words  ->  ~8 seconds, once
HSK 1-6:  5,301 words  ->  ~23 seconds, once
```

Coverage of the sample chapter is ~70% from HSK 1-4; the remaining quarter is
names and manga-specific vocabulary, which each chapter build adds lazily to the
same library.

---

## 3. How Chinese TTS actually works

Worth keeping, because it rules out one design and validates another.

**It is not concatenation.** Neural TTS reads the whole text, produces one
spectrogram for the entire utterance, and vocodes it in a single pass.

**So context changes a word.** Measured on identical characters in one call:

```
喜欢 mid-utterance 1.062s     喜欢 utterance-final 1.363s
```

~28% longer in final position. Plus contextual tone sandhi: 不 shifts bù→bú
before a 4th tone, and two 3rd tones in a row change the first (你好 = *ní* hǎo).

**But the two features want different things, which resolves it:**

- **Tap a word** → you want the *isolated citation form*. A library is not a
  compromise; it is more correct than slicing a word out of a sentence, where it
  would carry that sentence's intonation.
- **Play a bubble** → needs real sentence synthesis.

Hence: library for words, batched synthesis for bubbles. Do **not** build
sentences by gluing word clips.

> Correction to the previous version of this document, which said edge-tts gives
> no word timings. That was true of the 6.x line. **edge-tts 7.x accepts
> `boundary="WordBoundary"`** and returns per-word events for Chinese, which is
> what `tts_batch.py` aligns against. Hence the hard `>=7` requirement.

---

## 4. Designs considered

| # | Design | Verdict |
| --- | --- | --- |
| 1 | Web Speech API (browser TTS) | **Dead.** No Chinese voice in Chrome or Edge on Computer 1. |
| 2 | Syllable pack (~1,300 toned syllables, stitched per pinyin) | Never built. Superseded — sentences need real TTS anyway. |
| 3 | Per-word clips pre-baked at build time | Evolved into the shared library. |
| 4 | Per-bubble sentence TTS | **Shipped.** |
| 5 | Lazy server-side generation on demand | Not built. Still the answer if generation ever moves off the build. |
| 6 | Shared global word cache across chapters | **Shipped** as `data/word-library/`. |
| 7 | Non-blocking background generation | Not built. Now largely moot at ~13s. |
| 8 | Batched synthesis + offset playback | **Shipped.** |
| 9 | Pre-seeded HSK word library | **Shipped.** |

### Provider comparison

| Provider | Cost/chapter | Word timings | Notes |
| --- | --- | --- | --- |
| edge-tts | free | yes (7.x) | Unofficial Read Aloud endpoint, no SLA. **Current.** |
| Azure Speech | ~8.8¢ | yes | Bills each Chinese character as **two** — an early 4¢ estimate was wrong. |
| Gemini TTS | duration-billed | no | Quality worse than Azure for zh. |
| Kokoro (local) | free | no | Self-hosted. |

Chapters otherwise cost a fraction of a cent, so Azure would be a ~20x increase
in per-chapter cost. That is why edge-tts won despite having no SLA.

---

## 5. Decisions already made, and why

**Polyphones are excluded from the shared library.** `POLYPHONE_TEXTS` is
`地 得 着 了`, and any word whose readings differ across the chapter is skipped
too (`pys.size > 1`). A citation clip for 得 would be either `de` or `děi` and
wrong roughly half the time; better to show no word audio than the wrong
pronunciation. This replaces the older `text|pinyin` cache key, which solved the
same problem by storing both readings.

**Segment margins are asymmetric:**

```js
var SEG_LEAD = 0.02;        // batch sentences carry ~0.1s lead silence
var SEG_TAIL_WORD = 0.05;   // words sit flush against neighbours — stop tight
var SEG_TAIL_BUBBLE = 0.25; // let the last syllable decay
```

Isolated library words get wider margins still (−0.06 / +0.15) because they sit
in ≥0.2s of true silence on both sides.

**One `Audio` element, reused.** iOS only grants playback to the element that
was touched; a fresh `new Audio()` per clip works on desktop and silently fails
on a phone.

**`currentTime` is set before `play()`, but only once metadata has loaded** —
`readyState >= 1`, else on `loadedmetadata`. Seeking before that is a no-op.

**Audio is opt-in via `CHAPTER_AUDIO=1`.** Worth revisiting now that it is ~13s
rather than ~5 minutes.

---

## 6. Known issues

**The server must honour HTTP Range requests.** This is new and load-bearing.
Every clip is now a seek into a larger file, and Chrome cannot seek past what it
has buffered unless the server sends `206` with `Content-Range`. `express.static`
does this by default and Cloudflare does it for cached assets, so production is
fine — but a static host that does not, or any `file://` distribution, plays the
*wrong bubble* rather than failing loudly. Confirmed by testing: without Range,
clicking a bubble whose span starts at 23.37s played from 0s and ran on for
seconds.

**`file://` sharing is broken by design now.** Word clips are referenced by the
absolute route `/word-audio/…`, which has no meaning off a server.
`scripts/build-review-chapter.js` and the `inlineAudioManifest` option predate
this and will need rework if you want to share a zipped chapter again.

**Unofficial endpoint.** edge-tts has no SLA and concurrency 16 is assertive.

---

## 7. What's left

1. **Stage and commit.** ~370 files are untracked; only three are audio. See §8.
2. `pip install "edge-tts>=7"` on Computer 2.
3. `CHAPTER_AUDIO=1` on Computer 2.
4. **Run `node scripts/seed-word-library.js` on Computer 2** — `data/word-library/`
   is gitignored, so a fresh deploy serves an empty `/word-audio/` and word audio
   is silently absent until seeded.
5. Full chapter smoke test on Computer 2 (the pipeline machine).
6. Deploy, then **purge the Cloudflare cache** — mandatory, or visitors get
   unstyled pages.

Done in this pass: `.gitignore` rewritten (506 → ~370 untracked, 75 MB of
experiment dumps excluded), stale stores deleted (`data/word-audio` 4.8 MB,
`outputs/review-chapter` 57 MB, `outputs/simplified` 2.8 MB, 8 `.bak`/`.tmp`
files), sample chapter regenerated and verified.

---

## 8. Gotchas

- **`pip install edge-tts` is not enough** — `tts_batch.py` passes
  `boundary="WordBoundary"`, which only exists in 7.x. On a machine with 6.x
  cached it fails at runtime inside the pipeline's try/catch, silently producing
  no audio. Pin `"edge-tts>=7"`.
- **The directory is `data/word-library`, the URL is `/word-audio/`.** Not a
  typo; `WORD_LIBRARY_ROUTE` is the URL, `defaultWordLibraryDir()` is the disk
  path. `data/word-audio/` was the dead pre-batching store and has been deleted.
- **`generateInteractiveReaderHTML` is one enormous template literal.** A
  backtick anywhere inside it — including in a comment — breaks the build. This
  has bitten twice.
- **`recordEvent` silently drops unknown events.** Anything new must be added to
  `ALLOWED_EVENTS` or it vanishes with no error.
- **`offsetWidth` on a `display:none` element returns 0**, which silently breaks
  centring. Guard before measuring.
- **`python -m edge_tts`**, not the bare `edge-tts` script — on Windows the
  console script often lands outside PATH.
- Benchmark and probe scripts referenced here live in the session scratchpad,
  not the repo.
