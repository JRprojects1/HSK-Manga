#!/usr/bin/env python3
"""Batched chapter TTS for the manga pipeline (called by src/audio-generator.ts).

Reads a job as JSON on stdin, synthesizes every bubble in batched edge-tts
calls with word-boundary metadata, aligns the word events back to each
bubble's phrase segmentation, and prints one JSON result on stdout.
Progress and diagnostics go to stderr.

Requires:  pip install "edge-tts>=7"   (7.x adds the `boundary` parameter)

stdin:
{
  "outDir": "<dir for mp3s + sidecars>",
  "voice": "zh-CN-XiaoyiNeural",
  "batchChars": 350,
  "concurrency": 6,
  "items": [{"key": "p0b1", "text": "...", "phraseLens": [1, 2, ...]}, ...]
}

stdout:
{
  "ok": true, "batches": 7, "cachedBatches": 0, "wall": 11.1,
  "items": [{"key": "p0b1", "ok": true, "file": "b_<hash>.mp3",
             "ph": [[1.23, 1.56], null, ...]}, ...]
}
`ph` is parallel to phraseLens: [start, end] seconds within `file`, or null
for phrases with no voiced characters (punctuation, stripped decorations).

Design notes:
- Batches are CONTENT-ADDRESSED (hash of voice + bubble texts). Each mp3 gets
  a .json sidecar holding the raw word events, so re-running an unchanged
  chapter is pure cache hits and costs ~0s. Alignment is recomputed from the
  sidecar every run — it is pure math and may improve between versions.
- DECORATION CHARACTERS ARE STRIPPED before synthesis. The endpoint speaks
  them: a footnote '*' is read aloud as 星号. Stripped characters simply get
  no time span.
- Alignment consumes word-event characters against the bubble text, comparing
  BEFORE skipping unspeakable characters on either side — the TTS both
  vocalizes some symbols and may normalize others away.
- The endpoint streams each connection at ~28 hanzi/s, so wall time is
  ~ totalChars / (28 * concurrency) + ~2s. Concurrency 6 is deliberately
  gentle: this is an unofficial endpoint, and the pipeline hides the wall
  time behind the gloss LLM call anyway.
"""
import asyncio
import hashlib
import json
import os
import sys
import time
import unicodedata
from pathlib import Path

# NOTE: edge_tts is imported lazily inside synth_batch(). The import costs
# ~6s on Windows (measured; bare python starts in 0.2s), and a fully cached
# re-run never needs it — deferring it makes unchanged-chapter runs ~free.

# CRITICAL on Windows: piped stdin/stdout default to the locale codepage
# (cp1252) until PEP 686 lands in 3.15. Without this, the UTF-8 job JSON from
# Node decodes to mojibake — SELF-CONSISTENTLY: the TTS speaks the garbage,
# events echo the garbage, and alignment "succeeds" on gibberish audio.
sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")

SIDECAR_VERSION = 2  # v1 sidecars may predate the utf-8 stdin fix

# The service's boundary-event timeline runs EARLY relative to its own mp3
# stream by a near-constant offset. Measured over 205 bubbles / 27 batch
# files (energy-onset vs event offset): median 110ms, IQR 93-132, no drift
# with file position (~1 ms/s). Uncorrected, every word slice starts in the
# previous word's tail and stops before its own word finishes. Sidecars store
# RAW event times; this shift is applied on load, so it can be re-tuned
# without re-synthesis. Re-measure if the voice or service changes
# (scratchpad script: measure_drift.py — regenerate from this comment if
# lost: regress energy-onset-minus-span-start against file position).
SYNC = float(os.environ.get("TTS_SYNC_MS", "110")) / 1000.0
SENT_FINAL = "。！？!?"  # 。！？!?
# Characters the TTS reads out loud (or renders unpredictably) that are
# decoration, not dialogue. Stripped from synthesis input.
SILENT = set("*＊※☆★♪♫♡♥●◆■→←↑↓")


def unspeakable(ch):
    """Decoration marks, emoji, and stray surrogate halves — never synthesized.
    Comment-wall bubbles are full of emoji; the voice should read the words
    and skip the 💝. Non-BMP ideographs (CJK ext B+) are kept."""
    cp = ord(ch)
    return (ch in SILENT
            or 0xD800 <= cp <= 0xDFFF          # lone surrogate halves
            or 0x1F000 <= cp <= 0x1FAFF        # emoji & symbol blocks
            or 0x2600 <= cp <= 0x27BF          # misc symbols, dingbats
            or cp in (0xFE0F, 0x200D, 0x20E3))  # VS16, ZWJ, keycap


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def speakable(ch):
    return ch.isalnum()


def char_eq(a, b):
    return (unicodedata.normalize("NFKC", a).casefold()
            == unicodedata.normalize("NFKC", b).casefold())


def batch_hash(voice, texts):
    payload = json.dumps([SIDECAR_VERSION, voice, sorted(SILENT), texts], ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


# Isolated words ('w:' items) carry ~1s of audio each (word + list pause)
# regardless of character count. Packing them by chars made word batches ~3x
# the audio length of bubble batches, and the whole run waited on those
# stragglers. Words therefore pack by ITEM count, and never share a batch
# with bubbles.
WORD_BATCH_ITEMS = 24


def make_batches(items, batch_chars):
    batches, cur, chars = [], [], 0
    cur_words = False

    def flush():
        nonlocal cur, chars
        if cur:
            batches.append(cur)
        cur, chars = [], 0

    for it in items:
        is_word = it["key"].startswith("w:")
        if cur and is_word != cur_words:
            flush()
        cur_words = is_word
        if is_word:
            if len(cur) >= WORD_BATCH_ITEMS:
                flush()
        else:
            n = len(it["text"])
            if cur and chars + n > batch_chars:
                flush()
        cur.append(it)
        chars += len(it["text"])
    flush()
    return batches


def synth_text_for(item):
    """Synthesis input: decorations/emoji stripped, sentence-terminated."""
    t = "".join(ch for ch in item["text"] if not unspeakable(ch)).strip()
    if t and t[-1] not in SENT_FINAL:
        t += "。"
    return t


async def synth_batch(text, voice, retries=3):
    import edge_tts  # deferred: see note at top of file

    last = None
    for attempt in range(retries):
        try:
            comm = edge_tts.Communicate(
                text, voice, boundary="WordBoundary",
                connect_timeout=10, receive_timeout=30,
            )
            audio, events = bytearray(), []
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    audio.extend(chunk.get("data") or b"")
                elif chunk["type"] == "WordBoundary":
                    events.append([chunk["text"], chunk["offset"] / 1e7, chunk["duration"] / 1e7])
            if not audio:
                raise RuntimeError("empty audio")
            return bytes(audio), events
        except Exception as e:  # noqa: BLE001 - network path, retried
            last = e
            if attempt < retries - 1:
                await asyncio.sleep(0.8 * (attempt + 1))
    raise last


def align_item(target, events, ei):
    """Consume events starting at index ei to time-stamp `target`'s characters.

    Returns (times, next_ei, ok): times[i] = (start, end) for voiced
    characters, None otherwise. Characters in SILENT never match (they were
    stripped from synthesis input).
    """
    n = len(target)
    times = [None] * n
    cur = 0

    def rest_unvoiced(c):
        return all(not speakable(target[i]) or unspeakable(target[i]) for i in range(c, n))

    while cur < n:
        if rest_unvoiced(cur):
            return times, ei, True
        if ei >= len(events):
            return times, ei, False
        et, es, ed = events[ei]
        m = len(et)
        k = 0
        while k < m:
            if cur >= n:
                return times, ei, False  # event spills past this bubble
            tc = target[cur]
            # Compare BEFORE skipping: the TTS vocalizes some symbols, so a
            # non-alnum target char can still be the right match.
            if not unspeakable(tc) and char_eq(tc, et[k]):
                times[cur] = (es + ed * k / m, es + ed * (k + 1) / m)
                cur += 1
                k += 1
            elif unspeakable(tc) or not speakable(tc):
                cur += 1  # unvoiced in synthesis input; no time span
            elif not speakable(et[k]):
                k += 1  # TTS-invented char (normalization); skip it
            else:
                return times, ei, False
        ei += 1
    return times, ei, True


def align_batch(batch, events):
    """Returns {key: times|None} for every item in the batch."""
    out = {}
    ei = 0
    for idx, item in enumerate(batch):
        times, ei2, ok = align_item(item["text"], events, ei)
        out[item["key"]] = times if ok else None
        ei = ei2
        if not ok:
            # Resync: find the event that starts the next bubble.
            if idx + 1 < len(batch):
                nxt = synth_text_for(batch[idx + 1])
                first = next((c for c in nxt if speakable(c)), None)
                if first is not None:
                    for j in range(ei, min(ei + 40, len(events))):
                        if events[j][0] and char_eq(events[j][0][0], first):
                            ei = j
                            break
    return out


def phrase_spans(times, phrase_lens):
    """Fold per-char times into per-phrase [start, end] (or None)."""
    spans = []
    idx = 0
    for length in phrase_lens:
        chs = [t for t in times[idx:idx + length] if t]
        idx += length
        if chs:
            spans.append([round(min(c[0] for c in chs), 3),
                          round(max(c[1] for c in chs), 3)])
        else:
            spans.append(None)
    return spans


async def main():
    job = json.load(sys.stdin)
    out_dir = job["outDir"]
    voice = job["voice"]
    items = job["items"]
    batches = make_batches(items, int(job.get("batchChars") or 350))
    sem = asyncio.Semaphore(int(job.get("concurrency") or 6))
    total_chars = sum(len(it["text"]) for it in items)
    log(f"  audio: {len(items)} bubbles / {total_chars} chars -> {len(batches)} calls")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    done = 0
    cached = 0

    async def run(batch):
        nonlocal done, cached
        h = batch_hash(voice, [it["text"] for it in batch])
        mp3 = out_path / f"b_{h}.mp3"
        sidecar = out_path / f"b_{h}.json"
        events = None
        hit = False
        if mp3.exists() and sidecar.exists():
            try:
                side = json.loads(sidecar.read_text(encoding="utf-8"))
                if side.get("v") == SIDECAR_VERSION:
                    events = side["events"]
                    hit = True
                    cached += 1
            except (OSError, ValueError, KeyError):
                events = None
        if events is None:
            text = "".join(synth_text_for(it) for it in batch)
            async with sem:
                audio, events = await synth_batch(text, voice)
            mp3.write_bytes(audio)
            sidecar.write_text(json.dumps({"v": SIDECAR_VERSION, "events": events}),
                               encoding="utf-8")
        done += 1
        log(f"  audio: batch {done}/{len(batches)}" + (" (cached)" if hit else ""))
        return batch, mp3.name, events

    t0 = time.perf_counter()
    settled = await asyncio.gather(*(run(b) for b in batches), return_exceptions=True)
    wall = time.perf_counter() - t0

    by_key = {}
    failed_batches = 0
    for i, res in enumerate(settled):
        if isinstance(res, BaseException):
            failed_batches += 1
            log(f"  audio: batch failed after retries: {res}")
            for it in batches[i]:
                by_key[it["key"]] = {"key": it["key"], "ok": False, "file": None, "ph": None}
            continue
        batch, fname, events = res
        # Sidecar times are raw; shift onto the actual audio timeline here.
        events = [[ev[0], ev[1] + SYNC, ev[2]] for ev in events]
        aligned = align_batch(batch, events)
        for it in batch:
            times = aligned[it["key"]]
            if times is None:
                by_key[it["key"]] = {"key": it["key"], "ok": False, "file": None, "ph": None}
            else:
                by_key[it["key"]] = {
                    "key": it["key"], "ok": True, "file": fname,
                    "ph": phrase_spans(times, it["phraseLens"]),
                }

    result = {
        "ok": failed_batches < len(batches),
        "batches": len(batches),
        "cachedBatches": cached,
        "failedBatches": failed_batches,
        "wall": round(wall, 2),
        "items": [by_key[it["key"]] for it in items],
    }
    misaligned = sum(1 for r in result["items"] if not r["ok"])
    if misaligned:
        log(f"  audio: {misaligned}/{len(items)} bubbles without audio (alignment/synthesis)")
    json.dump(result, sys.stdout)
    sys.stdout.flush()


asyncio.run(main())
