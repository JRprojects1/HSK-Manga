/**
 * Chapter audio generation — batched synthesis with word-level timings.
 *
 * One mp3 per BATCH of speech bubbles (not per clip), plus a manifest mapping
 * every bubble to a time span inside its batch file, with per-phrase spans for
 * word playback and karaoke. Written into `<outputDir>/audio/`.
 *
 * Design notes worth keeping:
 *
 * - NO PER-WORD CLIPS. A tapped word plays its own span inside the bubble
 *   audio. That makes polyphones (得 de/děi, 过 guo/guò) and tone sandhi
 *   correct by construction — the TTS resolved them from sentence context,
 *   the same way `pinyin-context.ts` resolves the displayed reading — and it
 *   cuts synthesis work by ~two thirds. The old design's citation-form word
 *   library could not honor the reading its own cache key promised, because
 *   edge-tts accepts no pronunciation hints.
 *
 * - THE HEAVY LIFTING IS IN scripts/tts_batch.py. edge-tts is a Python
 *   library, and its word-boundary events plus the alignment logic were
 *   validated there first (see prototypes/word-slice-demo). This module only
 *   collects bubbles, spawns the helper once per chapter, and shapes the
 *   manifest. Swapping providers (Azure has native word boundaries) means
 *   swapping the helper, not this file.
 *
 * - STILL AN UNOFFICIAL ENDPOINT. Batching matters less for wall time than
 *   for call count: ~7 calls per chapter instead of ~575. Wall time itself is
 *   hidden by running concurrently with the gloss LLM call in the pipeline.
 *
 * - NOT WIRED IN BY DEFAULT. The pipeline only runs this when
 *   CHAPTER_AUDIO=1. A reader whose chapter has no manifest simply shows no
 *   audio controls; a bubble missing from the manifest just gets no speaker
 *   button.
 */
import * as fs from 'fs';
import * as path from 'path';
import { spawn } from 'child_process';
import type { ReaderData } from './image-processor';

export interface BubbleAudio {
  /** Batch mp3, relative to the audio dir. */
  file: string;
  /** Playable span of this bubble within the file, seconds. */
  s: number;
  e: number;
  /**
   * Per-phrase [start, end] spans, parallel to the block's `phrases` array.
   * null = unvoiced phrase (punctuation, stripped decoration marks).
   */
  ph: ([number, number] | null)[];
}

export interface WordAudio {
  file: string;
  s: number;
  e: number;
}

/**
 * Shared citation-clip library. A word's isolated pronunciation is
 * chapter-independent, so clips are synthesized ONCE into this store —
 * seeded from the HSK lists (scripts/seed-word-library.js), grown lazily by
 * each chapter's story-specific tail — and served to every chapter at
 * WORD_LIBRARY_ROUTE. `words` values hold bare file names within the dir.
 */
export interface WordLibraryIndex {
  format: 1;
  provider: string;
  voice: string;
  words: Record<string, WordAudio>;
}

export const WORD_LIBRARY_ROUTE = '/word-audio/';

export function defaultWordLibraryDir(): string {
  return process.env.WORD_LIBRARY_DIR || path.join(process.cwd(), 'data', 'word-library');
}

export interface AudioManifest {
  /** Bumped when the shape changes; the reader ignores formats it predates. */
  format: 2;
  provider: string;
  voice: string;
  bubbles: Record<string, BubbleAudio>;
  /**
   * Citation clips: each word synthesized ALONE, keyed by Simplified text.
   * Measured motivation: inside a sentence the TTS speaks short function
   * words 2-2.6× faster than in isolation (我 147ms vs 387ms) with
   * coarticulated edges — no slice of the sentence recording can sound like
   * a dictionary pronunciation, because that form was never articulated.
   * Words absent from this map (polyphones — see collectTargets) fall back
   * to the in-context slice in the reader, where playing the occurrence's
   * actual reading matters more than crispness.
   */
  words: Record<string, WordAudio>;
}

export interface GenerateAudioOptions {
  outputDir: string;
  voice?: string;
  /** Hanzi per batched call. Bigger = fewer calls, same throughput. */
  batchChars?: number;
  /** Concurrent edge-tts connections. Throughput is roughly 20-30 hanzi/s per
   *  connection (varies by day), so wall ≈ chars / (~25 × concurrency) + ~2s.
   *  Measured on a 3k-hanzi chapter: conc 6 ≈ 20s (leaks past the gloss
   *  window), conc 16 ≈ 10s (hides inside it). 16 is the default for that
   *  reason; the endpoint has tolerated it across every sweep so far. */
  concurrency?: number;
}

const CJK = /[一-鿿]/;

interface HelperItem {
  key: string;
  ok: boolean;
  file: string | null;
  ph: ([number, number] | null)[] | null;
}

interface HelperResult {
  ok: boolean;
  batches: number;
  cachedBatches: number;
  failedBatches: number;
  wall: number;
  items: HelperItem[];
}

/**
 * Texts whose ISOLATED reading may not match their common in-context reading
 * (particles read differently when standalone: 地 alone is dì, in text
 * usually de). These never get citation clips — the reader falls back to the
 * in-context slice, which is guaranteed to carry the displayed reading.
 */
const POLYPHONE_TEXTS = new Set(['地', '得', '着', '了']);

/** Every block worth speaking, with its phrase segmentation (in code points,
 *  matching Python's string indexing — .length would miscount surrogates),
 *  plus the chapter's citation-eligible word texts (resolved against the
 *  shared library rather than synthesized per chapter). */
function collectTargets(data: ReaderData) {
  const items: { key: string; text: string; phraseLens: number[] }[] = [];
  const readings = new Map<string, Set<string>>();

  data.pages.forEach((page, pi) => {
    (page.blocks || []).forEach((block: any, bi: number) => {
      const phrases: any[] = block.phrases || [];
      const text = phrases.map((p) => p.text).join('');
      if (!CJK.test(text)) return;
      items.push({
        key: `p${pi}b${bi}`,
        text,
        phraseLens: phrases.map((p) => Array.from(p.text as string).length),
      });
      for (const p of phrases) {
        if (p.pinyin && CJK.test(p.text)) {
          if (!readings.has(p.text)) readings.set(p.text, new Set());
          readings.get(p.text)!.add(p.pinyin);
        }
      }
    });
  });

  // Citation-safe = exactly one reading in this chapter AND not a text whose
  // standalone form is read differently. Everything else keeps the slice.
  const wordTexts: string[] = [];
  for (const [text, pys] of readings) {
    if (pys.size > 1 || POLYPHONE_TEXTS.has(text)) continue;
    wordTexts.push(text);
  }
  return { items, wordTexts };
}

function loadLibraryIndex(dir: string): WordLibraryIndex | null {
  try {
    return JSON.parse(fs.readFileSync(path.join(dir, 'index.json'), 'utf-8'));
  } catch {
    return null;
  }
}

/**
 * Ensure citation clips exist in the shared library for `texts`; synthesize
 * only the missing ones. Returns entries (bare file names) for every text it
 * can serve. Used by both the chapter pipeline and the HSK seed script.
 *
 * Not safe for concurrent writers — chapter jobs run one at a time
 * (jobService queue), which is what makes the read-merge-write below OK.
 */
export async function synthesizeWordsToLibrary(
  texts: string[],
  opts: { libraryDir: string; voice: string; onLog?: (msg: string) => void },
): Promise<Record<string, WordAudio>> {
  const { libraryDir, voice } = opts;
  fs.mkdirSync(libraryDir, { recursive: true });
  let index = loadLibraryIndex(libraryDir);
  if (index && index.voice !== voice) {
    throw new Error(
      `word library at ${libraryDir} was built with voice ${index.voice}, current is ${voice}; ` +
      're-seed it or point WORD_LIBRARY_DIR elsewhere');
  }
  if (!index) index = { format: 1, provider: 'edge-tts', voice, words: {} };

  const wanted = texts.filter((t) => CJK.test(t) && !POLYPHONE_TEXTS.has(t));
  const missing = wanted.filter((t) => !index!.words[t]);

  if (missing.length > 0) {
    if (opts.onLog) opts.onLog(`word library: synthesizing ${missing.length} new words`);
    const result = await runTtsHelper({
      outDir: libraryDir,
      voice,
      batchChars: Number(process.env.TTS_BATCH_CHARS) || 120,
      concurrency: Number(process.env.TTS_CONCURRENCY) || 16,
      items: missing.map((t) => ({ key: 'w:' + t, text: t, phraseLens: [Array.from(t).length] })),
    });
    for (const r of result.items) {
      if (!r.ok || !r.file || !r.ph || !r.ph[0]) continue;
      const span = r.ph[0];
      // Isolated words sit in ≥0.2s of true silence on both sides, so the
      // margins are safe and let the clip breathe instead of starting flush.
      index.words[r.key.slice(2)] = {
        file: r.file,
        s: Math.max(0, Math.round((span[0] - 0.06) * 1000) / 1000),
        e: Math.round((span[1] + 0.15) * 1000) / 1000,
      };
    }
    fs.writeFileSync(path.join(libraryDir, 'index.json'), JSON.stringify(index), 'utf-8');
  }

  const out: Record<string, WordAudio> = {};
  for (const t of wanted) {
    if (index.words[t]) out[t] = index.words[t];
  }
  return out;
}

function runTtsHelper(job: object): Promise<HelperResult> {
  return new Promise((resolve, reject) => {
    const script = path.resolve(__dirname, '..', 'scripts', 'tts_batch.py');
    // PYTHONUTF8: piped stdio on Windows Python defaults to the locale
    // codepage (cp1252), which mangles the UTF-8 job JSON — silently, because
    // the mojibake round-trips through synthesis and alignment consistently.
    // The helper also reconfigures its own stdin; both guards are deliberate.
    const proc = spawn('python', [script], {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, PYTHONUTF8: '1' },
    });
    let out = '';
    let errTail = '';
    proc.stdout.on('data', (d: Buffer) => { out += d.toString(); });
    proc.stderr.on('data', (d: Buffer) => {
      process.stderr.write(d); // live progress from the helper
      errTail = (errTail + d.toString()).slice(-2000);
    });
    proc.on('error', reject);
    proc.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`tts_batch.py exited ${code}: ${errTail.slice(-300)}`));
        return;
      }
      try {
        resolve(JSON.parse(out) as HelperResult);
      } catch {
        reject(new Error(`tts_batch.py produced unparseable output: ${out.slice(0, 200)}`));
      }
    });
    proc.stdin.write(JSON.stringify(job));
    proc.stdin.end();
  });
}

export async function generateChapterAudio(
  data: ReaderData,
  opts: GenerateAudioOptions,
): Promise<AudioManifest> {
  const voice = opts.voice || process.env.TTS_VOICE || 'zh-CN-XiaoyiNeural';
  const audioDir = path.join(opts.outputDir, 'audio');
  fs.mkdirSync(audioDir, { recursive: true });

  const { items, wordTexts } = collectTargets(data);
  const manifest: AudioManifest = { format: 2, provider: 'edge-tts', voice, bubbles: {}, words: {} };

  if (items.length > 0) {
    const result = await runTtsHelper({
      outDir: audioDir,
      voice,
      batchChars: opts.batchChars ?? (Number(process.env.TTS_BATCH_CHARS) || 120),
      concurrency: opts.concurrency ?? (Number(process.env.TTS_CONCURRENCY) || 16),
      items,
    });
    for (const r of result.items) {
      if (!r.ok || !r.file || !r.ph) continue;
      const spans = r.ph.filter((x): x is [number, number] => x !== null);
      if (spans.length === 0) continue;
      manifest.bubbles[r.key] = {
        file: r.file,
        s: Math.min(...spans.map((x) => x[0])),
        e: Math.max(...spans.map((x) => x[1])),
        ph: r.ph,
      };
    }
  }

  // Citation words come from the shared library: almost all of a chapter's
  // vocabulary is already there (HSK seed + previous chapters), so only the
  // novel tail is synthesized — into the library, where it benefits every
  // later chapter too. Failure leaves `words` empty and the reader falls
  // back to in-context slices.
  if (wordTexts.length > 0) {
    try {
      const lib = await synthesizeWordsToLibrary(wordTexts, {
        libraryDir: defaultWordLibraryDir(),
        voice,
        onLog: (m) => console.error('  ' + m),
      });
      for (const [text, w] of Object.entries(lib)) {
        manifest.words[text] = { file: WORD_LIBRARY_ROUTE + w.file, s: w.s, e: w.e };
      }
    } catch (err) {
      console.error(`  word library unavailable (${err instanceof Error ? err.message : String(err)}); word taps fall back to slices`);
    }
  }

  fs.writeFileSync(path.join(audioDir, 'manifest.json'), JSON.stringify(manifest), 'utf-8');

  // Batches are content-addressed and the simplifier is an LLM, so re-running
  // a chapter drifts the text and strands the previous run's batch files.
  // Sweep everything the manifest no longer references (audio/ holds only our
  // b_* batches, their sidecars, and the manifest).
  // (Word clips live in the shared library, not here — only bubble batches
  // occupy the chapter's audio dir.)
  const referenced = new Set(Object.values(manifest.bubbles).map((b) => b.file));
  for (const f of fs.readdirSync(audioDir)) {
    if (!f.startsWith('b_')) continue;
    const asMp3 = f.replace(/\.(mp3|json)$/, '.mp3');
    if (!referenced.has(asMp3)) {
      try { fs.unlinkSync(path.join(audioDir, f)); } catch { /* best-effort */ }
    }
  }
  return manifest;
}
