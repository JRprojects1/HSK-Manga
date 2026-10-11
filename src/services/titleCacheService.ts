// Per-series cache + call-serializer around the learner title resolver.
//
// WHY: resolveTitle() hits MangaDex + AniList. Every chapter of the same series
// would otherwise re-run those lookups — wasteful and a 429 risk, and it could
// even produce inconsistent titles across chapters. This caches the SERIES-level
// result (English name, cover, pinyin) keyed by a folded series name, so only
// the first chapter of a series pays the network cost and all chapters of a
// series render identically. Negative results are cached too (a series the DBs
// don't know stays a miss until the file is cleared).
//
// The chapter NUMBER is never cached — it's derived per-title by splitTitle().
//
// Storage: data/title-cache.json — atomic-write-via-tempfile, in-memory cache.
import * as fs from 'fs';
import * as path from 'path';
import { resolveTitle, splitTitle, seriesKey } from '../title-resolver';

const CACHE_PATH = path.join(__dirname, '..', '..', 'data', 'title-cache.json');

interface SeriesEntry {
  english: string | null;
  seriesPinyin: string;
  coverCdnUrl: string | null; // MangaDex CDN URL (downloaded per-job by the caller)
  source: string;
  confidence: string;
  ts: number;
}

/** What the pipeline needs to render + to fetch a cover for one chapter. */
export interface ResolvedChapterTitle {
  english: string | null;
  seriesZh: string;
  seriesPinyin: string;
  chapterLabel: string | null;
  coverCdnUrl: string | null;
}

let cache: Record<string, SeriesEntry> | null = null;

function load(): Record<string, SeriesEntry> {
  if (cache) return cache;
  try {
    if (fs.existsSync(CACHE_PATH)) {
      cache = JSON.parse(fs.readFileSync(CACHE_PATH, 'utf-8')) || {};
      return cache!;
    }
  } catch (e) {
    console.error('[TitleCache] Failed to load, starting fresh:', e);
  }
  cache = {};
  return cache;
}

function save(): void {
  if (!cache) return;
  try {
    fs.mkdirSync(path.dirname(CACHE_PATH), { recursive: true });
    const tmp = CACHE_PATH + '.tmp';
    fs.writeFileSync(tmp, JSON.stringify(cache, null, 2));
    fs.renameSync(tmp, CACHE_PATH);
  } catch (e) {
    console.error('[TitleCache] Failed to persist:', e);
  }
}

// Serialize resolveTitle calls process-wide. Jobs already run sequentially, but
// this guards against any future concurrency firing overlapping DB lookups.
let inFlight: Promise<unknown> = Promise.resolve();
function serialize<T>(fn: () => Promise<T>): Promise<T> {
  const run = inFlight.then(fn, fn);
  inFlight = run.then(() => undefined, () => undefined);
  return run;
}

/**
 * Resolve one raw chapter title to learner-friendly fields, using the per-series
 * cache. NO LLM fallback here (product decision: never show an unverified LLM
 * English title in the list — a DB miss degrades to hanzi + pinyin instead).
 */
export async function resolveChapterTitle(raw: string): Promise<ResolvedChapterTitle> {
  const { seriesZh, chapterLabel } = splitTitle(raw);
  const key = seriesKey(seriesZh);

  if (key) {
    const hit = load()[key];
    if (hit) {
      return {
        english: hit.english,
        seriesZh,
        seriesPinyin: hit.seriesPinyin,
        chapterLabel,
        coverCdnUrl: hit.coverCdnUrl,
      };
    }
  }

  // Miss → resolve (serialized, no LLM) and cache the series-level result.
  const r = await serialize(() => resolveTitle(raw));

  if (key) {
    load()[key] = {
      english: r.english,
      seriesPinyin: r.seriesPinyin,
      coverCdnUrl: r.coverUrl,
      source: r.source,
      confidence: r.confidence,
      ts: Date.now(),
    };
    save();
  }

  return {
    english: r.english,
    seriesZh: r.seriesZh,
    seriesPinyin: r.seriesPinyin,
    chapterLabel: r.chapterLabel,
    coverCdnUrl: r.coverUrl,
  };
}
