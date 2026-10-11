// Per-chapter text-only cache.
//
// Stores ONLY the bbox + simplified text per block. No images, no original
// text, no rendered HTML. Goal: legally-safe cache that lets us skip the
// expensive Phase 4b LLM simplification (~30s + ~$0.04 per chapter) AND
// skip the Phase 3 chapter analysis (~30s) on a cache hit.
//
// Why only bbox + simplifiedText:
//   - Original text is in the source manga site → caching it is copyright redistribution
//   - Inpainted images are derivative works → also a copyright issue
//   - Rendered reader.html embeds both → also copyright
//   - bbox + simplifiedText is the LLM-derived translation, which is the user's request
//     The downstream pipeline regenerates everything else fresh from the user's own
//     downloaded images.
//
// Cache layout on disk:
//   outputs/cache/{cacheKey}/
//     text-cache.json     { url, hskLevel, pipelineVersion, createdAt, pages: [...] }
//
// `cacheKey` = SHA-256 of `canonical(url) + ':' + hskLevel + ':' + PIPELINE_VERSION`.
//
// Pipeline version: bump this constant when pipeline output would change
// meaningfully (prompt edits, model changes, vocab updates). Bumping
// invalidates ALL existing cache entries.

import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';

// Bump when pipeline output changes meaningfully. Override via env for testing.
// v2: English OCR model routing (en_PP-OCRv5_mobile_rec) + tokenizer/definition
// fixes (Bug A: undefined-token decomposition, Bug B: KEEP-term glosses).
// v3: simplification model GLM-5.1 (replaced decommissioned DeepSeek V3.1) +
// GLM-tokenizer logit_bias maps — different vocab/phrasing output.
// v4: simplification model DeepSeek V4 Flash (~10x cheaper than GLM-5.1,
// comparable quality) + DeepSeek-V4 tokenizer logit_bias maps.
// v5: cluster-filter height cap (FILTER_HEIGHT_CAP_PX=2000). On long-strip
// images (e.g. asurascans serving a whole chapter as one 16000 px file), the
// proximity+size filter previously dropped most detected text because real
// 60 px dialogue collapsed to 0.375% of image height. Cap fixes that.
// Cached entries from v4 had near-empty text on long-strips; invalidating.
// v6: pre-merge image splitting (INPUT_SPLIT_THRESHOLD_PX=2500, target 2000).
// Source images taller than the threshold get divided into ~2000 px chunks
// before the existing page-merge logic. PaddleOCR previously crashed on
// >32,767 px images (OpenCV's cv::remap SHRT_MAX assertion — asurascans
// serves up to 48,000 px after pre-existing page merges stacked them) and
// degraded quality on anything >4000 px (internal downscale). Chunking
// before merge lets the merge logic re-stitch at panel gutters but cap each
// final processing unit closer to PaddleOCR's sweet spot. v5 cache entries
// for previously-broken pages are now stale; invalidating.
// v7: three English-source quality fixes.
//   (a) Solid-fill / inpaint mask padding now language-aware. English uses
//       SOLID_FILL_PADDING_SCALE_EN (default 0.005) instead of the Chinese
//       0.015 — English bboxes are naturally looser (Latin letters carry
//       their own whitespace from ascenders/descenders/baselines), so the
//       original scale over-expanded past speech-bubble outlines into
//       surrounding artwork. Even 0.008 was empirically still too much.
//   (b) Horizontal adjacency in groupRegionsIntoBlocks now uses
//       max(originalThreshold, avgHeightPct * 2). Very large stylized text
//       (SFX, splash panels) previously got over-segmented because the
//       flat threshold was too small for proportionally-larger inter-word
//       spacing. The × 2 compensates for the unit mismatch between
//       height-percent and width-percent on typical webtoon aspect ratios.
//   (c) English-input mode now runs a structure filter at the BLOCK level
//       (not per region): preliminary block grouping merges adjacent
//       regions, then any block whose JOINED text lacks EITHER whitespace
//       OR punctuation gets dropped. Catches UI labels ("BARBARIAN",
//       "INFORMATION"), UI grids of single words ("HUMAN BARBARIAN FAIRY"),
//       unpunctuated SFX ("BOOM", "CRASH"), bracketed titles
//       ("[BARBARIAN]"), and OCR-corrupted single tokens that got
//       artificially space-split by the English-spacing-fix
//       ("BARBAPTAN" → "BARB APT AN", "RMATION" → "RMA TION"). Per-region
//       checks were originally tried but misfired on PaddleOCR's
//       over-segmented multi-word dialogue. A looser "needs at least
//       space OR punctuation" rule was then tried but missed the
//       artificial-spacing and bracket failure modes. Joined-text +
//       AND-logic is the discriminator that handles all of these.
// v6 cache entries for English-source chapters captured the pre-fix output
// (over-large fills, split words, isolated UI translations); invalidating.
// v8: chained EN→CN simplification (translate first, then CN→CN simplify),
// per-chunk + merged KEEP from chapter analysis, unified positive/negative
// logit_bias map. All v7 entries reflect the older single-pass behavior.
// v9: CN chapter analysis prompt switched from "include EVERY noun" (T7
// aggressive) to strict-KEEP (T8): only preserve nouns that either are
// story-specific names/concepts or lack a simple HSK-target synonym. Everyday
// nouns like 袋子/晚餐/老公/表情/心情 now get paraphrased (包/晚饭/男人/样子/感觉)
// instead of forced through as HSK 4+ popup entries. v8 outputs have the
// aggressive KEEP profile; invalidating.
// v10: EN→CN switched to v4 two-stage pipeline (COMPOSE + STYLE REWRITE)
// with fragment rejoin, concept map, and KEEP repair. Replaces the single-pass
// EN path. CN→CN path unchanged. v9 EN entries would still work but reflect
// the old single-pass behavior — invalidate so new EN chapters pick up v4.
// v11: Reader now needs per-page imageWidth/imageHeight in data.json so the
// <img> tags can reserve correct vertical space (fixes scroll-to-bottom and
// eliminates layout shift). v10 caches are missing those fields; invalidate.
// Also swapped IntersectionObserver lazy loading for native loading="lazy"
// (fixes toggle-shows-blank-image bug from the sync gap between src and
// data-lazy-src). Reader code change; cache invalidation is defensive.
// v12: processMangaPageFromBuffer now deep-clones the cached OCR extraction
// on consumption instead of holding a shared reference. Each processing unit
// invokes the function three times (phase 4a scan, phase 4b inpaint, phase 4c
// render), and each pass mutated extraction.regions in-place — so phase 4a's
// structureless filter left the region set in a state that made phase 4c
// re-filter differently, producing off-by-one between the LLM input block
// array (built in 4a) and the render block array (in 4c). The symptom in
// cached v11 EN→CN chapters is dialogue blocks falling back to raw English
// on random pages (e.g. barbarian ch1 p15 shown as "I'M NOT SURE IF I'M
// EVEN GOING TO LIKE T", p100 "dungeon" untranslated). Invalidating so
// affected chapters re-render correctly.
// v13: two independent reasons, both affecting what this cache stores.
//
// Block geometry moved. 4cf12b1 fixed bubble/UI over-merging, 99fb69a fixed
// deleted bubbles and garbled UI blocks, 9e2faa3 replaced the per-region
// non-Chinese filter with a post-merge all-English-block drop, and 74d514f
// fixed the block-drop filters. matchCachedBlocks pairs fresh blocks to
// cached ones by center distance within MATCH_THRESHOLD (2.0 pct units). A
// block that no longer covers the same text but still centers within that
// radius — an un-merged pair, a re-split bubble — silently receives the old
// entry's simplifiedText. Failing to match is harmless (it re-simplifies);
// matching the wrong block is not.
//
// Cached definitions predate the dictionary rewrite. 627e585 rebuilt
// src/dictionary.ts specifically because pinyin came from one CC-CEDICT
// entry while the definition came from another. Entries written before it
// are wrong in exactly the way that commit fixed, so serving them from
// cache reintroduces the bug on every hit. Same for `keep` and
// `characters`, which e59733f changed when it narrowed the names toggle to
// real proper nouns.
// v14: language auto-detect overhaul. The old first-3-pages sample came back
// empty on chapters with textless openings and defaulted to chinese, so the
// all-English drop deleted every block of English chapters — those runs wrote
// near-empty cache entries (e.g. Hand Jumper Ep1 EN). Detection now samples
// pages spread across the chapter, rejects unsupported scripts, and a
// misdetection guard reruns as English when ~all blocks would be dropped.
// Invalidate so previously-poisoned empty entries get reprocessed.
//
// v16 (2026-08-13): CN OCR pool switched from PP-OCRv5_server to PP-OCRv6_medium
// (A/B verified: better accuracy + confidence calibration shift). Cached CN
// chapters were produced by the old model — invalidate so re-requests get
// v6-quality OCR. (EN pool unchanged — dedicated en model retained.)
//
// v17 (2026-08-29): font-size overhaul. renderedBlocks gains minFontSize /
// originalLineHeight (floor from the original lettering height) and the reader
// gained a measure-and-fit pass (forced line breaks, no round-down, no 0.92).
// Old cached readers render with the collapse-prone predict-and-shrink sizing
// and lack the per-block minimum — invalidate so re-requests pick up both.
//
// v18 (2026-08-30): v19 fit redesign + half-width quotes. v17's fullness
// preference let a width-pinning atomic group (KEEP terms like 好警察坏警察)
// inflate the line count into 1-2 character orphan towers. The fit is now a
// plain shrink-from-above: largest size whose NATURAL wrap fits (no layout
// search, no forced breaks — validated bubble-by-bubble on 196 bubbles, and
// agrees with the old search within 1px on 96% while never producing towers).
// compactPunctuation also converts curly quotes to ASCII (they were the one
// full-width pair it missed — a full em each, shrinking fonts and splitting
// names like 鞠炅 at line breaks; reader resolves open/close by parity).
// Readers stored under v17 have the tower layouts baked in — invalidate the
// (few) jobs processed in the hours v17 was live.
// v19: boundary merge now requires a box that straddles the seam (the old
// any-text-in-strip test merged whole watermarked chapters into one unit and
// lost their text); URL watermark tokens stripped from OCR text; oversized
// pages saved as PNG instead of failing silently. Chapters cached under v18
// from watermark-stamped sites carry the merged/empty results — regenerate.
export const PIPELINE_VERSION = process.env.PIPELINE_VERSION || '19';

const OUTPUTS_DIR = path.join(__dirname, '..', '..', 'outputs');
const CACHE_DIR = path.join(OUTPUTS_DIR, 'cache');

export interface CachedBlock {
  bbox: { x: number; y: number; width: number; height: number }; // percentages 0-100
  simplifiedText: string;
}

export interface CachedPage {
  pageNumber: number;
  blocks: CachedBlock[];
}

export interface CachedChapter {
  version: string;       // cache file format version (independent of pipelineVersion)
  url: string;
  hskLevel: string;
  pipelineVersion: string;
  createdAt: number;
  pages: CachedPage[];
  // Optional chapter-analysis derivatives — restore popup tooltip quality on
  // cache hit (role-based character definitions, contextual term glosses,
  // proper-noun tokenization). Deliberately excluded: setting + plot summaries
  // (synopsis-shaped content with copyright exposure).
  characters?: { name: string; role: string }[];
  keep?: string[];                                // proper nouns for tokenizer
  definitions?: { term: string; def: string }[]; // Map serialized as array
}

const CACHE_FILE_VERSION = '2'; // bumped from 1: characters/keep/definitions added

/**
 * Canonicalize a URL for cache keying. Strip tracking params (utm_*, ref, fbclid)
 * but keep functional params (chapter ID, episode number). Lowercase the host.
 */
export function canonicalizeUrl(rawUrl: string): string {
  try {
    const u = new URL(rawUrl);
    u.hostname = u.hostname.toLowerCase();

    const TRACKING_PARAMS = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content', 'fbclid', 'gclid', 'ref', 'referrer'];
    for (const p of TRACKING_PARAMS) {
      u.searchParams.delete(p);
    }

    u.hash = '';

    return u.toString();
  } catch {
    return rawUrl;
  }
}

/**
 * Cache key = SHA-256(canonical_url + ':' + hskLevel + ':' + inputLanguage + ':' + pipeline_version).
 *
 * `inputLanguage` is part of the key because Chinese-input simplification and
 * English-input translation produce different output for the same source URL.
 * Without it, the cache cross-contaminates between the two pipelines.
 */
export function getCacheKey(url: string, hskLevel: string, inputLanguage: string): string {
  const canonical = canonicalizeUrl(url);
  return crypto
    .createHash('sha256')
    .update(`${canonical}:${hskLevel}:${inputLanguage}:${PIPELINE_VERSION}`)
    .digest('hex');
}

export function getCachePath(url: string, hskLevel: string, inputLanguage: string): string {
  return path.join(CACHE_DIR, getCacheKey(url, hskLevel, inputLanguage));
}

/**
 * Look up a cached chapter result. Returns the parsed cache data on hit, null on miss.
 */
export function getCached(url: string, hskLevel: string, inputLanguage: string): { data: CachedChapter; cacheKey: string } | null {
  const cacheKey = getCacheKey(url, hskLevel, inputLanguage);
  const dir = path.join(CACHE_DIR, cacheKey);
  const cachePath = path.join(dir, 'text-cache.json');

  if (!fs.existsSync(cachePath)) {
    return null;
  }

  let data: CachedChapter;
  try {
    data = JSON.parse(fs.readFileSync(cachePath, 'utf-8'));
  } catch {
    return null;
  }

  // Defensive checks (cache key already encodes version; this catches manual corruption)
  if (data.pipelineVersion !== PIPELINE_VERSION || data.version !== CACHE_FILE_VERSION) {
    return null;
  }
  if (!Array.isArray(data.pages) || data.pages.length === 0) {
    return null;
  }

  return { data, cacheKey };
}

/**
 * Write the text-only cache for a finished chapter. Atomic-ish: write to .tmp
 * then rename, so partial writes don't leave a half-valid cache entry.
 *
 * `extras` holds chapter-analysis derivatives (characters/keep/definitions)
 * that restore popup tooltip quality on cache hit. Optional — entries written
 * without them still work, popups just fall back to dictionary lookups.
 */
export function commitTextCache(
  url: string,
  hskLevel: string,
  inputLanguage: string,
  pages: CachedPage[],
  extras?: {
    characters?: { name: string; role: string }[];
    keep?: string[];
    definitions?: Map<string, string> | { term: string; def: string }[];
  },
): void {
  if (pages.length === 0) {
    // Nothing useful to cache — refuse rather than write an empty entry that
    // would fail the load check.
    return;
  }

  const cacheKey = getCacheKey(url, hskLevel, inputLanguage);
  const cacheDir = path.join(CACHE_DIR, cacheKey);
  fs.mkdirSync(cacheDir, { recursive: true });

  // Normalize definitions: accept Map or array form, store as array (JSON-friendly).
  let definitions: { term: string; def: string }[] | undefined;
  if (extras?.definitions instanceof Map) {
    definitions = [...extras.definitions.entries()].map(([term, def]) => ({ term, def }));
  } else if (Array.isArray(extras?.definitions)) {
    definitions = extras.definitions;
  }

  const data: CachedChapter = {
    version: CACHE_FILE_VERSION,
    url,
    hskLevel,
    pipelineVersion: PIPELINE_VERSION,
    createdAt: Date.now(),
    pages,
    characters: extras?.characters,
    keep: extras?.keep,
    definitions,
  };

  const cachePath = path.join(cacheDir, 'text-cache.json');
  const tmpPath = cachePath + '.tmp';
  fs.writeFileSync(tmpPath, JSON.stringify(data));
  fs.renameSync(tmpPath, cachePath);
}

/**
 * Stats for admin/debugging.
 */
export function getCacheStats(): { entryCount: number; totalBytes: number } {
  if (!fs.existsSync(CACHE_DIR)) {
    return { entryCount: 0, totalBytes: 0 };
  }
  const entries = fs.readdirSync(CACHE_DIR, { withFileTypes: true }).filter(e => e.isDirectory());
  let totalBytes = 0;
  for (const entry of entries) {
    const cachePath = path.join(CACHE_DIR, entry.name, 'text-cache.json');
    if (fs.existsSync(cachePath)) {
      totalBytes += fs.statSync(cachePath).size;
    }
  }
  return { entryCount: entries.length, totalBytes };
}

/**
 * Match each fresh block bbox to the closest cached block bbox (greedy nearest-neighbor).
 * Returns simplifyIdx → simplifiedText for matched blocks; unmatched are omitted.
 *
 * Tolerance: bbox center distance ≤ MATCH_THRESHOLD (% of image). OCR is deterministic
 * on the same image, so close-to-zero distance is the norm; threshold mainly guards
 * against pathological rare drift.
 */
const MATCH_THRESHOLD = 2.0; // percentage units (Euclidean distance between centers)

export function matchCachedBlocks(
  freshBboxes: Array<{ x: number; y: number; width: number; height: number }>,
  cachedBlocks: CachedBlock[]
): Map<number, string> {
  const result = new Map<number, string>();
  const consumed = new Set<number>();

  // Pre-compute centers
  const cachedCenters = cachedBlocks.map(b => ({
    cx: b.bbox.x + b.bbox.width / 2,
    cy: b.bbox.y + b.bbox.height / 2,
  }));

  freshBboxes.forEach((freshBbox, simplifyIdx) => {
    const fcx = freshBbox.x + freshBbox.width / 2;
    const fcy = freshBbox.y + freshBbox.height / 2;

    let bestIdx = -1;
    let bestDist = Infinity;
    for (let i = 0; i < cachedBlocks.length; i++) {
      if (consumed.has(i)) continue;
      const dx = cachedCenters[i].cx - fcx;
      const dy = cachedCenters[i].cy - fcy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = i;
      }
    }

    if (bestIdx >= 0 && bestDist <= MATCH_THRESHOLD) {
      consumed.add(bestIdx);
      result.set(simplifyIdx, cachedBlocks[bestIdx].simplifiedText);
    }
  });

  return result;
}
