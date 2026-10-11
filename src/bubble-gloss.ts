/**
 * Per-bubble English glosses — a plain-English rendering of each speech bubble,
 * shown in the reader's word popup under "THIS BUBBLE" so a learner can check
 * their understanding of the whole line, not just the word they clicked.
 *
 * This is the one thing the rest of the pipeline cannot produce. Word popups
 * come from CC-CEDICT and chapter analysis; neither knows what a sentence means.
 *
 * Cost, measured on a 191-bubble chapter: 3.1k input tokens, 3.5k output,
 * 5 batched calls run concurrently, $0.0014 and 7.7s. Roughly doubles a
 * chapter's LLM bill (median is $0.0021) and its output tokens; in absolute
 * terms that is a third of a cent.
 *
 * About half the input tokens are the 589-char header repeated once per batch.
 * Left alone deliberately: deduplicating it means either one giant call (worse
 * failure blast radius) or a caching scheme, to save ~$0.0002.
 *
 * Deliberately NOT combined with per-word sense selection. That was prototyped
 * and measured: the dictionary already agrees with the LLM's pick on 481 of
 * 1,288 occurrences and the rest differ by length rather than correctness, so
 * the extra call buys conciseness, not accuracy. Combining also couples the
 * failure modes — one malformed JSON response would lose both the glosses and
 * the definitions for that batch.
 */

import pLimit from 'p-limit';
import { callLLMMultiTurn } from './simplifier';

/** Bubbles per LLM call. Smaller than the sense batches: whole-sentence output. */
const GLOSS_BATCH = 40;

/**
 * Batches in flight at once. Batches share nothing — each carries its own
 * numbering and writes disjoint keys — so this is bounded only by the provider.
 *
 * 8 rather than unbounded: Fireworks' 600 RPM is account-wide, and the server
 * can be running several jobs. A 1,000-bubble chapter would otherwise fire 25
 * calls simultaneously, and a 429 costs 30s of back-off — more than the
 * serialization it was trying to avoid. Any chapter up to 320 bubbles still
 * goes out in a single wave.
 */
const GLOSS_CONCURRENCY = 8;

export interface GlossTarget {
  /** Caller's key, echoed back on the result. */
  key: string;
  text: string;
}

interface CharacterEntry {
  name?: string;
  chineseName?: string;
  role?: string;
}

function stripFence(raw: string): string {
  return raw.replace(/^```(?:json)?/i, '').replace(/```\s*$/, '');
}

/**
 * Pull result rows out of a response, tolerating the two ways this has actually
 * failed rather than only the happy path.
 *
 * 1. Whole-array parse, as before.
 * 2. If that throws, recover object-by-object. A single malformed entry (or
 *    trailing prose containing a bracket, which breaks the outer slice) then
 *    costs one bubble instead of all forty.
 */
function extractRows(s: string): any[] {
  const start = s.indexOf('[');
  const end = s.lastIndexOf(']');
  if (start >= 0 && end > start) {
    try {
      const arr = JSON.parse(s.slice(start, end + 1));
      if (Array.isArray(arr) && arr.length) return arr;
    } catch { /* fall through to per-object recovery */ }
  }
  const rows: any[] = [];
  for (const m of s.matchAll(/\{[^{}]*\}/g)) {
    try { rows.push(JSON.parse(m[0])); } catch { /* skip this one only */ }
  }
  return rows;
}

// Key names drift. A batch once returned 715 completion tokens — a full, healthy
// response — and yielded 0/40 because every row failed the `n`/`t` lookup. Accept
// the obvious synonyms rather than discarding a whole batch over a schema whim.
const NUM_KEYS = ['n', 'index', 'i', 'id', 'num', 'number'];
const TEXT_KEYS = ['t', 'text', 'translation', 'en', 'english', 'gloss'];

function rowIndex(row: any): number {
  for (const k of NUM_KEYS) {
    const v = row?.[k];
    if (v !== undefined && v !== null) {
      const n = parseInt(v, 10);
      if (!Number.isNaN(n)) return n;
    }
  }
  return NaN;
}

function rowText(row: any): string {
  for (const k of TEXT_KEYS) {
    const v = row?.[k];
    if (typeof v === 'string' && v.trim()) return v.trim();
  }
  return '';
}

/** One retry per batch. At ~$0.0003 a batch, re-asking beats losing 40 bubbles. */
const MAX_BATCH_ATTEMPTS = 2;

async function glossBatch(
  items: GlossTarget[],
  title: string,
  nameMap: Record<string, string>,
  attempt = 1,
): Promise<Map<string, string>> {
  // Give the model the CORRESPONDENCE, not a bare list of English names. A flat
  // list leaves it to guess which Chinese name is which, and with several
  // similar transliterations it guesses wrong: a run with the flat list produced
  // "Paelman" and "Paidempus" for one character and "Saelen" and "Naelun" for
  // another, inside the same chapter.
  const pairs = Object.entries(nameMap).filter(([zh, en]) => zh && en && zh !== en);
  const nameLine = pairs.length
    ? 'Use exactly these names for these characters, and do not re-romanise them:\n'
      + pairs.map(([zh, en]) => `  ${zh} = ${en}`).join('\n') + '\n\n'
    : '';

  const prompt =
    `Comic chapter: ${title}\n\n${nameLine}` +
    'Give a natural English translation of each numbered speech bubble. Translate the WHOLE ' +
    'bubble as a sentence, not word by word. Keep it short and plain — this is a reading aid ' +
    'for a learner, not literary translation.\n\n' +
    `Return exactly ${items.length} entries, numbered 1 to ${items.length}, in order.\n` +
    'Reply with ONLY a JSON array: [{"n":1,"t":"..."}]\n\n' +
    // A sequential index, never the caller's key. Echoing a composite key back
    // proved unreliable in prototyping and one bubble received another bubble's
    // translation; the index is positional and cannot cross over.
    items.map((item, i) => `${i + 1}. ${item.text}`).join('\n');

  const out = new Map<string, string>();
  let raw = '';
  try {
    raw = await callLLMMultiTurn([{ role: 'user', content: prompt }], 0.2, 240000,
      undefined, undefined, 'hsk3', 'chinese', false);
  } catch (err) {
    // Distinguish a failed CALL from a failed PARSE — the old bare catch made
    // these indistinguishable in the logs, so a 0/40 batch was undiagnosable.
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`[Gloss]   batch call failed (attempt ${attempt}/${MAX_BATCH_ATTEMPTS}): ${msg}`);
    return attempt < MAX_BATCH_ATTEMPTS ? glossBatch(items, title, nameMap, attempt + 1) : out;
  }

  for (const row of extractRows(stripFence(raw))) {
    const n = rowIndex(row);
    const t = rowText(row);
    const item = items[n - 1];
    if (item && t) out.set(item.key, t);
  }

  if (out.size === 0) {
    // Log enough of the response to tell which failure this was next time.
    console.error(`[Gloss]   batch parsed 0/${items.length} from a ${raw.length}-char response `
      + `(attempt ${attempt}/${MAX_BATCH_ATTEMPTS}); head=${JSON.stringify(raw.slice(0, 200))}`);
    if (attempt < MAX_BATCH_ATTEMPTS) return glossBatch(items, title, nameMap, attempt + 1);
  } else if (out.size < items.length) {
    console.error(`[Gloss]   batch partial: ${out.size}/${items.length} parsed`);
  }
  // Whatever is still missing leaves those bubbles unglossed. The reader omits
  // the section when the field is absent, so this degrades rather than breaks.
  return out;
}

/**
 * Gloss every bubble. Returns key → English; missing keys simply have no gloss.
 * Never throws — a chapter with no glosses is still a usable chapter.
 */
export async function generateBubbleGlosses(
  targets: GlossTarget[],
  title: string,
  characters: CharacterEntry[] = [],
): Promise<Map<string, string>> {
  const glosses = new Map<string, string>();
  const usable = targets.filter(t => t.text && t.text.trim().length > 0);
  if (!usable.length) return glosses;

  // Names come from chapter analysis and nowhere else. Where it supplied both
  // forms — English-source chapters, where the English name is the source of
  // truth — the glosses are pinned to that spelling. Where it did not (Chinese
  // chapters put the Chinese form in `name` and omit an English one), the gloss
  // model romanises as it sees fit and consistency is not guaranteed across
  // chapters. That is accepted: inventing romanisations with an extra LLM call
  // just moves the inconsistency somewhere less visible and costs a round trip.
  const nameMap: Record<string, string> = {};
  for (const c of characters || []) {
    if (c?.chineseName && c?.name && c.chineseName !== c.name) nameMap[c.chineseName] = c.name;
  }
  const pinned = Object.keys(nameMap).length;
  console.error(`[Gloss] names pinned from analysis: ${pinned}/${(characters || []).length}`);

  const slices: GlossTarget[][] = [];
  for (let i = 0; i < usable.length; i += GLOSS_BATCH) {
    slices.push(usable.slice(i, i + GLOSS_BATCH));
  }
  console.error(`[Gloss] ${usable.length} bubbles in ${slices.length} call(s), ${GLOSS_CONCURRENCY} at a time`);

  // Concurrent, not sequential: measured 29.1s -> 7.7s on a 191-bubble chapter.
  // Merging after the fact is safe because keys are unique per bubble, so no
  // batch can overwrite another's result regardless of completion order.
  const limit = pLimit(GLOSS_CONCURRENCY);
  const results = await Promise.all(slices.map((slice, n) => limit(async () => {
    const got = await glossBatch(slice, title, nameMap);
    console.error(`[Gloss]   batch ${n + 1}/${slices.length}: ${got.size}/${slice.length}`);
    return got;
  })));
  for (const got of results) got.forEach((v, k) => glosses.set(k, v));
  return glosses;
}
