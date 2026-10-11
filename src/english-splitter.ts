// Split concatenated English chunks back into spaced words.
//
// Why this exists:
//   PaddleOCR is a Chinese-trained model. When it captures mixed Chinese+English
//   as a single line-level detection, it drops internal whitespace (Chinese has
//   no spaces, so the recognizer has no signal that ASCII letter sequences need
//   them). Manga bubbles like "[DUNGEON AND STONE]" come back as
//   "[DUNGEONANDSTONE]". This is upstream of our pipeline and unfixable
//   without retraining the model — so we patch it after the fact.
//
// Approach: word-frequency dynamic programming (the wordninja algorithm).
//   - Top-30k English words from Norvig's unigram corpus, ranked by frequency.
//   - Cost per word = log(rank * N) — common words are cheap, rare words pricey.
//   - For each input position, find the cheapest segmentation of the prefix.
//   - Walk the optimal path back to recover the word boundaries.
//   - O(n²) where n = chunk length. Tiny in practice (<1ms).
//
// Conservative by design — when in doubt we leave the input alone:
//   - Only fires on alphabetic chunks of length ≥ 6 (skips acronyms like NPC, RPG)
//   - Rejects splits where any word isn't in the dictionary
//   - Rejects splits where the average per-character cost exceeds an unknown-word
//     fallback (catches garbage strings from misread OCR)

import * as fs from 'fs';
import * as path from 'path';

let WORD_RANK: Map<string, number> | null = null;
let TOTAL_WORDS = 0;
let MAX_WORD_LEN = 0;

function loadWords(): void {
  if (WORD_RANK !== null) return;
  const wordsPath = path.join(__dirname, '..', 'data', 'english-words-30k.txt');
  if (!fs.existsSync(wordsPath)) {
    console.error(`[EnglishSplitter] data/english-words-30k.txt missing — splitter disabled`);
    WORD_RANK = new Map();
    return;
  }
  const lines = fs.readFileSync(wordsPath, 'utf-8').split(/\r?\n/);
  WORD_RANK = new Map();
  let maxLen = 0;
  lines.forEach((w, i) => {
    const trimmed = w.trim().toLowerCase();
    if (trimmed.length >= 2) {
      WORD_RANK!.set(trimmed, i + 1); // ranks 1-indexed
      if (trimmed.length > maxLen) maxLen = trimmed.length;
    }
  });
  TOTAL_WORDS = WORD_RANK.size;
  MAX_WORD_LEN = maxLen;
  console.error(`[EnglishSplitter] Loaded ${TOTAL_WORDS} words (max length ${MAX_WORD_LEN})`);
}

/**
 * Cost of using `word` as a segment. Lower = more likely to appear.
 *   common word ("and", rank 3): cost ≈ 11.5
 *   uncommon word ("dungeon", rank 14773): cost ≈ 19.7
 *   not-a-word: rejected outright (returns Infinity)
 */
function wordCost(word: string): number {
  if (!WORD_RANK) loadWords();
  const rank = WORD_RANK!.get(word);
  if (rank === undefined) return Infinity;
  return Math.log(rank * TOTAL_WORDS);
}

/**
 * Cost ceiling for an unknown-but-letters-only character: roughly the cost of a
 * very-rare-but-still-plausible word. Penalizes using non-words as fallback,
 * but doesn't infinitely penalize so a few unknown letters can pass through.
 */
function unknownCharCost(): number {
  if (!WORD_RANK) loadWords();
  return Math.log(TOTAL_WORDS * TOTAL_WORDS);
}

/**
 * Try to split a single concatenated alphabetic chunk into space-separated words
 * using word-frequency DP. Returns the spaced version on success, or null if no
 * confident segmentation found.
 *
 * Confidence rules:
 *   - Every produced word must be in the dictionary
 *   - Must produce ≥2 words (a single-word chunk shouldn't be split)
 *   - Average per-character cost must beat the unknown-word fallback
 */
export function tryWordSplit(chunk: string): string | null {
  if (!WORD_RANK) loadWords();
  if (WORD_RANK!.size === 0) return null;

  const lower = chunk.toLowerCase();
  const n = lower.length;

  // dp[i] = { cost, prevIdx } — cheapest way to segment prefix of length i
  const dp: Array<{ cost: number; prevIdx: number }> = new Array(n + 1);
  dp[0] = { cost: 0, prevIdx: -1 };

  for (let i = 1; i <= n; i++) {
    let best = { cost: Infinity, prevIdx: -1 };
    // Try every possible word ending at position i (start positions i-MAX_WORD_LEN..i-1)
    const minStart = Math.max(0, i - MAX_WORD_LEN);
    for (let j = minStart; j < i; j++) {
      if (dp[j].cost === Infinity) continue;
      const candidate = lower.substring(j, i);
      const w = wordCost(candidate);
      if (w === Infinity) continue;
      const total = dp[j].cost + w;
      if (total < best.cost) {
        best = { cost: total, prevIdx: j };
      }
    }
    dp[i] = best;
  }

  if (dp[n].cost === Infinity) return null;

  // Reject if average per-char cost exceeds the unknown-char fallback.
  // Catches garbage strings where DP found a segmentation by stringing together
  // rare/coincidental matches.
  const avgCost = dp[n].cost / n;
  if (avgCost >= unknownCharCost()) return null;

  // Walk back to recover words
  const segments: string[] = [];
  let cur = n;
  while (cur > 0) {
    const prev = dp[cur].prevIdx;
    segments.push(lower.substring(prev, cur));
    cur = prev;
  }
  segments.reverse();

  if (segments.length < 2) return null;

  // Sanity: average word length must be ≥3 chars. Garbage like "GLERPMOXTRIK"
  // can DP-segment into "GL ER PM OX TR IK" since rare 2-letter sequences may
  // exist in the dictionary; an avg-length floor rejects these without
  // banning legitimate 2-char words inside longer splits.
  const avgWordLen = n / segments.length;
  if (avgWordLen < 3) return null;

  // Restore original case from input. PaddleOCR usually returns all-uppercase
  // for ASCII text it captured, but mixed-case is possible too. We map the
  // lowercase split positions back to the original chars.
  const out: string[] = [];
  let pos = 0;
  for (const seg of segments) {
    out.push(chunk.substring(pos, pos + seg.length));
    pos += seg.length;
  }
  return out.join(' ');
}

/**
 * Main entry: scan a string for all-alphabetic runs of 6+ chars and try to
 * split each one. Leaves everything else (including short acronyms, mixed
 * Chinese, punctuation) untouched.
 *
 * Examples:
 *   "[DUNGEONANDSTONE]中"   → "[DUNGEON AND STONE]中"
 *   "我玩DUNGEONANDSTONE。"  → "我玩DUNGEON AND STONE。"
 *   "RPG"                  → "RPG"        (≤5 chars: skipped)
 *   "WEBTOON"              → "WEBTOON"    (single dict word: not split)
 *   "GLERPMOXTRIK"         → "GLERPMOXTRIK" (no valid segmentation: returned as-is)
 */
export function fixEnglishSpacing(text: string): string {
  if (!WORD_RANK) loadWords();
  if (WORD_RANK!.size === 0) return text;

  return text.replace(/[A-Za-z]{6,}/g, (chunk) => {
    const result = tryWordSplit(chunk);
    return result ?? chunk;
  });
}
