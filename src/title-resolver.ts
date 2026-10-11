/**
 * Learner-friendly title resolver.
 *
 * A raw title scraped from a Chinese source is a wall of characters a beginner
 * can't parse, e.g. `變成蠻族！遊戲生存戰 - 第30話 戰士 (2)`. This turns it into
 * something readable:
 *
 *   - series name split off from the chapter marker (第30話 → "Ch. 30")
 *   - an English series title when we can find one CONFIDENTLY
 *   - a cover image for the series
 *   - pinyin for the Chinese series name (always available, offline)
 *
 * Resolution is a LADDER, best source first, so a miss never leaves the user
 * worse off than today:
 *
 *   1. MangaDex / AniList lookup — curated, human-authored English titles +
 *      cover art. Accepted ONLY on a confident (near-exact) title match, so we
 *      never show a confidently-wrong title.
 *   2. LLM gloss — optional, injected by the caller (the chapter-analysis LLM we
 *      already call). Guaranteed coverage for the long tail the databases miss.
 *   3. Pinyin — offline, always present, so even a total miss is more readable
 *      than raw hanzi.
 *
 * The chapter marker is handled deterministically in every case — no lookup
 * needed to turn 第30話 into "Ch. 30".
 */
import { pinyin } from 'pinyin-pro';
import * as OpenCC from 'opencc-js';
import { mangadexTitleLookup } from './downloaders/mangadex';
import { aniListLookup } from './downloaders/anilist';

// Traditional/Hong-Kong → Simplified, so a Traditional scraped title matches a
// Simplified DB entry (and vice-versa) after normalization.
const toSimplified = OpenCC.Converter({ from: 'tw', to: 'cn' });

export type ResolveSource = 'mangadex' | 'anilist' | 'llm' | 'none';
export type MatchConfidence = 'exact' | 'strong' | 'none';

export interface ResolvedTitle {
  /** The original scraped title, untouched. */
  raw: string;
  /** Chinese series name with the chapter marker + site cruft stripped. */
  seriesZh: string;
  /** Pinyin for seriesZh (tone marks). Always present. */
  seriesPinyin: string;
  /** Normalized chapter label, e.g. "Ch. 30", or null if none was found. */
  chapterLabel: string | null;
  /** Leftover chapter subtitle after the marker, e.g. "戰士 (2)". */
  chapterSubtitle: string | null;
  /** Resolved English series title, or null if nothing confident was found. */
  english: string | null;
  /** Series cover image URL, or null. */
  coverUrl: string | null;
  /** Which rung of the ladder produced `english`. */
  source: ResolveSource;
  /** How strong the DB title match was (for `mangadex`/`anilist` sources). */
  confidence: MatchConfidence;
  /** Every candidate we considered, for debugging / the preview UI. */
  candidates: TitleCandidate[];
}

export interface TitleCandidate {
  english: string | null;
  cover: string | null;
  source: 'mangadex' | 'anilist';
  /** The alt-title that matched the query, so a human can sanity-check. */
  matchedOn: string;
  confidence: MatchConfidence;
}

/** Optional LLM fallback: series name in → concise English gloss out (or null). */
export type LlmGloss = (seriesZh: string) => Promise<string | null>;

/**
 * The slim, persisted subset of a resolution — what the job, library entry, and
 * chapter-list UI actually carry. `coverUrl` here is the SERVED path once the
 * cover has been downloaded (e.g. /outputs/job_x/cover.jpg), not the CDN URL.
 */
export interface ResolvedTitleFields {
  english: string | null;
  seriesZh: string;
  seriesPinyin: string;
  chapterLabel: string | null;
  coverUrl: string | null;
}

/** Stable cache/dedupe key for a series name (script- and punctuation-folded). */
export function seriesKey(seriesZh: string): string {
  return normCJK(seriesZh);
}

// --- title splitting ---------------------------------------------------------

// Site suffixes that ride along on scraped <title>s. Anything from a separator
// to end-of-string once we hit one of these markers is dropped.
const SITE_SUFFIX_RE = /\s*[-|｜–—]\s*(看漫画|漫画柜|WEBTOON|LINE WEBTOON|Tapas.*|Asura.*|哔哩哔哩|bilibili|快看漫画|动漫之家).*$/i;

// Chapter marker forms, most-specific first. Captures the chapter number.
//   第30話 / 第 30 话 / 第100章 / 第56回   (CJK)
//   EP12 / Episode 3 / Chapter 105 / Ch. 7  (latin, some sources localise)
const CHAPTER_MARKERS: RegExp[] = [
  /第\s*(\d+(?:\.\d+)?)\s*[话話章回節话]/,
  /(?:EP|Episode|Chapter|Ch\.?)\s*(\d+(?:\.\d+)?)/i,
];

interface SplitTitle {
  seriesZh: string;
  chapterLabel: string | null;
  chapterSubtitle: string | null;
}

/**
 * Split a raw title into { series, chapter number, chapter subtitle }.
 * Deterministic — no network. The chapter number becomes "Ch. N"; the series
 * name is everything before the marker (minus trailing separators); anything
 * after the marker becomes the subtitle (e.g. "戰士 (2)").
 */
export function splitTitle(raw: string): SplitTitle {
  let s = raw.replace(SITE_SUFFIX_RE, '').trim();

  for (const re of CHAPTER_MARKERS) {
    const m = s.match(re);
    if (m && m.index !== undefined) {
      const before = s.slice(0, m.index);
      const after = s.slice(m.index + m[0].length);
      // Series = text before the marker, stripped of trailing separators/space.
      const seriesZh = before.replace(/[\s\-|｜–—:：、,，。.]+$/, '').trim();
      // Subtitle = text after the marker, stripped of leading separators/space.
      const chapterSubtitle = after.replace(/^[\s\-|｜–—:：、,，。.]+/, '').trim() || null;
      return {
        seriesZh: seriesZh || s, // guard: marker at position 0 → keep whole
        chapterLabel: `Ch. ${m[1]}`,
        chapterSubtitle,
      };
    }
  }
  // No chapter marker at all — the whole string is the series name.
  return { seriesZh: s, chapterLabel: null, chapterSubtitle: null };
}

// --- matching ----------------------------------------------------------------

/** Fold away script + punctuation + case differences for robust comparison. */
function normCJK(s: string): string {
  return toSimplified(s)
    .replace(/[\s\p{P}\p{S}]/gu, '')
    .toLowerCase();
}

const isLatin = (t: string) => /^[\x00-\x7f]+$/.test(t) && /[A-Za-z]/.test(t);

/**
 * Compare a candidate's alt-titles against the query series name. Returns the
 * strongest match found:
 *   'exact'  — some alt equals the query after normalization
 *   'strong' — some alt contains (or is contained by) the query, lengths close
 *   'none'   — nothing credible
 * We deliberately do NOT accept loose token overlap: a wrong-but-plausible
 * title is worse than falling through to pinyin.
 */
function scoreMatch(queryNorm: string, alts: string[]): { confidence: MatchConfidence; matchedOn: string } {
  let strong = '';
  for (const alt of alts) {
    const an = normCJK(alt);
    if (!an) continue;
    if (an === queryNorm) return { confidence: 'exact', matchedOn: alt };
    // Containment counts only when the shorter is at least 60% of the longer,
    // so "斗破" doesn't match "斗破苍穹外传番外篇". Avoids greedy false hits.
    const [short, long] = an.length <= queryNorm.length ? [an, queryNorm] : [queryNorm, an];
    if (long.includes(short) && short.length / long.length >= 0.6) strong = strong || alt;
  }
  return strong ? { confidence: 'strong', matchedOn: strong } : { confidence: 'none', matchedOn: '' };
}

/** Pick the best English rendering out of a list of alt-titles. */
function pickEnglish(alts: string[]): string | null {
  return alts.find(isLatin) || null;
}

// --- public API --------------------------------------------------------------

/**
 * Resolve one raw title to learner-friendly form. Network calls (MangaDex +
 * AniList) are best-effort — any failure degrades gracefully to the next rung.
 *
 * @param raw    the scraped title
 * @param llm    optional LLM gloss fallback for DB misses
 */
export async function resolveTitle(raw: string, llm?: LlmGloss): Promise<ResolvedTitle> {
  const { seriesZh, chapterLabel, chapterSubtitle } = splitTitle(raw);
  // Pinyin only makes sense for a Han series name — for an English title it
  // would just echo the English back, so leave it empty.
  const seriesPinyin = /[一-鿿]/.test(seriesZh) ? pinyin(seriesZh, { toneType: 'symbol', type: 'string' }) : '';
  const queryNorm = normCJK(seriesZh);

  const candidates: TitleCandidate[] = [];

  // Rung 1a: MangaDex — the cover source, and the true English title when it
  // has one. We take ONLY MangaDex's `en`-keyed title (never a romaji alt that
  // merely looks Latin); AniList fills in English when MangaDex has none.
  try {
    const md = await mangadexTitleLookup(seriesZh);
    for (const r of md) {
      const { confidence, matchedOn } = scoreMatch(queryNorm, r.titles);
      if (confidence === 'none') continue;
      candidates.push({
        english: r.enTitle,
        cover: r.coverUrl,
        source: 'mangadex',
        matchedOn,
        confidence,
      });
    }
  } catch (e: any) {
    console.error(`[TitleResolver] MangaDex lookup failed for "${seriesZh}": ${e.message}`);
  }

  // Rung 1b: AniList (fills MangaDex gaps for manhua; no cover art).
  try {
    const al = await aniListLookup(seriesZh);
    for (const a of al) {
      const { confidence, matchedOn } = scoreMatch(queryNorm, a.alts);
      if (confidence === 'none') continue;
      candidates.push({
        english: pickEnglish(a.alts),
        cover: null,
        source: 'anilist',
        matchedOn,
        confidence,
      });
    }
  } catch (e: any) {
    console.error(`[TitleResolver] AniList lookup failed for "${seriesZh}": ${e.message}`);
  }

  // Choose the best candidate: exact over strong, then a candidate that has an
  // English title, then one that has a cover.
  const rank = (c: TitleCandidate) =>
    (c.confidence === 'exact' ? 4 : c.confidence === 'strong' ? 2 : 0) +
    (c.english ? 1 : 0);
  const best = [...candidates].sort((a, b) => rank(b) - rank(a))[0];

  let english: string | null = null;
  let coverUrl: string | null = null;
  let source: ResolveSource = 'none';
  let confidence: MatchConfidence = 'none';

  if (best && best.english) {
    english = best.english;
    source = best.source;
    confidence = best.confidence;
    // Cover: prefer the chosen candidate's; else borrow any candidate's cover
    // (they've all matched the same series confidently).
    coverUrl = best.cover || candidates.find(c => c.cover)?.cover || null;
    // If nobody had a cover but we now know the English name, ask MangaDex for
    // one by its English title (AniList-only matches often land here).
    if (!coverUrl) {
      try {
        const byEn = await mangadexTitleLookup(english);
        coverUrl = byEn.find(r => r.coverUrl)?.coverUrl || null;
      } catch { /* cover is a nice-to-have; ignore */ }
    }
  }

  // Rung 2: LLM gloss for DB misses.
  if (!english && llm) {
    try {
      const g = await llm(seriesZh);
      if (g && isLatin(g)) {
        english = g;
        source = 'llm';
      }
    } catch (e: any) {
      console.error(`[TitleResolver] LLM gloss failed for "${seriesZh}": ${e.message}`);
    }
  }

  return {
    raw,
    seriesZh,
    seriesPinyin,
    chapterLabel,
    chapterSubtitle,
    english,
    coverUrl,
    source,
    confidence,
    candidates,
  };
}
