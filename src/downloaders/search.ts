/**
 * Federated search across all site adapters.
 *
 * Flow:
 *   1. Hit MangaDex first to get title aliases across languages.
 *      MangaDex serves as the title-translator — its results give us the
 *      Chinese / Japanese / English forms of each series.
 *   2. In parallel, search every other source using the appropriate-language
 *      title. Chinese sites get the Chinese title; English sites get the
 *      English title. Any source can also be searched with the raw query as
 *      a fallback for series MangaDex doesn't know about.
 *   3. Merge results across sources by fuzzy-matching titles. Each merged
 *      "series" carries a list of per-source matches sorted by quality.
 *
 * Quality ordering (lower index = preferred):
 *   Chinese-source CN→simpler-CN pipeline outranks English-source EN→CN→simpler-CN
 *   because the latter is a translation chain with more drift opportunities.
 */
import { tryWordSplit } from '../english-splitter';
import { mangadexAdapter } from './mangadex';
import { manhuaguiAdapter } from './manhuagui';
import { tapasAdapter } from './tapas';
import { webtoonsEnAdapter, webtoonsCnAdapter } from './webtoons';
import { kuaikanAdapter } from './kuaikan';
import { dmzjAdapter } from './dmzj';
import { mangaplusAdapter } from './mangaplus';
import { aniListLookup } from './anilist';
import { SeriesSearchResult, AggregatedSearchResult, SourceId, SiteAdapter } from './types';

const QUALITY_ORDER: SourceId[] = [
  'manhuagui',      // Chinese, scanlation-quality
  'kuaikan',        // Chinese, official platform
  'webtoons-cn',    // Chinese, official translation
  'webtoons-en',    // English, official → requires EN→CN chain
  'mangadex',       // Multi-language, scanlation-quality
  'mangaplus',      // English/Spanish official → EN→CN chain
  'tapas',          // English indie → requires EN→CN chain
  'dmzj',           // Chinese but availability uncertain
  'bilibili-manga', // Auth-required, not actually usable
];

function qualityRank(s: SourceId): number {
  const i = QUALITY_ORDER.indexOf(s);
  return i < 0 ? 999 : i;
}

// Normalize a title for fuzzy comparison: lowercase, strip punctuation,
// collapse whitespace. CJK characters are preserved.
function normTitle(t: string): string {
  return t
    .toLowerCase()
    .replace(/[\p{P}\p{S}]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

// Significant tokens from a title — Latin words >= 3 chars, OR contiguous
// CJK runs >= 2 chars. Filters out trivial connectors ("the", "of", etc).
const STOP = new Set(['the', 'of', 'and', 'a', 'an', 'in', 'on', 'for', 'to', 'is']);
function tokens(t: string): string[] {
  const out: string[] = [];
  const latin = t.toLowerCase().match(/[a-z]{3,}/g) || [];
  for (const w of latin) if (!STOP.has(w)) out.push(w);
  const cjkRuns = t.match(/[一-鿿]+/g) || [];
  for (const r of cjkRuns) if (r.length >= 2) out.push(r);
  return out;
}

// True if `title` contains ALL significant tokens from `query`. For Latin
// tokens we require word-boundary matches so "solo" doesn't match inside
// "Solos" or "Solorigem". For CJK tokens substring is fine because CJK
// characters are whole morphemes and "包含" matches naturally.
//
// Returns false when query has no usable tokens (e.g. Japanese with only
// kana + single kanji) — otherwise an untokenizable variant would silently
// bypass the filter.
function titleMatchesQuery(title: string, query: string): boolean {
  const qTokens = tokens(query);
  if (qTokens.length === 0) return false;
  const titleLower = title.toLowerCase();
  return qTokens.every(t => {
    const tok = t.toLowerCase();
    // CJK token: substring match is correct (each char is a morpheme).
    if (/[一-鿿]/.test(tok)) return titleLower.includes(tok);
    // Latin token: require word boundaries. Word chars on either side
    // disqualify (so "solo" inside "Solos" or "Solorigem" fails).
    const escaped = tok.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    return new RegExp(`(^|[^a-z0-9])${escaped}([^a-z0-9]|$)`, 'i').test(titleLower);
  });
}

// True if two titles look like the same series. Strict equality after
// normalization only. We used to also do prefix matching, but it was too
// aggressive: "Solo" matched "Solo Leveling: Ragnarok", merging unrelated
// series into one bucket. Different titles are different series.
function titlesMatch(a: string, b: string): boolean {
  const na = normTitle(a);
  const nb = normTitle(b);
  if (!na || !nb) return false;
  return na === nb;
}

// Distinguish actual Simplified/Traditional Chinese titles from Japanese
// (which mixes kanji with kana) and Korean (Hangul). We feed only Chinese-
// pure titles to Chinese-site searches — sending Japanese kanji-runs to
// Manhuagui just rate-limits us and never matches anything.
function isChineseTitle(s: string): boolean {
  const hasCJK = /[一-鿿]/.test(s);
  if (!hasCJK) return false;
  const hasHiragana = /[぀-ゟ]/.test(s);
  const hasKatakana = /[゠-ヿ]/.test(s);
  const hasHangul = /[가-힯]/.test(s);
  return !hasHiragana && !hasKatakana && !hasHangul;
}

// Searchable adapters in the order we want to call them. (mangadex first
// because we use it for translation hints.)
// SCOPE NOTE: search results are intentionally restricted to Webtoons EN/CN.
// Other sources (KuaiKan, Tapas, Manhuagui, MangaPlus, DMZJ, etc.) are gated
// behind paywalls / app downloads for anything past the first 1-3 chapters,
// so they produce a worse user experience than helping. Their adapters are
// still in the codebase for direct URL processing (users who paste a chapter
// URL get full support); they're just not surfaced in the on-site search.
//
// MangaDex is NOT in this list either — but the orchestrator still calls
// mangadexAdapter.search() separately as a translator (to map "Solo Leveling"
// → "我独自升级" so Webtoons CN can find it). MangaDex results are never
// returned in the user-facing buckets.
function searchableAdapters(): SiteAdapter[] {
  const all: SiteAdapter[] = [
    webtoonsEnAdapter,
    webtoonsCnAdapter,
  ];
  return all.filter(a => typeof a.search === 'function');
}

// Circuit breaker: skip an adapter for COOLDOWN_MS after FAIL_THRESHOLD
// consecutive failures. Prevents a flaky/blocked site (e.g. Manhuagui
// temporarily IP-blocking us) from slowing every search request.
const FAIL_THRESHOLD = 3;
const COOLDOWN_MS = 5 * 60 * 1000;
const breaker = new Map<SourceId, { failures: number; openUntil: number }>();

function isOpen(sourceId: SourceId): boolean {
  const s = breaker.get(sourceId);
  return !!s && s.openUntil > Date.now();
}
function recordFailure(sourceId: SourceId): void {
  const s = breaker.get(sourceId) || { failures: 0, openUntil: 0 };
  s.failures++;
  if (s.failures >= FAIL_THRESHOLD) s.openUntil = Date.now() + COOLDOWN_MS;
  breaker.set(sourceId, s);
}
function recordSuccess(sourceId: SourceId): void {
  breaker.set(sourceId, { failures: 0, openUntil: 0 });
}

// Call an adapter's search with circuit-breaker protection. Returns [] for
// any failure mode (including breaker-open) so the caller doesn't have to
// guard each call.
async function safeSearch(a: SiteAdapter, query: string): Promise<SeriesSearchResult[]> {
  if (isOpen(a.sourceId)) {
    console.error(`[search] ${a.name} skipped (breaker open until ${new Date(breaker.get(a.sourceId)!.openUntil).toISOString()})`);
    return [];
  }
  try {
    const r = (await a.search!(query)) || [];
    recordSuccess(a.sourceId);
    return r;
  } catch (e) {
    recordFailure(a.sourceId);
    console.error(`[search] ${a.name} failed:`, (e as any).message);
    return [];
  }
}

/**
 * Build aggregated buckets from a flat list of search results.
 * Pure function — used by both the streaming and non-streaming paths.
 */
function buildBuckets(
  flatHits: SeriesSearchResult[],
  query: string,
  aliasGroups: string[][],
  chineseHints: Set<string>,
  splitVariant: string | null = null,
): AggregatedSearchResult[] {
  const queryVariants = [query, ...(splitVariant ? [splitVariant] : []), ...Array.from(chineseHints)];

  // Attach alias groups to results so cross-language matches merge.
  // E.g. a Webtoons EN hit titled "Surviving the Game as a Barbarian" and a
  // Webtoons CN hit titled "變成蠻族！遊戲生存戰" share no tokens but DO
  // share the same AniList alias group — augmenting each result's altTitles
  // with the group makes the bucket merger connect them.
  for (const hit of flatHits) {
    const existingAlts = ((hit as any)._altTitles as string[]) || [];
    const titlesToCheck = [hit.title, ...existingAlts];
    for (const group of aliasGroups) {
      const inGroup = group.some(g => titlesToCheck.some(t => titlesMatch(t, g)));
      if (inGroup) {
        const merged = new Set<string>([...existingAlts, ...group]);
        (hit as any)._altTitles = Array.from(merged);
        break;
      }
    }
  }

  // Relevance filter. Sites like Webtoons return "related" suggestions when
  // the exact title isn't found — those are noise. Keep only hits whose
  // title contains all significant tokens from EITHER the original query
  // OR any of the CJK translation hints.
  const allHits = flatHits.filter(hit => {
    const titlesToTest: string[] = [hit.title, ...(((hit as any)._altTitles as string[]) || [])];
    for (const t of titlesToTest) {
      for (const q of queryVariants) {
        if (titleMatchesQuery(t, q)) return true;
      }
    }
    return false;
  });

  const buckets: AggregatedSearchResult[] = [];
  for (const hit of allHits) {
    // Find an existing aggregated result whose title matches this hit.
    const altTitles = [(hit as any)._altTitles, [hit.title]].flat().filter(Boolean) as string[];
    const bucket = buckets.find(b =>
      altTitles.some(t => b.altTitles.some(bt => titlesMatch(t, bt)))
    );
    if (bucket) {
      // Add as another source; dedupe per source.
      if (!bucket.sources.some(s => s.source === hit.source && s.seriesUrl === hit.seriesUrl)) {
        bucket.sources.push(hit);
      }
      for (const t of altTitles) if (!bucket.altTitles.includes(t)) bucket.altTitles.push(t);
      if (!bucket.coverUrl && hit.coverUrl) bucket.coverUrl = hit.coverUrl;
    } else {
      buckets.push({
        title: hit.title,
        altTitles,
        coverUrl: hit.coverUrl,
        sources: [hit],
      });
    }
  }

  // Sort each bucket's sources by quality (best first), then pick a display
  // title. Priority:
  //   1. Latin-only title that CONTAINS the user's query (so the user sees
  //      "Solo Leveling" instead of "Na Honjaman Level-Up" for that query)
  //   2. Any Latin-only title (English form)
  //   3. Chinese title (Simplified or Traditional)
  //   4. Anything that isn't Korean (Hangul)
  // Canvas dedupe: when a bucket has BOTH a Webtoons /canvas/ URL and a
  // non-Canvas (Originals) URL for the same source-id, drop the Canvas one.
  // Canvas results for popular Originals are usually announcement/teaser
  // pages with 3 chapters of "ANNOUNCEMENT" / "LAUNCH DATE" content — not
  // the real comic. We only drop the Canvas version when an Originals
  // version exists in the same bucket; Canvas-only series (legitimate indie
  // comics with no Originals counterpart) are kept intact.
  for (const b of buckets) {
    const hasNonCanvasFor = new Set<string>();
    for (const s of b.sources) {
      if (!/\/canvas\//.test(s.seriesUrl)) hasNonCanvasFor.add(s.source);
    }
    b.sources = b.sources.filter(s =>
      !(/\/canvas\//.test(s.seriesUrl) && hasNonCanvasFor.has(s.source))
    );
  }

  // The split variant counts as "the user's query" for display-title selection
  // and scoring: for query "handjumper" the variant "hand jumper" is what
  // actually matches the title "Hand Jumper".
  const displayQueries = [query, ...(splitVariant ? [splitVariant] : [])].map(q => q.toLowerCase());
  const isHangul = (t: string) => /[가-힯]/.test(t);
  for (const b of buckets) {
    b.sources.sort((a, c) => qualityRank(a.source) - qualityRank(c.source));
    const enTitleMatching = b.altTitles.find(t => /^[\x00-\x7f]+$/.test(t) && displayQueries.some(q => t.toLowerCase().includes(q)));
    const enTitle = b.altTitles.find(t => /^[\x00-\x7f]+$/.test(t));
    const cnTitle = b.altTitles.find(t => isChineseTitle(t));
    const nonKoreanTitle = b.altTitles.find(t => !isHangul(t));
    if (enTitleMatching) b.title = enTitleMatching;
    else if (enTitle) b.title = enTitle;
    else if (cnTitle) b.title = cnTitle;
    else if (nonKoreanTitle && isHangul(b.title)) b.title = nonKoreanTitle;
  }

  // Relevance score: how well does this bucket's titles match the user's query?
  //   100  exact match (any variant equals the query, normalized)
  //   60   any variant CONTAINS the full query string (exact substring)
  //   30   token-only match (the existing relevance-filter pass)
  // Then break ties by source count + best-source quality. This puts series
  // whose title literally contains the user's query above fuzzy "related"
  // suggestions from sites like Webtoons EN.
  const scoreQueries = [query, ...(splitVariant ? [splitVariant] : [])]
    .map(q => ({ lowered: q.toLowerCase(), normalized: normTitle(q) }));
  function bucketScore(b: AggregatedSearchResult): number {
    let best = 30;
    for (const t of [b.title, ...b.altTitles]) {
      const tn = normTitle(t);
      for (const q of scoreQueries) {
        if (tn === q.normalized) { best = Math.max(best, 100); continue; }
        if (tn.includes(q.normalized) || t.toLowerCase().includes(q.lowered)) {
          best = Math.max(best, 60);
        }
      }
    }
    return best;
  }
  buckets.sort((a, c) => {
    const sa = bucketScore(a);
    const sc = bucketScore(c);
    if (sa !== sc) return sc - sa;
    if (a.sources.length !== c.sources.length) return c.sources.length - a.sources.length;
    return qualityRank(a.sources[0].source) - qualityRank(c.sources[0].source);
  });

  // Cap single-source results per source-id so one noisy site (typically
  // Tapas with its long tail of indie community comics) can't dominate the
  // visible results for generic queries like "solo". Multi-source buckets
  // are kept unconditionally — those are likely to be real popular series.
  const PER_SOURCE_CAP = 3;
  const perSourceSeen = new Map<SourceId, number>();
  const filtered: AggregatedSearchResult[] = [];
  for (const b of buckets) {
    if (b.sources.length >= 2) {
      filtered.push(b);
      continue;
    }
    const src = b.sources[0].source;
    const n = perSourceSeen.get(src) || 0;
    if (n >= PER_SOURCE_CAP) continue;
    perSourceSeen.set(src, n + 1);
    filtered.push(b);
  }

  return filtered;
}

/**
 * Shared orchestrator for federated search. Emits a snapshot of current
 * buckets each time an adapter completes. Used by both the SSE endpoint
 * (streaming) and the buffered endpoint (which just keeps the last snapshot).
 */
async function runFederatedSearchStream(
  query: string,
  onSnapshot: (buckets: AggregatedSearchResult[], translatorDone: boolean) => void,
): Promise<void> {
  const adapters = searchableAdapters();
  if (adapters.length === 0) { onSnapshot([], true); return; }

  // Query-split variant: users type multi-word titles without spaces
  // ("handjumper"), and both Webtoons search and MangaDex return nothing for
  // the concatenated form. When the query is a single alphabetic run that
  // confidently segments into dictionary words, search the spaced variant in
  // parallel with the original. tryWordSplit returns null for anything it
  // can't segment into ≥2 dictionary words, so typos never trigger the extra
  // requests.
  const trimmedQuery = query.trim();
  const splitVariant = /^[A-Za-z]{6,}$/.test(trimmedQuery) ? tryWordSplit(trimmedQuery) : null;
  if (splitVariant) {
    console.error(`[search] split variant: "${query}" → "${splitVariant}"`);
  }
  const searchQueries = splitVariant ? [query, splitVariant] : [query];

  const flatHits: SeriesSearchResult[] = [];
  const aliasGroups: string[][] = [];
  const chineseHints = new Set<string>();
  // `translatorDone` flips true after the MangaDex + AniList translator step
  // finishes. The frontend uses it to hold the source-button area on EN-only
  // buckets ("Checking for Chinese version…") until we know whether a CN
  // sibling exists — avoids the user clicking the EN button in the brief
  // window before the CN result merges into the bucket.
  let translatorDone = false;

  const emit = () => onSnapshot(buildBuckets([...flatHits], query, aliasGroups, chineseHints, splitVariant), translatorDone);

  // Translator step: feed a single query to MangaDex (with AniList fallback)
  // to mine Chinese alt-titles, then re-hit Webtoons CN with each new hint.
  // `requireMatch` controls how strict we are about which MangaDex result
  // counts — for the raw user query it's false (broad sweep), for the full
  // EN title from Webtoons it's true (avoid noise from substring matches).
  const translateAndRehit = async (q: string, requireMatch: boolean) => {
    let mdResults: SeriesSearchResult[] = [];
    try { mdResults = (await mangadexAdapter.search!(q)) || []; }
    catch (e) { console.error(`[search-translator] MangaDex(${q}) failed:`, (e as any).message); }

    const newHints = new Set<string>();
    for (const result of mdResults) {
      const alts = ((result as any)._altTitles as string[]) || [];
      const titlesToCheck = [result.title, ...alts];
      if (requireMatch && !titlesToCheck.some(t => titleMatchesQuery(t, q))) continue;
      if (alts.length) aliasGroups.push([result.title, ...alts]);
      for (const alt of alts) if (isChineseTitle(alt) && !chineseHints.has(alt)) {
        chineseHints.add(alt); newHints.add(alt);
      }
    }

    // AniList fallback when MangaDex gave us nothing usable for this query.
    if (newHints.size === 0) {
      try {
        const aniResults = await aniListLookup(q);
        for (const x of aniResults) {
          if (requireMatch && !x.alts.some(t => titleMatchesQuery(t, q))) continue;
          aliasGroups.push(x.alts);
          for (const alt of x.alts) if (isChineseTitle(alt) && !chineseHints.has(alt)) {
            chineseHints.add(alt); newHints.add(alt);
          }
        }
      } catch { /* AniList failure is non-fatal */ }
    }

    // Re-hit Webtoons CN with up to two new Chinese hints. Stop early on first
    // non-empty result — additional hints just add cost for no extra coverage.
    for (const hint of Array.from(newHints).slice(0, 2)) {
      if (isOpen('webtoons-cn')) break;
      const hr = await safeSearch(webtoonsCnAdapter, hint);
      flatHits.push(...hr);
      emit();
      if (hr.length > 0) break;
    }
  };

  // Stage 1a: Webtoons EN + CN with the raw query (and the split variant when
  // one exists). Capture the top EN hits' canonical titles so Stage 2 can use
  // them as much-more-precise translator queries than the user's partial input.
  const enHitTitles: string[] = [];
  const firstRoundPromises = adapters.flatMap(a => searchQueries.map(async q => {
    const r = await safeSearch(a, q);
    flatHits.push(...r);
    if (a.sourceId === 'webtoons-en') {
      for (const hit of r.slice(0, 3)) enHitTitles.push(hit.title);
    }
    emit();
  }));

  // Stage 1b: Raw-query translator (runs in parallel — fast when the user
  // already typed the full title, near-useless when they typed "surviving").
  const rawTranslatorPromise = Promise.all(searchQueries.map(q => translateAndRehit(q, false)));

  // Stage 2: Once Webtoons EN responds, fan out one translator lookup per
  // top EN title. Critical for partial queries — typing "surviving" returns
  // "Surviving the Game as a Barbarian" from Webtoons EN, and feeding THAT
  // full title to MangaDex reliably surfaces the CN sibling ("變成蠻族！…"
  // etc.) that a MangaDex search for "surviving" alone buries under dozens
  // of unrelated matches.
  const enTitleTranslatorPromise = (async () => {
    await Promise.all(firstRoundPromises);
    // Dedupe: the original query and the split variant often return the same
    // top EN titles — no reason to run the translator twice for one title.
    await Promise.all(Array.from(new Set(enHitTitles)).slice(0, 3).map(t => translateAndRehit(t, true)));
  })();

  await Promise.all([...firstRoundPromises, rawTranslatorPromise, enTitleTranslatorPromise]);
  translatorDone = true;
  emit();
}

export async function federatedSearchStream(
  query: string,
  onSnapshot: (buckets: AggregatedSearchResult[], translatorDone: boolean) => void,
): Promise<void> {
  return runFederatedSearchStream(query, onSnapshot);
}

export async function federatedSearch(query: string): Promise<AggregatedSearchResult[]> {
  let final: AggregatedSearchResult[] = [];
  await runFederatedSearchStream(query, snap => { final = snap; });
  return final;
}
