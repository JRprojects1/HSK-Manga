/**
 * Adapter interface for per-site manga downloaders.
 *
 * Each site we support has its own adapter that implements this interface.
 * The dispatcher in webtoon-downloader.ts picks the right adapter by URL.
 *
 * Adapters return image URLs + a chapter title. Image download itself stays
 * in the shared webtoon-downloader code so download concurrency, retry, and
 * referer-header logic don't get duplicated per site.
 */
export interface ChapterImages {
  title: string;
  imageUrls: string[];
  // Optional: some sites (Bilibili) sign image URLs with short-lived tokens.
  // Setting this true tells the downloader to NOT stage URLs — download them
  // immediately after extraction before tokens expire.
  urlsExpire?: boolean;
  // Optional: per-image headers required to fetch (e.g. specific Referer that
  // doesn't match the chapter URL). Passed through to the image downloader.
  imageHeaders?: Record<string, string>;
}

export interface SiteAdapter {
  /** Human-readable site name, used in logs + result cards. */
  name: string;
  /** Stable id for client-side ordering / quality ranking. */
  sourceId: SourceId;
  /** Returns true if this adapter handles the given chapter URL. */
  matches(url: string): boolean;
  /** Extracts the chapter's image URLs from the site. */
  getChapter(url: string): Promise<ChapterImages>;
  /** Optional: search this site for series matching a title. Adapters
   *  that don't support search can omit this and the aggregator will skip
   *  them. */
  search?(query: string): Promise<SeriesSearchResult[]>;
  /** Optional: given a series URL (the seriesUrl from a SeriesSearchResult),
   *  return the list of chapters. Each entry has a chapter URL the
   *  downloader can process, plus paywall metadata so the UI can warn the
   *  user before they pick a locked chapter. */
  getChapterList?(seriesUrl: string): Promise<ChapterListEntry[]>;
}

export interface ChapterListEntry {
  /** Display title for the chapter (e.g. "Ch. 47", "Episode 12: The Storm"). */
  title: string;
  /** URL the downloader will process — pass this to /api/process. */
  chapterUrl: string;
  /** Locked behind paywall (subscription, ink, Fast Pass, premium etc.) —
   *  download will fail without user auth. Surface to user, don't hide. */
  locked: boolean;
  /** Optional: chapter number for sort/display purposes. */
  number?: string;
  /** Optional: human-readable release date (YYYY-MM-DD or "2 days ago"). */
  releasedAt?: string;
}

/** Per-source quality + sort ordering. Lower number = preferred result. */
export type SourceId =
  | 'manhuagui'
  | 'kuaikan'
  | 'webtoons-cn'
  | 'webtoons-en'
  | 'mangadex'
  | 'tapas'
  | 'mangaplus'
  | 'dmzj'
  | 'bilibili-manga';

export interface SeriesSearchResult {
  /** Source this match came from. */
  source: SourceId;
  /** Display title for this source (in that source's native language). */
  title: string;
  /** URL of the series page on this source — opens in a new tab when the
   *  user clicks the source button. NOT a chapter URL. */
  seriesUrl: string;
  /** Optional cover image URL. */
  coverUrl?: string;
  /** Optional short description. */
  description?: string;
}

/** Aggregated result for a single series across multiple sources. */
export interface AggregatedSearchResult {
  /** Primary display title (English preferred, then native). */
  title: string;
  /** Alternate titles in other languages, for display + fuzzy-match dedup. */
  altTitles: string[];
  /** Cover URL (best available across sources). */
  coverUrl?: string;
  /** Per-source matches, sorted by quality (best first). */
  sources: SeriesSearchResult[];
}
