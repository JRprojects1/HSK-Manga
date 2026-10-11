/**
 * Adapter registry + dispatcher.
 *
 * Adapters are tried in order; the first one whose matches() returns true
 * gets the URL. If none match, the dispatcher returns null and the caller
 * falls back to the generic static-then-Puppeteer flow in webtoon-downloader.
 */
import { SiteAdapter, ChapterImages } from './types';
import { mangadexAdapter } from './mangadex';
import { tapasAdapter } from './tapas';
import { manhuaguiAdapter } from './manhuagui';
import { bilibiliMangaAdapter } from './bilibili-manga';
import { dmzjAdapter } from './dmzj';
import { kuaikanAdapter } from './kuaikan';
import { mangaplusAdapter } from './mangaplus';

const ADAPTERS: SiteAdapter[] = [
  mangadexAdapter,
  tapasAdapter,
  manhuaguiAdapter,
  bilibiliMangaAdapter,
  dmzjAdapter,
  kuaikanAdapter,
  mangaplusAdapter,
];

export function findAdapter(url: string): SiteAdapter | null {
  for (const a of ADAPTERS) {
    if (a.matches(url)) return a;
  }
  return null;
}

export { ChapterImages, SiteAdapter };
export { federatedSearch, federatedSearchStream } from './search';
export type { SeriesSearchResult, AggregatedSearchResult, SourceId } from './types';
