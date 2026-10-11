/**
 * Helper for looking up adapters by sourceId — used by the chapter-list
 * endpoint which receives a sourceId from the frontend and needs to find
 * the matching adapter. Separate file to avoid circular imports.
 */
import { SiteAdapter, SourceId } from './types';
import { mangadexAdapter } from './mangadex';
import { manhuaguiAdapter } from './manhuagui';
import { tapasAdapter } from './tapas';
import { webtoonsEnAdapter, webtoonsCnAdapter } from './webtoons';
import { kuaikanAdapter } from './kuaikan';
import { dmzjAdapter } from './dmzj';
import { bilibiliMangaAdapter } from './bilibili-manga';
import { mangaplusAdapter } from './mangaplus';

const REGISTRY: Record<SourceId, SiteAdapter> = {
  'mangadex': mangadexAdapter,
  'manhuagui': manhuaguiAdapter,
  'tapas': tapasAdapter,
  'webtoons-en': webtoonsEnAdapter,
  'webtoons-cn': webtoonsCnAdapter,
  'kuaikan': kuaikanAdapter,
  'mangaplus': mangaplusAdapter,
  'dmzj': dmzjAdapter,
  'bilibili-manga': bilibiliMangaAdapter,
};

export function findAdapterById(sourceId: string): SiteAdapter | undefined {
  return (REGISTRY as any)[sourceId];
}
