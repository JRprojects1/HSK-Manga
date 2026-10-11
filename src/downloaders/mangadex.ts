/**
 * MangaDex adapter — uses their public, documented REST API.
 *
 * Docs: https://api.mangadex.org/docs/
 *
 * URL forms accepted:
 *   https://mangadex.org/chapter/{uuid}
 *   https://mangadex.org/chapter/{uuid}/3        (any page number — we ignore it)
 *   https://canary.mangadex.dev/chapter/{uuid}   (test mirror)
 *
 * Flow:
 *   1. Regex-extract the chapter UUID from the URL.
 *   2. GET /at-home/server/{chapterId} — returns baseUrl + image filenames.
 *   3. Optionally GET /chapter/{chapterId}?includes[]=manga — for the chapter
 *      title (numeric chapter # + manga name).
 *   4. Build full image URLs: {baseUrl}/data/{hash}/{filename}.
 *
 * No auth required for public chapters. No rate-limit issues at single-chapter
 * scale (their rate limit is per IP, generous for non-bulk usage).
 */
import { SiteAdapter, ChapterImages, SeriesSearchResult, ChapterListEntry } from './types';
import * as https from 'https';

const API_BASE = 'https://api.mangadex.org';
const UUID_RE = /\/chapter\/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/i;

// MangaDex enforces app-identifying User-Agents and rejects generic browser
// strings ("You must set an appropriate User-Agent header"). Use a name that
// identifies us per their API docs.
const MANGADEX_UA = 'HSKManga/1.0 (https://hskmanga.com)';

function fetchJSON<T = any>(url: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const req = https.get(url, {
      headers: {
        'User-Agent': MANGADEX_UA,
        'Accept': 'application/json',
      },
    }, res => {
      if (res.statusCode !== 200) {
        reject(new Error(`MangaDex API HTTP ${res.statusCode} for ${url}`));
        return;
      }
      let body = '';
      res.on('data', c => body += c);
      res.on('end', () => {
        try { resolve(JSON.parse(body)); }
        catch (e: any) { reject(new Error('MangaDex API JSON parse failed: ' + e.message)); }
      });
      res.on('error', reject);
    });
    req.on('error', reject);
    req.setTimeout(30000, () => { req.destroy(); reject(new Error('MangaDex API timeout')); });
  });
}

/** One MangaDex series' title metadata, for cross-language title resolution. */
export interface MangaTitleMeta {
  /** Every title string across all languages + altTitles (for fuzzy matching). */
  titles: string[];
  /** The true English title (en / en-us key only), or null. NOT romaji. */
  enTitle: string | null;
  /** 256px cover URL, or null. */
  coverUrl: string | null;
}

/**
 * Title/cover resolver — DELIBERATELY different from the adapter's search():
 *   - NO "readable-language" filter. search() drops series with no EN/ZH
 *     *chapters* (right for a search UI, wrong here) — but such a series still
 *     has perfectly good cover art + title metadata, which is all we want.
 *   - Preserves the `en` language KEY so callers can take the true English
 *     title and never a romanized Japanese/Korean one that also looks Latin.
 * Used by the learner-title resolver; keep separate from search() so neither
 * change breaks the other.
 */
export async function mangadexTitleLookup(query: string): Promise<MangaTitleMeta[]> {
  const params = new URLSearchParams({ title: query, limit: '5' });
  params.append('includes[]', 'cover_art');
  params.append('contentRating[]', 'safe');
  params.append('contentRating[]', 'suggestive');
  params.append('contentRating[]', 'erotica');
  let resp: any;
  try { resp = await fetchJSON<any>(`${API_BASE}/manga?${params.toString()}`); }
  catch { return []; }
  const data: any[] = resp?.data || [];
  return data.map((m: any): MangaTitleMeta => {
    const attrs = m.attributes || {};
    const titleDict = attrs.title || {};
    const altList: any[] = attrs.altTitles || [];
    const titles: string[] = [];
    // The true English title lives under the `en`/`en-us` key in either the
    // primary title dict or an altTitle entry. Take ONLY those keys as English.
    let enTitle: string | null = titleDict.en || titleDict['en-us'] || null;
    for (const v of Object.values(titleDict)) if (typeof v === 'string') titles.push(v);
    for (const alt of altList) {
      for (const k of Object.keys(alt || {})) {
        const v = alt[k];
        if (typeof v !== 'string') continue;
        titles.push(v);
        if (!enTitle && (k === 'en' || k === 'en-us')) enTitle = v;
      }
    }
    const coverRel = (m.relationships || []).find((r: any) => r.type === 'cover_art');
    const coverFile = coverRel?.attributes?.fileName;
    const coverUrl = coverFile
      ? `https://uploads.mangadex.org/covers/${m.id}/${coverFile}.256.jpg`
      : null;
    return { titles, enTitle, coverUrl };
  });
}

export const mangadexAdapter: SiteAdapter = {
  name: 'MangaDex',
  sourceId: 'mangadex',

  matches(url) {
    return /^https?:\/\/(?:www\.|canary\.)?mangadex\.(?:org|dev)\//.test(url);
  },

  // Search MangaDex by title. Also serves as our cross-language title-resolver:
  // each result includes all language titles (zh, en, ja, etc.) plus altTitles,
  // which the aggregator uses to seed searches against language-specific sites.
  async search(query: string): Promise<SeriesSearchResult[]> {
    const params = new URLSearchParams({
      title: query,
      limit: '10',
      'includes[]': 'cover_art',
      'order[relevance]': 'desc',
      // contentRating: include everything except pornographic — picks up safe + suggestive + erotica
      'contentRating[]': 'safe',
    });
    // Tack on remaining contentRatings (URLSearchParams dedupes the key so we
    // append individually).
    params.append('contentRating[]', 'suggestive');
    params.append('contentRating[]', 'erotica');
    const url = `${API_BASE}/manga?${params.toString()}`;
    const resp = await fetchJSON<any>(url);
    const data = resp?.data || [];
    return data
      // Filter out:
      //   - Doujinshi (fan-made comics): MangaDex tags as "Doujinshi" or
      //     puts "(Doujinshi)" in the title.
      //   - Anthology compilations: not actual chapter content.
      //   - Entries with NO English or Chinese translation available: the user
      //     can't read them, so showing them in the result list is misleading.
      .filter((m: any) => {
        const title = (m.attributes?.title && Object.values(m.attributes.title)[0]) as string || '';
        if (/\(\s*doujinshi\s*\)/i.test(title)) return false;
        const tags: any[] = m.attributes?.tags || [];
        for (const t of tags) {
          const name = t.attributes?.name?.en;
          if (name === 'Doujinshi' || name === 'Anthology') return false;
        }
        // Require at least one English/Chinese translation. Korean/Japanese-
        // only entries are unreadable for our audience and clutter results.
        const langs: string[] = m.attributes?.availableTranslatedLanguages || [];
        const hasReadable = langs.some(l => l === 'en' || l === 'zh' || l === 'zh-hk');
        if (!hasReadable) return false;
        return true;
      })
      .map((m: any): SeriesSearchResult => {
      const attrs = m.attributes || {};
      const titleDict = attrs.title || {};
      const altList: any[] = attrs.altTitles || [];
      // Build a flat list of all titles to surface in the aggregator.
      const altTitles: string[] = [];
      for (const v of Object.values(titleDict)) if (typeof v === 'string') altTitles.push(v);
      for (const alt of altList) for (const v of Object.values(alt || {})) if (typeof v === 'string') altTitles.push(v as string);
      // Prefer English then any language for display.
      const displayTitle =
        titleDict.en || titleDict['en-us'] || (Object.values(titleDict)[0] as string) || 'Untitled';
      // Cover relationship → CDN URL
      const coverRel = (m.relationships || []).find((r: any) => r.type === 'cover_art');
      const coverFile = coverRel?.attributes?.fileName;
      const coverUrl = coverFile
        ? `https://uploads.mangadex.org/covers/${m.id}/${coverFile}.256.jpg`
        : undefined;
      // Description (en preferred)
      const desc = attrs.description?.en || (Object.values(attrs.description || {})[0] as string);

      return {
        source: 'mangadex',
        title: displayTitle,
        seriesUrl: `https://mangadex.org/title/${m.id}`,
        coverUrl,
        description: typeof desc === 'string' ? desc.slice(0, 200) : undefined,
        // We stash all-language titles in the aggregated result via a side channel:
        // the aggregator looks up MangaDex results specifically for translation hints.
        ...({ _altTitles: altTitles } as any),
      };
    });
  },

  // MangaDex chapter list: fetch /manga/{id}/feed for English + Chinese
  // translations. Chapters that link out to an external paid site
  // (externalUrl set) are marked locked since we can't fetch them.
  async getChapterList(seriesUrl: string): Promise<ChapterListEntry[]> {
    const m = seriesUrl.match(/\/title\/([0-9a-f-]{36})/i);
    if (!m) throw new Error('Not a MangaDex series URL: ' + seriesUrl);
    const titleId = m[1];
    const params = new URLSearchParams({
      'limit': '100',
      'order[chapter]': 'asc',
      'order[volume]': 'asc',
      'includes[]': 'scanlation_group',
      'contentRating[]': 'safe',
    });
    params.append('contentRating[]', 'suggestive');
    params.append('contentRating[]', 'erotica');
    params.append('translatedLanguage[]', 'en');
    params.append('translatedLanguage[]', 'zh');
    params.append('translatedLanguage[]', 'zh-hk');
    const resp = await fetchJSON<any>(`${API_BASE}/manga/${titleId}/feed?${params}`);
    const data: any[] = resp?.data || [];
    return data.map((c: any): ChapterListEntry => {
      const attrs = c.attributes || {};
      const chNum = attrs.chapter || '?';
      const chTitle = attrs.title ? ` — ${attrs.title}` : '';
      const lang = attrs.translatedLanguage ? ` [${attrs.translatedLanguage}]` : '';
      return {
        title: `Ch. ${chNum}${chTitle}${lang}`,
        chapterUrl: `https://mangadex.org/chapter/${c.id}`,
        // externalUrl means MangaDex doesn't host the images — the chapter
        // redirects to MangaPlus, ComiKey etc. We can't download from those.
        locked: !!attrs.externalUrl,
        number: attrs.chapter || undefined,
        releasedAt: attrs.publishAt ? attrs.publishAt.slice(0, 10) : undefined,
      };
    });
  },

  async getChapter(url): Promise<ChapterImages> {
    const m = url.match(UUID_RE);
    if (!m) {
      const err: any = new Error('MangaDex URL must contain a chapter UUID (looks like https://mangadex.org/chapter/<uuid>)');
      err.code = 'BAD_URL';
      throw err;
    }
    const chapterId = m[1];

    // 1. Get image server + filenames
    const server = await fetchJSON<any>(`${API_BASE}/at-home/server/${chapterId}`);
    if (!server.baseUrl || !server.chapter?.data?.length) {
      const err: any = new Error('MangaDex returned no chapter images. Chapter may be removed or restricted.');
      err.code = 'NO_IMAGES_FOUND';
      throw err;
    }
    const hash = server.chapter.hash;
    // data/ = full-quality. dataSaver/ = compressed. We always pick full.
    const imageUrls: string[] = server.chapter.data.map((f: string) =>
      `${server.baseUrl}/data/${hash}/${f}`
    );

    // 2. Title: try chapter+manga lookup, but don't fail the download if it errors.
    let title = `chapter ${chapterId.slice(0, 8)}`;
    try {
      const chapter = await fetchJSON<any>(`${API_BASE}/chapter/${chapterId}?includes[]=manga`);
      const attrs = chapter?.data?.attributes;
      const mangaRel = chapter?.data?.relationships?.find((r: any) => r.type === 'manga');
      const mangaTitle = mangaRel?.attributes?.title
        ? Object.values(mangaRel.attributes.title)[0] as string
        : null;
      const chNum = attrs?.chapter ? `Ch ${attrs.chapter}` : '';
      const chTitle = attrs?.title ? ` — ${attrs.title}` : '';
      if (mangaTitle) title = `${mangaTitle} ${chNum}${chTitle}`.trim();
    } catch { /* fall back to default title */ }

    return { title, imageUrls };
  },
};
