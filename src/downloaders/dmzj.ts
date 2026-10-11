/**
 * DMZJ (动漫之家 / dmzj.com) adapter — speculative, untested.
 *
 * The site was unreachable from the dev environment when this adapter was
 * written (consistent ECONNREFUSED/ECONNRESET), so the implementation is
 * based on public documentation rather than live verification. It may need
 * adjustment to work once a reachable test URL is available.
 *
 * URL forms expected (based on dmzj's typical structure):
 *   https://www.dmzj.com/view/{comicId}/{chapterId}.html
 *   https://manhua.dmzj.com/{slug}/{chapterSlug}.shtml
 *   https://m.dmzj.com/view/{comicId}/{chapterId}.html
 *
 * API pattern (per dmzj's docs / 3rd-party scraper references):
 *   GET https://api.dmzj.com/dynamic/comicinfo/{comicId}.json
 *     → series metadata
 *   GET https://api.dmzj.com/comic/chapter/{comicId}/{chapterId}.json
 *     → { page_url: [...] } with full image URLs
 *
 *   Image host: https://images.dmzj.com/...
 *
 * If this adapter fails in production, falling back to the generic
 * Puppeteer flow will likely also fail (DMZJ's reader is JS-rendered).
 * Realistic path then: scope-cut DMZJ.
 */
import { SiteAdapter, ChapterImages, SeriesSearchResult } from './types';
import * as https from 'https';

// Two URL formats. The /view/ one has explicit numeric IDs we can grab.
const VIEW_RE = /^https?:\/\/(?:www|m)\.dmzj\.com\/view\/(\d+)\/(\d+)\.html/;
// Slug-form URLs require an extra series lookup; skip in this initial impl.

const REFERER = 'https://www.dmzj.com/';

function fetchJSON<T = any>(url: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const req = https.get(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json',
        'Referer': REFERER,
      },
    }, res => {
      if (res.statusCode !== 200) {
        reject(new Error(`DMZJ API HTTP ${res.statusCode}`));
        return;
      }
      let body = '';
      res.on('data', c => body += c);
      res.on('end', () => {
        try { resolve(JSON.parse(body)); }
        catch (e: any) { reject(new Error('DMZJ API JSON parse failed: ' + e.message)); }
      });
      res.on('error', reject);
    });
    req.on('error', reject);
    // Short timeout — DMZJ is often unreachable from outside China. Failing
    // fast lets the circuit breaker skip it for 5 minutes after 3 failures.
    req.setTimeout(3000, () => { req.destroy(); reject(new Error('DMZJ API timeout (site may be region-locked)')); });
  });
}

export const dmzjAdapter: SiteAdapter = {
  name: 'DMZJ',
  sourceId: 'dmzj',

  matches(url) {
    return VIEW_RE.test(url);
  },

  async getChapter(url): Promise<ChapterImages> {
    const m = url.match(VIEW_RE);
    if (!m) {
      const err: any = new Error('DMZJ URL must be of the form https://www.dmzj.com/view/{comicId}/{chapterId}.html');
      err.code = 'BAD_URL';
      throw err;
    }
    const comicId = m[1];
    const chapterId = m[2];

    try {
      const chap = await fetchJSON<any>(`https://api.dmzj.com/comic/chapter/${comicId}/${chapterId}.json`);
      // Response shape (per dmzj docs): { page_url: [...], chapter_name: ..., title: ... }
      const pages = chap?.page_url || chap?.chapter?.page_url || chap?.data?.page_url;
      if (!Array.isArray(pages) || pages.length === 0) {
        throw new Error('DMZJ API returned no page_url array');
      }
      const title = chap?.chapter_name || chap?.title || `DMZJ chapter ${chapterId}`;
      return {
        title,
        imageUrls: pages,
        imageHeaders: { Referer: REFERER },
      };
    } catch (e: any) {
      // Bubble up with a useful hint if connectivity is the issue
      if (e.message.includes('timeout') || e.message.includes('ECONN')) {
        const err: any = new Error('DMZJ unreachable. The site may be region-locked or temporarily down.');
        err.code = 'SITE_UNREACHABLE';
        throw err;
      }
      throw e;
    }
  },

  // DMZJ has a search-suggest endpoint at /api/v1/search/suggest. Returns
  // JSON with title + cover + comic_id. Like the chapter adapter, this is
  // untested (site unreachable from the dev environment). If the API path
  // changed or the response shape is different, results will be empty.
  async search(query: string): Promise<SeriesSearchResult[]> {
    try {
      const suggest = await fetchJSON<any>(`https://api.dmzj.com/api/v1/search/suggest?keyword=${encodeURIComponent(query)}`);
      const data = suggest?.data || suggest?.suggest || suggest?.results || suggest;
      if (!Array.isArray(data)) return [];
      return data.slice(0, 10).map((d: any): SeriesSearchResult => ({
        source: 'dmzj',
        title: d.name || d.title || d.comic_name || 'Untitled',
        seriesUrl: `https://www.dmzj.com/info/${d.comic_id || d.id}.html`,
        coverUrl: d.cover || d.cover_url,
      }));
    } catch (e: any) {
      // Site unreachable — circuit breaker handles this upstream.
      throw e;
    }
  },
};
