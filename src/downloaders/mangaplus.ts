/**
 * MangaPlus (Shueisha official, free, English/Spanish/French/Thai) adapter.
 *
 * Why Puppeteer instead of their gRPC API: MangaPlus's backend speaks
 * protobuf over HTTP and would require schema definitions + a decoder. The
 * web UI exposes the same data in plain HTML/IMG tags after JS renders, so
 * a headless-browser scrape gets us search + chapter images without the
 * protobuf integration.
 *
 * Strategy:
 *   - Search: Load /manga_list/all (their complete catalog page) once, scrape
 *     all ~110 ongoing series as { id, title, coverUrl }, cache in memory for
 *     1 hour, do fuzzy match locally on subsequent searches.
 *   - Chapter list: Series page /titles/{id} shows the chapter list with
 *     viewer URLs. Scrape with Puppeteer.
 *   - Chapter download: Reader page /viewer/{chapterId} renders the chapter's
 *     images via JS. Wait for them to load, scrape <img> tags.
 *
 * Coverage scope: Japanese manga officially translated to English / Spanish /
 * French / Thai. Excellent for major Shonen Jump series (Kagurabachi, One
 * Piece, Sakamoto Days, etc.). All free, no auth.
 */
import { SiteAdapter, ChapterImages, SeriesSearchResult, ChapterListEntry } from './types';

const BASE = 'https://mangaplus.shueisha.co.jp';

// Catalog cache. Map<lowercase-title, entry> for O(1) lookups + a flat list
// for substring matching. Refresh every CATALOG_TTL_MS.
interface CatalogEntry { id: string; title: string; coverUrl?: string; author?: string; }
let catalogCache: CatalogEntry[] | null = null;
let catalogCachedAt = 0;
const CATALOG_TTL_MS = 60 * 60 * 1000; // 1 hour
let catalogFetchPromise: Promise<CatalogEntry[]> | null = null;

declare const document: any;
declare const window: any;

async function loadCatalog(): Promise<CatalogEntry[]> {
  if (catalogCache && Date.now() - catalogCachedAt < CATALOG_TTL_MS) return catalogCache;
  if (catalogFetchPromise) return catalogFetchPromise;
  catalogFetchPromise = (async () => {
    const puppeteer = await import('puppeteer');
    const browser = await puppeteer.default.launch({
      headless: true,
      args: ['--no-sandbox', '--disable-setuid-sandbox'],
    });
    try {
      const page = await browser.newPage();
      await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36');
      await page.goto(`${BASE}/manga_list/all`, { waitUntil: 'networkidle2', timeout: 25000 });
      // Brief wait for the title-list React component to fully render.
      await new Promise(r => setTimeout(r, 1500));
      const entries: CatalogEntry[] = await page.evaluate(() => {
        const links = Array.from(document.querySelectorAll('a[href*="/titles/"]')) as any[];
        const out: any[] = [];
        for (const a of links) {
          const m = a.href.match(/\/titles\/(\d+)/);
          if (!m) continue;
          const img = a.querySelector('img');
          // The clean title is in the alt attribute; author is a separate <p>.
          const title = (img?.alt || '').trim();
          if (!title) continue;
          // Authors are in <p> with class containing 'author' — fuzzy class match.
          const ps = Array.from(a.querySelectorAll('p')) as any[];
          let author: string | undefined;
          for (const p of ps) {
            const cls = (p.className || '') as string;
            if (cls.includes('author')) { author = (p.textContent || '').trim(); break; }
          }
          out.push({
            id: m[1],
            title,
            coverUrl: img?.src || undefined,
            author,
          });
        }
        return out;
      });
      catalogCache = entries;
      catalogCachedAt = Date.now();
      return entries;
    } finally {
      await browser.close();
      catalogFetchPromise = null;
    }
  })();
  return catalogFetchPromise;
}

// Read a single page (series page OR viewer page) with Puppeteer. Used for
// chapter list + chapter download. Wraps the boilerplate.
async function withPage<T>(url: string, fn: (page: any) => Promise<T>, waitMs: number = 1500): Promise<T> {
  const puppeteer = await import('puppeteer');
  const browser = await puppeteer.default.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });
  try {
    const page = await browser.newPage();
    await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36');
    await page.setViewport({ width: 1280, height: 1600 });
    await page.goto(url, { waitUntil: 'networkidle2', timeout: 30000 });
    await new Promise(r => setTimeout(r, waitMs));
    return await fn(page);
  } finally {
    await browser.close();
  }
}

export const mangaplusAdapter: SiteAdapter = {
  name: 'MangaPlus',
  sourceId: 'mangaplus',

  matches(url: string): boolean {
    return /^https?:\/\/mangaplus\.shueisha\.co\.jp\/viewer\/\d+/.test(url);
  },

  // Chapter download: navigate to viewer URL, wait for images to load, scrape.
  async getChapter(url: string): Promise<ChapterImages> {
    const title = `MangaPlus ${url.match(/\/viewer\/(\d+)/)?.[1] || 'chapter'}`;
    const imageUrls = await withPage(url, async page => {
      // Scroll all the way through to trigger lazy-loaded pages.
      await page.evaluate(async () => {
        await new Promise<void>(resolve => {
          let stable = 0, iter = 0, lastCount = 0;
          const t = setInterval(() => {
            iter++;
            window.scrollBy(0, 1200);
            const c = document.querySelectorAll('img').length;
            if (c > lastCount) { lastCount = c; stable = 0; } else stable++;
            if (stable >= 8 || iter >= 100) { clearInterval(t); resolve(); }
          }, 300);
        });
      });
      await new Promise(r => setTimeout(r, 2500));
      return await page.evaluate(() => {
        const imgs = Array.from(document.querySelectorAll('img')) as any[];
        return imgs
          .map(i => ({ src: i.currentSrc || i.src || '', w: i.naturalWidth, h: i.naturalHeight }))
          .filter(i => i.src && i.w >= 500 && i.h >= 500 && /jumpg-assets|cdn\.tokyo-cdn/.test(i.src))
          .map(i => i.src);
      });
    }, 2500);
    return {
      title,
      imageUrls,
      urlsExpire: true,
      imageHeaders: { Referer: BASE + '/' },
    };
  },

  // Search: load the cached catalog and fuzzy-match by title substring.
  async search(query: string): Promise<SeriesSearchResult[]> {
    const catalog = await loadCatalog();
    const q = query.toLowerCase().trim();
    if (!q) return [];
    // Substring + word-boundary scoring. Exact-match prefix → highest; substring
    // anywhere → lower; nothing → drop.
    const scored: Array<{ score: number; e: CatalogEntry }> = [];
    for (const e of catalog) {
      const t = e.title.toLowerCase();
      if (t === q) { scored.push({ score: 100, e }); continue; }
      if (t.startsWith(q + ' ') || t.startsWith(q + '-') || t.startsWith(q + ':')) { scored.push({ score: 80, e }); continue; }
      // Token presence
      const queryTokens = q.split(/\s+/).filter(Boolean);
      const titleTokens = t.split(/\s+/);
      const allTokens = queryTokens.every(qt => titleTokens.some(tt => tt.startsWith(qt)));
      if (allTokens) { scored.push({ score: 50, e }); continue; }
      if (t.includes(q)) { scored.push({ score: 30, e }); continue; }
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, 10).map(s => ({
      source: 'mangaplus',
      title: s.e.title,
      seriesUrl: `${BASE}/titles/${s.e.id}`,
      coverUrl: s.e.coverUrl,
      description: s.e.author,
    }));
  },

  // Chapter list: NOT IMPLEMENTED. MangaPlus's series page uses React with
  // onclick handlers for chapter navigation (no plain <a href="/viewer/...">
  // tags), so DOM scraping returns nothing. The actual data lives in their
  // protobuf-encoded title_detailV3 API which we don't decode.
  //
  // Workaround: search result still surfaces MangaPlus as a source. Clicking
  // it opens the MangaPlus series page in a new tab; the user picks a
  // chapter there and pastes the /viewer/X URL into our URL input to
  // process. Less polished than the inline chapter list other sources have,
  // but functional.
  //
  // Future fix: implement a minimal protobuf decoder for the chapter list
  // response shape, OR brute-force click each chapter button and capture
  // navigation URLs (slow but generic).
  async getChapterList(_seriesUrl: string): Promise<ChapterListEntry[]> {
    const err: any = new Error('MangaPlus chapter list isn\'t supported yet — open the series in a new tab and paste a chapter URL directly.');
    err.code = 'CHAPTER_LIST_UNAVAILABLE';
    throw err;
  },
};
