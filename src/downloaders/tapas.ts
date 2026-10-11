// Tapas.io adapter.
//
// Chapter download: handled by the GENERIC headless-browser fallback in
// webtoon-downloader.ts. Tapas serves comic images via standard <img> tags
// with signed URLs (us-a.tapas.io/c/...) after the page JS renders.
//
// Search: Tapas's search page is server-rendered HTML — no JSON API. We
// fetch /search?q=X and regex out series links + cover images.
import { SiteAdapter, SeriesSearchResult, ChapterListEntry } from './types';
import * as https from 'https';

// /series/{slug} links inside search results. Each result also has a
// preceding <img src="..."> for the cover; we capture them in one regex
// that anchors on the series link's <a> tag.
const SEARCH_RE = /<a[^>]+href="(\/series\/[^"\/?#]+)"[^>]*>\s*<img[^>]+src="([^"]+)"[^>]+alt="([^"]+)"/g;

// Tapas search HTML tags each result with `data-series-type="COMMUNITY"`
// for indie creator uploads vs `"ORIGINAL"` / `"PREMIUM"` for the curated
// catalog. Community series are usually small/personal projects we don't
// want surfacing for casual queries like "solo". Map series-slug → type by
// scanning the inner block for the data attribute.
const SERIES_TYPE_RE = /href="\/series\/([^"\/?#]+)"[\s\S]{0,2000}?data-series-type="([^"]+)"/g;
function buildSeriesTypeMap(html: string): Map<string, string> {
  const m = new Map<string, string>();
  let match: RegExpExecArray | null;
  SERIES_TYPE_RE.lastIndex = 0;
  while ((match = SERIES_TYPE_RE.exec(html)) !== null) {
    if (!m.has(match[1])) m.set(match[1], match[2]);
  }
  return m;
}

function fetchHTML(url: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const req = https.get(url, {
      headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36' },
    }, res => {
      if (res.statusCode !== 200) { reject(new Error(`Tapas HTTP ${res.statusCode}`)); return; }
      let body = '';
      res.on('data', c => body += c);
      res.on('end', () => resolve(body));
      res.on('error', reject);
    });
    req.on('error', reject);
    req.setTimeout(20000, () => { req.destroy(); reject(new Error('Tapas timeout')); });
  });
}

// Tapas wraps the matched search term in #_h_i_g_h_L_i_g_h_t_#...#/_h_i_g_h_L_i_g_h_t_#
// markers in the alt attribute. Strip them for a clean title. Also decode
// common HTML entities since Tapas HTML often has &#39; / &amp; etc. in
// titles ("Don't" → "Don&#39;t").
function cleanTitle(s: string): string {
  return s
    .replace(/#_h_i_g_h_L_i_g_h_t_#/g, '')
    .replace(/#\/_h_i_g_h_L_i_g_h_t_#/g, '')
    .replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&apos;/g, "'")
    .trim();
}

export const tapasAdapter: SiteAdapter = {
  name: 'Tapas',
  sourceId: 'tapas',
  matches: () => false, // delegate chapter downloads to generic Puppeteer fallback
  getChapter: async () => { throw new Error('Tapas adapter unreachable — should use generic fallback'); },

  // Chapter list: /series/{slug}/info renders all episodes server-side.
  // Each <li class="episode-list__item"> has data-is-charging="true|false"
  // (paid?) and data-is-wait-or-pay="true|false" (subscription gated?).
  // We mark locked=true if either is set, since both block our downloader.
  async getChapterList(seriesUrl: string): Promise<ChapterListEntry[]> {
    // Normalize: ensure /info suffix so we get the full episode list, not
    // the landing page that only shows the latest few.
    const u = seriesUrl.replace(/\/$/, '');
    const infoUrl = u.endsWith('/info') ? u : `${u}/info`;
    const html = await fetchHTML(infoUrl);
    const out: ChapterListEntry[] = [];
    // Match each episode block: href, data-id, paid flags, scene label, title.
    const EP_RE = /<a[^>]+href="(\/episode\/(\d+))"[^>]+data-is-wait-or-pay="(true|false)"[^>]+data-is-wuf="(true|false)"[^>]+data-is-wuf-plus="(true|false)"[^>]+data-is-charging="(true|false)"[\s\S]{0,2500}?<p\s+class="scene">([^<]+)<\/p>[\s\S]{0,500}?<span\s+class="title__body">([^<]+)<\/span>/g;
    let m: RegExpExecArray | null;
    EP_RE.lastIndex = 0;
    const seen = new Set<string>();
    while ((m = EP_RE.exec(html)) !== null) {
      const chapterUrl = `https://tapas.io${m[1]}`;
      if (seen.has(chapterUrl)) continue;
      seen.add(chapterUrl);
      const isWaitOrPay = m[3] === 'true';
      const isCharging = m[6] === 'true';
      const sceneLabel = m[7].trim().replace(/\s+/g, ' '); // "Episode 1"
      const title = cleanTitle(m[8]).trim();
      out.push({
        title: `${sceneLabel}: ${title}`,
        chapterUrl,
        locked: isWaitOrPay || isCharging,
        number: m[2],
      });
    }
    return out;
  },

  async search(query: string): Promise<SeriesSearchResult[]> {
    const html = await fetchHTML(`https://tapas.io/search?q=${encodeURIComponent(query)}`);
    const typeBySlug = buildSeriesTypeMap(html);
    const results: SeriesSearchResult[] = [];
    const seen = new Set<string>();
    let m: RegExpExecArray | null;
    SEARCH_RE.lastIndex = 0;
    while ((m = SEARCH_RE.exec(html)) !== null) {
      const slug = m[1].replace(/^\/series\//, '');
      const seriesUrl = `https://tapas.io${m[1]}`;
      if (seen.has(seriesUrl)) continue;
      seen.add(seriesUrl);
      // Drop indie community-uploaded comics. These dominate search results
      // for short queries with low-quality matches (e.g. "Solo-Q" for "solo").
      // ORIGINAL + PREMIUM are Tapas-curated series.
      const seriesType = typeBySlug.get(slug);
      if (seriesType === 'COMMUNITY') continue;
      results.push({
        source: 'tapas',
        title: cleanTitle(m[3]),
        seriesUrl,
        coverUrl: m[2],
      });
      if (results.length >= 10) break;
    }
    return results;
  },
};
