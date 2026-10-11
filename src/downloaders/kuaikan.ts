/**
 * KuaiKan (快看漫画) adapter.
 *
 * Chapter download: handled by the GENERIC headless-browser fallback in
 * webtoon-downloader.ts. KuaiKan serves chapter images via signed CDN URLs
 * that appear in standard <img> tags after JS renders.
 *
 * Search: KuaiKan's search results page (/sou/{query}) is server-rendered
 * with topic links — direct HTTP scrape works. Each result links to
 * /web/topic/{seriesId} (the series page), not a specific chapter.
 */
import { SiteAdapter, SeriesSearchResult } from './types';
import * as https from 'https';

// Match each search-result <a href="/web/topic/N" title="TITLE">. Cover URLs
// aren't in the static HTML for most results (lazy-loaded after JS); we set
// coverUrl undefined and let the frontend show a blank thumbnail.
const RESULT_RE = /<a\s+href="(\/web\/topic\/\d+)"\s+target="_blank"\s+title="([^"]+)"/g;

function fetchHTML(url: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const req = https.get(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html',
      },
    }, res => {
      if (res.statusCode !== 200) { reject(new Error(`KuaiKan HTTP ${res.statusCode}`)); return; }
      let body = '';
      res.on('data', c => body += c);
      res.on('end', () => resolve(body));
      res.on('error', reject);
    });
    req.on('error', reject);
    req.setTimeout(10000, () => { req.destroy(); reject(new Error('KuaiKan timeout')); });
  });
}

export const kuaikanAdapter: SiteAdapter = {
  name: 'KuaiKan',
  sourceId: 'kuaikan',
  matches: () => false, // chapter downloads use generic Puppeteer fallback
  getChapter: async () => { throw new Error('KuaiKan adapter unreachable — chapter downloads use generic flow'); },

  async search(query: string): Promise<SeriesSearchResult[]> {
    const html = await fetchHTML(`https://www.kuaikanmanhua.com/sou/${encodeURIComponent(query)}`);
    const results: SeriesSearchResult[] = [];
    const seen = new Set<string>();
    let m: RegExpExecArray | null;
    RESULT_RE.lastIndex = 0;
    while ((m = RESULT_RE.exec(html)) !== null) {
      const seriesUrl = `https://www.kuaikanmanhua.com${m[1]}`;
      if (seen.has(seriesUrl)) continue;
      seen.add(seriesUrl);
      results.push({
        source: 'kuaikan',
        title: m[2],
        seriesUrl,
      });
      if (results.length >= 10) break;
    }
    return results;
  },
};
