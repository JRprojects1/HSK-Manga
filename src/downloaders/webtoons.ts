/**
 * Webtoons.com adapter — search + lightweight chapter metadata.
 *
 * Chapter downloads for Webtoons go through the static-HTML scrape fast path
 * in webtoon-downloader.ts (Webtoons server-renders `_images` containers with
 * data-url attributes — the original case the generic extractor was built for).
 * No chapter-extraction logic here.
 *
 * Chapter selection model:
 *   Webtoons URLs already contain a globally-unique chapter identifier
 *   (`episode_no=N`). The frontend builds chapter URLs directly from this
 *   number — no chapter-list scrape, no title parsing, no pagination logic.
 *   Trade-off: for series with prologues episode_no=1 is the prologue rather
 *   than "Chapter 1", and for multi-season series the user sees a global
 *   number rather than the per-season number on the source page. The "Open
 *   chapter to verify" link mitigates both — the user can confirm the URL
 *   before processing, and the URL bar on Webtoons shows episode_no directly.
 *
 * This file provides two adapters, one each for the EN and CN locales, so the
 * federated search can hit both endpoints in parallel and aggregate the
 * cross-language results.
 */
import { SiteAdapter, SeriesSearchResult } from './types';
import * as https from 'https';

const RESULT_RE = /<a\s+href="(https?:\/\/www\.webtoons\.com\/[^"]+\/list\?title_no=\d+)"[^>]*>\s*<div[^>]+>\s*<img[^>]+src="([^"]+)"[\s\S]*?<strong\s+class="title">([^<]+)<\/strong>/g;

function fetchHTML(url: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const req = https.get(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html',
        'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
      },
    }, res => {
      if (res.statusCode !== 200) { reject(new Error(`Webtoons HTTP ${res.statusCode}`)); return; }
      let body = '';
      res.on('data', c => body += c);
      res.on('end', () => resolve(body));
      res.on('error', reject);
    });
    req.on('error', reject);
    req.setTimeout(20000, () => { req.destroy(); reject(new Error('Webtoons timeout')); });
  });
}

function decodeEntities(s: string): string {
  return s.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>')
          .replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&apos;/g, "'");
}

// Lightweight series metadata: max episode_no on the series page. ONE fetch
// (page 1) tells us the latest chapter, which is also the upper bound of the
// "1 to N" range shown in the chapter picker. Replaces the prior 70+ page
// chapter-list scrape that took ~10 seconds for long series (Tower of God).
export async function fetchWebtoonsChapterCount(seriesUrl: string): Promise<number | null> {
  try {
    const html = await fetchHTML(seriesUrl);
    const nums = [...html.matchAll(/data-episode-no="(\d+)"/g)].map(m => parseInt(m[1], 10));
    if (nums.length === 0) return null;
    return Math.max(...nums);
  } catch {
    return null;
  }
}

// In-process cache. New chapters drop weekly so a 1-hour TTL is fine and
// keeps repeat searches instant. Cache dies on server restart which serves
// as the natural "refresh" trigger.
const chapterCountCache = new Map<string, { at: number; count: number | null }>();
const CACHE_TTL_MS = 60 * 60 * 1000;
export async function getWebtoonsChapterCountCached(seriesUrl: string): Promise<number | null> {
  const cached = chapterCountCache.get(seriesUrl);
  if (cached && Date.now() - cached.at < CACHE_TTL_MS) return cached.count;
  const count = await fetchWebtoonsChapterCount(seriesUrl);
  chapterCountCache.set(seriesUrl, { at: Date.now(), count });
  return count;
}

function parseResults(html: string, source: 'webtoons-en' | 'webtoons-cn'): SeriesSearchResult[] {
  const results: SeriesSearchResult[] = [];
  const seen = new Set<string>();
  let m: RegExpExecArray | null;
  RESULT_RE.lastIndex = 0;
  while ((m = RESULT_RE.exec(html)) !== null) {
    const seriesUrl = m[1];
    if (seen.has(seriesUrl)) continue;
    seen.add(seriesUrl);
    results.push({
      source,
      title: decodeEntities(m[3]).trim(),
      seriesUrl,
      coverUrl: m[2],
    });
    if (results.length >= 10) break;
  }
  return results;
}

// Webtoons adapters intentionally omit getChapterList — Webtoons chapters
// are selected by episode_no via direct URL construction in the frontend,
// not via paginated chapter-list scraping. The frontend uses the dedicated
// /api/webtoons-chapter-count endpoint to discover the chapter range.
export const webtoonsEnAdapter: SiteAdapter = {
  name: 'Webtoons EN',
  sourceId: 'webtoons-en',
  matches: () => false, // chapter downloads use the generic static-HTML scrape
  getChapter: async () => { throw new Error('Webtoons EN adapter unreachable — chapter downloads use generic flow'); },
  async search(query: string): Promise<SeriesSearchResult[]> {
    const html = await fetchHTML(`https://www.webtoons.com/en/search?keyword=${encodeURIComponent(query)}`);
    return parseResults(html, 'webtoons-en');
  },
};

export const webtoonsCnAdapter: SiteAdapter = {
  name: 'Webtoons CN',
  sourceId: 'webtoons-cn',
  matches: () => false,
  getChapter: async () => { throw new Error('Webtoons CN adapter unreachable — chapter downloads use generic flow'); },
  async search(query: string): Promise<SeriesSearchResult[]> {
    const html = await fetchHTML(`https://www.webtoons.com/zh-hant/search?keyword=${encodeURIComponent(query)}`);
    return parseResults(html, 'webtoons-cn');
  },
};
