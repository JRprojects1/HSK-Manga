/**
 * AniList title resolver — secondary translation source.
 *
 * Fills MangaDex's coverage gaps. AniList has stronger indexing for modern
 * Korean manhwa and Chinese manhua, and exposes more alternate-language
 * titles per entry (often including Simplified + Traditional Chinese).
 *
 * NOT a full SiteAdapter — we don't download chapters from AniList (they
 * don't host content). Only used to discover CJK title variants for queries
 * MangaDex doesn't index well.
 *
 * Free public GraphQL API at https://graphql.anilist.co. No auth required
 * for read-only queries. Generous rate limit (~90 req/min/IP).
 */
import * as https from 'https';

export interface TitleAliases {
  /** Display title (English preferred, falls back to romaji/native). */
  primary: string;
  /** All known title variants across languages. Includes the primary. */
  alts: string[];
}

const ENDPOINT = 'https://graphql.anilist.co';
// Search query: returns up to 5 manga matches with their title fields +
// synonyms. We only need title data, not chapters/coverArt — those come
// from MangaDex when it has them.
const SEARCH_QUERY = `
query ($q: String) {
  Page(perPage: 5) {
    media(search: $q, type: MANGA) {
      title { english native romaji }
      synonyms
    }
  }
}`;

function postGraphQL<T = any>(query: string, variables: object): Promise<T> {
  const payload = JSON.stringify({ query, variables });
  return new Promise((resolve, reject) => {
    const u = new URL(ENDPOINT);
    const req = https.request({
      method: 'POST',
      hostname: u.hostname,
      path: u.pathname,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Content-Length': Buffer.byteLength(payload),
        'User-Agent': 'HSKManga/1.0',
      },
    }, res => {
      if (res.statusCode !== 200) {
        reject(new Error(`AniList HTTP ${res.statusCode}`));
        return;
      }
      let body = '';
      res.on('data', c => body += c);
      res.on('end', () => {
        try { resolve(JSON.parse(body)); }
        catch (e: any) { reject(new Error('AniList JSON parse failed: ' + e.message)); }
      });
      res.on('error', reject);
    });
    req.on('error', reject);
    // Short timeout — AniList is fast (~200-500ms) when reachable; failing
    // quickly here matters because we only call AniList as a fallback.
    req.setTimeout(3000, () => { req.destroy(); reject(new Error('AniList timeout')); });
    req.write(payload);
    req.end();
  });
}

/**
 * Look up a series by name on AniList. Returns up to 5 candidate matches
 * with their multilingual title aliases. Caller decides which (if any)
 * matches the original query and feeds altTitles into the search aggregator.
 */
export async function aniListLookup(query: string): Promise<TitleAliases[]> {
  let resp: any;
  try {
    resp = await postGraphQL(SEARCH_QUERY, { q: query });
  } catch (e) {
    return [];
  }
  const media: any[] = resp?.data?.Page?.media || [];
  return media.map((m: any): TitleAliases => {
    const titles: string[] = [];
    if (m.title?.english) titles.push(m.title.english);
    if (m.title?.romaji) titles.push(m.title.romaji);
    if (m.title?.native) titles.push(m.title.native);
    if (Array.isArray(m.synonyms)) for (const s of m.synonyms) if (typeof s === 'string') titles.push(s);
    const primary = m.title?.english || m.title?.romaji || m.title?.native || 'Untitled';
    return { primary, alts: titles };
  });
}
