/**
 * Manhuagui (漫画柜) adapter.
 *
 * URL forms accepted:
 *   https://www.manhuagui.com/comic/{seriesId}/{chapterId}.html
 *   https://m.manhuagui.com/comic/{seriesId}/{chapterId}.html  (mobile mirror)
 *   https://tw.manhuagui.com/comic/{seriesId}/{chapterId}.html  (traditional)
 *
 * Strategy:
 *   The site uses the classic "Dean Edwards packer" to obfuscate the chapter's
 *   image file list. The packed IIFE is in an inline <script>; we extract it,
 *   run it in a Node vm sandbox (it just returns a JS string), then run the
 *   unpacked JS with a mocked SMH global that captures the imgData() call.
 *
 * Image URLs are built from { host, path, files, sl } in the captured data:
 *   {host}{path}{file}?e={sl.e}&m={sl.m}
 *
 * Default CDN host is eu1.hamreus.com (Manhuagui rotates CDNs; eu1 works
 * worldwide). Images are .webp regardless of the file extension in the list.
 *
 * Images REQUIRE Referer: https://www.manhuagui.com/ — without it the CDN
 * returns 403. We pass this via imageHeaders so the shared downloader uses it.
 */
import { SiteAdapter, ChapterImages, SeriesSearchResult, ChapterListEntry } from './types';
import * as https from 'https';
import * as vm from 'vm';

const CHAPTER_RE = /https?:\/\/(?:www|m|tw)\.manhuagui\.com\/comic\/\d+\/\d+\.html/;
// Match the eval(function(p,a,c,k,e,d){...return p;}('pkg', a, c, 'keys',
// 0, {})) IIFE. The outer wrapper is `window["\x65\x76\x61\x6c"](...)`
// (obfuscated `window.eval`). The body contains nested braces, so we anchor
// the function-body match on the literal `return p;}` epilogue.
const PACKER_RE = /window\["\\x65\\x76\\x61\\x6c"\]\((function\(p,a,c,k,e,d\)\{[\s\S]*?return p;\}\([\s\S]+?\}\))\)/;

const CDN_HOST = 'https://eu1.hamreus.com';
const REFERER = 'https://www.manhuagui.com/';

function fetchHTML(url: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const req = https.get(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
      },
    }, res => {
      if (res.statusCode !== 200) {
        reject(new Error(`Manhuagui HTTP ${res.statusCode}`));
        return;
      }
      let body = '';
      res.on('data', c => body += c);
      res.on('end', () => resolve(body));
      res.on('error', reject);
    });
    req.on('error', reject);
    // Very short timeout (3s) because Manhuagui frequently rate-limits/blocks
    // us at the IP level. When that happens, every request just hangs. Failing
    // fast lets the circuit breaker trip after ~9s of cumulative failures and
    // skip Manhuagui entirely for the next 5 minutes.
    req.setTimeout(3000, () => { req.destroy(); reject(new Error('Manhuagui timeout')); });
  });
}

// Match individual search-result <li> blocks. Each block has a `bcover` <a>
// with the series URL + title, plus an <img> for the cover. We grab those
// in one regex per result.
const SEARCH_RESULT_RE = /<a[^>]+class="bcover"[^>]+href="(\/comic\/\d+\/)"[^>]+title="([^"]+)"[^>]*>\s*<img[^>]+src="([^"]+)"/g;

// Chapter-list entry pattern. Manhuagui series pages list chapters as
// `<li><a href="/comic/{seriesId}/{chapterId}.html" title="ChTitle">ChTitle</a></li>`.
// Same chapter may appear multiple times (different volumes / formats); dedupe
// by URL. All chapters on Manhuagui are free (aggregator), so locked is false.
const CHAPTER_LIST_RE = /<a\s+href="(\/comic\/\d+\/\d+\.html)"\s+title="([^"]+)"[^>]*>([^<]+)<\/a>/g;

export const manhuaguiAdapter: SiteAdapter = {
  name: 'Manhuagui',
  sourceId: 'manhuagui',

  matches(url) {
    return CHAPTER_RE.test(url);
  },

  async getChapterList(seriesUrl: string): Promise<ChapterListEntry[]> {
    // Normalize: /comic/{id}/ or /comic/{id}/{whatever}.html — strip to series root.
    const m = seriesUrl.match(/(https?:\/\/(?:www|m|tw)\.manhuagui\.com\/comic\/\d+\/)/);
    if (!m) throw new Error('Not a Manhuagui series URL: ' + seriesUrl);
    const html = await fetchHTML(m[1]);
    const out: ChapterListEntry[] = [];
    const seen = new Set<string>();
    let match: RegExpExecArray | null;
    CHAPTER_LIST_RE.lastIndex = 0;
    while ((match = CHAPTER_LIST_RE.exec(html)) !== null) {
      const url = `https://www.manhuagui.com${match[1]}`;
      if (seen.has(url)) continue;
      seen.add(url);
      out.push({
        title: match[2],
        chapterUrl: url,
        locked: false, // Manhuagui is a free aggregator
      });
    }
    // Manhuagui usually lists chapters oldest-first; reverse so newest first.
    return out.reverse();
  },

  async search(query: string): Promise<SeriesSearchResult[]> {
    // Manhuagui's search URL is /s/{query-url-encoded}.html. CJK chars need
    // standard URI encoding; their server accepts UTF-8 percent-encoded queries.
    // When their IP-block kicks in this fails — handled by the circuit breaker
    // upstream so the user doesn't wait.
    const searchUrl = `https://www.manhuagui.com/s/${encodeURIComponent(query)}.html`;
    const html = await fetchHTML(searchUrl);
    const results: SeriesSearchResult[] = [];
    const seen = new Set<string>();
    let m: RegExpExecArray | null;
    SEARCH_RESULT_RE.lastIndex = 0;
    while ((m = SEARCH_RESULT_RE.exec(html)) !== null) {
      const seriesUrl = `https://www.manhuagui.com${m[1]}`;
      if (seen.has(seriesUrl)) continue;
      seen.add(seriesUrl);
      // Cover URLs come back protocol-relative (//cf.mhgui.com/...). Make absolute.
      const cover = m[3].startsWith('//') ? `https:${m[3]}` : m[3];
      results.push({
        source: 'manhuagui',
        title: m[2],
        seriesUrl,
        coverUrl: cover,
      });
      if (results.length >= 10) break;
    }
    return results;
  },

  async getChapter(url): Promise<ChapterImages> {
    const html = await fetchHTML(url);

    // Title from <title> tag, trimmed to drop the site suffix
    const titleMatch = html.match(/<title>([^<]+)<\/title>/);
    const title = (titleMatch?.[1] || 'manhuagui chapter').replace(/\s*-\s*看漫画\s*$/, '').trim();

    const packed = html.match(PACKER_RE);
    if (!packed) {
      const err: any = new Error('Manhuagui: could not find packed image data in chapter page (site may have changed format)');
      err.code = 'EXTRACTION_FAILED';
      throw err;
    }

    // Step 1: run the IIFE to get the unpacked JS string. The function is
    // a pure string transformation — no side effects — so an empty sandbox
    // with a 5s timeout is safe.
    // Manhuagui's packer calls 'BASE64'['\x73\x70\x6c\x69\x63']('\x7c') for
    // the keys array. `\x73\x70\x6c\x69\x63` decodes to "splic" — not a real
    // String method. The site's core.js polyfills it to do LZ-string-style
    // decompression. Easiest workaround: install a sandbox polyfill that
    // makes `splic` decompress the LZ-string-compressed packed-key blob.
    let unpackedCode: string;
    try {
      // Install LZ-string + String.prototype.splic in the sandbox before
      // running the packed IIFE. LZ-string decompress is exposed under
      // the global LZString variable.
      const sandbox: any = { console };
      // Lazy-require lz-string at runtime so we don't pay the import cost
      // on every adapter load.
      const LZString = require('lz-string');
      sandbox.LZString = LZString;
      // The polyfill: splic(sep) treats `this` as LZ-string-compressed and
      // returns the decoded string split by sep.
      const polyfill = `String.prototype.splic = function(sep) { return LZString.decompressFromBase64(String(this)).split(sep); };`;
      vm.createContext(sandbox);
      vm.runInContext(polyfill, sandbox);
      // Wrap in parens so vm parses it as an expression, not a function
      // statement (which would require a name and error out).
      unpackedCode = vm.runInContext('(' + packed[1] + ')', sandbox, { timeout: 5000 });
    } catch (e: any) {
      throw new Error('Manhuagui: failed to unpack image data: ' + e.message);
    }
    if (typeof unpackedCode !== 'string') {
      throw new Error('Manhuagui: unpacker returned non-string');
    }

    // Step 2: run the unpacked code with a mocked SMH global. The unpacked
    // code calls SMH.imgData({...}).preInit(). We capture the imgData arg.
    let imgData: any = null;
    const sandbox: any = {
      SMH: {
        imgData(d: any) {
          imgData = d;
          return { preInit() { /* no-op */ } };
        },
      },
    };
    try {
      vm.runInNewContext(unpackedCode, sandbox, { timeout: 5000 });
    } catch (e: any) {
      throw new Error('Manhuagui: failed to execute unpacked code: ' + e.message);
    }
    if (!imgData?.files?.length || !imgData?.path) {
      throw new Error('Manhuagui: unpacked imgData missing files/path');
    }

    // Step 3: build image URLs. The `sl` object has `e` (expiry timestamp)
    // and `m` (hmac sig) query params required for the CDN to serve the
    // image. Older chapters may have empty sl values, which is fine.
    const sl = imgData.sl || {};
    const qs = sl.m ? `?e=${sl.e || ''}&m=${encodeURIComponent(sl.m)}` : '';
    // Filenames from imgData.files are already URL-encoded by Manhuagui's
    // packer (parens, CJK chars are percent-encoded). DO NOT re-encode — the
    // CDN serves files using the exact percent-encoded names.
    const imageUrls: string[] = imgData.files.map((f: string) =>
      `${CDN_HOST}${imgData.path}${f}${qs}`
    );

    return {
      title,
      imageUrls,
      urlsExpire: !!sl.e,
      imageHeaders: { Referer: REFERER },
    };
  },
};
