import * as fs from 'fs';
import * as path from 'path';
import * as https from 'https';
import * as http from 'http';
import pLimit from 'p-limit';
import { findAdapter } from './downloaders';

// Configurable download concurrency (default 5 parallel downloads)
const DOWNLOAD_CONCURRENCY = parseInt(process.env.DOWNLOAD_CONCURRENCY || '5', 10);

export interface WebtoonChapter {
  title: string;
  url: string;
  images: string[];
}

export interface DownloadResult {
  chapterTitle: string;
  imagePaths: string[];
  outputDir: string;
  totalImages: number;
  downloadedImages: number;
  failedImages: number;
  timing?: DownloadTiming;
}

export interface DownloadTiming {
  totalWallTimeMs: number;
  cumulativeDownloadTimeMs: number;
  avgPerImageMs: number;
  parallelism: number;
}

/**
 * Download an image from a URL to a file
 */
async function downloadImage(
  url: string,
  outputPath: string,
  referer?: string
): Promise<boolean> {
  return new Promise((resolve) => {
    const protocol = url.startsWith('https://') ? https : http;

    const options: http.RequestOptions = {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
        // Percent-encode the Referer. Node's HTTP module enforces strict
        // ASCII in header values, so a URL containing raw non-ASCII (common
        // when a user pastes a chapter URL from WeChat's in-app browser,
        // which copies the decoded form) crashes the request with
        // ERR_INVALID_CHAR. encodeURI leaves already-encoded URLs untouched.
        ...(referer && { 'Referer': encodeURI(referer) }),
      },
    };

    const request = protocol.get(url, options, (response) => {
      // Handle redirects
      if (response.statusCode === 301 || response.statusCode === 302) {
        const redirectUrl = response.headers.location;
        if (redirectUrl) {
          downloadImage(redirectUrl, outputPath, referer).then(resolve);
          return;
        }
      }

      if (response.statusCode !== 200) {
        console.error(`  Failed to download: HTTP ${response.statusCode}`);
        resolve(false);
        return;
      }

      const fileStream = fs.createWriteStream(outputPath);
      response.pipe(fileStream);

      fileStream.on('finish', () => {
        fileStream.close();
        resolve(true);
      });

      fileStream.on('error', (err) => {
        fs.unlink(outputPath, () => {}); // Delete partial file
        console.error(`  Write error: ${err.message}`);
        resolve(false);
      });
    });

    request.on('error', (err) => {
      console.error(`  Download error: ${err.message}`);
      resolve(false);
    });

    request.setTimeout(30000, () => {
      request.destroy();
      console.error(`  Download timeout`);
      resolve(false);
    });
  });
}

/**
 * Fetch HTML content from a URL
 */
async function fetchHTML(url: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const protocol = url.startsWith('https://') ? https : http;

    const options: http.RequestOptions = {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
      },
    };

    const request = protocol.get(url, options, (response) => {
      // Handle redirects
      if (response.statusCode === 301 || response.statusCode === 302) {
        const redirectUrl = response.headers.location;
        if (redirectUrl) {
          // Handle relative redirects
          const fullUrl = redirectUrl.startsWith('http')
            ? redirectUrl
            : new URL(redirectUrl, url).href;
          fetchHTML(fullUrl).then(resolve).catch(reject);
          return;
        }
      }

      if (response.statusCode !== 200) {
        // 403/429/503 are anti-bot signatures from manga sites that detect
        // automated access. Tag with SITE_BLOCKED so the frontend can render
        // a useful message ("try a different reading source") rather than the
        // generic "HTTP 403" string.
        const blocked = [403, 429, 503].includes(response.statusCode || 0);
        const err: any = new Error(blocked
          ? `Site blocked the request (HTTP ${response.statusCode})`
          : `HTTP ${response.statusCode}`);
        if (blocked) err.code = 'SITE_BLOCKED';
        reject(err);
        return;
      }

      let body = '';
      response.on('data', (chunk) => (body += chunk));
      response.on('end', () => resolve(body));
      response.on('error', reject);
    });

    request.on('error', reject);
    request.setTimeout(30000, () => {
      request.destroy();
      const err: any = new Error('Request timeout fetching chapter HTML');
      err.code = 'SITE_BLOCKED'; // upstream was unreachable/silent — same UX bucket
      reject(err);
    });
  });
}

/**
 * Extract image URLs from a webtoon/manhua page
 * Supports multiple common platforms
 */
export function extractImageUrls(html: string, baseUrl: string): string[] {
  const images: string[] = [];
  const seen = new Set<string>();

  // Common patterns for webtoon/manhua image containers
  const patterns = [
    // WEBTOON official
    /<img[^>]+class="[^"]*_images[^"]*"[^>]+data-url="([^"]+)"/gi,
    // Generic manga readers - img tags with data-src (lazy loading)
    /<img[^>]+data-src="([^"]+\.(jpg|jpeg|png|webp|gif)(\?[^"]*)?)"/gi,
    // Direct src images in content areas
    /<img[^>]+src="([^"]+\.(jpg|jpeg|png|webp|gif)(\?[^"]*)?)"/gi,
    // Background images
    /background-image:\s*url\(['"]?([^'")\s]+\.(jpg|jpeg|png|webp|gif)(\?[^'")\s]*)?)['"]?\)/gi,
  ];

  for (const pattern of patterns) {
    let match;
    while ((match = pattern.exec(html)) !== null) {
      let imgUrl = match[1];
      const lower = imgUrl.toLowerCase();

      // Drop UI noise by URL keywords. Most webtoon sites cluster these in
      // predictable path segments. `/icon`, `/logo`, `/button` are common;
      // `/avatar`, `/profile`, `/thumb` and `/sa/` (small avatar), `/ua/`
      // (user avatar), `/resources/` (site assets) catch the rest. We anchor
      // most on a leading `/` so we don't false-match a real comic URL that
      // happens to contain the substring (e.g. a comic titled "Iconoclast").
      if (lower.includes('/icon') || lower.includes('/logo') || lower.includes('button')) continue;
      if (lower.includes('/avatar') || lower.includes('/profile') || lower.includes('thumb')) continue;
      if (lower.includes('/sa/') || lower.includes('/ua/') || lower.includes('/resources/')) continue;
      if (lower.includes('/default')) continue; // defaultuser-XXX.png on Tapas etc.

      // Convert relative URLs to absolute
      if (!imgUrl.startsWith('http')) {
        try {
          imgUrl = new URL(imgUrl, baseUrl).href;
        } catch {
          continue;
        }
      }

      // Deduplicate
      if (!seen.has(imgUrl)) {
        seen.add(imgUrl);
        images.push(imgUrl);
      }
    }
  }

  return images;
}

/**
 * Parse chapter title from HTML
 */
function extractChapterTitle(html: string, url: string): string {
  // Try common title patterns
  const patterns = [
    // og:title is the cleanest source — the episode/series name as the site
    // itself presents it. Preferred over <title>, which often has site suffixes.
    /<meta[^>]+property=["']og:title["'][^>]+content=["']([^"']+)["']/i,
    /<title>([^<]+)<\/title>/i,
    /<h1[^>]*>([^<]+)<\/h1>/i,
    /<h2[^>]*class="[^"]*title[^"]*"[^>]*>([^<]+)<\/h2>/i,
    /class="[^"]*episode[^"]*title[^"]*"[^>]*>([^<]+)</i,
  ];

  for (const pattern of patterns) {
    const match = html.match(pattern);
    if (match && match[1]) {
      // Clean up the title
      let title = match[1].trim();
      // Remove common suffixes
      title = title.replace(/\s*[-|]\s*WEBTOON$/i, '');
      title = title.replace(/\s*[-|]\s*LINE WEBTOON$/i, '');
      if (title.length > 0 && title.length < 200) {
        return title;
      }
    }
  }

  // Fallback: use URL path
  try {
    const urlObj = new URL(url);
    const pathParts = urlObj.pathname.split('/').filter(Boolean);
    return pathParts[pathParts.length - 1] || 'chapter';
  } catch {
    return 'chapter';
  }
}

/**
 * Headless-browser fallback for JS-rendered chapter pages (Bilibili Manga,
 * KuaiKan, Manhuagui, DMZJ etc. all serve near-empty HTML shells and inject
 * image URLs after JS executes). Used only when the static regex extractor
 * returns no images so the fast path stays fast for sites like Webtoons.
 *
 * Spawns Puppeteer per call rather than sharing a browser. Per-call launch
 * costs ~1-2s but avoids leaking a Chromium process if the pipeline crashes
 * and avoids "stale browser" failure modes. Can be pooled later if call
 * volume warrants it.
 */
// Browser globals used inside page.evaluate() bodies. tsc compiles in a Node
// lib context so these names need ambient declarations — the actual evaluate
// functions execute inside Chromium where the real globals exist.
declare const document: any;
declare const window: any;

async function extractImagesWithBrowser(url: string): Promise<string[]> {
  // Dynamic import so a missing/broken Chromium install doesn't crash the
  // whole module — only the browser fallback path errors out.
  const puppeteer = await import('puppeteer');
  const browser = await puppeteer.default.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-blink-features=AutomationControlled'],
  });
  try {
    const page = await browser.newPage();
    await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36');
    await page.setViewport({ width: 1280, height: 2400 });

    // networkidle2 = wait until <=2 network connections for 500ms. Better than
    // domcontentloaded for sites that lazy-fetch images.
    await page.goto(url, { waitUntil: 'networkidle2', timeout: 45000 });

    // Lazy-loaded vertical readers (KuaiKan, Webtoons, generic manhua) load
    // each image only after it enters the viewport (via IntersectionObserver
    // or scroll listeners). The observer fires instantly on ANY scroll past
    // an element — fetch latency is the real cost, not scroll speed. So:
    //
    //   Phase 1 — Rapid scroll to the bottom (kicks off every observer/fetch).
    //   Phase 2 — One settle wait to let queued fetches complete.
    //   Phase 3 — Batched fix-up for any images still at naturalWidth==0:
    //             scroll all of them into view in a single pass with no
    //             per-image delay, then one wait for the batch.
    //
    // Old design did "slow scroll + per-image scrollIntoView with 120ms wait"
    // which spent ~40s on a 113-page chapter. New design hits the same count
    // in ~8-12s by leaning on the observer's instant fire and batching waits.

    // Phase 1: scroll to the bottom, paced by imgCount growth. KuaiKan-style
    // sites only inject the next batch of <img> elements after the current
    // batch finishes loading. If you scroll faster than they load, the next
    // batch never gets created → lost pages.
    //
    // Strategy: scroll one viewport per tick. If imgCount grew OR scroll
    // advanced, keep going at the same pace. Bail when both stay flat for
    // STUCK_THRESHOLD ticks (we're at the bottom AND no more batches coming).
    await page.evaluate(async () => {
      await new Promise<void>(resolve => {
        let lastY = -1;
        let lastImgCount = 0;
        let stuckTicks = 0;
        let iterations = 0;
        const MAX_ITERATIONS = 200; // hard cap: 200 × 150ms = 30s
        const STUCK_THRESHOLD = 5;  // 750ms flat = done
        const step = 2400;
        const timer = setInterval(() => {
          iterations++;
          window.scrollBy(0, step);
          const y = window.scrollY;
          const imgCount = document.querySelectorAll('img').length;
          if (y > lastY || imgCount > lastImgCount) {
            lastY = y;
            lastImgCount = imgCount;
            stuckTicks = 0;
          } else {
            stuckTicks++;
          }
          if (stuckTicks >= STUCK_THRESHOLD || iterations >= MAX_ITERATIONS) {
            clearInterval(timer);
            resolve();
          }
        }, 150);
      });
    });

    // Phase 2: brief settle for in-flight fetches kicked off during scroll.
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await new Promise(r => setTimeout(r, 800));

    // Phase 3: batched fix-up. Scroll every still-unloaded <img> into view
    // in a tight loop (no per-image wait — observers fire instantly), then
    // one settle wait for the whole batch's fetches.
    const unloadedCount = await page.evaluate(async () => {
      const unloaded = Array.from(document.querySelectorAll('img'))
        .filter((img: any) => img.naturalWidth === 0 && img.naturalHeight === 0)
        .slice(0, 200);
      for (const img of unloaded) {
        (img as any).scrollIntoView({ block: 'center' });
      }
      return unloaded.length;
    });
    if (unloadedCount > 0) {
      await new Promise(r => setTimeout(r, 2000));
      // Tail pass for stragglers (slow CDN, big file).
      const tailRemaining = await page.evaluate(async () => {
        const stillUnloaded = Array.from(document.querySelectorAll('img'))
          .filter((img: any) => img.naturalWidth === 0 && img.naturalHeight === 0)
          .slice(0, 30);
        for (const img of stillUnloaded) {
          (img as any).scrollIntoView({ block: 'center' });
          await new Promise(r => setTimeout(r, 80));
        }
        return stillUnloaded.length;
      });
      if (tailRemaining > 0) await new Promise(r => setTimeout(r, 700));
    }

    // Pull every <img> the rendered DOM exposes, with size hints. We use the
    // size to filter UI icons (logos, avatars, tab buttons) from real comic
    // pages. After full scroll, real pages have loaded so naturalWidth is
    // accurate; icons are typically <200px wide, comic pages 600px+.
    const rawImages: Array<{ src: string; w: number; h: number }> = await page.evaluate(() => {
      const imgs = Array.from(document.querySelectorAll('img'));
      return imgs
        .map((img: any) => ({
          src: img.currentSrc || img.src || '',
          w: img.naturalWidth || 0,
          h: img.naturalHeight || 0,
        }))
        .filter((i: any) => !!i.src);
    });

    // Filter: drop UI icons by both URL noise-keywords AND size. Comic pages
    // are usually ≥400px wide. Anything smaller is almost certainly chrome.
    // 0-dimension images (didn't load) are dropped too — if we can't tell
    // their size after a 30s scroll, they're not real content.
    const seen = new Set<string>();
    const result: string[] = [];
    for (const img of rawImages) {
      let imgUrl = img.src;
      if (!imgUrl || imgUrl.startsWith('data:')) continue;
      const lower = imgUrl.toLowerCase();
      if (lower.includes('/icon') || lower.includes('logo') || lower.includes('button')) continue;
      if (lower.includes('avatar') || lower.includes('profile') || lower.includes('thumb')) continue;
      // Size filter: width is the reliable signal — real comic pages are
      // 800-1600px wide; icons/avatars cluster around 100-450px. Height has
      // no such floor: sites like KuaiKan serve narrow transition strips
      // (1280x103 bubble curves, 1280x45 panel separators, etc.) as part of
      // the chapter. The old `h < 500` threshold dropped ~7% of KuaiKan
      // content. The 30px floor here keeps out 1-pixel layout artifacts /
      // hairline dividers but admits every substantive chunk.
      if (img.w < 500 || img.h < 30) continue;
      if (!imgUrl.startsWith('http')) {
        try { imgUrl = new URL(imgUrl, url).href; } catch { continue; }
      }
      if (seen.has(imgUrl)) continue;
      seen.add(imgUrl);
      result.push(imgUrl);
    }
    return result;
  } finally {
    await browser.close();
  }
}

/**
 * Sanitize filename for Windows/Mac/Linux
 */
function sanitizeFilename(name: string): string {
  return name
    .replace(/[<>:"/\\|?*]/g, '_')
    .replace(/\s+/g, '_')
    .substring(0, 100);
}

/**
 * Download a webtoon/manhua chapter from a URL
 */
export async function downloadWebtoonChapter(
  url: string,
  outputDir: string
): Promise<DownloadResult> {
  console.error(`Fetching chapter page: ${url}`);

  // First: check for a per-site adapter. Sites with adapters bypass the
  // generic HTML/Puppeteer flow entirely — they know exactly how to talk
  // to their site (API, deobfuscation, signed URLs, etc.) and return clean
  // image URLs deterministically.
  const adapter = findAdapter(url);
  let chapterTitle: string;
  let imageUrls: string[];
  let imageHeaders: Record<string, string> | undefined;

  if (adapter) {
    console.error(`Using ${adapter.name} adapter`);
    const result = await adapter.getChapter(url);
    chapterTitle = result.title;
    imageUrls = result.imageUrls;
    imageHeaders = result.imageHeaders;
    console.error(`${adapter.name} adapter found ${imageUrls.length} images`);
  } else {
    // Generic fallback: fetch the chapter HTML, try regex extraction first,
    // then headless browser if static scrape returns nothing.
    const html = await fetchHTML(url);
    chapterTitle = extractChapterTitle(html, url);
    console.error(`Chapter: ${chapterTitle}`);

    imageUrls = extractImageUrls(html, url);
    console.error(`Static scrape found ${imageUrls.length} images`);
    // Fall back to Puppeteer when static scrape returns a suspiciously low
    // count. Real manga chapters are 10+ pages; lower counts usually mean
    // we're getting UI icons (logos, avatars, share buttons) from a JS-
    // rendered page where the actual comic images load via JS after page load.
    const STATIC_THRESHOLD = 10;
    if (imageUrls.length < STATIC_THRESHOLD) {
      console.error(`Static count below threshold (${STATIC_THRESHOLD}); falling back to headless browser…`);
      const browserStart = Date.now();
      const browserUrls = await extractImagesWithBrowser(url);
      console.error(`Browser fallback found ${browserUrls.length} images in ${((Date.now() - browserStart) / 1000).toFixed(1)}s`);
      // Prefer browser results if it found more — otherwise stick with what
      // static scrape gave us (handles "short chapter" edge cases where the
      // chapter genuinely has <10 images).
      if (browserUrls.length > imageUrls.length) imageUrls = browserUrls;
    }
  }

  const safeTitle = sanitizeFilename(chapterTitle);
  console.error(`Chapter: ${chapterTitle}`);

  if (imageUrls.length === 0) {
    // Phase 1.7: typed error so the frontend can render a specific message.
    // Most common cause is the user pasting a homepage URL instead of a chapter.
    const err: any = new Error('No chapter images at this URL. Did you paste a homepage instead of a specific chapter?');
    err.code = 'NO_IMAGES_FOUND';
    throw err;
  }

  // Phase 1.3 (per-URL page cap): bound the cost of any single submission.
  // Default 300 pages covers virtually every legitimate chapter (webtoon: 50-200,
  // manga: 20-50, manhua: 50-100). Anything bigger is likely a multi-chapter
  // archive or abuse — reject before consuming GPU/LLM resources.
  const MAX_PAGES_PER_URL = parseInt(process.env.MAX_PAGES_PER_URL || '300', 10);
  if (imageUrls.length > MAX_PAGES_PER_URL) {
    const err: any = new Error(`This URL has ${imageUrls.length} pages, max is ${MAX_PAGES_PER_URL}. Submit a single chapter, not an archive.`);
    err.code = 'TOO_MANY_PAGES';
    err.pageCount = imageUrls.length;
    err.maxPages = MAX_PAGES_PER_URL;
    throw err;
  }

  // Create output directory
  const chapterDir = path.join(outputDir, safeTitle);
  if (!fs.existsSync(chapterDir)) {
    fs.mkdirSync(chapterDir, { recursive: true });
  }

  // Download images in parallel with timing
  const downloadStartTime = Date.now();
  let cumulativeDownloadTime = 0;
  const limit = pLimit(DOWNLOAD_CONCURRENCY);

  console.error(`  Downloading with concurrency=${DOWNLOAD_CONCURRENCY}`);

  // Whitelist of known image extensions. Anything else (e.g. KuaiKan's
  // `.h` CDN transform suffix, query-string artifacts in the pathname,
  // novelty extensions) gets normalized to `.jpg` so downstream tools that
  // sniff by extension don't choke. The actual file bytes are content-
  // sniffed by Sharp regardless, so this is mostly cosmetic for the
  // pipeline but matters for user-visible file lists.
  const KNOWN_IMG_EXTS = new Set(['.jpg', '.jpeg', '.png', '.webp', '.gif']);
  const downloadPromises = imageUrls.map((imgUrl, i) =>
    limit(async () => {
      const rawExt = path.extname(new URL(imgUrl).pathname).toLowerCase();
      const ext = KNOWN_IMG_EXTS.has(rawExt) ? rawExt : '.jpg';
      const filename = `${String(i + 1).padStart(3, '0')}${ext}`;
      const outputPath = path.join(chapterDir, filename);

      const imageStartTime = Date.now();
      // Adapter-provided headers (e.g. Bilibili's image CDN requires a
      // specific Referer) override the default referer behavior.
      const refererForImage = imageHeaders?.['Referer'] || imageHeaders?.['referer'] || url;
      const success = await downloadImage(imgUrl, outputPath, refererForImage);
      const imageTime = Date.now() - imageStartTime;

      console.error(`  Downloaded ${i + 1}/${imageUrls.length}: ${filename} (${imageTime}ms)`);

      return { success, outputPath, imageTime, index: i };
    })
  );

  const results = await Promise.all(downloadPromises);

  // Sort results by index to maintain order
  results.sort((a, b) => a.index - b.index);

  const imagePaths: string[] = [];
  let downloadedCount = 0;
  let failedCount = 0;

  for (const result of results) {
    cumulativeDownloadTime += result.imageTime;
    if (result.success) {
      imagePaths.push(result.outputPath);
      downloadedCount++;
    } else {
      failedCount++;
    }
  }

  const totalWallTime = Date.now() - downloadStartTime;
  const avgPerImage = downloadedCount > 0 ? cumulativeDownloadTime / downloadedCount : 0;

  // Print timing summary
  console.error(`\n[DOWNLOAD TIMING]`);
  console.error(`  Total wall time: ${(totalWallTime / 1000).toFixed(2)}s`);
  console.error(`  Cumulative download time: ${(cumulativeDownloadTime / 1000).toFixed(2)}s`);
  console.error(`  Avg per image: ${avgPerImage.toFixed(0)}ms`);
  console.error(`  Parallelism: ${DOWNLOAD_CONCURRENCY}x`);
  console.error(`  Speedup from parallel: ${(cumulativeDownloadTime / totalWallTime).toFixed(1)}x`);

  console.error(`\nDownloaded ${downloadedCount}/${imageUrls.length} images`);
  if (failedCount > 0) {
    console.error(`Failed: ${failedCount} images`);
  }

  return {
    chapterTitle,
    imagePaths,
    outputDir: chapterDir,
    totalImages: imageUrls.length,
    downloadedImages: downloadedCount,
    failedImages: failedCount,
    timing: {
      totalWallTimeMs: totalWallTime,
      cumulativeDownloadTimeMs: cumulativeDownloadTime,
      avgPerImageMs: avgPerImage,
      parallelism: DOWNLOAD_CONCURRENCY,
    },
  };
}

/**
 * Download and process a webtoon chapter in one step
 */
export async function downloadAndProcessChapter(
  url: string,
  outputDir: string
): Promise<DownloadResult> {
  return downloadWebtoonChapter(url, outputDir);
}
