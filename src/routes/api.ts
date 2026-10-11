import { Router, Request, Response } from 'express';
import { createJob, getJob, queueJob, canStartNewJob, getActiveJobCountForIp, getChaptersAheadForIp, getJobsAhead } from '../services/jobService';
import { checkQuota, recordSubmission } from '../services/quotaService';
import { getSpendStatus, canSpend, pricingAt } from '../services/spendCap';
import { getCached } from '../services/cacheService';
import { getEta, getAllSamples } from '../services/jobMetricsService';
import { recordSubmission as recordSubmissionHistory } from '../services/submissionHistoryService';
import {
  addToLibrary,
  removeFromLibrary,
  getLibrary,
  touchLastRead,
  getStats as getLibraryStats,
} from '../services/libraryService';
import { recordEvent, getReaderStats } from '../services/eventLogService';
import { InputLanguage, VALID_INPUT_LANGUAGES } from '../language';
import { readMetricsFile } from '../services/metricsSnapshotService';
import { getUserBySessionToken } from '../services/authService';
import { getRecentErrorLines } from '../services/logService';
import { randomUUID } from 'crypto';
import { federatedSearch, federatedSearchStream } from '../downloaders';
import { ResolvedTitleFields } from '../title-resolver';
import multer from 'multer';
import * as os from 'os';
import * as fs from 'fs';
import * as path from 'path';

const router = Router();

const TURNSTILE_SECRET_KEY = process.env.TURNSTILE_SECRET_KEY;
const TURNSTILE_SITE_KEY = process.env.TURNSTILE_SITE_KEY;

/**
 * Extract the real client IP. Cloudflare Tunnel passes the original IP via
 * `cf-connecting-ip`. Falls back to standard Express `req.ip` for direct access.
 */
function getClientIp(req: Request): string {
  const cfIp = req.headers['cf-connecting-ip'];
  if (typeof cfIp === 'string' && cfIp) return cfIp;
  const xff = req.headers['x-forwarded-for'];
  if (typeof xff === 'string' && xff) return xff.split(',')[0].trim();
  return req.ip || 'unknown';
}

/**
 * Verify a Cloudflare Turnstile token by calling Cloudflare's siteverify API.
 * Returns true if no secret key is configured (dev mode — Turnstile disabled).
 */
async function verifyTurnstile(token: string | undefined, ip: string): Promise<boolean> {
  if (!TURNSTILE_SECRET_KEY) {
    // Dev mode: no key configured, skip verification.
    return true;
  }
  if (!token) return false;
  try {
    const res = await fetch('https://challenges.cloudflare.com/turnstile/v0/siteverify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        secret: TURNSTILE_SECRET_KEY,
        response: token,
        remoteip: ip,
      }),
    });
    const result: any = await res.json();
    return result.success === true;
  } catch (e) {
    console.error('[Turnstile] Verification failed:', e);
    return false;
  }
}

/**
 * GET /api/config
 * Returns public configuration the frontend needs at load time
 * (e.g., Turnstile site key for widget rendering).
 */
/**
 * GET /api/search?q=...
 * Federated cross-language manga search. Returns aggregated results across
 * MangaDex (translator + source), Manhuagui, Webtoons EN/CN, Tapas.
 *
 * Result shape: AggregatedSearchResult[] — each entry has multilingual title
 * variants and a list of per-source matches sorted by quality (best first).
 *
 * Cached for 5 minutes per query to avoid hammering source sites on repeated
 * keypresses from the same user.
 */
const searchCache = new Map<string, { at: number; data: any }>();
const SEARCH_CACHE_TTL_MS = 5 * 60 * 1000;
router.get('/search', async (req: Request, res: Response) => {
  const q = String(req.query.q || '').trim().slice(0, 100);
  if (!q || q.length < 2) {
    res.json({ results: [] });
    return;
  }
  const cached = searchCache.get(q);
  if (cached && Date.now() - cached.at < SEARCH_CACHE_TTL_MS) {
    res.json({ results: cached.data, cached: true });
    return;
  }
  try {
    const results = await federatedSearch(q);
    searchCache.set(q, { at: Date.now(), data: results });
    res.json({ results, cached: false });
  } catch (e: any) {
    console.error('[search] failed:', e.message);
    res.status(500).json({ error: 'Search failed', message: e.message });
  }
});

/**
 * GET /api/webtoons-chapter-count?url=Y
 *
 * Lightweight chapter-range probe for Webtoons series. Fetches ONLY the
 * series page (one HTTP request) and returns the highest data-episode-no
 * value — that's both the latest chapter and the upper bound of the
 * "1 to N" range the chapter picker shows.
 *
 * Replaces the prior heavy /api/chapter-list flow for Webtoons (which
 * scraped 70+ pages and broke for series like Tower of God with non-standard
 * pagination). Chapter URLs are built directly from episode_no on the
 * frontend, so there's no need to know individual chapter titles.
 *
 * Result cached server-side for 1 hour. See getWebtoonsChapterCountCached.
 */
router.get('/webtoons-chapter-count', async (req: Request, res: Response) => {
  const url = String(req.query.url || '');
  if (!url || !/^https?:\/\/www\.webtoons\.com\//.test(url)) {
    res.status(400).json({ error: 'Webtoons URL required' });
    return;
  }
  try {
    const { getWebtoonsChapterCountCached } = await import('../downloaders/webtoons');
    const count = await getWebtoonsChapterCountCached(url);
    res.json({ count });
  } catch (e: any) {
    res.status(500).json({ error: e.message });
  }
});

/**
 * GET /api/chapter-list?source=X&url=Y
 * Returns the chapter list for a given series URL. The frontend calls this
 * when the user expands a NON-Webtoons search result card. Webtoons uses
 * /api/webtoons-chapter-count instead (much lighter — one fetch vs many).
 *
 * `source` tells us which adapter to use (must match a registered SourceId).
 * `url` is the series URL from the search result's seriesUrl field.
 */
const chapterListCache = new Map<string, { at: number; data: any }>();
const CHAPTER_LIST_TTL_MS = 5 * 60 * 1000;
router.get('/chapter-list', async (req: Request, res: Response) => {
  const source = String(req.query.source || '');
  const url = String(req.query.url || '');
  if (!source || !url) {
    res.status(400).json({ code: 'BAD_REQUEST', error: 'source and url required' });
    return;
  }
  const cacheKey = `${source}::${url}`;
  const cached = chapterListCache.get(cacheKey);
  if (cached && Date.now() - cached.at < CHAPTER_LIST_TTL_MS) {
    res.json({ chapters: cached.data, cached: true });
    return;
  }
  try {
    // Lazy import so we don't pay the cost on the (more common) /search hit.
    const { findAdapterById } = await import('../downloaders/registry-helper');
    const adapter = findAdapterById(source);
    if (!adapter || typeof adapter.getChapterList !== 'function') {
      res.status(404).json({ code: 'UNSUPPORTED_SOURCE', error: `Source "${source}" does not support chapter lists` });
      return;
    }
    const chapters = await adapter.getChapterList(url);
    chapterListCache.set(cacheKey, { at: Date.now(), data: chapters });
    res.json({ chapters, cached: false });
  } catch (e: any) {
    console.error('[chapter-list] failed:', e.message);
    res.status(500).json({ error: 'Chapter list failed', message: e.message });
  }
});

/**
 * GET /api/chapter-by-number?source=X&url=Y&n=N
 * Resolve a chapter NUMBER (as the user typed it) to its chapter URL.
 * Reuses the chapter-list cache so this is usually instant after the first
 * /api/chapter-list call for the same series.
 *
 * Returns { chapterUrl, title, locked } or 404 if no matching chapter.
 *
 * `n` is the chapter number as a string ("1", "1.5", "47") — string match
 * against the source's reported chapter number so .5 chapters work correctly.
 */
router.get('/chapter-by-number', async (req: Request, res: Response) => {
  const source = String(req.query.source || '');
  const url = String(req.query.url || '');
  const n = String(req.query.n || '').trim();
  if (!source || !url || !n) {
    res.status(400).json({ code: 'BAD_REQUEST', error: 'source, url, n required' });
    return;
  }
  const cacheKey = `${source}::${url}`;
  let chapters: any[];
  const cached = chapterListCache.get(cacheKey);
  if (cached && Date.now() - cached.at < CHAPTER_LIST_TTL_MS) {
    chapters = cached.data;
  } else {
    try {
      const { findAdapterById } = await import('../downloaders/registry-helper');
      const adapter = findAdapterById(source);
      if (!adapter || typeof adapter.getChapterList !== 'function') {
        res.status(404).json({ code: 'UNSUPPORTED_SOURCE' });
        return;
      }
      chapters = await adapter.getChapterList(url);
      chapterListCache.set(cacheKey, { at: Date.now(), data: chapters });
    } catch (e: any) {
      res.status(500).json({ error: 'Chapter lookup failed', message: e.message });
      return;
    }
  }
  // Match by source-reported chapter number (string equality). Handles .5
  // chapters: user types "1.5" → matches { number: "1.5" }. Some sites use
  // sequential URL indices that don't match the displayed chapter number
  // (e.g. Webtoons episode_no=2 for chapter "1.5") — we trust the source's
  // `number` field, not the URL.
  //
  // Multi-season collision: in series like Hand Jumper or SubZero where each
  // season resets its episode count, the same chapter number can match
  // multiple entries (S1 Ep. 5, S2 Ep. 5, S3 Ep. 5 all become number="5").
  // We return ALL matches so the frontend can show a picker; matches[0] is
  // the newest one (chapter list is already sorted newest-first).
  const matches = chapters.filter((c: any) => (c.number || '').trim() === n);
  if (matches.length === 0) {
    res.status(404).json({ code: 'CHAPTER_NOT_FOUND', error: `No chapter ${n}. Available: ${chapters.length} chapters.` });
    return;
  }
  const match = matches[0];
  res.json({
    chapterUrl: match.chapterUrl,
    title: match.title,
    locked: !!match.locked,
    matches: matches.length > 1 ? matches.map((m: any) => ({
      chapterUrl: m.chapterUrl,
      title: m.title,
      locked: !!m.locked,
    })) : undefined,
  });
});

/**
 * GET /api/search/stream?q=...
 * Server-Sent Events variant: pushes a snapshot of current results each time
 * one of the federated source adapters completes. Frontend renders each
 * snapshot, so fast sources (MangaDex, Tapas) paint immediately while slow
 * ones (Manhuagui, DMZJ) either appear later or never block the response.
 *
 * Event format:
 *   event: snapshot
 *   data: { results: AggregatedSearchResult[] }
 * Plus a final `event: done` so the client can stop indicating "searching..."
 */
router.get('/search/stream', async (req: Request, res: Response) => {
  const q = String(req.query.q || '').trim().slice(0, 100);

  // SSE headers — must be sent before any data
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  // Some proxies (Cloudflare in particular) buffer responses by default;
  // disable it for this endpoint so events flush immediately.
  res.setHeader('X-Accel-Buffering', 'no');

  if (!q || q.length < 2) {
    res.write('event: snapshot\ndata: {"results":[]}\n\n');
    res.write('event: done\ndata: {}\n\n');
    res.end();
    return;
  }

  // Serve from cache if recent — but still as SSE for frontend symmetry.
  const cached = searchCache.get(q);
  if (cached && Date.now() - cached.at < SEARCH_CACHE_TTL_MS) {
    res.write(`event: snapshot\ndata: ${JSON.stringify({ results: cached.data, cached: true })}\n\n`);
    res.write('event: done\ndata: {}\n\n');
    res.end();
    return;
  }

  // Track the latest snapshot — cache it after `done` so subsequent searches
  // for the same query are instant.
  let latestSnapshot: any[] = [];
  let closed = false;
  req.on('close', () => { closed = true; });

  try {
    await federatedSearchStream(q, (snapshot, translatorDone) => {
      if (closed) return;
      latestSnapshot = snapshot;
      res.write(`event: snapshot\ndata: ${JSON.stringify({ results: snapshot, translatorDone })}\n\n`);
    });
    if (!closed) {
      searchCache.set(q, { at: Date.now(), data: latestSnapshot });
      res.write('event: done\ndata: {}\n\n');
      res.end();
    }
  } catch (e: any) {
    console.error('[search-stream] failed:', e.message);
    if (!closed) {
      res.write(`event: error\ndata: ${JSON.stringify({ message: e.message })}\n\n`);
      res.end();
    }
  }
});

/**
 * GET /api/proxy-image?url=...
 * Server-side image proxy. Bypasses CDN hotlink protection (Manhuagui /
 * KuaiKan / Tapas etc. block image requests with a non-matching Referer)
 * by fetching with the correct Referer for that host and streaming the
 * bytes back. Result is served from our origin so it always loads.
 *
 * Caches recent fetches in memory (~5 MB total, LRU) to avoid hammering
 * source CDNs on repeated searches.
 */
// Per-host Referer overrides. CDNs check this header to detect hotlinking;
// sending the right value makes them serve the image normally.
const IMAGE_PROXY_REFERERS: Record<string, string> = {
  'eu1.hamreus.com': 'https://www.manhuagui.com/',
  'cf.mhgui.com': 'https://www.manhuagui.com/',
  'tn1.kkmh.com': 'https://www.kuaikanmanhua.com/',
  'tn2.kkmh.com': 'https://www.kuaikanmanhua.com/',
  'tn3.kkmh.com': 'https://www.kuaikanmanhua.com/',
  'us-a.tapas.io': 'https://tapas.io/',
  'images.dmzj.com': 'https://www.dmzj.com/',
  'webtoon-phinf.pstatic.net': 'https://www.webtoons.com/',
  'swebtoon-phinf.pstatic.net': 'https://www.webtoons.com/',  // series covers (og:image)
  'jumpg-assets.tokyo-cdn.com': 'https://mangaplus.shueisha.co.jp/',
};

interface CachedImage { data: Buffer; contentType: string; at: number; }
const imageCache = new Map<string, CachedImage>();
const IMAGE_CACHE_MAX_BYTES = 5 * 1024 * 1024; // 5 MB
const IMAGE_CACHE_TTL_MS = 60 * 60 * 1000; // 1 hour
function imageCacheBytes(): number {
  let n = 0; for (const v of imageCache.values()) n += v.data.length; return n;
}
function evictOldestImage() {
  let oldestKey: string | null = null; let oldestAt = Infinity;
  for (const [k, v] of imageCache) if (v.at < oldestAt) { oldestAt = v.at; oldestKey = k; }
  if (oldestKey) imageCache.delete(oldestKey);
}

router.get('/proxy-image', async (req: Request, res: Response) => {
  const url = String(req.query.url || '');
  if (!url || !/^https?:\/\//.test(url)) {
    res.status(400).send('Bad URL');
    return;
  }

  // Cache check
  const cached = imageCache.get(url);
  if (cached && Date.now() - cached.at < IMAGE_CACHE_TTL_MS) {
    res.setHeader('Content-Type', cached.contentType);
    res.setHeader('Cache-Control', 'public, max-age=3600');
    res.send(cached.data);
    return;
  }

  let targetHost: string;
  try { targetHost = new URL(url).hostname; }
  catch { res.status(400).send('Bad URL'); return; }
  const referer = IMAGE_PROXY_REFERERS[targetHost] || `https://${targetHost}/`;

  // Use the existing https module pattern (consistent with other endpoints).
  const https_mod = await import('https');
  const reqOpts = {
    headers: {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
      'Accept': 'image/webp,image/avif,image/*,*/*;q=0.8',
      'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
      'Referer': referer,
    },
  };
  const upstream = https_mod.get(url, reqOpts, (proxyRes) => {
    if ((proxyRes.statusCode || 0) >= 400) {
      res.status(502).send('Upstream HTTP ' + proxyRes.statusCode);
      return;
    }
    // Handle redirects (CDN edges sometimes 302 to a regional CDN)
    if ([301, 302, 303, 307, 308].includes(proxyRes.statusCode || 0) && proxyRes.headers.location) {
      // Redirect through the proxy too so Referer stays correct.
      const next = new URL(proxyRes.headers.location, url).href;
      res.redirect(`/api/proxy-image?url=${encodeURIComponent(next)}`);
      return;
    }
    const contentType = proxyRes.headers['content-type'] || 'image/jpeg';
    res.setHeader('Content-Type', contentType);
    res.setHeader('Cache-Control', 'public, max-age=3600');
    const chunks: Buffer[] = [];
    proxyRes.on('data', c => chunks.push(c));
    proxyRes.on('end', () => {
      const buf = Buffer.concat(chunks);
      // Cache if under 200KB (covers, not full chapter pages) and total cache
      // budget allows.
      if (buf.length < 200 * 1024) {
        while (imageCacheBytes() + buf.length > IMAGE_CACHE_MAX_BYTES && imageCache.size > 0) {
          evictOldestImage();
        }
        imageCache.set(url, { data: buf, contentType, at: Date.now() });
      }
      res.send(buf);
    });
    proxyRes.on('error', err => {
      if (!res.headersSent) res.status(502).send('Upstream stream error: ' + err.message);
    });
  });
  upstream.on('error', err => {
    if (!res.headersSent) res.status(502).send('Upstream connect error: ' + err.message);
  });
  upstream.setTimeout(10000, () => {
    upstream.destroy();
    if (!res.headersSent) res.status(504).send('Upstream timeout');
  });
});

/**
 * POST /api/upload-images
 * Multipart upload: user-supplied chapter images. Creates a job that skips
 * the URL-download step and processes the uploaded images directly. Useful
 * for users with their own scans, screenshots, or chapters from sites the
 * downloader can't handle.
 *
 * Limits: 300 images per upload, 10MB per image, 1GB total (matches the
 * URL-path MAX_PAGES_PER_URL ceiling). Images are stored in OS temp dir
 * and cleaned up after job processing.
 */
const UPLOAD_MAX_FILES = 300;
const UPLOAD_MAX_FILE_SIZE = 10 * 1024 * 1024;       // 10 MB per image
const UPLOAD_MAX_TOTAL_SIZE = 1 * 1024 * 1024 * 1024; // 1 GB total

// Multer storage: write each upload to a per-job temp dir so we can pass
// concrete file paths to the job worker.
const uploadDirRoot = path.join(os.tmpdir(), 'manga-uploads');
fs.mkdirSync(uploadDirRoot, { recursive: true });
const upload = multer({
  storage: multer.diskStorage({
    destination: (req: any, _file, cb) => {
      // Reuse the same uploadId across all files in one request so they land
      // in the same job folder.
      if (!req._uploadDir) {
        const id = randomUUID();
        req._uploadDir = path.join(uploadDirRoot, id);
        fs.mkdirSync(req._uploadDir, { recursive: true });
      }
      cb(null, req._uploadDir);
    },
    filename: (_req, file, cb) => {
      // Use the upload index so filenames are sortable and preserve order.
      // Original extension preserved so pipeline can detect format.
      const ext = path.extname(file.originalname).toLowerCase() || '.png';
      // Multer doesn't expose file index here; use a timestamp+rand for uniqueness
      // and sort by upload completion order on the worker side later.
      cb(null, `${Date.now()}-${Math.random().toString(36).slice(2, 8)}${ext}`);
    },
  }),
  limits: { fileSize: UPLOAD_MAX_FILE_SIZE, files: UPLOAD_MAX_FILES },
  fileFilter: (_req, file, cb) => {
    // Reject obviously-wrong file types. Pipeline expects jpg/png/webp/gif.
    const ok = /^image\/(jpe?g|png|webp|gif)$/i.test(file.mimetype);
    if (!ok) cb(new Error(`Unsupported file type: ${file.mimetype}`));
    else cb(null, true);
  },
});

router.post('/upload-images', upload.array('images', UPLOAD_MAX_FILES), async (req: Request, res: Response) => {
  const files = req.files as Express.Multer.File[] | undefined;
  if (!files || files.length === 0) {
    res.status(400).json({ code: 'NO_FILES', error: 'No images uploaded.' });
    return;
  }
  // Enforce total-size cap (multer's fileSize is per-file).
  const totalSize = files.reduce((s, f) => s + f.size, 0);
  if (totalSize > UPLOAD_MAX_TOTAL_SIZE) {
    for (const f of files) try { fs.unlinkSync(f.path); } catch {}
    res.status(413).json({ code: 'TOO_LARGE', error: `Upload exceeds ${UPLOAD_MAX_TOTAL_SIZE / 1024 / 1024}MB total.` });
    return;
  }

  const level = String(req.body?.level || 'hsk3');
  const inputLanguage = String(req.body?.inputLanguage || 'auto') as InputLanguage;
  const validLangs: InputLanguage[] = VALID_INPUT_LANGUAGES;
  const lang = validLangs.includes(inputLanguage) ? inputLanguage : 'auto';
  const validLevels = ['hsk1', 'hsk2', 'hsk3', 'hsk4', 'hsk5', 'hsk6'];
  if (!validLevels.includes(level)) {
    res.status(400).json({ code: 'BAD_LEVEL', error: `Level must be one of ${validLevels.join(', ')}.` });
    return;
  }

  const ip = getClientIp(req);
  const imagePaths = files.map(f => f.path);
  // Sort by filename so user-controlled order (lex / date) is preserved.
  imagePaths.sort();

  const jobId = createJob(`upload://${(req as any)._uploadDir}`, [level], false, lang, undefined, ip, imagePaths);
  queueJob(jobId);
  res.json({ jobs: [{ jobId, url: `upload://${files.length} images` }] });
});

router.get('/config', (_req: Request, res: Response) => {
  res.json({
    turnstileSiteKey: TURNSTILE_SITE_KEY || null,
    spendStatus: getSpendStatus(),
  });
});

/**
 * GET /api/eta
 * Phase 2.2: returns median + Q1/Q3 of past chapter processing times,
 * bucketed into cold (no cache) and cache-hit (skipped LLM + chapter analysis).
 * Returns null per bucket when sample size is below threshold; the frontend
 * falls back to the static `fallback` string in that case.
 */
router.get('/eta', (_req: Request, res: Response) => {
  res.json(getEta());
});

/**
 * GET /api/health
 * Lightweight liveness + dependency check. Used as a manual "is the site up"
 * ping and as an integration point for any uptime monitor (UptimeRobot, etc.).
 *
 * The LM Studio reachability check is the only non-trivial part — we cache the
 * result for 30s so frequent health checks don't hammer the LLM server.
 */
const SERVER_STARTED_AT = Date.now();
const QWEN_API_URL = process.env.QWEN_API_URL || 'http://localhost:1234';
let lmStudioCache: { reachable: boolean; checkedAt: number } | null = null;
const LM_STUDIO_CACHE_TTL_MS = 30_000;

async function checkLmStudioReachable(): Promise<boolean> {
  const now = Date.now();
  if (lmStudioCache && now - lmStudioCache.checkedAt < LM_STUDIO_CACHE_TTL_MS) {
    return lmStudioCache.reachable;
  }
  let reachable = false;
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 1000);
    const r = await fetch(`${QWEN_API_URL}/v1/models`, { signal: controller.signal });
    clearTimeout(timer);
    reachable = r.ok;
  } catch {
    reachable = false;
  }
  lmStudioCache = { reachable, checkedAt: now };
  return reachable;
}

router.get('/health', async (_req: Request, res: Response) => {
  const lmStudioReachable = await checkLmStudioReachable();
  res.json({
    ok: true,
    uptimeSeconds: Math.floor((Date.now() - SERVER_STARTED_AT) / 1000),
    version: process.env.npm_package_version || '0.0.0',
    lmStudioReachable,
  });
});

// ============================================================================
// Admin dashboard — token-gated operational stats
// ============================================================================

const ADMIN_TOKEN = process.env.ADMIN_TOKEN;

/** Pull the admin token from an `Authorization: Bearer` header or a `?key=` query param. */
function getAdminToken(req: Request): string | undefined {
  const auth = req.headers['authorization'];
  if (typeof auth === 'string' && auth.startsWith('Bearer ')) return auth.slice(7).trim();
  const key = req.query?.key;
  if (typeof key === 'string' && key) return key;
  return undefined;
}

/**
 * GET /api/admin/stats
 * Aggregated operational data for the /admin dashboard. Gated by the ADMIN_TOKEN
 * env var — if it is unset the endpoint is disabled rather than served open.
 */
router.get('/admin/stats', (req: Request, res: Response) => {
  if (!ADMIN_TOKEN) {
    res.status(503).json({ code: 'ADMIN_DISABLED', error: 'Admin dashboard is not configured (ADMIN_TOKEN unset).' });
    return;
  }
  if (getAdminToken(req) !== ADMIN_TOKEN) {
    res.status(401).json({ code: 'UNAUTHORIZED', error: 'Invalid or missing admin token.' });
    return;
  }

  // Fireworks (DeepSeek V4 Flash) pricing — single source of truth in
  // spendCap.ts, so the dashboard cost and the spend cap can never drift apart.
  // Each sample is priced at the rate in effect WHEN IT RAN, so a provider price
  // change doesn't retroactively rewrite what past chapters appear to have cost.

  // Recent chapters: last 50 job samples, newest first, each with its cost.
  const chapters = getAllSamples().slice(-50).reverse().map((s) => {
    const fw = s.fireworks;
    const rate = pricingAt('fireworks', s.ts);
    const costUsd = fw && rate
      ? (fw.inputTokens / 1000) * rate.inputPer1K
        + (fw.cachedInputTokens / 1000) * (rate.cachedInputPer1K ?? rate.inputPer1K)
        + (fw.outputTokens / 1000) * rate.outputPer1K
      : null;
    return {
      ts: s.ts,
      title: s.title ?? null,
      pages: s.pages,
      cacheHit: s.cacheHit,
      avgPageWidth: s.avgPageWidth ?? null,
      avgPageHeight: s.avgPageHeight ?? null,
      fireworks: s.fireworks ?? null,
      costUsd,
    };
  });

  res.json({
    generatedAt: Date.now(),
    spend: getSpendStatus(),
    chapters,
    recentErrors: getRecentErrorLines(20),
    readers: getReaderStats(),
    library: getLibraryStats(),
  });
});

/**
 * GET /api/admin/metrics
 * The full weekly-metrics history (aggregate counts only, no personal data).
 * Token-gated like /admin/stats. Pulled by `npm run pull-metrics` on Computer 1
 * to write and commit the published evidence copy (docs/metrics-weekly.json).
 */
router.get('/admin/metrics', (req: Request, res: Response) => {
  if (!ADMIN_TOKEN) {
    res.status(503).json({ code: 'ADMIN_DISABLED', error: 'Admin dashboard is not configured (ADMIN_TOKEN unset).' });
    return;
  }
  if (getAdminToken(req) !== ADMIN_TOKEN) {
    res.status(401).json({ code: 'UNAUTHORIZED', error: 'Invalid or missing admin token.' });
    return;
  }
  res.json(readMetricsFile());
});

/**
 * POST /api/beacon
 * Anonymous usage events from the generated reader (reader_open, word_click,
 * toggle_mode, flashcard_save, sample_view). Fire-and-forget by design:
 * always 204, validation happens inside recordEvent (event whitelist, size
 * caps), nothing user-identifiable is stored. This is the raw feed behind
 * the admin dashboard's reader stats and scripts/weekly-snapshot.js.
 */
router.post('/beacon', (req: Request, res: Response) => {
  try {
    const b = req.body || {};
    recordEvent({ event: b.event, bid: b.bid, props: b.props });
  } catch { /* never fail a beacon */ }
  res.status(204).end();
});

/**
 * POST /api/process
 * Start one or more manga processing jobs
 *
 * Body: { urls: string[] | url: string, levels?: string[], level?: string, includeEnglish?: boolean, inputLanguage?: 'auto' | 'chinese' | 'english' | 'japanese' | 'korean' }
 * Response: { jobs: [{ jobId, status }] } or error
 */
router.post('/process', async (req: Request, res: Response) => {
  try {
    const ip = getClientIp(req);
    const { url, urls, level, levels, includeEnglish, inputLanguage, turnstileToken } = req.body;

    // Phase 1.2: Cloudflare Turnstile gate. No-op if TURNSTILE_SECRET_KEY unset.
    const tokenOk = await verifyTurnstile(turnstileToken, ip);
    if (!tokenOk) {
      res.status(403).json({ code: 'TURNSTILE_FAILED', error: 'Bot check failed. Please refresh and try again.' });
      return;
    }

    // Phase 1.1: spend cap gate. Refuse new submissions when today's budget
    // is exhausted. Chapters already in flight finish; new submissions are
    // rejected at submit time (better UX than failing partway through).
    if (!canSpend()) {
      const status = getSpendStatus();
      res.status(429).json({
        code: 'BUDGET_REACHED',
        error: `Daily processing budget reached ($${status.spentUsd.toFixed(2)} / $${status.capUsd.toFixed(2)}). Resets at UTC midnight.`,
        spendStatus: status,
      });
      return;
    }

    // Support both single URL (url) and multiple URLs (urls)
    let urlList: string[] = [];
    if (urls && Array.isArray(urls)) {
      urlList = urls;
    } else if (url && typeof url === 'string') {
      urlList = [url];
    }

    // Validate we have at least one URL
    if (urlList.length === 0) {
      res.status(400).json({ code: 'BAD_URL', error: 'At least one URL is required' });
      return;
    }

    // Validate URL shape only — no hostname allowlist. Generic extraction
    // (data-src / og:title fallbacks in webtoon-downloader) covers most
    // standard manga / manhua sites; sites that don't match cleanly will
    // surface as NO_IMAGES_FOUND, which the frontend already handles.
    for (const u of urlList) {
      if (typeof u !== 'string') {
        res.status(400).json({ code: 'BAD_URL', error: 'All URLs must be strings' });
        return;
      }
      if (!u.startsWith('http://') && !u.startsWith('https://')) {
        res.status(400).json({ code: 'BAD_URL', error: `URL must start with http:// or https://: ${u}` });
        return;
      }
      try {
        new URL(u);
      } catch {
        res.status(400).json({ code: 'BAD_URL', error: `Invalid URL: ${u}` });
        return;
      }
    }

    // Validate levels (support both single 'level' and array 'levels')
    const validLevels = ['hsk1', 'hsk2', 'hsk3', 'hsk4', 'hsk5', 'hsk6'];
    let targetLevels: string[];
    if (levels && Array.isArray(levels) && levels.length > 0) {
      targetLevels = levels;
    } else {
      targetLevels = [level || 'hsk3'];
    }
    for (const l of targetLevels) {
      if (!validLevels.includes(l)) {
        res.status(400).json({ code: 'BAD_URL', error: `Level must be one of: ${validLevels.join(', ')}` });
        return;
      }
    }

    // Per-URL: cache lookup → if hit, create job with cachedData (still uses
    //   user's quota since the pipeline still runs OCR + inpaint + render — it
    //   just skips the LLM and chapter analysis). If miss, normal flow.
    //
    // Cache is disabled for multi-level requests because the cache is keyed on
    // a single primary level — using it would silently downgrade the user's
    // multi-level request to single-level output.
    const jobs: { jobId: string; status: string; fromCache?: boolean }[] = [];
    const validLangs: string[] = VALID_INPUT_LANGUAGES;
    const lang = validLangs.includes(inputLanguage) ? inputLanguage : 'auto';
    const primaryLevel = targetLevels[0]; // cache keyed on primary level
    const isMultiLevel = targetLevels.length > 1 || includeEnglish === true;

    // Phase 3.4: queue-depth cap. Per-IP fairness — a power user can have at
    // most ACTIVE_JOB_CAP jobs in flight at once so others don't wait too long.
    // Counts cache-hit jobs too (they still run download + OCR + inpaint + render).
    const ACTIVE_JOB_CAP = parseInt(process.env.ACTIVE_JOB_CAP || '3', 10);

    for (const u of urlList) {
      // Active-job cap before quota — quota records a submission timestamp
      // that we don't want to consume if the cap rejects this request.
      // getActiveJobCountForIp already counts jobs created earlier in THIS
      // request — createJob registers each in the job map immediately — so it
      // is the complete in-flight count. Adding jobs.length on top double-counts
      // them and wrongly rejects a legitimate 3-URL submit on the 3rd URL.
      const activeCount = getActiveJobCountForIp(ip);
      if (activeCount >= ACTIVE_JOB_CAP) {
        res.status(429).json({
          code: 'TOO_MANY_ACTIVE',
          error: `You have ${activeCount} chapters processing. Wait for one to finish before submitting more.`,
          activeCount,
          limit: ACTIVE_JOB_CAP,
        });
        return;
      }

      // Quota check (cache hit still does real work, so it counts).
      const check = checkQuota(ip);
      if (!check.ok) {
        res.status(429).json({
          code: 'QUOTA_EXCEEDED',
          error: check.reason,
          retryAfterSeconds: check.retryAfterSeconds,
          limit: check.limit,
        });
        return;
      }

      recordSubmission(ip);

      const cached = !isMultiLevel ? getCached(u, primaryLevel, lang) : null;
      if (cached) {
        console.log(`[Cache] HIT for ${u} @ ${primaryLevel} (${cached.data.pages.length} pages cached) — pipeline will skip LLM + chapter analysis`);
      }

      // Submission history feeds per-browser analytics only. The library is
      // NOT auto-populated here: the client adds the finished chapter to the account library on completion (with its
      // resultUrl) via /api/library/add — so cross-device entries open directly
      // instead of being reprocess stubs, and nothing enters an account without
      // the user's explicit action.
      const browserId = req.cookies?.browser_id as string | undefined;
      recordSubmissionHistory(browserId, { url: u, level: primaryLevel, lang, title: '' });

      const jobId = createJob(u, targetLevels, includeEnglish || false, lang, cached?.data, ip);
      jobs.push({ jobId, status: 'queued', fromCache: !!cached });
      queueJob(jobId);
    }

    res.json({ jobs });
  } catch (error) {
    console.error('Error starting jobs:', error);
    res.status(500).json({ code: 'INTERNAL', error: 'Failed to start processing' });
  }
});

/**
 * GET /api/status/:jobId
 * Get the status of a processing job
 *
 * Response: { status, progress, total, message, resultUrl?, error? }
 */
router.get('/status/:jobId', (req: Request, res: Response) => {
  const { jobId } = req.params;

  const job = getJob(jobId);
  if (!job) {
    res.status(404).json({ error: 'Job not found' });
    return;
  }

  const response: {
    status: string;
    progress: number;
    total: number;
    message: string;
    title?: string;
    resolved?: ResolvedTitleFields;
    resultUrl?: string;
    error?: string;
    errorCode?: string;
    chaptersAhead?: number;
    jobsAhead?: number;
  } = {
    status: job.status,
    progress: job.progress,
    total: job.total,
    message: job.message,
  };

  // Phase 3.4: how many chapters from OTHER users are ahead of this user's
  // batch. Same value for every job this IP polls; the frontend renders it
  // once as a single "N chapters ahead of you" box.
  response.chaptersAhead = getChaptersAheadForIp(getClientIp(req));

  // Absolute queue position for THIS job (counts everyone ahead, including the
  // user's own earlier chapters + the running one). The frontend renders it
  // per-card as "#N in queue" while the job is still pending.
  response.jobsAhead = getJobsAhead(jobId);

  // Chapter/episode name (once download has parsed it) — UI shows it in place
  // of the raw URL in the Recent list and on the processing screen.
  if (job.title) response.title = job.title;

  // Learner-friendly resolution (English name + cover + pinyin + chapter #),
  // when available. The client carries it into the library on completion.
  if (job.resolved) response.resolved = job.resolved;

  // Include resultUrl when complete
  if (job.status === 'complete' && job.resultUrl) {
    response.resultUrl = job.resultUrl;
  }

  // Include error message + code when failed (Phase 1.6: frontend keys off code)
  if (job.status === 'error') {
    if (job.error) response.error = job.error;
    response.errorCode = job.errorCode || 'INTERNAL';
  }

  res.json(response);
});

// ============================================================================
// Per-user chapter library (cross-device sync for signed-in users). The email
// is derived server-side from the session cookie; it is never taken from
// client state.
// ============================================================================

function getLoggedInEmail(req: Request): string | null {
  const sessionToken = req.cookies?.session as string | undefined;
  if (!sessionToken) return null;
  return getUserBySessionToken(sessionToken)?.email ?? null;
}

/**
 * GET /api/library
 * Returns the logged-in user's chapter library, newest first. Returns an
 * empty list (200) for anonymous users — frontend treats that as "not signed
 * in" and prompts them to sign in via the existing restore modal.
 */
router.get('/library', (req: Request, res: Response) => {
  const email = getLoggedInEmail(req);
  if (!email) {
    res.json({ ok: true, signedIn: false, library: [] });
    return;
  }
  res.json({ ok: true, signedIn: true, email, library: getLibrary(email) });
});

/**
 * POST /api/library/remove
 * Remove one chapter by URL from the logged-in user's library.
 */
router.post('/library/remove', (req: Request, res: Response) => {
  const email = getLoggedInEmail(req);
  if (!email) {
    res.status(401).json({ code: 'NOT_SIGNED_IN', error: 'Sign in to manage your library.' });
    return;
  }
  const url = String(req.body?.url || '');
  if (!url) {
    res.status(400).json({ code: 'BAD_URL', error: 'url is required.' });
    return;
  }
  const level = typeof req.body?.level === 'string' ? req.body.level : undefined;
  const removed = removeFromLibrary(email, url, level);
  res.json({ ok: true, removed });
});

/**
 * Only accept a same-origin reader path (/outputs/...). Prevents a client from
 * stashing an arbitrary external URL in their own library.
 */
function sanitizeResultUrl(v: unknown): string | undefined {
  return typeof v === 'string' && v.startsWith('/outputs/') ? v : undefined;
}

/**
 * Sanitize the client-echoed `resolved` title fields before persisting them.
 * The client sends back what our own /api/status produced, but never trust it:
 * cap string lengths and only accept a cover path we actually serve (a
 * self-hosted /outputs/.../cover.<ext>) — never an arbitrary external URL.
 */
function sanitizeResolved(v: unknown): ResolvedTitleFields | undefined {
  if (!v || typeof v !== 'object') return undefined;
  const o = v as Record<string, unknown>;
  const str = (x: unknown, max: number): string | null =>
    typeof x === 'string' && x.length <= max ? x : null;
  const seriesZh = str(o.seriesZh, 120);
  const english = str(o.english, 200);
  const seriesPinyin = str(o.seriesPinyin, 300);
  const chapterLabel = str(o.chapterLabel, 40);
  const coverUrl =
    typeof o.coverUrl === 'string' && /^\/outputs\/[\w.-]+\/cover\.(jpg|jpeg|png|webp)$/i.test(o.coverUrl)
      ? o.coverUrl
      : null;
  // Nothing useful → drop it entirely so the row falls back to the raw title.
  if (!english && !seriesZh && !coverUrl) return undefined;
  return { english, seriesZh: seriesZh || '', seriesPinyin: seriesPinyin || '', chapterLabel, coverUrl };
}

/**
 * POST /api/library/add
 * Add ONE finished chapter to the signed-in user's library. Called by the
 * client when a job completes while signed in; carries the openable reader URL
 * so other devices open it directly (no reprocessing).
 * Body: { url, resultUrl, title?, level?, lang? }
 */
router.post('/library/add', (req: Request, res: Response) => {
  const email = getLoggedInEmail(req);
  if (!email) {
    res.status(401).json({ code: 'NOT_SIGNED_IN', error: 'Sign in to save chapters.' });
    return;
  }
  const b = req.body || {};
  if (typeof b.url !== 'string' || !b.url.startsWith('http')) {
    res.status(400).json({ code: 'BAD_URL', error: 'A valid url is required.' });
    return;
  }
  addToLibrary(email, {
    url: b.url,
    resultUrl: sanitizeResultUrl(b.resultUrl),
    level: typeof b.level === 'string' ? b.level : 'hsk3',
    lang: typeof b.lang === 'string' ? b.lang : 'auto',
    title: typeof b.title === 'string' ? b.title : '',
    resolved: sanitizeResolved(b.resolved),
  });
  res.json({ ok: true, total: getLibrary(email).length });
});

/**
 * POST /api/library/touch
 * Mark a library chapter as read just now (sets lastReadAt). Called
 * fire-and-forget by the frontend when a signed-in user opens a chapter from
 * their library — this is what makes per-user "chapters read over time"
 * measurable. Silent ok:false for anonymous users (opening still works).
 * Body: { url, level? }
 */
router.post('/library/touch', (req: Request, res: Response) => {
  const email = getLoggedInEmail(req);
  const url = String(req.body?.url || '');
  if (!email || !url) {
    res.json({ ok: false });
    return;
  }
  const level = typeof req.body?.level === 'string' ? req.body.level : undefined;
  res.json({ ok: touchLastRead(email, url, level) });
});

/**
 * POST /api/library/claim
 * EXPLICIT one-time claim: the user chose to add anonymous chapters saved in
 * their browser to their account. This is the ONLY path by which pre-existing
 * chapters enter an account — there is deliberately no automatic browser→account
 * sweep (that leaked chapters between accounts on a shared browser).
 * Body: { entries: [{ url, resultUrl?, title?, level?, lang? }, ...] }  (max 500)
 */
router.post('/library/claim', (req: Request, res: Response) => {
  const email = getLoggedInEmail(req);
  if (!email) {
    res.status(401).json({ code: 'NOT_SIGNED_IN', error: 'Sign in to claim chapters.' });
    return;
  }
  const entries = Array.isArray(req.body?.entries) ? req.body.entries : [];
  if (entries.length > 500) {
    res.status(400).json({ code: 'TOO_MANY', error: 'Max 500 entries per call.' });
    return;
  }
  const before = getLibrary(email).length;
  for (const e of entries) {
    if (!e || typeof e.url !== 'string' || !e.url.startsWith('http')) continue;
    addToLibrary(email, {
      url: e.url,
      resultUrl: sanitizeResultUrl(e.resultUrl),
      level: typeof e.level === 'string' ? e.level : 'hsk3',
      lang: typeof e.lang === 'string' ? e.lang : 'auto',
      title: typeof e.title === 'string' ? e.title : '',
      resolved: sanitizeResolved(e.resolved),
    });
  }
  const after = getLibrary(email).length;
  res.json({ ok: true, added: after - before, total: after });
});

export default router;
