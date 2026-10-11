import { v4 as uuidv4 } from 'uuid';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import * as https from 'https';
import { downloadWebtoonChapter } from '../webtoon-downloader';
import { resolveChapterTitle } from './titleCacheService';
import { ResolvedTitleFields } from '../title-resolver';
import { commitTextCache, CachedChapter } from './cacheService';
import { processMangaPages, getTimingStats } from '../manga-pipeline';
import { InputLanguage, isTranslationSource } from '../language';
import { getLLMUsage } from '../simplifier';
import { startJobLogCapture, appendErrorLog } from './logService';
import { recordJobCompletion } from './jobMetricsService';
import { HSKLevel } from '../types';

export interface Job {
  id: string;
  url: string;
  // When set, the job skips the URL-download step entirely and uses these
  // image paths directly. Used for user-uploaded images (/api/upload-images).
  // Paths must already exist on disk and be readable by the worker.
  localImagePaths?: string[];
  // Chapter/episode name parsed from the source page (og:title), surfaced via
  // /api/status so the UI can show a real name instead of the raw URL.
  title?: string;
  // Learner-friendly title resolution (English name + cover + pinyin + chapter
  // number), when the raw title is Chinese and a source could be found. Rides
  // /api/status → client → library so the chapter list renders it. Absent =
  // fall back to the raw `title`.
  resolved?: ResolvedTitleFields;
  level: HSKLevel;
  levels: HSKLevel[];
  includeEnglish: boolean;
  inputLanguage: InputLanguage;
  status: 'pending' | 'downloading' | 'processing' | 'complete' | 'error';
  progress: number;
  total: number;
  message: string;
  resultUrl?: string;
  error?: string;
  errorCode?: string; // Phase 1.6: typed error code surfaced from /api/status to the frontend
  createdAt: Date;
  // Source IP for the submission. Used by getActiveJobCountForIp to enforce
  // the queue-depth cap (Phase 3.4) — counts cache-hit jobs too since they
  // still run download + OCR + inpaint + render.
  ip?: string;
  // When set, the job will skip Phase 4b LLM by bbox-matching cached
  // simplifications and skip Phase 3 chapter analysis.
  cachedData?: CachedChapter;
}

const ACTIVE_STATUSES: ReadonlyArray<Job['status']> = ['pending', 'downloading', 'processing'];

/**
 * Count how many jobs from `ip` are currently active (queued or running).
 * Used by the submit handler to enforce the per-IP queue-depth cap.
 */
export function getActiveJobCountForIp(ip: string): number {
  let count = 0;
  for (const job of jobs.values()) {
    if (job.ip === ip && ACTIVE_STATUSES.includes(job.status)) count++;
  }
  return count;
}

// In-memory job storage
const jobs = new Map<string, Job>();

// Job queue for sequential processing
const jobQueue: string[] = [];
let isProcessing = false;
// ID of the job currently running (null when idle). Tracked so the queue-ahead
// count can tell whether the in-flight job belongs to another user.
let runningJobId: string | null = null;

// Outputs directory path
const outputsDir = path.join(__dirname, '..', '..', 'outputs');

/**
 * Download a series cover to `dest` (self-hosted, so we don't hotlink the
 * MangaDex CDN). Their covers require an app-identifying User-Agent. Best-effort:
 * any failure resolves false and the chapter simply renders without art.
 */
function downloadCoverToFile(url: string, dest: string): Promise<boolean> {
  return new Promise((resolve) => {
    const req = https.get(url, {
      headers: { 'User-Agent': 'HSKManga/1.0 (https://hskmanga.com)', 'Accept': 'image/*' },
    }, (res) => {
      if (res.statusCode !== 200) { res.resume(); resolve(false); return; }
      const chunks: Buffer[] = [];
      res.on('data', (c) => chunks.push(c as Buffer));
      res.on('end', () => {
        try { fs.writeFileSync(dest, Buffer.concat(chunks)); resolve(true); }
        catch { resolve(false); }
      });
      res.on('error', () => resolve(false));
    });
    req.on('error', () => resolve(false));
    req.setTimeout(10000, () => { req.destroy(); resolve(false); });
  });
}

/**
 * Check if a new job can be started (for backwards compatibility)
 */
export function canStartNewJob(): boolean {
  return !isProcessing;
}

/**
 * Absolute queue position for a single job: how many jobs (from anyone,
 * including this user's own earlier chapters) will run before it. Counts the
 * currently-running job since it occupies the front slot. Returns 0 when the
 * job is already running or finished (it's no longer waiting in jobQueue).
 * Surfaced via /api/status as `jobsAhead` so each queued card can show its
 * real "#N in queue" position — distinct from getChaptersAheadForIp, which
 * intentionally counts only OTHER users for the fairness banner.
 */
export function getJobsAhead(jobId: string): number {
  const idx = jobQueue.indexOf(jobId);
  if (idx === -1) return 0; // running or done → at the front / not waiting
  return idx + (isProcessing ? 1 : 0);
}

/**
 * How many chapters from OTHER users are scheduled before this user's batch —
 * i.e. ahead of `ip`'s earliest still-queued job. The running job counts if it
 * belongs to someone else. The user's own jobs are never counted (their order
 * among themselves is self-evident). Returns 0 when nothing of `ip`'s is
 * waiting, or when only their own jobs are ahead. Surfaced via /api/status so
 * the frontend can show a single "N chapters ahead of you" box.
 */
export function getChaptersAheadForIp(ip: string): number {
  const myFirstIdx = jobQueue.findIndex((id) => jobs.get(id)?.ip === ip);
  if (myFirstIdx === -1) return 0; // nothing of this user's is queued
  let count = 0;
  // The running job is ahead of everything still in the queue.
  if (isProcessing && runningJobId && jobs.get(runningJobId)?.ip !== ip) count++;
  // Queued jobs sitting before this user's first one.
  for (let i = 0; i < myFirstIdx; i++) {
    if (jobs.get(jobQueue[i])?.ip !== ip) count++;
  }
  return count;
}

/**
 * Create a new job (does not start processing).
 *
 * `cachedData`, when present, lets the pipeline skip Phase 4b LLM and Phase 3
 * chapter analysis — the cached bbox→simplifiedText map is bbox-matched to
 * fresh blocks downstream.
 */
export function createJob(
  url: string,
  levels: string[],
  includeEnglish: boolean = false,
  inputLanguage: InputLanguage = 'auto',
  cachedData?: CachedChapter,
  ip?: string,
  localImagePaths?: string[],
): string {
  const id = uuidv4();
  const hskLevels = levels as HSKLevel[];
  const job: Job = {
    id,
    url,
    localImagePaths,
    level: hskLevels[0], // Primary level (used for simplification target)
    levels: hskLevels,
    includeEnglish,
    inputLanguage,
    status: 'pending',
    progress: 0,
    total: 0,
    message: cachedData ? 'Queued (cache hit — will skip translation)...' : 'Queued...',
    createdAt: new Date(),
    ip,
    cachedData,
  };
  jobs.set(id, job);
  return id;
}

/**
 * Get a job by ID
 */
export function getJob(id: string): Job | undefined {
  return jobs.get(id);
}

/**
 * Update a job's properties
 */
function updateJob(id: string, updates: Partial<Job>): void {
  const job = jobs.get(id);
  if (job) {
    Object.assign(job, updates);
  }
}

/**
 * Add a job to the processing queue
 */
export function queueJob(id: string): void {
  const job = jobs.get(id);
  if (!job) {
    console.error(`Cannot queue job ${id}: not found`);
    return;
  }

  jobQueue.push(id);
  console.log(`Job ${id} added to queue. Queue length: ${jobQueue.length}`);

  // Try to process next job
  processNextJob();
}

/**
 * Process the next job in the queue
 */
async function processNextJob(): Promise<void> {
  // If already processing or queue is empty, do nothing
  if (isProcessing || jobQueue.length === 0) {
    return;
  }

  // Get next job from queue
  const jobId = jobQueue.shift();
  if (!jobId) return;

  const job = jobs.get(jobId);
  if (!job) {
    console.error(`Job ${jobId} not found, skipping`);
    processNextJob(); // Try next
    return;
  }

  isProcessing = true;
  runningJobId = jobId;
  console.log(`Starting job ${jobId}. Remaining in queue: ${jobQueue.length}`);

  try {
    await runJob(jobId);
  } catch (error) {
    console.error(`Job ${jobId} failed with error:`, error);
  } finally {
    isProcessing = false;
    runningJobId = null;
    // Process next job in queue
    processNextJob();
  }
}

/**
 * Run the processing pipeline for a job
 */
async function runJob(id: string): Promise<void> {
  const job = jobs.get(id);
  if (!job) {
    throw new Error(`Job ${id} not found`);
  }

  let tempDir: string | null = null;
  // Tee server stdout/stderr to logs/job_{id}.log for the duration of this job.
  // The file is deleted on success (unless PRESERVE_ALL_LOGS=1) and kept on error.
  const logHandle = startJobLogCapture(id);
  // Phase 2.2: track wall-clock start so we can record the total time on success.
  const jobStartedAtMs = Date.now();

  try {
    // Create temp directory for downloads
    tempDir = path.join(os.tmpdir(), `manga-${id}`);
    fs.mkdirSync(tempDir, { recursive: true });

    // Create output directory
    const jobOutputDir = path.join(outputsDir, `job_${id}`);
    fs.mkdirSync(jobOutputDir, { recursive: true });

    // Step 1: Acquire chapter images. Two paths:
    //   - User-uploaded files: skip download, use the paths the upload
    //     endpoint stashed on the job.
    //   - URL submission: hit the downloader to fetch the chapter.
    let imagePaths: string[];
    let chapterTitle: string;
    if (job.localImagePaths && job.localImagePaths.length > 0) {
      updateJob(id, {
        status: 'processing',
        message: `Processing ${job.localImagePaths.length} uploaded images…`,
      });
      imagePaths = job.localImagePaths;
      chapterTitle = job.url || `upload-${id.slice(0, 8)}`;
    } else {
      updateJob(id, {
        status: 'downloading',
        message: 'Downloading images from URL...',
      });
      const downloadResult = await downloadWebtoonChapter(job.url, tempDir);
      if (downloadResult.downloadedImages === 0) {
        throw new Error('No images were downloaded from the URL');
      }
      imagePaths = downloadResult.imagePaths;
      chapterTitle = downloadResult.chapterTitle;
    }

    updateJob(id, {
      progress: 0,
      total: imagePaths.length,
      title: chapterTitle,
      message: `Got ${imagePaths.length} images. Starting processing...`,
    });

    // Learner-friendly title: resolve a clean series name + cover (+ pinyin for
    // Chinese) for the scraped title. Runs for BOTH Chinese and English sources —
    // English aggregator titles ("... - Read Online | Asura Scans") get the site
    // cruft stripped, the chapter marker normalized, a cover, and a canonical
    // name so the same series doesn't show up several different ways. Best-effort:
    // any failure leaves the raw title in place. Skipped only for uploads (no
    // scraped series title) and empty titles.
    if (chapterTitle && !(job.localImagePaths && job.localImagePaths.length)) {
      try {
        const r = await resolveChapterTitle(chapterTitle);
        let coverUrl: string | null = null;
        if (r.coverCdnUrl) {
          const saved = await downloadCoverToFile(r.coverCdnUrl, path.join(jobOutputDir, 'cover.jpg'));
          if (saved) coverUrl = `/outputs/job_${id}/cover.jpg`;
        }
        updateJob(id, {
          resolved: {
            english: r.english,
            seriesZh: r.seriesZh,
            seriesPinyin: r.seriesPinyin,
            chapterLabel: r.chapterLabel,
            coverUrl,
          },
        });
      } catch (resolveErr) {
        console.error(`[Job ${id}] title resolution failed (non-fatal):`, (resolveErr as any)?.message);
      }
    }

    // Step 2: Process images
    updateJob(id, {
      status: 'processing',
      message: `Processing 0/${imagePaths.length} images...`,
    });

    // On cache hit with v2+ entry (has characters/keep/definitions), synthesize a
    // ChapterContext from the cached metadata. Pipeline's Phase 3 short-circuits
    // when options.chapterContext is set, AND buildPhraseData reuses the cached
    // definitions/keep for popup tooltips → cache-hit reader matches cold-run quality.
    let chapterContextFromCache;
    if (job.cachedData && (job.cachedData.characters || job.cachedData.definitions || job.cachedData.keep)) {
      const defsArr = job.cachedData.definitions || [];
      chapterContextFromCache = {
        setting: '',
        characters: job.cachedData.characters || [],
        plot: [],
        keep: job.cachedData.keep || [],
        definitions: new Map(defsArr.map(d => [d.term, d.def])),
      };
    }

    // Browser-tab title for the reader: resolved English name + chapter number,
    // else the resolved (cleaned) Chinese name, else the raw scraped title. For
    // uploads there's no meaningful chapter title, so leave it undefined and let
    // the pipeline fall back to the uploaded filename.
    const rt = job.resolved;
    const isUpload = !!(job.localImagePaths && job.localImagePaths.length);
    const readerTitle =
      rt && rt.english
        ? (rt.chapterLabel ? `${rt.english} — ${rt.chapterLabel}` : rt.english)
        : rt && rt.seriesZh
          ? (rt.chapterLabel ? `${rt.seriesZh} ${rt.chapterLabel}` : rt.seriesZh)
          : (isUpload ? undefined : (chapterTitle || undefined));

    const result = await processMangaPages(imagePaths, {
      targetLevel: job.level,
      targetLevels: job.levels,
      includeEnglish: job.includeEnglish || undefined,
      inputLanguage: job.inputLanguage,
      outputDir: jobOutputDir,
      readerTitle,
      convertTraditional: isTranslationSource(job.inputLanguage) ? false : undefined, // Skip for translation sources (EN nothing to convert; JA shinjitai must not pass t2s); when 'auto', pipeline resolves after detection
      visionProvider: 'paddleocr',
      // URL-sourced chapters fail fast when the language-detection sample
      // finds zero readable text (textless chapter / unsupported language).
      // Uploads keep the historic default-to-chinese behavior.
      failOnNoText: !isUpload,
      separateFiles: false,
      // Cache-hit fast path: feed cached simplifications to pipeline so Phase 4b
      // LLM is skipped (bbox-matched). chapterContext (if present) makes Phase 3
      // skip naturally; skipChapterAnalysis is the belt-and-suspenders fallback
      // for v1 cache entries that don't carry chapterContext fields.
      cachedSimplifications: job.cachedData ? { pages: job.cachedData.pages } : undefined,
      chapterContext: chapterContextFromCache,
      skipChapterAnalysis: job.cachedData ? true : undefined,
      onProgress: (current, total, message) => {
        updateJob(id, {
          progress: current,
          total: total,
          message: message,
        });
      },
    });

    // A reader only exists if the pipeline could turn the download into real
    // pages. If it's missing — e.g. an unsupported site where we scraped only
    // junk images, or reader generation failed — fail loudly with a clear
    // message instead of marking the job complete and handing back a resultUrl
    // that 404s ("Cannot GET /outputs/job_.../reader.html").
    if (!fs.existsSync(path.join(jobOutputDir, 'reader.html'))) {
      const err: any = new Error('Processing produced no reader (source likely unsupported)');
      err.code = 'NO_READER';
      throw err;
    }

    // Step 3: Write the text-only cache for future requests.
    // Always points users to the per-job reader (cache stores text only — no
    // images/HTML, so it's never user-facing).
    const resultUrl = `/outputs/job_${id}/reader.html`;
    if (result.cacheBlocks && result.cacheBlocks.length > 0) {
      try {
        // Persist popup-restoring chapter metadata alongside the simplification
        // bboxes. Skipped: setting + plot summaries (synopsis-shaped exposure).
        const ctx = result.chapterContext;
        const extras = ctx ? {
          characters: ctx.characters,
          keep: ctx.keep,
          definitions: ctx.definitions, // Map — commitTextCache normalizes
        } : undefined;
        commitTextCache(job.url, job.level, job.inputLanguage, result.cacheBlocks, extras);
        const extrasNote = extras
          ? ` + ${ctx!.characters.length} chars / ${ctx!.keep.length} keep / ${ctx!.definitions.size} defs`
          : '';
        console.log(`[Cache] WROTE text-only entry for ${job.url} @ ${job.level} lang=${job.inputLanguage} (${result.cacheBlocks.length} pages${extrasNote})`);
      } catch (cacheErr) {
        console.error(`[Cache] Failed to write cache for job ${id}:`, cacheErr);
      }
    }

    // Step 4: Mark complete
    updateJob(id, {
      status: 'complete',
      progress: imagePaths.length,
      total: imagePaths.length,
      message: 'Processing complete!',
      resultUrl,
    });

    // Phase 2.2: record success-only into job-metrics.json so /api/eta can
    // give users an accurate median ETA. Failures are NOT recorded (they'd
    // skew the distribution toward fail-fast times).
    try {
      const stats = getTimingStats();
      recordJobCompletion({
        ts: Date.now(),
        pages: imagePaths.length,
        elapsedMs: Date.now() - jobStartedAtMs,
        cacheHit: !!job.cachedData,
        title: job.title,
        avgPageWidth: stats.avgPageWidth,
        avgPageHeight: stats.avgPageHeight,
        fireworks: getLLMUsage().fireworks,
        phases: {
          loading: stats.loading,
          stitching: stats.stitching,
          ocrScan: stats.ocrScan,
          chapterAnalysis: stats.chapterAnalysis,
          ocr: stats.ocr,
          llm: stats.llm,
          inpaintServer: stats.inpaintServer,
          textRendering: stats.textRendering,
        },
      });
    } catch (metricsErr) {
      // Never let a metrics write failure crash a finished job — telemetry
      // is best-effort, the user-facing job has already completed successfully.
      console.error(`[Metrics] Failed to record job ${id} timings:`, metricsErr);
    }

    console.log(`Job ${id} completed. Result: ${resultUrl}`);

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    const errorCode = (error as any)?.code || 'INTERNAL';
    console.error(`Job ${id} failed [${errorCode}]:`, errorMessage);
    updateJob(id, {
      status: 'error',
      message: 'Processing failed',
      error: errorMessage,
      errorCode,
    });
    // Phase 1.6: append to errors.log so the user's secondary terminal
    // (Get-Content errors.log -Wait) sees it immediately.
    appendErrorLog({
      jobId: id,
      code: errorCode,
      url: job.url,
      message: errorMessage,
      hasLogFile: true, // we keep the per-job log on error
    });
  } finally {
    // Clean up temp directory
    if (tempDir && fs.existsSync(tempDir)) {
      try {
        fs.rmSync(tempDir, { recursive: true, force: true });
        console.log(`Cleaned up temp directory: ${tempDir}`);
      } catch (e) {
        console.error(`Failed to clean up temp directory: ${e}`);
      }
    }
    // Restore stdout/stderr; delete log file if successful.
    // Awaited so the next job in the queue starts with a fully-flushed log.
    const succeeded = jobs.get(id)?.status === 'complete';
    await logHandle.finalize(succeeded);
  }
}

/**
 * Rehydrate jobs from disk on startup.
 *
 * Two cases:
 *   - Directory has reader.html → job completed before shutdown → mark `complete`
 *   - Directory has no .html    → job was in-flight when the process died →
 *                                 mark `error` with INTERRUPTED code so the
 *                                 frontend renders a clean retry message instead
 *                                 of polling 404 forever.
 *
 * Without the second case, users whose chapter was processing during a restart
 * see "submit and infinitely spin" — the job is gone from memory, the frontend
 * gets 404s, and the UX is dead-ended.
 */
export function rehydrateCompletedJobs(): void {
  if (!fs.existsSync(outputsDir)) {
    fs.mkdirSync(outputsDir, { recursive: true });
    console.log(`Created outputs directory: ${outputsDir}`);
    return;
  }

  const entries = fs.readdirSync(outputsDir, { withFileTypes: true });
  let completed = 0;
  let interrupted = 0;

  for (const entry of entries) {
    if (!entry.isDirectory() || !entry.name.startsWith('job_')) continue;

    const jobId = entry.name.replace('job_', '');
    const jobDir = path.join(outputsDir, entry.name);
    const files = fs.readdirSync(jobDir);
    const htmlFile = files.find(f => f.endsWith('.html'));
    const createdAt = new Date(fs.statSync(jobDir).mtime);

    if (htmlFile) {
      const job: Job = {
        id: jobId,
        url: 'Unknown (rehydrated)',
        level: 'hsk3',
        levels: ['hsk3'],
        includeEnglish: false,
        inputLanguage: 'chinese',
        status: 'complete',
        progress: 1,
        total: 1,
        message: 'Processing complete!',
        resultUrl: `/outputs/${entry.name}/${htmlFile}`,
        createdAt,
      };
      jobs.set(jobId, job);
      completed++;
    } else {
      // No HTML output → in-flight when the process died. Surface as a clean
      // error so the frontend shows a retry message rather than spinning.
      const job: Job = {
        id: jobId,
        url: 'Unknown (rehydrated)',
        level: 'hsk3',
        levels: ['hsk3'],
        includeEnglish: false,
        inputLanguage: 'chinese',
        status: 'error',
        progress: 0,
        total: 0,
        message: 'Server restarted before this finished — please resubmit.',
        error: 'Server restarted before this finished — please resubmit.',
        errorCode: 'INTERRUPTED',
        createdAt,
      };
      jobs.set(jobId, job);
      interrupted++;
    }
  }

  console.log(`Rehydrated ${completed} completed job(s) and ${interrupted} interrupted job(s) from disk`);
}
