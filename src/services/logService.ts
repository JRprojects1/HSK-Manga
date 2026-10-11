// Per-job log capture + structured errors.log for fast error scanning.
//
// Architecture:
//   - During a job, tee process.stdout/stderr to logs/job_{id}.log
//   - On job completion (success): delete the file (most logs aren't useful)
//   - On job failure: keep the file + append a one-line entry to errors.log
//   - At server startup: prune kept logs older than LOG_RETENTION_DAYS (default 14)
//   - PRESERVE_ALL_LOGS=1 keeps successful-job logs for debugging
//
// Why this design:
//   The user runs `Get-Content errors.log -Wait` in a second terminal to see
//   errors as they happen. Each entry points at a per-job log so they can
//   open the full context for any failure without scrolling the live server
//   terminal.
//
// Single-threaded queue assumption:
//   Tee uses process.stdout.write/process.stderr.write swap. Safe because
//   only one job runs at a time. When fair-share queueing (Phase 3.4) lands,
//   revisit — likely move to AsyncLocalStorage or pass logger objects.

import * as fs from 'fs';
import * as path from 'path';

const PROJECT_ROOT = path.join(__dirname, '..', '..');
export const LOGS_DIR = path.join(PROJECT_ROOT, 'logs');
export const ERRORS_LOG_PATH = path.join(PROJECT_ROOT, 'errors.log');
const LOG_RETENTION_DAYS = parseInt(process.env.LOG_RETENTION_DAYS || '14', 10);
const PRESERVE_ALL_LOGS = process.env.PRESERVE_ALL_LOGS === '1';

/**
 * Ensure logs/ exists and prune anything older than LOG_RETENTION_DAYS.
 * Call once at server startup.
 */
export function initLogService(): void {
  fs.mkdirSync(LOGS_DIR, { recursive: true });
  const cutoffMs = Date.now() - LOG_RETENTION_DAYS * 24 * 60 * 60 * 1000;
  let pruned = 0;
  try {
    for (const entry of fs.readdirSync(LOGS_DIR)) {
      const filePath = path.join(LOGS_DIR, entry);
      try {
        const stat = fs.statSync(filePath);
        if (stat.isFile() && stat.mtimeMs < cutoffMs) {
          fs.unlinkSync(filePath);
          pruned++;
        }
      } catch { /* skip — file may have been deleted concurrently */ }
    }
  } catch { /* dir may not exist yet — already created above */ }
  if (pruned > 0) {
    console.log(`[LogService] Pruned ${pruned} log file(s) older than ${LOG_RETENTION_DAYS} days`);
  }
}

/**
 * Per-job tee handle returned by startJobLogCapture().
 * Caller MUST call .finalize(success) when the job ends. Awaiting the
 * returned promise guarantees the log file is fully closed (and possibly
 * deleted, on success) before the next job runs.
 */
export interface JobLogHandle {
  jobId: string;
  filePath: string;
  finalize(success: boolean): Promise<void>;
}

/**
 * Begin capturing stdout/stderr for the duration of a job to logs/job_{id}.log.
 * Output continues to flow to the original streams (server terminal still sees
 * everything live). The file is closed and conditionally deleted via finalize().
 */
export function startJobLogCapture(jobId: string): JobLogHandle {
  fs.mkdirSync(LOGS_DIR, { recursive: true });
  const filePath = path.join(LOGS_DIR, `job_${jobId}.log`);
  const fileStream = fs.createWriteStream(filePath, { flags: 'w' });

  const origStdoutWrite = process.stdout.write.bind(process.stdout);
  const origStderrWrite = process.stderr.write.bind(process.stderr);

  process.stdout.write = ((chunk: any, ...rest: any[]) => {
    try { fileStream.write(typeof chunk === 'string' ? chunk : chunk); } catch { /* never block console */ }
    return origStdoutWrite(chunk, ...rest);
  }) as typeof process.stdout.write;

  process.stderr.write = ((chunk: any, ...rest: any[]) => {
    try { fileStream.write(typeof chunk === 'string' ? chunk : chunk); } catch { /* never block console */ }
    return origStderrWrite(chunk, ...rest);
  }) as typeof process.stderr.write;

  // Header row for the file so it's identifiable when opened standalone
  fileStream.write(`=== Job ${jobId} started at ${new Date().toISOString()} ===\n`);

  return {
    jobId,
    filePath,
    finalize(success: boolean): Promise<void> {
      return new Promise<void>((resolve) => {
        fileStream.write(`=== Job ${jobId} ended at ${new Date().toISOString()} (status: ${success ? 'complete' : 'error'}) ===\n`);
        // Restore original streams BEFORE closing the file
        process.stdout.write = origStdoutWrite;
        process.stderr.write = origStderrWrite;
        fileStream.end(() => {
          // Successful jobs: delete unless PRESERVE_ALL_LOGS=1 (most successful runs don't need investigation)
          if (success && !PRESERVE_ALL_LOGS) {
            try { fs.unlinkSync(filePath); } catch { /* file may not exist */ }
          }
          resolve();
        });
      });
    },
  };
}

/**
 * Append a single-line error entry to errors.log for fast scanning via
 * `Get-Content errors.log -Wait`.
 *
 * Format (one entry = 2 lines for kept logs, 1 line for retried/recovered events):
 *   [2026-05-09T15:23:45Z] [LLM_TIMEOUT] [job_abc] [https://...] message text
 *     → logs/job_abc.log
 */
export function appendErrorLog(args: {
  jobId: string;
  code: string;
  url?: string;
  message: string;
  hasLogFile?: boolean;
}): void {
  const { jobId, code, url, message, hasLogFile } = args;
  const timestamp = new Date().toISOString();
  const urlPart = url ? `[${url}] ` : '';
  let entry = `[${timestamp}] [${code}] [job_${jobId}] ${urlPart}${message}\n`;
  if (hasLogFile) {
    entry += `  → logs/job_${jobId}.log\n`;
  }
  try {
    fs.appendFileSync(ERRORS_LOG_PATH, entry);
  } catch (e) {
    // errors.log write should never crash the server — if disk is full or path
    // is bad, log to console as a last resort
    console.error(`[LogService] Failed to write to errors.log: ${e}`);
  }
}

/**
 * Return the last `n` non-empty lines of errors.log (oldest first), for the
 * admin dashboard. Empty array if the file doesn't exist yet.
 */
export function getRecentErrorLines(n: number): string[] {
  try {
    if (!fs.existsSync(ERRORS_LOG_PATH)) return [];
    const lines = fs.readFileSync(ERRORS_LOG_PATH, 'utf-8').split('\n').filter((l) => l.trim());
    return lines.slice(-n);
  } catch {
    return [];
  }
}
