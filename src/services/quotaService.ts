// Per-IP chapter submission rate limiter — daily-only.
//
// Phase 1.6: simplified from two-window (short + daily) to daily-only.
// The serial queue means each chapter takes ~75s, so a user can't submit
// faster than the daily cap allows anyway. The short window added UX
// confusion ("5-min cooldown") without meaningful protection.
//
// State is in-memory only — losing it on restart is acceptable since attackers
// would still hit per-IP limits within a few requests after restart.

const LONG_WINDOW_MS = 24 * 60 * 60 * 1000;
const LONG_WINDOW_MAX = parseInt(process.env.QUOTA_DAILY_MAX || '30', 10);

// IP → array of submission timestamps (newest last)
const submissions = new Map<string, number[]>();

/**
 * Periodic cleanup so old IPs don't leak memory. Runs every hour.
 * Drops timestamps older than the long window.
 */
function pruneAll(): void {
  const now = Date.now();
  const cutoff = now - LONG_WINDOW_MS;
  for (const [ip, timestamps] of submissions) {
    const fresh = timestamps.filter(t => t >= cutoff);
    if (fresh.length === 0) {
      submissions.delete(ip);
    } else if (fresh.length !== timestamps.length) {
      submissions.set(ip, fresh);
    }
  }
}
setInterval(pruneAll, 60 * 60 * 1000).unref();

export interface QuotaCheckResult {
  ok: boolean;
  reason?: string;
  retryAfterSeconds?: number;
  limit?: { window: 'long'; max: number };
}

/**
 * Check whether `ip` may submit a new chapter right now. Read-only — does NOT
 * record the submission. Caller should call `recordSubmission(ip)` only after
 * other validations pass and the job is actually queued.
 */
export function checkQuota(ip: string): QuotaCheckResult {
  const now = Date.now();
  let timestamps = submissions.get(ip) || [];
  // Prune in-place for this IP
  timestamps = timestamps.filter(t => t >= now - LONG_WINDOW_MS);
  submissions.set(ip, timestamps);

  if (timestamps.length >= LONG_WINDOW_MAX) {
    const oldest = Math.min(...timestamps);
    const retryAfterSeconds = Math.ceil((oldest + LONG_WINDOW_MS - now) / 1000);
    const hours = Math.ceil(retryAfterSeconds / 3600);
    return {
      ok: false,
      reason: `Daily limit: ${LONG_WINDOW_MAX} chapters per 24h per IP. Try again in ${hours}h.`,
      retryAfterSeconds,
      limit: { window: 'long', max: LONG_WINDOW_MAX },
    };
  }

  return { ok: true };
}

/**
 * Record that `ip` just submitted a chapter. Call only after the job has
 * actually been queued (not before validation passes).
 */
export function recordSubmission(ip: string): void {
  const timestamps = submissions.get(ip) || [];
  timestamps.push(Date.now());
  submissions.set(ip, timestamps);
}

/**
 * Snapshot of current quota state for `ip` — for debugging or admin panels.
 */
export function getQuotaStatus(ip: string): {
  ip: string;
  dailyUsed: number;
  dailyMax: number;
} {
  const now = Date.now();
  const timestamps = (submissions.get(ip) || []).filter(t => t >= now - LONG_WINDOW_MS);
  return {
    ip,
    dailyUsed: timestamps.length,
    dailyMax: LONG_WINDOW_MAX,
  };
}

export class QuotaExceededError extends Error {
  code = 'QUOTA_EXCEEDED';
  retryAfterSeconds: number;
  limit: { window: 'long'; max: number };
  constructor(reason: string, retryAfterSeconds: number, limit: { window: 'long'; max: number }) {
    super(reason);
    this.name = 'QuotaExceededError';
    this.retryAfterSeconds = retryAfterSeconds;
    this.limit = limit;
  }
}
