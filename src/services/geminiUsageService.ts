// Daily Gemini API usage counter.
//
// Tracks how many requests + tokens the app has sent to Gemini today, so the
// operator can see how close they are to the free-tier daily quota. This is a
// monitor, not a circuit breaker — it never blocks calls (unlike spendCap.ts).
//
// Modeled on spendCap.ts, with two deliberate differences:
//   1. Reset boundary is PACIFIC midnight, not UTC. Google resets Gemini
//      free-tier quotas at midnight Pacific; resetting on any other clock would
//      leave this counter hours out of sync with the real quota.
//   2. Atomic temp-and-rename writes (like creditsService.ts), so a crash
//      mid-write can't leave the file half-written.

import * as fs from 'fs';
import * as path from 'path';

const GEMINI_USAGE_PATH = path.join(__dirname, '..', '..', 'data', 'gemini-usage.json');
// Gemini free-tier daily token cap. Configurable because Google changes its
// free-tier numbers; the dashboard shows "used / cap" against this value.
const DAILY_TOKEN_CAP = parseInt(process.env.GEMINI_DAILY_TOKEN_CAP || '1000000', 10);

interface GeminiUsageData {
  date: string; // YYYY-MM-DD in America/Los_Angeles (matches Google's quota reset)
  requests: number;
  inputTokens: number;
  outputTokens: number;
}

let cache: GeminiUsageData | null = null;

/** Today's date in Pacific time (YYYY-MM-DD) — Gemini quotas reset at Pacific midnight. */
function todayPacific(): string {
  return new Date().toLocaleDateString('en-CA', { timeZone: 'America/Los_Angeles' });
}

function fresh(): GeminiUsageData {
  return { date: todayPacific(), requests: 0, inputTokens: 0, outputTokens: 0 };
}

function load(): GeminiUsageData {
  if (cache) return cache;
  try {
    if (fs.existsSync(GEMINI_USAGE_PATH)) {
      cache = JSON.parse(fs.readFileSync(GEMINI_USAGE_PATH, 'utf-8'));
      return cache!;
    }
  } catch (e) {
    console.error('[GeminiUsage] Failed to load, starting fresh:', e);
  }
  cache = fresh();
  return cache;
}

function save(data: GeminiUsageData): void {
  try {
    fs.mkdirSync(path.dirname(GEMINI_USAGE_PATH), { recursive: true });
    // Atomic: write to temp, then rename. Rename is atomic on the same volume.
    const tmp = GEMINI_USAGE_PATH + '.tmp';
    fs.writeFileSync(tmp, JSON.stringify(data, null, 2));
    fs.renameSync(tmp, GEMINI_USAGE_PATH);
  } catch (e) {
    console.error('[GeminiUsage] Failed to persist:', e);
  }
}

function rolloverIfNeeded(): GeminiUsageData {
  const data = load();
  const today = todayPacific();
  if (data.date !== today) {
    console.log(`[GeminiUsage] New day (${today}), resetting from ${data.requests} req / ${data.inputTokens + data.outputTokens} tok`);
    data.date = today;
    data.requests = 0;
    data.inputTokens = 0;
    data.outputTokens = 0;
    save(data);
  }
  return data;
}

/**
 * Record one Gemini API call: counts the request and adds its tokens to today's
 * running total. Never blocks — this is monitoring only.
 */
export function recordGeminiUsage(inputTokens: number, outputTokens: number): void {
  const data = rolloverIfNeeded();
  data.requests += 1;
  data.inputTokens += inputTokens || 0;
  data.outputTokens += outputTokens || 0;
  save(data);
}

/**
 * Snapshot of today's Gemini usage, for the admin dashboard.
 */
export function getGeminiUsage(): {
  date: string;
  requests: number;
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  tokenCap: number;
} {
  const data = rolloverIfNeeded();
  return {
    date: data.date,
    requests: data.requests,
    inputTokens: data.inputTokens,
    outputTokens: data.outputTokens,
    totalTokens: data.inputTokens + data.outputTokens,
    tokenCap: DAILY_TOKEN_CAP,
  };
}
