// Weekly metrics snapshot — the impact record, written automatically.
//
// This used to be a manual `npm run snapshot` chore. Now the running server
// calls writeWeeklySnapshot() on startup and once a day (see server.ts), so
// the current ISO week's row stays fresh with zero human action and rolls to
// a new row when the week turns over. The write is idempotent per week
// (upsert, not append), so running it repeatedly is safe.
//
// Output: data/metrics-weekly.json — aggregate counts ONLY (no emails, ids, or
// words), so it is safe to publish. It is gitignored on the server (like every
// other runtime data/ file) to keep Computer 2's `git pull` conflict-free; the
// committed evidence copy is pulled to Computer 1 via `npm run pull-metrics`
// (GET /api/admin/metrics) and committed there.

import * as fs from 'fs';
import * as path from 'path';
import { readEvents, isoWeekKey, getReaderStats } from './eventLogService';

const DATA_DIR = path.join(__dirname, '..', '..', 'data');
const OUT_PATH = path.join(DATA_DIR, 'metrics-weekly.json');
const WEEK_MS = 7 * 24 * 60 * 60 * 1000;

function readJson<T>(file: string, fallback: T): T {
  try { return JSON.parse(fs.readFileSync(file, 'utf-8')) as T; } catch { return fallback; }
}

export interface WeeklyRow {
  week: string;
  generatedAt: string;
  weeklyActiveReaders: number;
  returningReaders28d: number;
  readerOpens7d: number;
  wordClicks7d: number;
  toggles7d: number;
  flashcardSaves7d: number;
  sampleViews7d: number;
  chaptersProcessed7d: number;
  pagesProcessed7d: number;
  totalUsers: number;
  signups7d: number;
  activeAccounts7d: number;
  libraryEntries: number;
  libraryRead7d: number;
}

export interface MetricsFile {
  note: string;
  weeks: WeeklyRow[];
}

/** Compute the current week's aggregate row from the live runtime files. */
export function computeWeeklyRow(now: number = Date.now()): WeeklyRow {
  const since7d = now - WEEK_MS;
  const readers = getReaderStats(now);

  // Chapters processed (job metrics).
  const jobs = readJson<{ samples?: Array<{ ts: number; pages?: number }> }>(
    path.join(DATA_DIR, 'job-metrics.json'), { samples: [] },
  ).samples || [];
  const jobs7d = jobs.filter(s => s.ts >= since7d);

  // Accounts (users.json). loginHistory is the retention signal; fall back to
  // lastLoginAt for users created before that field existed.
  const usersFile = readJson<{ users?: Record<string, { createdAt?: number; lastLoginAt?: number; loginHistory?: number[] }> }>(
    path.join(DATA_DIR, 'users.json'), { users: {} },
  ).users || {};
  const users = Object.values(usersFile);
  const signups7d = users.filter(u => (u.createdAt || 0) >= since7d).length;
  const activeAccounts7d = users.filter(u =>
    (u.loginHistory && u.loginHistory.length ? u.loginHistory : [u.lastLoginAt || 0]).some(t => (t || 0) >= since7d),
  ).length;

  // Library (account-level reads).
  const lib = readJson<{ byEmail?: Record<string, Array<{ lastReadAt?: number }>> }>(
    path.join(DATA_DIR, 'library.json'), { byEmail: {} },
  ).byEmail || {};
  let libraryEntries = 0;
  let libraryRead7d = 0;
  for (const list of Object.values(lib)) {
    libraryEntries += list.length;
    for (const e of list) if ((e.lastReadAt || 0) >= since7d) libraryRead7d++;
  }

  return {
    week: isoWeekKey(now),
    generatedAt: new Date(now).toISOString(),
    weeklyActiveReaders: readers.weeklyActiveReaders,
    returningReaders28d: readers.returningReaders28d,
    readerOpens7d: readers.readerOpens7d,
    wordClicks7d: readers.wordClicks7d,
    toggles7d: readers.toggles7d,
    flashcardSaves7d: readers.flashcardSaves7d,
    sampleViews7d: readers.sampleViews7d,
    chaptersProcessed7d: jobs7d.length,
    pagesProcessed7d: jobs7d.reduce((a, s) => a + (s.pages || 0), 0),
    totalUsers: users.length,
    signups7d,
    activeAccounts7d,
    libraryEntries,
    libraryRead7d,
  };
}

/**
 * Upsert the current week's row into data/metrics-weekly.json and return it.
 * Idempotent per ISO week. Never throws — a metrics write must never take the
 * server down; failures are logged and swallowed.
 */
export function writeWeeklySnapshot(now: number = Date.now()): WeeklyRow | null {
  try {
    const row = computeWeeklyRow(now);
    const file = readJson<MetricsFile>(OUT_PATH, {
      note: 'Aggregate weekly usage metrics — no personal data. Written automatically by the server; published copy committed via npm run pull-metrics.',
      weeks: [],
    });
    if (!Array.isArray(file.weeks)) file.weeks = [];
    const idx = file.weeks.findIndex(r => r.week === row.week);
    if (idx >= 0) file.weeks[idx] = row; else file.weeks.push(row);
    fs.mkdirSync(DATA_DIR, { recursive: true });
    fs.writeFileSync(OUT_PATH, JSON.stringify(file, null, 2) + '\n');
    return row;
  } catch (e) {
    console.error('[Metrics] weekly snapshot failed:', e instanceof Error ? e.message : e);
    return null;
  }
}

/** Read the full committed-shape metrics file (for the admin endpoint / CLI). */
export function readMetricsFile(): MetricsFile {
  return readJson<MetricsFile>(OUT_PATH, { note: '', weeks: [] });
}
