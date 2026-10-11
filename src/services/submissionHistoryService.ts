// Per-browser submission history.
//
// Tracks the URLs (+ level, lang, title) that have been submitted from each
// browser_id, going forward. Two purposes:
//
//   1. Library auto-add: when a logged-in user submits a chapter, we record
//      the submission here AND push it to libraryService. When they log in
//      from a new device, the initial sync pulls submissions from all
//      browser_ids previously linked to their email.
//   2. "Recently submitted" history for users who haven't logged in yet —
//      not currently exposed but the data is here when we want it.
//
// Storage: data/submission-history.json — same atomic-write-via-tempfile
// pattern as the other services. State is small (a few KB per active browser).

import * as fs from 'fs';
import * as path from 'path';

const DATA_PATH = path.join(__dirname, '..', '..', 'data', 'submission-history.json');

export interface SubmissionEntry {
  url: string;
  level: string;
  lang: string;
  title: string;
  submittedAt: number;
}

interface SubmissionHistoryData {
  byBrowser: Record<string, SubmissionEntry[]>;
}

let cache: SubmissionHistoryData | null = null;

function load(): SubmissionHistoryData {
  if (cache) return cache;
  try {
    if (fs.existsSync(DATA_PATH)) {
      cache = JSON.parse(fs.readFileSync(DATA_PATH, 'utf-8'));
      return cache!;
    }
  } catch (e) {
    console.error('[SubmissionHistory] Failed to load, starting fresh:', e);
  }
  cache = { byBrowser: {} };
  return cache;
}

function save(): void {
  if (!cache) return;
  try {
    fs.mkdirSync(path.dirname(DATA_PATH), { recursive: true });
    const tmp = DATA_PATH + '.tmp';
    fs.writeFileSync(tmp, JSON.stringify(cache, null, 2));
    fs.renameSync(tmp, DATA_PATH);
  } catch (e) {
    console.error('[SubmissionHistory] Failed to persist:', e);
  }
}

/**
 * Record one submission against a browser_id. No-op if browserId is missing.
 * Dedupes by URL — multiple submissions of the same URL keep just the latest
 * entry (refreshes title + timestamp).
 */
export function recordSubmission(
  browserId: string | undefined,
  entry: Omit<SubmissionEntry, 'submittedAt'>,
): void {
  if (!browserId) return;
  const data = load();
  if (!data.byBrowser[browserId]) data.byBrowser[browserId] = [];
  const list = data.byBrowser[browserId];
  const idx = list.findIndex(e => e.url === entry.url);
  const fullEntry: SubmissionEntry = { ...entry, submittedAt: Date.now() };
  if (idx >= 0) list[idx] = fullEntry;
  else list.push(fullEntry);
  save();
}

/**
 * All entries recorded for a single browser, newest first. Empty list if none.
 */
export function getSubmissions(browserId: string | undefined): SubmissionEntry[] {
  if (!browserId) return [];
  const data = load();
  const list = data.byBrowser[browserId] || [];
  return [...list].sort((a, b) => b.submittedAt - a.submittedAt);
}

/**
 * Merge entries from MANY browser_ids into a single deduped list (by URL,
 * keeping the newest submission). Used by libraryService.syncFromBrowsers
 * to backfill a user's library on first login.
 */
export function getSubmissionsForBrowsers(browserIds: string[]): SubmissionEntry[] {
  const data = load();
  const byUrl = new Map<string, SubmissionEntry>();
  for (const bid of browserIds) {
    const list = data.byBrowser[bid] || [];
    for (const e of list) {
      const existing = byUrl.get(e.url);
      if (!existing || existing.submittedAt < e.submittedAt) byUrl.set(e.url, e);
    }
  }
  return Array.from(byUrl.values()).sort((a, b) => b.submittedAt - a.submittedAt);
}
