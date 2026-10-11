// Per-account chapter library.
//
// Identity = email (Google OAuth or magic-link both resolve to it). A library
// entry is a POINTER to a finished chapter — the input URL plus the openable
// reader URL (`resultUrl`) — NOT the chapter content. The rendered chapter
// (reader.html + images) is served from disk on Computer 2 (Phase A); Phase B
// moves those to cloud storage.
//
// OWNERSHIP MODEL (important): chapters only ever enter an account by an
// EXPLICIT action — the client calls addToLibrary (on completion while signed
// in) or the claim endpoint (user chose to add anonymous chapters). There is
// deliberately NO automatic "link this browser to whoever signs in" sweep:
// that mechanism leaked one account's chapters into the next account that
// signed in on the same browser. Do not reintroduce it.
//
// Storage: data/library.json — atomic-write-via-tempfile, in-memory cache.

import * as fs from 'fs';
import * as path from 'path';
import { ResolvedTitleFields } from '../title-resolver';

const DATA_PATH = path.join(__dirname, '..', '..', 'data', 'library.json');

export interface LibraryEntry {
  url: string;        // the input URL the user submitted
  resultUrl?: string; // openable reader URL, e.g. /outputs/job_<id>/reader.html
  level: string;
  lang: string;
  title: string;      // raw scraped title — the fallback display + legacy entries
  // Learner-friendly resolution (English name + cover + pinyin + chapter #).
  // Absent on old entries and on chapters that couldn't be resolved.
  resolved?: ResolvedTitleFields;
  addedAt: number;
  lastReadAt?: number;
}

interface LibraryData {
  // lowercased email → list of entries (any order, frontend sorts)
  byEmail: Record<string, LibraryEntry[]>;
}

let cache: LibraryData | null = null;

function load(): LibraryData {
  if (cache) return cache;
  try {
    if (fs.existsSync(DATA_PATH)) {
      const parsed = JSON.parse(fs.readFileSync(DATA_PATH, 'utf-8'));
      cache = { byEmail: parsed.byEmail || {} };
      return cache!;
    }
  } catch (e) {
    console.error('[Library] Failed to load, starting fresh:', e);
  }
  cache = { byEmail: {} };
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
    console.error('[Library] Failed to persist:', e);
  }
}

function norm(email: string): string {
  return email.trim().toLowerCase();
}

/**
 * Full library for an email, newest entry first. Uncapped — a user keeps every
 * chapter they've ever added.
 */
export function getLibrary(email: string | undefined): LibraryEntry[] {
  if (!email) return [];
  const data = load();
  const list = data.byEmail[norm(email)] || [];
  return [...list].sort((a, b) => b.addedAt - a.addedAt);
}

/**
 * Add (or refresh) a chapter in a user's library. Identity is (url, level) — the
 * SAME url at a different HSK level is a distinct entry, so a chapter can live in
 * the library at more than one level. Re-adding an existing (url, level) updates
 * its fields (title, resultUrl) and bumps addedAt. A missing resultUrl on an
 * update does NOT clobber an existing one. No-op if email is missing.
 */
export function addToLibrary(
  email: string | undefined,
  entry: Omit<LibraryEntry, 'addedAt'>,
): void {
  if (!email) return;
  const data = load();
  const key = norm(email);
  if (!data.byEmail[key]) data.byEmail[key] = [];
  const list = data.byEmail[key];
  const idx = list.findIndex(e => e.url === entry.url && e.level === entry.level);
  if (idx >= 0) {
    const prev = list[idx];
    list[idx] = {
      ...prev,
      ...entry,
      // never overwrite a known reader URL with an empty one
      resultUrl: entry.resultUrl || prev.resultUrl,
      // keep a previously-resolved title if this re-add couldn't resolve one
      resolved: entry.resolved || prev.resolved,
      addedAt: Date.now(),
    };
  } else {
    list.push({ ...entry, addedAt: Date.now() });
  }
  save();
}

/**
 * Mark a chapter as read just now. Identity matches addToLibrary: (url, level)
 * when level is given, otherwise every entry for the url (a multi-level open
 * from an ambiguous client is rare enough that over-touching is harmless).
 * Returns whether anything was updated.
 */
export function touchLastRead(email: string | undefined, url: string, level?: string): boolean {
  if (!email || !url) return false;
  const data = load();
  const list = data.byEmail[norm(email)];
  if (!list) return false;
  let touched = false;
  for (const e of list) {
    if (e.url === url && (level === undefined || e.level === level)) {
      e.lastReadAt = Date.now();
      touched = true;
    }
  }
  if (touched) save();
  return touched;
}

/**
 * Remove a chapter. Identity is (url, level): if `level` is given, only the
 * entry at that level is removed (the same url at other levels stays). If
 * `level` is omitted, all entries for that url are removed. Silent no-op if not
 * found.
 */
export function removeFromLibrary(email: string | undefined, url: string, level?: string): boolean {
  if (!email || !url) return false;
  const data = load();
  const key = norm(email);
  const list = data.byEmail[key];
  if (!list) return false;
  const before = list.length;
  data.byEmail[key] = list.filter(e => !(e.url === url && (level === undefined || e.level === level)));
  if (data.byEmail[key].length === before) return false;
  save();
  return true;
}

/**
 * Snapshot for admin/debug. readLast7d counts entries opened from a library in
 * the last week — account-level reading activity, complementing the anonymous
 * browser-level reader stats from eventLogService.
 */
export function getStats(): { emailsTracked: number; totalEntries: number; readLast7d: number } {
  const data = load();
  const cutoff = Date.now() - 7 * 24 * 60 * 60 * 1000;
  let totalEntries = 0;
  let readLast7d = 0;
  for (const list of Object.values(data.byEmail)) {
    totalEntries += list.length;
    for (const e of list) if ((e.lastReadAt || 0) >= cutoff) readLast7d++;
  }
  return {
    emailsTracked: Object.keys(data.byEmail).length,
    totalEntries,
    readLast7d,
  };
}
