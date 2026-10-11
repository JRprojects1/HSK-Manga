// Anonymous usage events — the evidence layer for real-usage metrics.
//
// Why this exists: chapters *processed* were already recorded (jobMetrics),
// but nothing recorded chapters being *read* or in-reader engagement, so
// "weekly active readers" and "returning readers" were unmeasurable. The
// generated reader posts small events here via navigator.sendBeacon.
//
// Privacy: no IP, no email, no account linkage. `bid` is a random per-browser
// id from localStorage — enough to count distinct/returning readers, nothing
// more. Word text is capped at 32 chars (it's dictionary vocabulary, useful
// for a "most looked-up words" view).
//
// Storage: data/events.jsonl, append-only (one JSON object per line). Append
// is used instead of the usual read-modify-write cache because events arrive
// concurrently and per-line JSONL needs no whole-file rewrite. A size cap
// stops runaway growth; weekly aggregates are snapshotted by
// scripts/weekly-snapshot.js so trimming old raw lines loses nothing.

import * as fs from 'fs';
import * as path from 'path';

const EVENTS_PATH = path.join(__dirname, '..', '..', 'data', 'events.jsonl');
// ~50 MB ≈ 300k events — far beyond current scale; guard only.
const MAX_FILE_BYTES = 50 * 1024 * 1024;
const SIZE_CHECK_EVERY = 500;

// Anything not listed here is dropped on the floor, silently — so a reader
// that emits a new event without an entry produces no data and no error.
const ALLOWED_EVENTS = new Set([
  'reader_open',
  'word_click',
  'toggle_mode',
  'flashcard_save',
  'sample_view',
  'toggle_script',   // Traditional characters on/off
  'bubble_select',   // first tap of two-stage selection
  'audio_bubble',    // played a whole line
  'audio_word',      // played a single word
  'gloss_reveal',    // reader un-veiled a bubble's English translation
  'gloss_hide',      // reader re-veiled it
  'pref_translation', // sentence-translation mode changed (tap / always / off)
  'rec_click',        // recommended-series tile clicked on /start (props.id)
]);

export interface UsageEvent {
  ts: number;
  event: string;
  bid: string;
  props: Record<string, unknown>;
}

let writesSinceSizeCheck = 0;
let sizeCapHit = false;

/**
 * Validate and append one event. Fire-and-forget: invalid input is dropped
 * silently (beacons come from the public internet), disk errors are logged
 * once per process, and the caller never sees a failure.
 */
export function recordEvent(input: { event?: unknown; bid?: unknown; props?: unknown }): void {
  if (sizeCapHit) return;
  const event = typeof input.event === 'string' ? input.event : '';
  if (!ALLOWED_EVENTS.has(event)) return;

  const bid = (typeof input.bid === 'string' ? input.bid : 'anon').slice(0, 64);

  let props: Record<string, unknown> = {};
  if (input.props && typeof input.props === 'object' && !Array.isArray(input.props)) {
    for (const [k, v] of Object.entries(input.props as Record<string, unknown>)) {
      // reader_open alone now carries 8 (level, pages, script, vw, vh, ptr,
      // hover, dpr). At a cap of 8 it fit with nothing to spare, and the
      // overflow is silent and order-dependent — one more field and something
      // vanishes without a trace.
      if (Object.keys(props).length >= 12) break;
      if (typeof v === 'string') props[k.slice(0, 24)] = v.slice(0, 32);
      else if (typeof v === 'number' || typeof v === 'boolean' || v === null) props[k.slice(0, 24)] = v;
    }
  }

  const line = JSON.stringify({ ts: Date.now(), event, bid, props }) + '\n';

  writesSinceSizeCheck++;
  if (writesSinceSizeCheck >= SIZE_CHECK_EVERY) {
    writesSinceSizeCheck = 0;
    try {
      if (fs.existsSync(EVENTS_PATH) && fs.statSync(EVENTS_PATH).size > MAX_FILE_BYTES) {
        sizeCapHit = true;
        console.error('[Events] events.jsonl exceeded size cap — logging disabled until trimmed');
        return;
      }
    } catch { /* stat failure is not fatal */ }
  }

  fs.mkdirSync(path.dirname(EVENTS_PATH), { recursive: true });
  fs.appendFile(EVENTS_PATH, line, (err) => {
    if (err) console.error('[Events] append failed:', err.message);
  });
}

/**
 * Read events since a timestamp. Whole-file read is fine at this scale
 * (size-capped above); malformed lines are skipped.
 */
export function readEvents(sinceMs: number): UsageEvent[] {
  try {
    if (!fs.existsSync(EVENTS_PATH)) return [];
    const out: UsageEvent[] = [];
    for (const line of fs.readFileSync(EVENTS_PATH, 'utf-8').split('\n')) {
      if (!line) continue;
      try {
        const e = JSON.parse(line);
        if (e && typeof e.ts === 'number' && e.ts >= sinceMs) out.push(e);
      } catch { /* skip malformed line */ }
    }
    return out;
  } catch {
    return [];
  }
}

export interface ReaderStats {
  weeklyActiveReaders: number;   // distinct browsers with a reader_open in the last 7 days
  readerOpens7d: number;
  wordClicks7d: number;
  toggles7d: number;
  flashcardSaves7d: number;
  sampleViews7d: number;
  returningReaders28d: number;   // browsers with reader_opens in ≥2 distinct ISO weeks of the last 28 days
}

/** ISO week key like "2026-W29" — used to define a "returning" reader. */
export function isoWeekKey(ts: number): string {
  const d = new Date(ts);
  // ISO week: Thursday of the current week determines the year/week number.
  const target = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()));
  const day = target.getUTCDay() || 7;
  target.setUTCDate(target.getUTCDate() + 4 - day);
  const yearStart = new Date(Date.UTC(target.getUTCFullYear(), 0, 1));
  const week = Math.ceil(((target.getTime() - yearStart.getTime()) / 86400000 + 1) / 7);
  return `${target.getUTCFullYear()}-W${String(week).padStart(2, '0')}`;
}

export function getReaderStats(now: number = Date.now()): ReaderStats {
  const since28d = now - 28 * 24 * 60 * 60 * 1000;
  const since7d = now - 7 * 24 * 60 * 60 * 1000;
  const events = readEvents(since28d);

  const recent = events.filter(e => e.ts >= since7d);
  const count = (name: string) => recent.filter(e => e.event === name).length;

  const weeklyBids = new Set(
    recent.filter(e => e.event === 'reader_open' && e.bid !== 'anon').map(e => e.bid),
  );

  // Returning = same browser opened a reader in two different ISO weeks.
  const weeksByBid = new Map<string, Set<string>>();
  for (const e of events) {
    if (e.event !== 'reader_open' || e.bid === 'anon') continue;
    if (!weeksByBid.has(e.bid)) weeksByBid.set(e.bid, new Set());
    weeksByBid.get(e.bid)!.add(isoWeekKey(e.ts));
  }
  let returning = 0;
  for (const weeks of weeksByBid.values()) if (weeks.size >= 2) returning++;

  return {
    weeklyActiveReaders: weeklyBids.size,
    readerOpens7d: count('reader_open'),
    wordClicks7d: count('word_click'),
    toggles7d: count('toggle_mode'),
    flashcardSaves7d: count('flashcard_save'),
    sampleViews7d: count('sample_view'),
    returningReaders28d: returning,
  };
}
