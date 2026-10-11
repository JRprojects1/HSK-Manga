// Phase 2.2: Data-driven ETA.
//
// Records per-job timings to data/job-metrics.json, exposes percentile-based
// estimates so the frontend can replace its hardcoded "Usually under 2 minutes"
// with an actual median from past runs.
//
// Why median + Q1/Q3 (not mean ± std dev):
//   Processing-time distributions are right-skewed (rate limits, retry storms,
//   server hiccups push a long tail). Std dev squares deviations so one outlier
//   double the apparent variability; mean - 2σ can be negative on skewed data.
//   Median + IQR are rank-based, robust to outliers, and match the actual
//   shape of the distribution.
//
// Two buckets (cold vs cache-hit) because cache hits skip Phase 4b LLM and
// Phase 3 chapter analysis (~60s saved). Mixing them would give a bimodal
// distribution that doesn't predict either case well.
//
// Storage: data/job-metrics.json, append-only. File grows slowly (~250 bytes/job)
// and is filtered to ETA_LOOKBACK_DAYS at read time, so old samples don't drag
// the displayed median when pipeline performance changes (new model, faster
// server, etc.). Soft size cap exists only as runaway-protection.

import * as fs from 'fs';
import * as path from 'path';

const PROJECT_ROOT = path.join(__dirname, '..', '..');
const METRICS_PATH = path.join(PROJECT_ROOT, 'data', 'job-metrics.json');
// Soft cap: prevents unbounded growth from buggy/abuse cases. Real lifecycle
// is governed by ETA_LOOKBACK_DAYS in getEta(), not by this cap.
const MAX_SAMPLES = 10000;
const CURRENT_VERSION = 1;

/**
 * Per-phase timing snapshot. All values in milliseconds.
 * Populated from manga-pipeline's getTimingStats() at job completion.
 */
export interface PhaseTimings {
  loading: number;          // image load
  stitching: number;        // page stitching
  ocrScan: number;          // Phase 3 parallel OCR scan
  chapterAnalysis: number;  // Phase 3 chapter analysis (extra time beyond OCR)
  ocr: number;              // cumulative per-page OCR time
  llm: number;              // cumulative LLM time across all pages
  inpaintServer: number;    // cumulative LaMa time
  textRendering: number;    // cumulative metadata + render time
}

export interface JobSample {
  ts: number;            // unix ms when the job completed
  pages: number;         // chapter page count
  elapsedMs: number;     // total wall time from queue → complete
  cacheHit: boolean;     // whether the job used a cached entry (skipped Phase 4b LLM)
  phases?: PhaseTimings; // per-phase breakdown — recorded for transparency, not used in ETA
  // Cost-tracking fields (optional — samples written before this feature lack them).
  title?: string;          // chapter title, for the admin dashboard
  avgPageWidth?: number;   // mean page image width, pixels
  avgPageHeight?: number;  // mean page image height, pixels
  fireworks?: {            // Fireworks LLM token split for this chapter
    cachedInputTokens: number;
    inputTokens: number;
    outputTokens: number;
  };
}

interface MetricsFile {
  version: number;
  samples: JobSample[];
}

let cache: MetricsFile | null = null;

function loadFromDisk(): MetricsFile {
  if (cache) return cache;
  if (!fs.existsSync(METRICS_PATH)) {
    cache = { version: CURRENT_VERSION, samples: [] };
    return cache;
  }
  try {
    const parsed = JSON.parse(fs.readFileSync(METRICS_PATH, 'utf-8'));
    if (!parsed || typeof parsed !== 'object' || !Array.isArray(parsed.samples)) {
      cache = { version: CURRENT_VERSION, samples: [] };
      return cache;
    }
    cache = parsed as MetricsFile;
    return cache;
  } catch {
    cache = { version: CURRENT_VERSION, samples: [] };
    return cache;
  }
}

function saveToDisk(): void {
  if (!cache) return;
  fs.mkdirSync(path.dirname(METRICS_PATH), { recursive: true });
  // Soft size cap — prevents unbounded growth from runaway/buggy cases.
  // Real "freshness" is enforced by ETA_LOOKBACK_DAYS at read time, not here.
  if (cache.samples.length > MAX_SAMPLES) {
    cache.samples = cache.samples.slice(-MAX_SAMPLES);
  }
  fs.writeFileSync(METRICS_PATH, JSON.stringify(cache, null, 2));
}

/**
 * Record a successfully-completed job's timings.
 * Failures are NOT recorded (would skew the distribution toward fail-fast times).
 */
export function recordJobCompletion(sample: JobSample): void {
  const data = loadFromDisk();
  data.samples.push(sample);
  saveToDisk();
}

/**
 * Compute median, Q1 (p25), Q3 (p75) of a numeric array.
 * Returns null if the array is empty.
 */
function quartiles(values: number[]): { median: number; q1: number; q3: number } | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const at = (q: number) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
  return { median: at(0.5), q1: at(0.25), q3: at(0.75) };
}

export interface EtaBucket {
  median: number;  // milliseconds
  q1: number;
  q3: number;
  samples: number;
}

export interface EtaResult {
  cold: EtaBucket | null;       // typical cold-run time (no cache)
  cacheHit: EtaBucket | null;   // typical cache-hit time (skipped LLM + chapter analysis)
  fallback: string;             // shown when sample size is too small
}

// 30 samples gives a robust median (one outlier moves nothing). Below this,
// we'd rather show the fallback than a noisy estimate.
const MIN_SAMPLES_FOR_ETA = 30;

// Only consider samples from this many days back. Stale data from before a
// pipeline change (new model, faster server) shouldn't drag the displayed median.
// 90 days gives ~3 months for the new pipeline performance to dominate.
const ETA_LOOKBACK_DAYS = 90;

/**
 * Get current ETA estimates by cache bucket.
 *
 * Filters to samples from the last ETA_LOOKBACK_DAYS days, then groups by
 * cache state. Returns null for a bucket if it has fewer than
 * MIN_SAMPLES_FOR_ETA recent samples — the frontend falls back to the
 * static `fallback` string in that case.
 */
export function getEta(): EtaResult {
  const data = loadFromDisk();
  const cutoffMs = Date.now() - ETA_LOOKBACK_DAYS * 24 * 60 * 60 * 1000;
  const recent = data.samples.filter(s => s.ts >= cutoffMs);
  const cold = recent.filter(s => !s.cacheHit).map(s => s.elapsedMs);
  const hit = recent.filter(s => s.cacheHit).map(s => s.elapsedMs);

  function bucket(values: number[]): EtaBucket | null {
    if (values.length < MIN_SAMPLES_FOR_ETA) return null;
    const q = quartiles(values)!;
    return { median: q.median, q1: q.q1, q3: q.q3, samples: values.length };
  }

  return {
    cold: bucket(cold),
    cacheHit: bucket(hit),
    fallback: 'Usually under 2 minutes',
  };
}

/**
 * Read all samples (admin/debug use). The frontend should consume getEta()
 * which collapses to per-bucket summaries.
 */
export function getAllSamples(): JobSample[] {
  const data = loadFromDisk();
  return [...data.samples]; // defensive copy
}

/**
 * Test-only: reset the in-memory cache so a fresh module load happens. The
 * disk file is untouched. Used by test-safety-gates.js to test against a
 * controlled set of samples.
 */
export function _resetCache(): void {
  cache = null;
}
