// Daily LLM spend tracking + hard circuit breaker.
//
// Tracks total $ spent on paid LLM providers across all users in a day. When
// the daily cap is hit, refuses new LLM calls with BUDGET_REACHED until midnight.
//
// Persists state to disk so a server restart doesn't reset the counter (which
// would let an attacker DoS the cap by triggering crashes).

import * as fs from 'fs';
import * as path from 'path';

const SPEND_DATA_PATH = path.join(__dirname, '..', '..', 'data', 'spend-cap.json');
const DAILY_CAP_USD = parseFloat(process.env.LLM_DAILY_CAP_USD || '5');

// Per-1k-token costs in USD. Hardcoded, not fetched — UPDATE THESE if the model
// or a provider's pricing changes; a stale entry silently mis-counts real spend
// (this has happened twice: DeepSeek prices left behind after the GLM-5.1
// migration, and the Aug 2026 DeepSeek price rise recorded below).
//
// Rates are DATE-TIERED because the two consumers need different answers.
// recordSpend() banks dollars as calls happen, so it only ever wants today's
// rate. But /admin/stats re-derives per-chapter cost from token counts stored
// long after the fact — handing it today's rate would silently reprice history
// and distort the unit economics the dashboard exists to show.
//
// Each entry applies from `from` (epoch ms, inclusive) until the next entry's
// `from`; ordered oldest → newest. cachedInputPer1K falls back to inputPer1K for
// providers that don't report a cached-token split.
// Free tiers (Gemini Flash-Lite at low volume) excluded — recordSpend no-ops.

export interface TokenRate {
  inputPer1K: number;
  outputPer1K: number;
  cachedInputPer1K?: number;
}

// DeepSeek's V4 price rise took effect 16:00 UTC 2026-08-16, introducing
// peak/off-peak tiers. Fireworks mirrors DeepSeek's OFF-PEAK rate as its flat
// serverless price ($0.22/M uncached in, $0.007/M cached in, $0.66/M out), up
// from a flat $0.14 / $0.0028 / $0.28. Fireworks may have adopted it a day or
// two late; at our volume that gap is sub-cent, so we anchor to DeepSeek's date.
const DEEPSEEK_AUG_2026_RISE = Date.UTC(2026, 7, 16, 16, 0, 0);

const PROVIDER_RATES: Record<string, Array<TokenRate & { from: number }>> = {
  fireworks: [
    { from: 0,                      inputPer1K: 0.00014, cachedInputPer1K: 0.0000028, outputPer1K: 0.00028 },
    { from: DEEPSEEK_AUG_2026_RISE, inputPer1K: 0.00022, cachedInputPer1K: 0.000007,  outputPer1K: 0.00066 },
  ],
  // NOTE: the entries below predate the Aug 2026 DeepSeek repricing and have
  // not been re-verified. None are the active provider (SIMPLIFY_PROVIDER=
  // fireworks), so they're left alone rather than guessed at — re-check before
  // relying on any of them.
  deepseek: [{ from: 0, inputPer1K: 0.00027, outputPer1K: 0.0011 }],
  groq: [{ from: 0, inputPer1K: 0.00079, outputPer1K: 0.00079 }],
  openrouter: [{ from: 0, inputPer1K: 0.00027, outputPer1K: 0.0011 }],
  cerebras: [{ from: 0, inputPer1K: 0.00060, outputPer1K: 0.00120 }],
  // Free or unmetered (skipped):
  // gemini: { ... }  -- Flash-Lite is free at our volume
  // qwen: { ... }    -- self-hosted
  // anthropic: { ... } -- not currently used as primary; add if/when relevant
};

/**
 * The rate in effect for `provider` at `tsMs`. Undefined for free/unknown
 * providers. A timestamp older than the first tier falls back to that tier, so
 * a sample with a malformed ts still costs something rather than vanishing.
 */
export function pricingAt(provider: string, tsMs: number): TokenRate | undefined {
  const schedule = PROVIDER_RATES[provider];
  if (!schedule || schedule.length === 0) return undefined;
  let rate: TokenRate = schedule[0];
  for (const tier of schedule) {
    if (tier.from > tsMs) break;
    rate = tier;
  }
  return rate;
}

interface SpendData {
  date: string; // YYYY-MM-DD (UTC)
  spentUsd: number;
  // Lifetime LLM spend in USD. Increments with every recordSpend() call and
  // NEVER resets. Used by the profit dashboard. Pre-deploy spend (from test
  // runs prior to this field existing) is not backfilled — starts at 0 on
  // first deploy and grows from there.
  totalSpentAllTimeUsd?: number;
}

let cache: SpendData | null = null;

function todayUtc(): string {
  return new Date().toISOString().slice(0, 10);
}

function loadSpend(): SpendData {
  if (cache) return cache;
  try {
    if (fs.existsSync(SPEND_DATA_PATH)) {
      cache = JSON.parse(fs.readFileSync(SPEND_DATA_PATH, 'utf-8'));
      return cache!;
    }
  } catch (e) {
    console.error('[SpendCap] Failed to load spend data, starting fresh:', e);
  }
  cache = { date: todayUtc(), spentUsd: 0 };
  return cache;
}

function saveSpend(data: SpendData): void {
  try {
    fs.mkdirSync(path.dirname(SPEND_DATA_PATH), { recursive: true });
    fs.writeFileSync(SPEND_DATA_PATH, JSON.stringify(data, null, 2));
  } catch (e) {
    console.error('[SpendCap] Failed to persist spend data:', e);
  }
}

function rolloverIfNeeded(): SpendData {
  const data = loadSpend();
  const today = todayUtc();
  if (data.date !== today) {
    console.log(`[SpendCap] New day (${today}), resetting counter from $${data.spentUsd.toFixed(4)}`);
    data.date = today;
    data.spentUsd = 0;
    saveSpend(data);
  }
  return data;
}

/**
 * Returns true if it's safe to make a new LLM call. Pre-call gate.
 * Pessimistic estimate: if the next call could push us over, refuse.
 */
export function canSpend(estimateUsd: number = 0): boolean {
  const data = rolloverIfNeeded();
  return data.spentUsd + estimateUsd <= DAILY_CAP_USD;
}

/**
 * Records actual spend after an LLM call completes. Computes cost from the
 * provider's usage stats. No-op for unknown/free providers.
 */
export function recordSpend(
  provider: string,
  promptTokens: number,
  completionTokens: number,
  cachedTokens: number = 0,
): void {
  // Cost is banked as dollars at call time, so this always wants today's rate —
  // spend already recorded stays frozen at whatever rate was in effect then.
  const pricing = pricingAt(provider, Date.now());
  if (!pricing) return; // free/unknown provider, don't track

  // Split the prompt into cached vs fresh input — cached input is billed at a
  // much lower rate. Providers without a separate cached rate fall back to the
  // standard input rate.
  const uncachedTokens = Math.max(0, promptTokens - cachedTokens);
  const cachedRate = pricing.cachedInputPer1K ?? pricing.inputPer1K;
  const cost = (uncachedTokens / 1000) * pricing.inputPer1K
             + (cachedTokens / 1000) * cachedRate
             + (completionTokens / 1000) * pricing.outputPer1K;

  const data = rolloverIfNeeded();
  data.spentUsd += cost;
  data.totalSpentAllTimeUsd = (data.totalSpentAllTimeUsd || 0) + cost;
  saveSpend(data);

  if (data.spentUsd > DAILY_CAP_USD * 0.8) {
    console.warn(`[SpendCap] At ${(data.spentUsd / DAILY_CAP_USD * 100).toFixed(1)}% of daily cap ($${data.spentUsd.toFixed(4)} / $${DAILY_CAP_USD}). +$${cost.toFixed(4)} from ${provider} (${promptTokens}+${completionTokens} tokens).`);
  }
}

/**
 * Snapshot of current cap status. For /api/status or admin debugging.
 */
export function getSpendStatus(): { spentUsd: number; capUsd: number; date: string; remainingUsd: number } {
  const data = rolloverIfNeeded();
  return {
    spentUsd: data.spentUsd,
    capUsd: DAILY_CAP_USD,
    date: data.date,
    remainingUsd: Math.max(0, DAILY_CAP_USD - data.spentUsd),
  };
}

/**
 * Lifetime LLM spend. Used by the profit dashboard. Returns 0 for fresh
 * deploys (the field didn't exist before this change).
 */
export function getLifetimeSpendUsd(): number {
  const data = loadSpend();
  return data.totalSpentAllTimeUsd || 0;
}

/**
 * Thrown when a caller tries to make an LLM call but the daily cap is exhausted.
 * Phase 1.6 will formalize the typed-error pattern; for now the .code field is
 * the convention.
 */
export class BudgetReachedError extends Error {
  code = 'BUDGET_REACHED';
  constructor() {
    const status = getSpendStatus();
    super(`Daily LLM budget reached ($${status.spentUsd.toFixed(2)} / $${status.capUsd.toFixed(2)}). Resets at UTC midnight.`);
    this.name = 'BudgetReachedError';
  }
}
