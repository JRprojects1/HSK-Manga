import Anthropic from '@anthropic-ai/sdk';
import { GoogleGenerativeAI } from '@google/generative-ai';
import * as http from 'http';
import * as fs from 'fs';
import * as path from 'path';
import { HSKLevel, SimplificationResult, WordMapping } from './types';
import { getFlaggedWords } from './hsk-checker';
import { getDefinition } from './dictionary';
import { fixEnglishSpacing } from './english-splitter';
import { InputLanguage, isTranslationSource, sourceLanguageName, toBiasLanguage } from './language';
import { recordSpend } from './services/spendCap';
import { recordGeminiUsage } from './services/geminiUsageService';

// ---------------------------------------------------------------------------
// Per-chapter LLM token accumulator.
//
// callLLMMultiTurn is the single choke point every LLM call (simplification AND
// chapter analysis) flows through, so Fireworks token usage is summed here and
// read once at job completion by jobService via getLLMUsage(). Safe as
// module-level state because jobs run strictly one at a time (the global queue).
// ---------------------------------------------------------------------------
interface LLMUsage {
  fireworks: { cachedInputTokens: number; inputTokens: number; outputTokens: number };
}

function emptyLLMUsage(): LLMUsage {
  return { fireworks: { cachedInputTokens: 0, inputTokens: 0, outputTokens: 0 } };
}

let llmUsage: LLMUsage = emptyLLMUsage();

/** Reset the token accumulator — called by the pipeline at the start of each chapter. */
export function resetLLMUsage(): void {
  llmUsage = emptyLLMUsage();
}

/** Snapshot of Fireworks tokens consumed since the last reset. */
export function getLLMUsage(): LLMUsage {
  return { fireworks: { ...llmUsage.fireworks } };
}

/**
 * Smart join for block text segments.
 * Adds a space between segments when the previous ends with [A-Za-z]
 * and the next starts with [A-Za-z] (English word boundary).
 * Otherwise concatenates directly (correct for Chinese).
 */
/** Prefix for Qwen 3 to disable thinking mode. Other providers ignore or misinterpret it. */
function noThinkPrefix(): string {
  try {
    return detectProvider() === 'qwen' ? '/no_think\n' : '';
  } catch {
    return '';
  }
}

/** Normalize English punctuation to Chinese and strip LLM artifacts from output. */
function normalizePunctuation(text: string): string {
  return text
    .replace(/['''"]/g, '')           // Strip quotes LLM adds around KEEP terms
    .replace(/(?<!\.)\.(?!\.)(\s|$)/g, '。$1')   // Single period (not part of ellipsis) at end or before space → 。
    .replace(/,\s*/g, '，')          // Comma → ，
    .replace(/;\s*/g, '；');          // Semicolon → ；
}

/**
 * Returns true if `token` is a KEEP term, or a KEEP term with a trailing 的/们/啊/吗/吧/了 particle.
 * Segmentit sometimes glues these particles to high-frequency injected names (e.g. 卡督亚 + 的 → 卡督亚的).
 */
function isKeepOrSuffixed(token: string, keepSet: Set<string>): boolean {
  if (keepSet.has(token)) return true;
  const stripped = token.replace(/[的们啊吗吧了]$/, '');
  return stripped !== token && keepSet.has(stripped);
}

export function smartJoin(segments: string[]): string {
  if (segments.length === 0) return '';
  let result = segments[0];
  for (let i = 1; i < segments.length; i++) {
    const prev = result;
    const next = segments[i];
    if (prev.length > 0 && next.length > 0 &&
        /[A-Za-z]$/.test(prev) && /^[A-Za-z]/.test(next)) {
      result += ' ' + next;
    } else {
      result += next;
    }
  }
  return result;
}

// ─── English OCR fragment rejoin ─────────────────────────────────────────────
// English OCR frequently splits single words on whitespace ("STINK ING",
// "BAR BARI ANS", "EX IT ED", "ADVENTURE RS"). Downstream simplification
// then hallucinates around the fragments. Reassemble by looking up whether
// adjacent caps-only tokens' concatenation is a real English word.
//
// Only merges when at least one of the boundary tokens is NOT a common
// function word — protects normal bigrams like "OF THE", "TO GO", "NO WAY"
// which are also dictionary-valid concatenations ("ofthe" isn't in the
// dict, but "togo" and "noway" are; the stop-word check catches these).
let _alphaWordSet: Set<string> | null = null;
function getAlphaWordSet(): Set<string> {
  if (_alphaWordSet) return _alphaWordSet;
  const wordsPath = path.join(__dirname, '..', 'data', 'english-words-alpha.txt');
  try {
    _alphaWordSet = new Set(
      fs.readFileSync(wordsPath, 'utf-8')
        .split(/\r?\n/)
        .map(s => s.trim().toLowerCase())
        .filter(Boolean)
    );
  } catch {
    _alphaWordSet = new Set();  // Missing dict → no-op rejoin (safe default)
  }
  return _alphaWordSet;
}

const REJOIN_STOP_WORDS = new Set([
  'a', 'an', 'the',
  'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'you', 'your',
  'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their',
  'of', 'in', 'on', 'at', 'to', 'by', 'as', 'for', 'from', 'with',
  'into', 'onto', 'off', 'over', 'under',
  'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
  'do', 'did', 'does', 'have', 'has', 'had', 'get', 'got', 'go', 'went',
  'no', 'yes', 'not', 'and', 'or', 'if', 'so', 'but',
  'all', 'any', 'one', 'two', 'three', 'some', 'many', 'more', 'most',
  'here', 'there', 'now', 'then', 'way', 'day', 'time', 'man', 'men',
  'who', 'what', 'when', 'why', 'where', 'how', 'this', 'that', 'these', 'those',
]);

const REJOIN_ALPHA_TOK_RE = /^[A-Z][A-Z']*[.,!?;:'"]?$/;
const REJOIN_TRAIL_PUNCT_RE = /[.,!?;:'"]+$/;

/**
 * Reassemble OCR-split English words. Applied to raw region text before
 * simplification. Idempotent — safe to call on already-clean text.
 */
export function fragmentRejoin(text: string): string {
  const alphaSet = getAlphaWordSet();
  if (alphaSet.size === 0) return text;  // No dict loaded → skip
  const parts = text.split(/(\s+)/);
  const out: string[] = [];
  let i = 0;
  while (i < parts.length) {
    const tok = parts[i];
    if (!REJOIN_ALPHA_TOK_RE.test(tok)) { out.push(tok); i++; continue; }
    const tokCore = tok.replace(REJOIN_TRAIL_PUNCT_RE, '');
    if (tokCore.length === 1) { out.push(tok); i++; continue; }

    let bestEnd = i;
    let bestMerge = tok;
    let acc = tokCore;
    let lastPart = tokCore;
    for (let j = i + 2; j < parts.length && (j - i) / 2 <= 5; j += 2) {
      const nxt = parts[j];
      if (!REJOIN_ALPHA_TOK_RE.test(nxt)) break;
      const nxtCore = nxt.replace(REJOIN_TRAIL_PUNCT_RE, '');
      if (nxtCore.length === 1) break;
      acc = acc + nxtCore;
      const bothStop = REJOIN_STOP_WORDS.has(lastPart.toLowerCase()) && REJOIN_STOP_WORDS.has(nxtCore.toLowerCase());
      lastPart = nxtCore;
      if (bothStop) break;
      if (!alphaSet.has(acc.toLowerCase())) continue;
      bestEnd = j;
      const trail = nxt.match(REJOIN_TRAIL_PUNCT_RE)?.[0] || '';
      bestMerge = acc + trail;
    }
    out.push(bestMerge);
    if (bestEnd > i) i = bestEnd + 1;
    else i++;
  }
  return out.join('');
}


type Provider = 'anthropic' | 'gemini' | 'qwen' | 'openrouter' | 'groq' | 'deepseek' | 'cerebras' | 'fireworks';

// LLM provider configuration
const QWEN_API_URL = process.env.QWEN_API_URL || 'http://localhost:1234';
const QWEN_MODEL = process.env.QWEN_MODEL || 'qwen2.5-7b';

// Provider configs: base URL, API key env var, default model, auth header format
const PROVIDER_CONFIGS: Record<string, { baseUrl: string; apiKey: string; defaultModel: string; useHttps: boolean }> = {
  qwen: {
    baseUrl: QWEN_API_URL,
    apiKey: '',  // Local Qwen doesn't need auth
    defaultModel: QWEN_MODEL,
    useHttps: false,
  },
  openrouter: {
    baseUrl: 'https://openrouter.ai/api',
    apiKey: process.env.OPENROUTER_API_KEY || '',
    defaultModel: process.env.SIMPLIFY_MODEL || 'qwen/qwen3-coder:free',
    useHttps: true,
  },
  groq: {
    baseUrl: 'https://api.groq.com/openai',
    apiKey: process.env.GROQ_API_KEY || '',
    defaultModel: process.env.SIMPLIFY_MODEL || 'qwen/qwen3-32b',
    useHttps: true,
  },
  deepseek: {
    baseUrl: 'https://api.deepseek.com',
    apiKey: process.env.DEEPSEEK_API_KEY || '',
    defaultModel: process.env.SIMPLIFY_MODEL || 'deepseek-chat',
    useHttps: true,
  },
  gemini: {
    baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai',
    apiKey: (process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY || '').split(',')[0],
    defaultModel: process.env.SIMPLIFY_MODEL || 'gemini-2.5-flash',
    useHttps: true,
  },
  cerebras: {
    baseUrl: 'https://api.cerebras.ai',
    apiKey: process.env.CEREBRAS_API_KEY || '',
    defaultModel: process.env.SIMPLIFY_MODEL || 'qwen-3-235b-a22b-instruct-2507',
    useHttps: true,
  },
  fireworks: {
    baseUrl: 'https://api.fireworks.ai/inference',
    apiKey: process.env.FIREWORKS_API_KEY || '',
    defaultModel: process.env.SIMPLIFY_MODEL || 'accounts/fireworks/models/glm-5p1',
    useHttps: true,
  },
};

// Lazy-loaded logit_bias maps for Fireworks soft-constraint simplification.
// One map per HSK level: data/hsk-token-ids-1.json … hsk-token-ids-1to4.json.
// At simplification time we look up the right map by target level so HSK 1
// users get a tighter vocabulary push than HSK 4 users.
//
// Per-level bias overrides via env:
//   BIAS         (default 2.5) — global default
//   HSK1_BIAS    (optional)    — per-level override
//   HSK2_BIAS    (optional)
//   HSK3_BIAS    (optional)
//   HSK4_BIAS    (optional)
//
// Why per-level overrides exist:
//   HSK 1 has only ~250 chars / 4.6k tokens. Pushing the model into that vocab
//   too aggressively produces robotic output. Lowering HSK1_BIAS gives the
//   model more flexibility to use slightly above-level words when forced.
// Cache keyed by `${level}:${inputLanguage}` — English-input and Chinese-input
// use different default bias strengths, so the same level needs two entries.
const LOGIT_BIAS_CACHE = new Map<string, Record<number, number> | null>();

// Bias maps exist in exactly two flavors (Chinese-input vs translation-input);
// japanese/korean normalize onto the english side via toBiasLanguage at the
// entry points below, so cache keys and map selection collapse to cn/en.
type BiasInputLanguage = InputLanguage;

// Per-language positive bias magnitude for the unified bias map. The map file
// encodes WHICH tokens are positive vs negative (the structure); the actual
// MAGNITUDE is applied here at load time so we can tune bias strength without
// rebuilding the bias map files.
//
// Defaults:
//   CN→CN (chinese input): +2.5. Source is already Chinese, so the simplifier
//     only needs gentle pressure to land at HSK 1-3 vocab. Higher values risk
//     over-paraphrasing or repetition collapse.
//   EN→CN (english input): +3. Translation+simplification (single-pass) needs
//     stronger pull to land translations at HSK 1-3 vocab rather than the
//     model's natural higher-register Chinese. Verified historically via the
//     legacy biasForLevel function which used +3.5 for EN→CN HSK 3.
//
// Override via env: POS_BIAS_CN, POS_BIAS_EN.
const POS_BIAS_CN = parseFloat(process.env.POS_BIAS_CN || '2.5');
const POS_BIAS_EN = parseFloat(process.env.POS_BIAS_EN || '3');

// Token IDs are tokenizer-specific — a bias map built for one model's
// tokenizer is meaningless on another's. Pick the file suffix matching the
// configured Fireworks model so logit_bias lands on the right token IDs.
//   glm-*  → data/hsk-token-ids-1toN.glm.json   (GLM tokenizer)
//   kimi-* → .kimi.json,  qwen-* → .qwen.json
//   deepseek / anything else → no suffix (legacy DeepSeek-V3 tokenizer)
function tokenFileSuffix(): string {
  const model = (process.env.SIMPLIFY_MODEL || '').toLowerCase();
  if (model.includes('glm')) return '.glm';
  if (model.includes('kimi')) return '.kimi';
  if (model.includes('qwen')) return '.qwen';
  if (model.includes('deepseek-v4')) return '.deepseek-v4';
  return '';
}

function biasFilePath(level: HSKLevel): string {
  // Cumulative range — HSK 3 means 1+2+3, etc.
  const range = level === 'hsk1' ? '1'
    : level === 'hsk2' ? '1to2'
    : level === 'hsk3' ? '1to3'
    : level === 'hsk4' ? '1to4'
    : level === 'hsk5' ? '1to5'
    : '1to6';
  return path.join(process.cwd(), 'data', `hsk-token-ids-${range}${tokenFileSuffix()}.json`);
}

function biasForLevel(level: HSKLevel, inputLanguage: BiasInputLanguage = 'chinese'): number {
  inputLanguage = toBiasLanguage(inputLanguage); // ja/ko ride the english-side bias config
  const envKey = level === 'hsk1' ? 'HSK1_BIAS'
    : level === 'hsk2' ? 'HSK2_BIAS'
    : level === 'hsk3' ? 'HSK3_BIAS'
    : level === 'hsk4' ? 'HSK4_BIAS'
    : level === 'hsk5' ? 'HSK5_BIAS'
    : 'HSK6_BIAS';
  const override = process.env[envKey];
  if (override !== undefined) return parseFloat(override);
  if (process.env.BIAS !== undefined) return parseFloat(process.env.BIAS);
  // Both EN→CN and CN→CN default to +2.5. An EN bias sweep at +2.5 / +3.5 /
  // unified pos+neg (2026-06-30, Surviving the Game as a Barbarian ep 1)
  // showed +2.5 produces more natural HSK-3 Chinese; higher magnitudes gain
  // marginal HSK-3 strictness at the cost of clunky periphrases (敏捷→快,
  // 沉浸→喜欢) where the register was appropriate.
  return 2.5;
}

function unifiedBiasFilePath(level: HSKLevel): string {
  const range = level === 'hsk1' ? '1' : level === 'hsk2' ? '1to2' : level === 'hsk3' ? '1to3'
    : level === 'hsk4' ? '1to4' : level === 'hsk5' ? '1to5' : '1to6';
  return path.join(process.cwd(), 'data', `hsk-token-bias-${range}${tokenFileSuffix()}.json`);
}

// ------------ KEEP-term bias override (production fix) ------------
//
// The unified bias map assigns -2.5 to every token whose decoded text contains
// an HSK 4+ word. KEEP terms above HSK 3 (e.g. `猎人` HSK 7) get suppressed by
// this even though the system prompt says "do NOT simplify these." The model
// then probabilistically loses the bias-vs-prompt fight and produces paraphrases.
//
// Fix: for each KEEP term, find every bias-map token whose decoded text appears
// as a substring of the term (these are the tokens the model uses to generate
// the term). Zero the negative bias on those token IDs so they're no longer
// actively suppressed. Positive biases are left alone — we never lift HSK 1-3
// tokens by accident.
//
// Decoding uses the GPT-2 byte-to-unicode scheme that the DeepSeek-V4 tokenizer
// inherits. Cached after first build; ~22k tokens decoded once per process.

let DECODED_TOKEN_CACHE: Array<{ id: number; decoded: string }> | null = null;

function buildGpt2ByteToUnicode(): Map<string, number> {
  const bs: number[] = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  }
  const charToByte = new Map<string, number>();
  for (let i = 0; i < bs.length; i++) {
    charToByte.set(String.fromCodePoint(cs[i]), bs[i]);
  }
  return charToByte;
}

function loadDecodedTokens(): Array<{ id: number; decoded: string }> {
  if (DECODED_TOKEN_CACHE) return DECODED_TOKEN_CACHE;
  const tokFile = path.join(process.cwd(), 'data', `dsv4-tokenizer.json`);
  if (!fs.existsSync(tokFile)) {
    console.error(`[Simplifier] ${tokFile} missing — KEEP-token override disabled`);
    DECODED_TOKEN_CACHE = [];
    return DECODED_TOKEN_CACHE;
  }
  const t = JSON.parse(fs.readFileSync(tokFile, 'utf-8'));
  const vocab = t.model?.vocab || {};
  const charToByte = buildGpt2ByteToUnicode();
  const out: Array<{ id: number; decoded: string }> = [];
  for (const [tokenStr, idAny] of Object.entries(vocab)) {
    const id = +(idAny as number);
    const bytes: number[] = [];
    let ok = true;
    for (const ch of tokenStr) {
      const b = charToByte.get(ch);
      if (b === undefined) { ok = false; break; }
      bytes.push(b);
    }
    if (!ok) continue;
    try {
      const decoded = Buffer.from(bytes).toString('utf-8');
      if (decoded.length > 0) out.push({ id, decoded });
    } catch { /* skip un-decodable */ }
  }
  DECODED_TOKEN_CACHE = out;
  console.error(`[Simplifier] Decoded ${out.length} tokens from dsv4-tokenizer.json (cached)`);
  return out;
}

/**
 * Find every token ID that (a) decodes to text appearing as a substring of one
 * of the KEEP terms AND (b) has a negative bias in the bias map. These are the
 * IDs whose negative bias should be zeroed so the model can freely generate
 * KEEP terms.
 *
 * Exported for the manga-pipeline post-analysis hook + for testing.
 */
export function getKeepNegBiasIds(keep: string[], level: HSKLevel = 'hsk3', inputLanguage: BiasInputLanguage = 'chinese'): number[] {
  if (!keep || keep.length === 0) return [];
  const bias = getLogitBias(level, inputLanguage);
  if (!bias) return [];
  const tokens = loadDecodedTokens();
  if (tokens.length === 0) return [];
  const out: number[] = [];
  for (const { id, decoded } of tokens) {
    if (bias[id] === undefined || bias[id] >= 0) continue;
    // Trim a leading space (BPE tokens often start with " ") for the substring test
    const probe = decoded.startsWith(' ') ? decoded.slice(1) : decoded;
    if (probe.length === 0) continue;
    for (const term of keep) {
      if (term.includes(probe) || probe.includes(term)) {
        out.push(id);
        break;
      }
    }
  }
  return out;
}

function getLogitBias(level: HSKLevel = 'hsk3', inputLanguage: BiasInputLanguage = 'chinese'): Record<number, number> | null {
  inputLanguage = toBiasLanguage(inputLanguage); // ja/ko ride the english-side bias config (and share its cache key)
  // A/B test escape hatch: completely disable logit bias when this env is set.
  if (process.env.DISABLE_LOGIT_BIAS === '1') {
    console.error(`[Simplifier] DISABLE_LOGIT_BIAS=1: logit bias disabled for this run`);
    return null;
  }
  const cacheKey = `${level}:${inputLanguage}`;
  if (LOGIT_BIAS_CACHE.has(cacheKey)) {
    return LOGIT_BIAS_CACHE.get(cacheKey) || null;
  }

  // Bias-map default policy:
  //   - CN→CN: unified pos+neg map by default. Combined with the noun-first
  //     chapter-analysis KEEP list and getKeepNegBiasIds' negative-bias-zero
  //     override on KEEP-term tokens, unified pushes V4-Flash harder away
  //     from HSK 4+ verbs/adjectives while leaving KEEP nouns unaffected.
  //     T7 evaluation (2026-07-01) beat V3.1's strict simplification floor
  //     (3.41% vs 3.64%) using this config.
  //   - EN→CN: posonly (legacy). The negative half of the unified map
  //     suppresses acronyms and register-carrying vocab that EN input
  //     needs to translate cleanly. Kept as-is per the earlier EN bias sweep.
  //   - FORCE_UNIFIED_BIAS=1 / FORCE_POSONLY_BIAS=1 override the default.
  const forceUnified = process.env.FORCE_UNIFIED_BIAS === '1';
  const forcePosonly = process.env.FORCE_POSONLY_BIAS === '1';
  const cnUnifiedDefault = inputLanguage === 'chinese' && !forcePosonly;
  const useUnified = forceUnified || cnUnifiedDefault;
  const unifiedPath = useUnified ? unifiedBiasFilePath(level) : '';
  if (useUnified && fs.existsSync(unifiedPath)) {
    const parsed = JSON.parse(fs.readFileSync(unifiedPath, 'utf-8'));
    if (parsed.bias_map && typeof parsed.bias_map === 'object') {
      // Per-token value encodes SIGN (positive vs negative). The MAGNITUDE of
      // the positive side is set per-language at load time — see the
      // POS_BIAS_CN / POS_BIAS_EN constants.
      const posBias = inputLanguage === 'english' ? POS_BIAS_EN : POS_BIAS_CN;
      const map: Record<number, number> = {};
      let pos = 0, neg = 0;
      for (const [id, bias] of Object.entries(parsed.bias_map)) {
        const v = bias as number;
        if (v > 0) {
          if (posBias !== 0) { map[+id] = posBias; pos++; }
        } else if (v < 0) {
          map[+id] = v; neg++;
        }
      }
      LOGIT_BIAS_CACHE.set(cacheKey, map);
      console.error(`[Simplifier] Loaded ${level} unified bias map (${inputLanguage}): ${pos} positive @ +${posBias} + ${neg} negative entries`);
      return map;
    }
  }
  let tokenPath = biasFilePath(level);
  // Backwards-compat: if the per-level file is missing, fall back to the
  // legacy HSK 1-3 file (data/hsk-token-ids.json) — but ONLY for the legacy
  // DeepSeek tokenizer. For a model-specific tokenizer (GLM/Kimi/Qwen) the
  // legacy file holds the WRONG token IDs, so we disable bias instead of
  // silently steering on garbage tokens.
  if (!fs.existsSync(tokenPath)) {
    const legacyPath = path.join(process.cwd(), 'data', 'hsk-token-ids.json');
    if (tokenFileSuffix() === '' && fs.existsSync(legacyPath)) {
      console.error(`[Simplifier] ${path.basename(tokenPath)} missing — falling back to legacy HSK 1-3 file`);
      tokenPath = legacyPath;
    } else {
      console.error(`[Simplifier] ${path.basename(tokenPath)} missing — Fireworks logit_bias disabled for ${level}`);
      LOGIT_BIAS_CACHE.set(cacheKey, null);
      return null;
    }
  }
  const parsed = JSON.parse(fs.readFileSync(tokenPath, 'utf-8'));
  const ids: number[] = parsed.all_unique_token_ids || [];
  const bias = biasForLevel(level, inputLanguage);
  const map: Record<number, number> = {};
  for (const id of ids) map[id] = bias;
  LOGIT_BIAS_CACHE.set(cacheKey, map);
  console.error(`[Simplifier] Loaded ${level} logit_bias (${inputLanguage}): ${ids.length} tokens at +${bias}`);
  return map;
}

const SYSTEM_PROMPT = `You are a Chinese language simplification assistant. Your task is to rewrite Chinese text so that it uses only beginner-level vocabulary and grammar.

Rules:
1. Output simplified Chinese only - no English, no pinyin, no explanations
2. Preserve the original meaning as closely as possible
3. Replace advanced vocabulary with beginner equivalents (HSK 1-2 level words)
4. Split long sentences into shorter ones (max 15-20 characters per sentence)
5. Remove idioms (成语) and replace with plain language
6. Keep proper nouns (names, places, brands) unchanged
7. Use simple grammar: basic SVO structure, avoid 把/被 constructions, avoid complex complements
8. If a word absolutely cannot be simplified without losing critical meaning, keep it and mark with [[word]]

Common simplifications to use:
- 购买 → 买
- 非常/特别/极其 → 很
- 迅速/快速 → 快
- 美丽/漂亮/好看 → 好看
- 因此/所以/故 → 所以
- 然而/但是/不过 → 但是
- 获得/取得 → 得到/有
- 使用/运用/利用 → 用
- 能够/可以 → 能/可以

You MUST respond with valid JSON in this exact format:
{
  "simplified": "the simplified Chinese text here",
  "mappings": [
    {"original": "advanced word", "simplified": "simple word", "reason": "vocabulary"},
    {"original": "成语", "simplified": "plain explanation", "reason": "idiom"}
  ],
  "preserved": ["word1", "word2"]
}

The "reason" field should be one of: "vocabulary", "idiom", "grammar", "preserved"
The "preserved" array should contain any words marked with [[word]] that couldn't be simplified.`;

// Simpler prompt for Qwen - it works better with concise instructions
const QWEN_SYSTEM_PROMPT = `你是中文简化工具。把复杂的中文句子改成简单的句子。

规则：
1. 只输出简化后的中文，不要英文、拼音或解释
2. 用简单词汇代替难词
3. 长句子分成短句子
4. 去掉成语，用简单的话说
5. 保持原意

用JSON格式回复：
{
  "simplified": "简化后的文字",
  "mappings": [{"original": "难词", "simplified": "简单词", "reason": "vocabulary"}],
  "preserved": []
}`;

// Full vocabulary batch prompt - each item must be translated independently

function getUserPrompt(text: string, level: HSKLevel, flaggedWords: string[]): string {
  const levelNum = level === 'hsk1' ? 1 : level === 'hsk2' ? 2 : level === 'hsk3' ? 3
    : level === 'hsk4' ? 4 : level === 'hsk5' ? 5 : 6;
  const flaggedList = flaggedWords.length > 0 ? flaggedWords.join(', ') : '(none detected)';

  return `请将以下中文文本简化到HSK${levelNum}水平。

原文：
${text}

已标记的高级词汇: ${flaggedList}

请用JSON格式回复。`;
}

function getQwenUserPrompt(text: string, level: HSKLevel): string {
  const levelNum = level === 'hsk1' ? 1 : level === 'hsk2' ? 2 : level === 'hsk3' ? 3
    : level === 'hsk4' ? 4 : level === 'hsk5' ? 5 : 6;
  return `简化到HSK${levelNum}水平：${text}`;
}

interface LLMResponse {
  simplified: string;
  mappings: Array<{
    original: string;
    simplified: string;
    reason: string;
  }>;
  preserved: string[];
}

export function detectProvider(): Provider {
  // Check for explicit provider override
  const explicitProvider = process.env.SIMPLIFY_PROVIDER?.toLowerCase();
  if (explicitProvider && ['qwen', 'gemini', 'anthropic', 'openrouter', 'groq', 'deepseek', 'cerebras', 'fireworks'].includes(explicitProvider)) {
    return explicitProvider as Provider;
  }

  // Auto-detect based on available keys (cloud providers first, local Qwen last)
  if (process.env.FIREWORKS_API_KEY) return 'fireworks';
  if (process.env.CEREBRAS_API_KEY) return 'cerebras';
  if (process.env.OPENROUTER_API_KEY) return 'openrouter';
  if (process.env.GROQ_API_KEY) return 'groq';
  if (process.env.DEEPSEEK_API_KEY) return 'deepseek';
  if (process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY) return 'gemini';
  if (process.env.ANTHROPIC_API_KEY) return 'anthropic';
  if (process.env.QWEN_API_URL) return 'qwen';

  throw new Error(
    'No LLM provider found. Set one of:\n' +
    '  OPENROUTER_API_KEY (free models available)\n' +
    '  GROQ_API_KEY (free tier)\n' +
    '  DEEPSEEK_API_KEY (free tier)\n' +
    '  GEMINI_API_KEY (free tier)\n' +
    '  QWEN_API_URL (local model)\n' +
    'Get a free OpenRouter key at: https://openrouter.ai/keys'
  );
}

async function callAnthropic(userPrompt: string): Promise<string> {
  const client = new Anthropic();
  const message = await client.messages.create({
    model: 'claude-sonnet-4-20250514',
    max_tokens: 2048,
    messages: [{ role: 'user', content: userPrompt }],
    system: SYSTEM_PROMPT,
  });
  return message.content[0].type === 'text' ? message.content[0].text : '';
}

async function callGemini(userPrompt: string): Promise<string> {
  const apiKey = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY not found');
  }

  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });

  const fullPrompt = `${SYSTEM_PROMPT}\n\n${userPrompt}`;
  const result = await model.generateContent(fullPrompt);
  return result.response.text();
}

// Keep-alive agent for connection reuse
const keepAliveAgent = new http.Agent({
  keepAlive: true,
  keepAliveMsecs: 30000,
  maxSockets: 10,
});




export async function simplifyText(
  text: string,
  level: HSKLevel = 'hsk2'
): Promise<SimplificationResult> {
  const provider = detectProvider();
  console.error(`Using ${provider} for simplification...`);

  let responseText: string;

  if (provider === 'qwen') {
    const userPrompt = getQwenUserPrompt(text, level);
    responseText = await callQwenChat(QWEN_SYSTEM_PROMPT, userPrompt);
  } else if (provider === 'anthropic') {
    const flaggedWords = getFlaggedWords(text, level);
    const userPrompt = getUserPrompt(text, level, flaggedWords);
    responseText = await callAnthropic(userPrompt);
  } else {
    const flaggedWords = getFlaggedWords(text, level);
    const userPrompt = getUserPrompt(text, level, flaggedWords);
    responseText = await callGemini(userPrompt);
  }

  // Parse JSON response
  let parsed: LLMResponse;
  try {
    const jsonMatch = responseText.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      // Qwen might return plain text if JSON parsing fails
      // Fall back to treating the response as simplified text
      if (provider === 'qwen' && responseText.trim()) {
        console.error('Qwen returned plain text, using as simplified output');
        return {
          original: text,
          simplified: responseText.trim(),
          mappings: [],
          preservedAdvanced: [],
          confidence: 0.8,
        };
      }
      throw new Error('No JSON found in response');
    }
    parsed = JSON.parse(jsonMatch[0]);
  } catch (e) {
    console.error('Failed to parse LLM response:', responseText);
    throw new Error('Failed to parse simplification response');
  }

  const mappings: WordMapping[] = (parsed.mappings || []).map((m) => ({
    original: m.original,
    simplified: m.simplified,
    reason: m.reason as WordMapping['reason'],
  }));

  const preservedAdvanced = (parsed.preserved || []).map((w) =>
    w.replace(/^\[\[|\]\]$/g, '')
  );

  const cleanedSimplified = parsed.simplified.replace(/\[\[([^\]]+)\]\]/g, '$1');

  return {
    original: text,
    simplified: cleanedSimplified,
    mappings,
    preservedAdvanced,
    confidence: 0.9,
  };
}

export async function simplifyTextBatch(
  texts: string[],
  level: HSKLevel = 'hsk2'
): Promise<SimplificationResult[]> {
  const results: SimplificationResult[] = [];
  for (const text of texts) {
    const result = await simplifyText(text, level);
    results.push(result);
  }
  return results;
}

/**
 * Text block representing grouped text regions that form a speech bubble
 */
export interface TextBlock {
  regionIndices: number[];  // Indices into the original regions array
  texts: string[];          // Text from each region in the block
}

/**
 * Chapter-level context gathered by multi-turn analysis (Approach 4)
 */
export interface ChapterContext {
  setting: string;
  characters: { name: string; role: string }[];
  plot: { pages: string; summary: string }[];
  keep: string[];  // Protected terms (after cleanup)
  glosses?: Record<string, string>;  // Raw glosses preserved for dump/round-trip (definitions Map is rebuilt from this)
  definitions: Map<string, string>;  // term → contextual definition (from characters + KEEP)
  keepTokenIds?: number[];  // Cached: token IDs whose negative logit-bias should be zeroed (computed via getKeepNegBiasIds after analysis)
  // EN input only: canonical English concept → Chinese mapping. Used by the
  // EN two-stage pipeline to tell Stage 1 what Chinese term to output for each
  // English concept in the source. Absent for CN input. Extracted from the
  // FINAL turn's `concepts` array + characters' name/chineseName pairs.
  concepts?: { english: string; chinese: string }[];
}

/**
 * Build a definitions map from chapter analysis results.
 * Characters get role-based definitions; KEEP terms get a CC-CEDICT lookup,
 * falling back to the chapter-analysis gloss for invented terms (skill names,
 * made-up words) that no dictionary contains — this is the Bug B fix: without
 * a gloss, such terms render an empty popup in the reader.
 */
export function buildChapterDefinitions(
  characters: { name: string; chineseName?: string; role: string }[],
  keep: string[],
  glosses?: Record<string, string>
): Map<string, string> {
  const defs = new Map<string, string>();

  // Character names → role descriptions.
  //
  // This map is looked up with Chinese tokens, so chineseName is the key that
  // actually matters. Keying only on `name` (the English name, per the analysis
  // prompt) meant character roles never matched anything, and any Chinese name
  // that also exists in CC-CEDICT fell through to the dictionary instead —
  // 克拉克 displayed "(name) Clark or Clarke" while its own bubble gloss said
  // "Klak". The English key is kept for chapters whose analysis puts the Chinese
  // surface form in `name`.
  for (const char of characters) {
    if (!char.role) continue;
    if (char.chineseName) defs.set(char.chineseName, `character: ${char.role}`);
    if (char.name) defs.set(char.name, `character: ${char.role}`);
  }

  // KEEP terms → CC-CEDICT lookup, then gloss fallback (if not already a character)
  for (const term of keep) {
    if (!defs.has(term)) {
      const dictDef = getDefinition(term);
      if (dictDef) {
        defs.set(term, dictDef);
      } else if (glosses && glosses[term]) {
        // Invented term with no dictionary entry — use the LLM's gloss.
        defs.set(term, glosses[term]);
      }
    }
  }

  return defs;
}

function rateLimitedCall<T>(fn: () => Promise<T>): Promise<T> {
  return fn();
}

/**
 * Generic helper to call Qwen chat API with a single system+user message
 */
async function callQwenChat(systemPrompt: string, userPrompt: string, temperature = 0.1, level: HSKLevel = 'hsk3', inputLanguage: BiasInputLanguage = 'chinese', keepTokenIds?: number[], biasSourceLanguage?: BiasInputLanguage): Promise<string> {
  return rateLimitedCall(() => callLLMMultiTurn(
    [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt },
    ],
    temperature,
    240000, // 4 min — a large chapter batch can run ~90s+ even with reasoning off
    undefined,
    undefined,
    level,
    inputLanguage,
    true,            // applyLogitBias
    keepTokenIds,    // per-call KEEP-token override
    biasSourceLanguage, // override: original source language for bias magnitude
  ));
}

/**
 * Dedicated entry point for chapter analysis LLM calls.
 *
 * Chapter analysis must NEVER have logit_bias applied — the analysis output is
 * structured JSON describing the chapter (KEEP terms, character names, plot
 * summaries, glosses) and a positive-on-HSK1-3 / negative-on-HSK4+ bias
 * actively distorts it: KEEP lists drop technical/character vocabulary,
 * plot summaries get reworded into simpler-but-less-accurate Chinese, glosses
 * lose precision. The simplifier downstream then operates on degraded context.
 *
 * All chapter analysis turns (both the initial `analyzeChapter` and the
 * streaming variant + its FINAL turn) MUST route through here so the
 * unbiased guarantee is enforced in one place.
 */
async function callAnalysisLLM(messages: Array<{ role: string; content: string }>): Promise<string> {
  return callLLMMultiTurn(
    messages,
    0.3,
    600000,
    process.env.ANALYSIS_MODEL,
    process.env.ANALYSIS_PROVIDER as Provider | undefined,
    'hsk3',     // level — ignored when applyLogitBias=false
    'chinese',  // inputLanguage — ignored when applyLogitBias=false
    false,      // applyLogitBias: chapter analysis must be unconstrained
  );
}

/**
 * Post-analysis English normalization for `role` / `glosses` values.
 *
 * The multi-turn CN analysis reliably drifts into Chinese for these fields — by
 * the FINAL turn the conversation is saturated with the Chinese chapter text, so
 * the "in English" instruction (and even a negative example) can't overcome it
 * (verified against the real pipeline: 10/10 roles came back Chinese). The fix is
 * an ISOLATED translation call — a fresh conversation with NO chapter context to
 * re-prime Chinese — which converts them reliably (verified 30/30 across runs).
 *
 * Runs only when Chinese is actually present, so English-input chapters (whose
 * fields are already English) make no extra call. Best-effort: any failure
 * (network, parse, wrong length, still-Chinese output) leaves the original value
 * in place — no worse than before.
 */
async function normalizeAnalysisToEnglish(
  characters: any[],
  glosses: Record<string, string>,
): Promise<{ characters: any[]; glosses: Record<string, string> }> {
  const hasHan = (s: string) => /[一-鿿]/.test(s || '');
  const jobs: { kind: 'role' | 'gloss'; key: number | string; text: string }[] = [];
  characters.forEach((c, i) => { if (c && hasHan(c.role)) jobs.push({ kind: 'role', key: i, text: c.role }); });
  for (const [k, v] of Object.entries(glosses)) if (hasHan(v)) jobs.push({ kind: 'gloss', key: k, text: v });
  if (jobs.length === 0) return { characters, glosses };

  const sys = 'You translate short Chinese descriptions into concise English for language learners. Keep proper names as-is or romanize them. Return ONLY a JSON array of the English strings, in the same order and same length as the input.';
  const user = 'Translate each to English:\n' + jobs.map((j, i) => `${i + 1}. ${j.text}`).join('\n');
  let out: string[] | null = null;
  try {
    const resp = await callAnalysisLLM([{ role: 'system', content: sys }, { role: 'user', content: user }]);
    const a = resp.indexOf('['), b = resp.lastIndexOf(']');
    if (a !== -1 && b !== -1) {
      const arr = JSON.parse(resp.slice(a, b + 1));
      if (Array.isArray(arr) && arr.length === jobs.length) out = arr.map((x: any) => String(x));
    }
  } catch (e: any) {
    console.error(`[Simplifier] English normalization call failed: ${e.message}`);
  }
  if (!out) return { characters, glosses };

  const outChars = characters.map(c => ({ ...c }));
  const outGlosses = { ...glosses };
  let applied = 0;
  jobs.forEach((j, i) => {
    const en = out![i];
    if (!en || hasHan(en)) return; // still Chinese → keep original (informative beats empty)
    if (j.kind === 'role') { outChars[j.key as number].role = en; applied++; }
    else { outGlosses[j.key as string] = en; applied++; }
  });
  console.error(`[Simplifier] English normalization: translated ${applied}/${jobs.length} Chinese role/gloss values`);
  return { characters: outChars, glosses: outGlosses };
}

/**
 * Generic OpenAI-compatible chat API helper — works with Qwen, OpenRouter, Groq, DeepSeek.
 * Automatically uses the detected provider's config (URL, API key, model).
 */
export async function callLLMMultiTurn(
  messages: Array<{ role: string; content: string }>,
  temperature = 0.1,
  timeoutMs = 120000,
  modelOverride?: string,
  providerOverride?: Provider,
  level: HSKLevel = 'hsk3',
  inputLanguage: BiasInputLanguage = 'chinese',
  // Set false for any pass that must NOT be HSK-constrained:
  //   - chapter analysis (callAnalysisLLM): the analysis output is structured
  //     metadata about the chapter; biasing it distorts KEEP lists and plot
  //     summaries and degrades the simplifier's downstream context
  //   - chained pipeline's stage-1 EN→CN translation: produces natural Chinese
  //     unconstrained, then stage-2 applies the HSK constraint
  // Defaults true so simplification callers keep current behavior.
  applyLogitBias: boolean = true,
  // Per-call override: token IDs whose negative logit-bias should be zeroed
  // (KEEP-term tokens). Applied on a copy of the cached bias map; doesn't
  // pollute the cache or affect concurrent requests.
  keepTokenIds?: number[],
  // Bias-magnitude language override. The single-pass EN→CN flow re-enters
  // simplifyPageBatch with inputLanguage='chinese' to break recursion, but
  // we want EN bias magnitudes for the original-English chapters. This param
  // preserves the original source language for getLogitBias only.
  biasSourceLanguage?: BiasInputLanguage,
): Promise<string> {
  const provider = providerOverride || detectProvider();
  const config = PROVIDER_CONFIGS[provider];

  if (!config) {
    throw new Error(`No config for provider: ${provider}. Use qwen, openrouter, groq, deepseek, cerebras, or fireworks.`);
  }

  // Note: spend cap is checked once per chapter at the API submit gate
  // (src/routes/api.ts), not here. Once a chapter starts processing, all its
  // LLM calls run to completion to avoid half-finished chapters. recordSpend()
  // below still tracks actual cost from each response.

  const url = new URL(config.baseUrl);
  const model = modelOverride || process.env.SIMPLIFY_MODEL || config.defaultModel;
  const endpoint = (url.pathname === '/' ? '' : url.pathname) + '/v1/chat/completions';
  const isHttps = config.useHttps || url.protocol === 'https:';

  // For local HTTP (Qwen), use the old http.request with keepAlive agent
  if (!isHttps) {
    const requestData = JSON.stringify({ model, messages, temperature });
    const headers: Record<string, string> = {
      'Content-Type': 'application/json; charset=utf-8',
      'Content-Length': String(Buffer.byteLength(requestData)),
    };
    if (config.apiKey) headers['Authorization'] = `Bearer ${config.apiKey}`;

    return new Promise((resolve, reject) => {
      const req = http.request({
        hostname: url.hostname,
        port: url.port || 80,
        path: endpoint,
        method: 'POST',
        agent: keepAliveAgent,
        headers,
      }, (res) => {
        let body = '';
        res.on('data', (chunk: string) => (body += chunk));
        res.on('end', () => {
          try {
            const result = JSON.parse(body);
            if (result.error) {
              reject(new Error(`[${provider}] ${result.error.message || JSON.stringify(result.error)}`));
              return;
            }
            let content = result.choices?.[0]?.message?.content || '';
            const thinkMatch = content.match(/<think>[\s\S]*?<\/think>/);
            if (thinkMatch) {
              console.error(`[Simplifier] Stripped ${thinkMatch[0].length} chars of thinking tokens`);
              content = content.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
            }
            resolve(content);
          } catch (e) {
            reject(new Error(`[${provider}] Failed to parse response: ${body.substring(0, 200)}`));
          }
        });
      });
      req.on('error', (e) => reject(new Error(`[${provider}] Request failed: ${e.message}`)));
      req.setTimeout(timeoutMs, () => { req.destroy(); reject(new Error(`[${provider}] Request timed out after ${timeoutMs / 1000}s`)); });
      req.write(requestData);
      req.end();
    });
  }

  // For cloud HTTPS providers, use native fetch() — avoids Node https.request connection issues
  const fetchUrl = `https://${url.hostname}${url.port ? ':' + url.port : ''}${endpoint}`;
  const fetchHeaders: Record<string, string> = { 'Content-Type': 'application/json' };
  if (config.apiKey) fetchHeaders['Authorization'] = `Bearer ${config.apiKey}`;

  const MAX_RETRIES = 3;
  const RATE_LIMIT_DELAYS = [30000, 60000, 90000]; // 429: 30s, 60s, 90s
  const OVERLOAD_DELAYS = [3000, 8000, 15000];     // 503: 3s, 8s, 15s

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const requestBody: any = { model, messages, temperature };
      if (applyLogitBias && (provider === 'fireworks' || provider === 'deepseek')) {
        const bias = getLogitBias(level, biasSourceLanguage || inputLanguage);
        if (bias) {
          if (keepTokenIds && keepTokenIds.length > 0) {
            // Per-call override: zero negative bias on KEEP-term token IDs so
            // the model isn't actively suppressed from generating KEEP terms.
            // Applied on a shallow copy to keep the cached map clean.
            const overridden: Record<number, number> = { ...bias };
            let zeroed = 0;
            for (const id of keepTokenIds) {
              if (overridden[id] !== undefined && overridden[id] < 0) {
                overridden[id] = 0;
                zeroed++;
              }
            }
            if (zeroed > 0 && attempt === 0) {
              console.error(`[${provider}] KEEP override: zeroed ${zeroed} negative bias entries on KEEP-term tokens`);
            }
            requestBody.logit_bias = overridden;
          } else {
            requestBody.logit_bias = bias;
          }
        }
      }
      if (provider === 'fireworks') {
        // Reasoning is wasted overhead for translation/simplification and,
        // left on, makes a chapter batch run far past the request timeout
        // (GLM-5.1 reasons before every answer). Disable it. max_tokens caps
        // a degenerate generation loop — logit_bias can otherwise trigger
        // runaway repetition that would spew tokens until the timeout.
        // Overridable for A/B work: deepseek-v4-flash-0731 is far more
        // reasoning-dependent than v4-flash (24k reasoning tokens vs 4k on the
        // same trivial prompt), so 'none' may cost it more on structured
        // extraction than it costs v4-flash. Default stays 'none'.
        requestBody.reasoning_effort = process.env.FIREWORKS_REASONING_EFFORT || 'none';
        requestBody.max_tokens = parseInt(process.env.FIREWORKS_MAX_TOKENS || '16000', 10);
      }
      const response = await fetch(fetchUrl, {
        method: 'POST',
        headers: fetchHeaders,
        body: JSON.stringify(requestBody),
        signal: controller.signal,
      });

      const body = await response.text();
      clearTimeout(timer);

      // 429: rate limited — slow back-off
      if (response.status === 429 && attempt < MAX_RETRIES) {
        const delay = RATE_LIMIT_DELAYS[attempt];
        console.error(`[${provider}] Rate limited (429), retrying in ${delay / 1000}s (attempt ${attempt + 1}/${MAX_RETRIES})...`);
        await new Promise(r => setTimeout(r, delay));
        continue;
      }

      // 503: model overloaded — short back-off (transient capacity issue)
      if (response.status === 503 && attempt < MAX_RETRIES) {
        const delay = OVERLOAD_DELAYS[attempt];
        console.error(`[${provider}] Overloaded (503), retrying in ${delay / 1000}s (attempt ${attempt + 1}/${MAX_RETRIES})...`);
        await new Promise(r => setTimeout(r, delay));
        continue;
      }

      if (!response.ok) {
        throw new Error(`[${provider}] HTTP ${response.status}: ${body.substring(0, 300)}`);
      }

      const result = JSON.parse(body);
      if (result.error) {
        throw new Error(`[${provider}] ${result.error.message || JSON.stringify(result.error)}`);
      }
      // Cerebras-style errors: {"message":"...","type":"too_many_requests_error"} without "error" wrapper
      if (result.type && result.message && !result.choices) {
        if (attempt < MAX_RETRIES) {
          const delay = RATE_LIMIT_DELAYS[attempt];
          console.error(`[${provider}] ${result.type}, retrying in ${delay / 1000}s (attempt ${attempt + 1}/${MAX_RETRIES})...`);
          await new Promise(r => setTimeout(r, delay));
          continue;
        }
        throw new Error(`[${provider}] ${result.type}: ${result.message}`);
      }

      let content = result.choices?.[0]?.message?.content || '';

      // Strip thinking tags if present (Qwen3, DeepSeek R1, etc.)
      const thinkMatch = content.match(/<think>[\s\S]*?<\/think>/);
      if (thinkMatch) {
        console.error(`[Simplifier] Stripped ${thinkMatch[0].length} chars of thinking tokens`);
        content = content.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
      }

      // Record actual spend from response usage. No-op for free/unknown providers.
      const promptTokens = result.usage?.prompt_tokens || 0;
      const completionTokens = result.usage?.completion_tokens || 0;
      // Cached (prompt-cache hit) portion of the prompt — billed at a discount.
      const cachedTokens = result.usage?.prompt_tokens_details?.cached_tokens || 0;
      if (promptTokens || completionTokens) {
        recordSpend(provider, promptTokens, completionTokens, cachedTokens);
        console.log(`[Tokens] ${provider}: ${promptTokens} prompt (${cachedTokens} cached) + ${completionTokens} completion`);
      }
      // Per-chapter Fireworks token split (cached vs fresh input vs output) —
      // read at job completion for the cost dashboard.
      if (provider === 'fireworks') {
        llmUsage.fireworks.cachedInputTokens += cachedTokens;
        llmUsage.fireworks.inputTokens += Math.max(0, promptTokens - cachedTokens);
        llmUsage.fireworks.outputTokens += completionTokens;
      }
      // Daily Gemini usage — tracked against the free-tier quota.
      if (provider === 'gemini' && (promptTokens || completionTokens)) {
        recordGeminiUsage(promptTokens, completionTokens);
      }

      return content;
    } catch (e: any) {
      clearTimeout(timer);
      if (e.name === 'AbortError') {
        const err: any = new Error(`[${provider}] Request timed out after ${timeoutMs / 1000}s`);
        err.code = 'LLM_TIMEOUT';
        throw err;
      }
      const err: any = new Error(`[${provider}] Request failed: ${e.message}`);
      // Preserve the underlying code if the inner error already had one
      if (e.code) err.code = e.code;
      throw err;
    }
  }

  const err: any = new Error(`[${provider}] All ${MAX_RETRIES} retries exhausted`);
  err.code = 'LLM_TIMEOUT';
  throw err;
}

/**
 * Extract JSON object from LLM response, handling markdown code blocks
 */
function extractJsonFromResponse(content: string): string | null {
  let jsonContent = content.trim();
  const codeBlockMatch = jsonContent.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (codeBlockMatch) {
    jsonContent = codeBlockMatch[1].trim();
  }
  const objectMatch = jsonContent.match(/\{[\s\S]*\}/);
  return objectMatch ? objectMatch[0] : null;
}

/**
 * Extract per-chunk KEEP candidates from a structured chunk response.
 * Accepts either {term: "..."} objects or bare strings; returns clean string list.
 * Empty list on parse failure — caller must tolerate.
 */
function extractKeepCandidates(response: string): string[] {
  const jsonStr = extractJsonFromResponse(response);
  if (!jsonStr) return [];
  try {
    const parsed = JSON.parse(jsonStr);
    if (!Array.isArray(parsed.keep_candidates)) return [];
    return parsed.keep_candidates
      .map((c: any) => typeof c === 'string' ? c : c?.term)
      .filter((t: any): t is string => typeof t === 'string' && t.trim().length > 0)
      .map((t: string) => t.trim());
  } catch {
    return [];
  }
}

/**
 * Merge per-chunk KEEP candidates with the FINAL turn's keep list.
 * - All FINAL entries are kept (the model's most confident picks).
 * - Chunk-only entries are kept if they appear in >=2 chunks (filters one-off
 *   hallucinations) OR if there was only one chunk (no frequency signal possible).
 *
 * Empirically (test-chapter-analysis-merge.js on a Solo Leveling chapter):
 * raised KEEP recall from 7 to 17 terms — captured chapter concepts like 突袭,
 * 治疗师, 首尔 that the single-shot FINAL turn was missing.
 */
function mergeChunkKeepCandidates(
  chunkKeepLists: string[][],
  finalKeep: string[]
): string[] {
  const termCount = new Map<string, number>();
  for (const list of chunkKeepLists) {
    const seenInThisChunk = new Set<string>();
    for (const term of list) {
      if (!term || seenInThisChunk.has(term)) continue;
      seenInThisChunk.add(term);
      termCount.set(term, (termCount.get(term) || 0) + 1);
    }
  }
  const finalSet = new Set(finalKeep);
  const totalChunks = chunkKeepLists.length;
  const merged: string[] = [...finalKeep];
  for (const [term, count] of termCount) {
    if (finalSet.has(term)) continue;
    if (count >= 2 || totalChunks <= 1) merged.push(term);
  }
  return merged;
}

/**
 * Clean up KEEP list from LLM analysis — remove phrases, overly long entries, etc.
 *
 * For English-input mode the chapterText source-presence check is skipped, because
 * KEEP terms are Chinese transliterations of English names that won't appear in the
 * English source by definition (e.g. "Bjorn" → "比约恩" never matches).
 */
function cleanKeepList(rawKeep: string[], chapterText?: string, inputLanguage: InputLanguage = 'chinese'): string[] {
  return rawKeep.filter(term => {
    // Allow English terms of any length
    const isEnglishOnly = /^[a-zA-Z\s]+$/.test(term);
    const isMixed = /[a-zA-Z]/.test(term) && term.length <= 10;

    if (!isEnglishOnly && !isMixed) {
      // Remove Chinese entries longer than 6 chars (likely phrases)
      if (term.replace(/[a-zA-Z\s]/g, '').length > 6) {
        console.error(`[Simplifier] KEEP removed (too long): "${term}"`);
        return false;
      }
      // Remove entries containing 的 (likely descriptive phrases)
      if (term.includes('的') && term.length > 3) {
        console.error(`[Simplifier] KEEP removed (contains 的): "${term}"`);
        return false;
      }
    }

    // Drop entries the LLM hallucinated — must appear in the chapter text.
    // Skipped for English-input mode (Chinese transliterations won't be in English source).
    if (!isTranslationSource(inputLanguage) && chapterText && !chapterText.includes(term)) {
      console.error(`[Simplifier] KEEP removed (not in chapter): "${term}"`);
      return false;
    }

    return true;
  });
}

/**
 * Analyze an entire chapter's text using multi-turn conversation.
 * Gathers setting, characters, plot summaries, and KEEP terms.
 * Uses think mode for deeper analysis.
 *
 * @param pages Array of { pageNum, text } for each page
 * @param level Target HSK level
 * @returns ChapterContext with setting, characters, plot, and cleaned KEEP list
 */
export async function analyzeChapter(
  pages: { pageNum: number; text: string }[],
  level: HSKLevel,
  inputLanguage: InputLanguage = 'chinese'
): Promise<ChapterContext> {
  const levelNum = level === 'hsk1' ? 1 : level === 'hsk2' ? 2 : level === 'hsk3' ? 3
    : level === 'hsk4' ? 4 : level === 'hsk5' ? 5 : 6;
  const MIN_CHUNK_SIZE = 20;
  const TARGET_TURNS = 2;
  const PAGES_PER_CHUNK = Math.max(MIN_CHUNK_SIZE, Math.ceil(pages.length / TARGET_TURNS));
  const allChapterText = pages.map(p => p.text).join('\n');

  // Split pages into chunks
  const chunks: typeof pages[] = [];
  for (let i = 0; i < pages.length; i += PAGES_PER_CHUNK) {
    chunks.push(pages.slice(i, i + PAGES_PER_CHUNK));
  }

  console.error(`[Simplifier] Chapter analysis: ${pages.length} pages in ${chunks.length} chunks (inputLanguage=${inputLanguage})`);

  const systemPrompt = isTranslationSource(inputLanguage)
    ? `You are analyzing ${inputLanguage === 'english' ? 'an English' : inputLanguage === 'japanese' ? 'a Japanese' : 'a Korean'} manga/manhua chapter that will be translated into HSK${levelNum} Chinese.

I will send you the text in batches of ~${PAGES_PER_CHUNK} pages. For EACH non-final batch, output JSON with:
  - "summary": 1-sentence plot summary of the batch (in Chinese)
  - "keep_candidates": list of { "term": "<Chinese term>", "type": "character|place|concept" }
    capturing chapter-specific terms you saw in THIS batch — character names (Chinese
    transliterations), place names, world concepts, game-system terms, special items, races.
    Be inclusive — aim for 5-15 candidates per batch.

After the final batch, I will append FINAL instructions and you produce the complete chapter
analysis JSON (setting, characters, plot, keep, glosses). The FINAL keep does NOT need to
repeat every per-batch candidate; I will merge them automatically.`
    : `You are analyzing a Chinese manhua chapter for text simplification to HSK${levelNum}.

I will send you the text in batches of ~${PAGES_PER_CHUNK} pages. For EACH non-final batch, output JSON with:
  - "summary": 1-sentence plot summary of the batch (in Chinese)
  - "keep_candidates": list of { "term": "<Chinese term>", "type": "character|place|concept" }
    capturing chapter-specific terms you saw in THIS batch — character names, place names,
    world concepts, special items, slang, censored words. Be inclusive — aim for 5-15
    candidates per batch.

After the final batch, I will append FINAL instructions and you produce the complete chapter
analysis JSON (setting, characters, plot, keep, glosses). The FINAL keep does NOT need to
repeat every per-batch candidate; I will merge them automatically.`;

  const messages: Array<{ role: string; content: string }> = [
    { role: 'system', content: systemPrompt },
  ];

  // Per-chunk KEEP candidates accumulated across non-final turns, then merged
  // with the FINAL turn's keep list (see mergeChunkKeepCandidates).
  const chunkKeepLists: string[][] = [];

  for (let chunkIdx = 0; chunkIdx < chunks.length; chunkIdx++) {
    const chunk = chunks[chunkIdx];
    const startPage = chunk[0].pageNum;
    const endPage = chunk[chunk.length - 1].pageNum;
    const isFinal = chunkIdx === chunks.length - 1;

    let userMsg = `Pages ${startPage}-${endPage}:\n`;
    for (const page of chunk) {
      userMsg += `\n--- Page ${page.pageNum} ---\n${page.text}\n`;
    }
    if (!isFinal) {
      userMsg += `\nOutput JSON for this batch (summary + keep_candidates).`;
    }

    if (isFinal) {
      if (isTranslationSource(inputLanguage)) {
        userMsg += `\nFINAL — Produce the complete chapter analysis as JSON:
{
  "setting": "1-2 sentence genre/setting description",
  "characters": [{"name": "English Name", "chineseName": "中文名", "role": "brief description IN ENGLISH — shown to learners in the reader's dictionary popup"}],
  "concepts": [{"english": "<source-language term as written in the chapter>", "chinese": "canonical Chinese"}],
  "plot": [{"pages": "1-20", "summary": "what happens"}],
  "keep": ["中文名1", "中文名2"],
  "glosses": {"中文名1": "a few-word English definition of this term", "中文名2": "a few-word English definition of this term"}
}

CRITICAL — Output language rules:
- "characters" role descriptions: ALWAYS in English (these are shown to English-speaking learners in popup tooltips, never in Chinese)
- "glosses" values: ALWAYS in English (short, few-word definitions)
- "name" stays in the source-language form as written in the chapter (e.g. "Sung Jin-Woo", "田中太郎")
- "chineseName" is the Chinese transliteration
- "setting", "plot" summaries, "keep" terms stay in Chinese

Guidelines for "concepts":
- Every non-character noun/term in "keep" MUST also appear in "concepts" with its source-language word in the "english" field.
  Example: if "keep" has "魔石", concepts must have {"english": "mana stone", "chinese": "魔石"}.
- Pick SHORT canonical Chinese: mana stone → 魔石 (not 魔法石); portal → 传送门; barbarian → 蛮族 (not 野蛮人); adventurer → 冒险者.
- These will be enforced literally in the translation. Choose Chinese that sounds like manga dialogue, not a translation textbook.

CRITICAL — Validity rules (apply to "keep" AND "characters"):
- Every entry MUST appear verbatim in the chapter text you have just read above. Before adding a term, verify you saw it in at least one page batch.
- Do NOT include game/UI labels (RPG, FPS, NPC, UI, AOS), race/class names the chapter doesn't actually use (HUMAN, DWARF, FAIRY, FURRY), or one-off English phrases that appeared only once.
- For "characters", entries must be proper names. Common nouns describing a role (大叔, 老师, 神官, 长老) are NOT characters.
- For "keep", a term should appear at least twice OR be unambiguously a proper name/place/concept. One-off terms that are not names should be excluded.

Examples of bad outputs to AVOID:
- "keep": ["HUMAN", "DWARF", "FAIRY"] when the chapter never mentions these races
- "characters": [{"name": "大叔"}] (a common noun, not a name)
- "keep": ["GATE OF THE ABYSS"] if it appears once and isn't reused

CRITICAL — "keep" entry format rules:
- Each entry MUST be a single word or short term (1-4 Chinese characters, or a single English/mixed word)
- NEVER use phrases, sentences, or descriptions as entries
- Break compound references into separate entries. Example: "波嫩的三儿子克拉克" → add "波嫩" and "克拉克" as separate entries
- No entry should contain 的 unless it is part of an official name
- Character names, place names, race/class names, rank labels, and core concept nouns should each be their own entry

Guidelines for "characters":
- Include all named characters that satisfy the validity rules above
- "chineseName" should be the standard Chinese translation of the name (e.g., "Charlie" → "查理")
- Each character should be a separate entry

Guidelines for "keep":
- Include the Chinese translations of all character names, place names, and important terms
- These Chinese terms will be enforced consistently throughout the translation
- Each entry should be 1-4 Chinese characters

Guidelines for "plot":
- One entry per ~${PAGES_PER_CHUNK}-page batch
- Focus on: who is present, what's happening, key events`;
      } else {
        userMsg += `\nFINAL — Produce the complete chapter analysis as JSON:
{
  "setting": "1-2 sentence genre/setting description",
  "characters": [{"name": "English Name", "chineseName": "中文名", "role": "brief description IN ENGLISH — shown to learners in the reader's dictionary popup"}],
  "plot": [{"pages": "1-20", "summary": "what happens"}],
  "keep": ["terms that should NOT be simplified"],
  "glosses": {"term": "a few-word English definition of this term"}
}

CRITICAL — Output language rules:
- "characters" role descriptions: ALWAYS in English (these are shown to English-speaking learners)
- "glosses" values: ALWAYS in English (short, few-word definitions)
- "characters" name: ALWAYS English — see the romanisation rules below
- "characters" chineseName: the Chinese form exactly as it appears in the chapter
- "setting", "plot" summaries and "keep" terms stay in Chinese

CRITICAL — Validity rules (apply to "keep" AND "characters"):
- Every entry MUST appear verbatim in the chapter text you have just read above. Before adding a term, verify you saw it in at least one page batch.
- Do NOT include game/UI labels (RPG, FPS, NPC, UI, AOS), race/class names the chapter doesn't actually use (HUMAN, DWARF, FAIRY, FURRY), or one-off English phrases that appeared only once.
- For "characters", entries must be proper names. Common nouns describing a role (大叔, 老师, 神官, 长老) are NOT characters.
- For "keep", a term should appear at least twice OR be unambiguously a proper name/place/concept. One-off terms that are not names should be excluded.

Examples of bad outputs to AVOID:
- "keep": ["HUMAN", "DWARF", "FAIRY"] when the chapter never mentions these races
- "characters": [{"name": "大叔"}] (a common noun, not a name)
- "keep": ["GATE OF THE ABYSS"] if it appears once and isn't reused

CRITICAL — "keep" entry format rules:
- Each entry MUST be a single word or short term (1-4 Chinese characters, or a single English/mixed word)
- NEVER use phrases, sentences, or descriptions as entries
- Break compound references into separate entries. Example: "波嫩的三儿子克拉克" → add "波嫩" and "克拉克" as separate entries
- No entry should contain 的 unless it is part of an official name
- Character names, place names, race/class names, rank labels, and core concept nouns should each be their own entry

Guidelines for "characters":
- "chineseName" is the name as written in the chapter; "name" is its English form
- Many of these names are transliterations of Western names. Recover the original
  rather than romanising syllable by syllable: 比约恩 is Bjorn (not Biyue'en),
  克拉克 is Clark, 罗宾 is Robin, 洋德尔 is Yondel
- If the name is genuinely Chinese, use standard pinyin, capitalised, no tone marks
- Names must be consistent with each other: if one role description refers to
  another character, spell it the same way in both places

Guidelines for "keep":
- Character names, place names, titles, race/class names, game concepts
- Slang and emotional expressions natural in speech
- Censored words (X蛋, X的, etc.)
- Terms above HSK${levelNum} where no natural simpler alternative exists
- Do NOT include normal vocabulary, adjectives, or verbs that have simple synonyms

Guidelines for "characters":
- Each character should be a separate entry with their own name

Guidelines for "plot":
- One entry per ~${PAGES_PER_CHUNK}-page batch
- Focus on: who is present, what's happening, key events`;
      }
    }

    messages.push({ role: 'user', content: userMsg });

    const turnStart = Date.now();
    try {
      const response = await callAnalysisLLM(messages);
      const turnTime = ((Date.now() - turnStart) / 1000).toFixed(1);

      if (isFinal) {
        console.error(`[Simplifier] Chapter analysis turn ${chunkIdx + 1} (FINAL, pages ${startPage}-${endPage}): ${turnTime}s`);
        const jsonStr = extractJsonFromResponse(response);
        if (jsonStr) {
          const parsed = JSON.parse(jsonStr);
          const finalKeep: string[] = parsed.keep || [];
          // Merge: union of per-chunk keep_candidates with FINAL keep, then
          // clean. Chunk candidates capture chapter-specific terms the FINAL
          // turn sometimes forgets (e.g. 突袭 / 治疗师 / 首尔 in test runs).
          const mergedKeep = mergeChunkKeepCandidates(chunkKeepLists, finalKeep);
          const cleanedKeep = cleanKeepList(mergedKeep, allChapterText, inputLanguage);
          console.error(`[Simplifier] Setting: ${parsed.setting || '(none)'}`);
          console.error(`[Simplifier] Characters (${(parsed.characters || []).length}): ${(parsed.characters || []).map((c: any) => c.name).join(', ')}`);
          console.error(`[Simplifier] Plot sections: ${(parsed.plot || []).length}`);
          console.error(`[Simplifier] KEEP merged from ${chunkKeepLists.length} chunks + FINAL: ${finalKeep.length} → ${mergedKeep.length} pre-clean → ${cleanedKeep.length} final`);
          console.error(`[Simplifier] KEEP (${cleanedKeep.length}): ${cleanedKeep.join(', ')}`);

          const characters = parsed.characters || [];
          // glosses is optional + defensively typed — a malformed/missing field
          // degrades gracefully (KEEP terms fall back to CC-CEDICT or nothing).
          const glosses = (parsed.glosses && typeof parsed.glosses === 'object' && !Array.isArray(parsed.glosses))
            ? parsed.glosses : {};
          // concepts is EN-only. Merge with characters' name/chineseName pairs
          // so downstream Stage 1 has one unified EN→CN map. Malformed entries
          // are silently skipped; empty concepts is fine (CN input path).
          let concepts: { english: string; chinese: string }[] | undefined;
          if (isTranslationSource(inputLanguage)) {
            const map = new Map<string, string>();
            for (const c of characters) {
              const chineseName = (c as any).chineseName;
              if (c.name && chineseName) map.set(c.name.trim(), chineseName.trim());
            }
            if (Array.isArray(parsed.concepts)) {
              for (const c of parsed.concepts) {
                if (c && typeof c.english === 'string' && typeof c.chinese === 'string') {
                  map.set(c.english.trim(), c.chinese.trim());
                }
              }
            }
            concepts = [...map].map(([english, chinese]) => ({ english, chinese }));
          }
          const norm = await normalizeAnalysisToEnglish(characters, glosses);
          return {
            setting: parsed.setting || '',
            characters: norm.characters,
            plot: parsed.plot || [],
            keep: cleanedKeep,
            glosses: norm.glosses,
            definitions: buildChapterDefinitions(norm.characters, cleanedKeep, norm.glosses),
            concepts,
          };
        } else {
          console.error('[Simplifier] No JSON found in final chapter analysis response');
        }
      } else {
        // Non-FINAL chunk — extract its keep_candidates for the merge.
        // extractKeepCandidates returns [] on parse failure, so a malformed
        // chunk response just contributes nothing instead of crashing.
        const candidates = extractKeepCandidates(response);
        chunkKeepLists.push(candidates);
        console.error(`[Simplifier] Chapter analysis turn ${chunkIdx + 1} (pages ${startPage}-${endPage}): ${turnTime}s, ${candidates.length} keep_candidates`);
      }

      messages.push({ role: 'assistant', content: response });
    } catch (e: any) {
      console.error(`[Simplifier] Chapter analysis turn ${chunkIdx + 1} FAILED: ${e.message}`);
      // A failed non-FINAL turn contributes no candidates; the FINAL turn can
      // still produce a usable analysis on its own.
      if (!isFinal) chunkKeepLists.push([]);
      messages.push({ role: 'assistant', content: '(error - continuing)' });
    }
  }

  // Return empty context if analysis failed
  return { setting: '', characters: [], plot: [], keep: [], definitions: new Map() };
}

/**
 * Streaming version of analyzeChapter — accepts chunks incrementally via async callback.
 * This allows OCR scan and chapter analysis to run concurrently:
 * as soon as a chunk of pages finishes OCR, it can be sent for analysis
 * while remaining pages are still being scanned.
 *
 * @param getNextChunk Async function that returns the next chunk of pages, or null when done.
 *   Each chunk includes { pages, isLast } where isLast signals the final chunk.
 * @param level Target HSK level
 * @param estimatedChunkSize Approximate pages per chunk (used in system prompt)
 * @returns ChapterContext with setting, characters, plot, and cleaned KEEP list
 */
export async function analyzeChapterStreaming(
  getNextChunk: () => Promise<{ pages: { pageNum: number; text: string }[]; isLast: boolean } | null>,
  level: HSKLevel,
  estimatedChunkSize: number = 20,
  inputLanguage: InputLanguage = 'chinese'
): Promise<ChapterContext> {
  const levelNum = level === 'hsk1' ? 1 : level === 'hsk2' ? 2 : level === 'hsk3' ? 3
    : level === 'hsk4' ? 4 : level === 'hsk5' ? 5 : 6;

  const systemPrompt = isTranslationSource(inputLanguage)
    ? `You are analyzing ${inputLanguage === 'english' ? 'an English' : inputLanguage === 'japanese' ? 'a Japanese' : 'a Korean'} manga/manhua chapter that will be translated into HSK${levelNum} Chinese.

I will send you the text in batches of ~${estimatedChunkSize} pages. For EACH batch, output JSON with:
  - "summary": 1-sentence plot summary of the batch (in Chinese)
  - "keep_candidates": list of { "term": "<Chinese term>", "type": "character|place|concept" }
    capturing chapter-specific terms you saw in THIS batch — character names (Chinese
    transliterations), place names, world concepts, game-system terms, special items, races.
    Be inclusive — aim for 5-15 candidates per batch.

After the final batch, I will say FINAL — produce the complete chapter analysis JSON
(setting, characters, plot, keep, glosses). The FINAL keep does NOT need to repeat every
per-batch candidate; I will merge them automatically.`
    : `You are analyzing a Chinese manhua chapter for text simplification to HSK${levelNum}.

Your task: produce a KEEP list to prevent over-simplification of story-important nouns.
Be STRICT: only preserve nouns that either (a) are story-specific names/concepts, or
(b) genuinely have no simple HSK${levelNum} equivalent in Chinese.

Everyday nouns that DO have a simple HSK${levelNum} synonym should NOT be in KEEP — the
simplifier will paraphrase them naturally. Examples of what to EXCLUDE:
  袋子 (has 包), 晚餐 (has 晚饭), 老公/大叔 (has 男人), 表情/个性 (has 样子),
  心情 (has 感觉), 氛围 (has 环境), 全名 (has 名字), 富裕 (has 有钱),
  疯子 (has 奇怪的人), 时代 (has 时候), 姓氏 (has 姓).

I will send you the text in batches of ~${estimatedChunkSize} pages. For EACH batch, output JSON with:
  - "summary": 1-sentence plot summary of the batch (in Chinese)
  - "keep_candidates": list of { "term": "<Chinese term>", "type": "character|place|concept" }.
    Include ONLY: character names, place names, race/class names, weapon/item names, world
    concepts, game titles, and concept nouns with NO simple HSK${levelNum} alternative
    (e.g., 迷宫, 冒险家, 战士, 妖精, 战斧, 尖耳怪, 蛮族, 怀表, 传送门, 异世界).
    Do NOT include verbs (打斗, 沉迷, 消失), adjectives (强悍, 敏捷), or emotion words
    (羡慕, 惊讶). When unsure whether a term has a simple synonym, err on the side of
    EXCLUDING it — over-inclusion forces HSK 4+ vocabulary on the learner.

After the final batch, I will say FINAL — produce the complete chapter analysis JSON
(setting, characters, plot, keep, glosses). The FINAL keep does NOT need to repeat every
per-batch candidate; I will merge them automatically.`;

  const messages: Array<{ role: string; content: string }> = [
    { role: 'system', content: systemPrompt },
  ];

  // Per-chunk KEEP candidates accumulated across turns, then merged with the
  // FINAL turn's keep list (see mergeChunkKeepCandidates).
  const chunkKeepLists: string[][] = [];

  let chunkIdx = 0;
  let totalPages = 0;
  let allChapterText = '';

  while (true) {
    const result = await getNextChunk();
    if (!result) break;

    const { pages: chunk, isLast } = result;
    if (chunk.length === 0 && !isLast) continue;

    totalPages += chunk.length;
    const startPage = chunk.length > 0 ? chunk[0].pageNum : 0;
    const endPage = chunk.length > 0 ? chunk[chunk.length - 1].pageNum : 0;

    let userMsg = `Pages ${startPage}-${endPage}:\n`;
    for (const page of chunk) {
      userMsg += `\n--- Page ${page.pageNum} ---\n${page.text}\n`;
      allChapterText += page.text + '\n';
    }
    userMsg += `\nOutput JSON for this batch (summary + keep_candidates).`;

    messages.push({ role: 'user', content: userMsg });

    const turnStart = Date.now();
    try {
      const response = await callAnalysisLLM(messages);
      const turnTime = ((Date.now() - turnStart) / 1000).toFixed(1);

      // Extract this chunk's KEEP candidates for the merge. extractKeepCandidates
      // returns [] on parse failure, so a malformed response just contributes
      // nothing — the FINAL turn can still produce a usable analysis.
      const candidates = extractKeepCandidates(response);
      chunkKeepLists.push(candidates);
      console.error(`[Simplifier] Chapter analysis turn ${chunkIdx + 1} (pages ${startPage}-${endPage}): ${turnTime}s, ${candidates.length} keep_candidates`);
      console.error(`[Simplifier] Turn ${chunkIdx + 1} response (${response.length} chars):\n${response}`);

      messages.push({ role: 'assistant', content: response });

      // After the last batch, send a separate FINAL turn to produce the combined analysis
      if (isLast) {
        const finalMsg = isTranslationSource(inputLanguage)
          ? `FINAL — Produce the complete chapter analysis as JSON:
{
  "setting": "1-2 sentence genre/setting description",
  "characters": [{"name": "English Name", "chineseName": "中文名", "role": "brief description IN ENGLISH — shown to learners in the reader's dictionary popup"}],
  "concepts": [{"english": "<source-language term as written in the chapter>", "chinese": "canonical Chinese"}],
  "plot": [{"pages": "1-20", "summary": "what happens"}],
  "keep": ["中文名1", "中文名2"],
  "glosses": {"中文名1": "a few-word English definition of this term", "中文名2": "a few-word English definition of this term"}
}

CRITICAL — Output language rules:
- "characters" role descriptions: ALWAYS in English (these are shown to English-speaking learners in popup tooltips, never in Chinese)
- "glosses" values: ALWAYS in English (short, few-word definitions)
- "name" stays in the source-language form as written in the chapter (e.g. "Sung Jin-Woo", "田中太郎")
- "chineseName" is the Chinese transliteration
- "setting", "plot" summaries, "keep" terms stay in Chinese

Guidelines for "concepts":
- Every non-character noun/term in "keep" MUST also appear in "concepts" with its source-language word in the "english" field.
  Example: if "keep" has "魔石", concepts must have {"english": "mana stone", "chinese": "魔石"}.
- Pick SHORT canonical Chinese: mana stone → 魔石 (not 魔法石); portal → 传送门; barbarian → 蛮族 (not 野蛮人); adventurer → 冒险者.
- These will be enforced literally in the translation. Choose Chinese that sounds like manga dialogue, not a translation textbook.

CRITICAL — Validity rules (apply to "keep" AND "characters"):
- Every entry MUST appear verbatim in the chapter text you have just read above. Before adding a term, verify you saw it in at least one page batch.
- Do NOT include game/UI labels (RPG, FPS, NPC, UI, AOS), race/class names the chapter doesn't actually use (HUMAN, DWARF, FAIRY, FURRY), or one-off English phrases that appeared only once.
- For "characters", entries must be proper names. Common nouns describing a role (大叔, 老师, 神官, 长老) are NOT characters.
- For "keep", a term should appear at least twice OR be unambiguously a proper name/place/concept. One-off terms that are not names should be excluded.

Examples of bad outputs to AVOID:
- "keep": ["HUMAN", "DWARF", "FAIRY"] when the chapter never mentions these races
- "characters": [{"name": "大叔"}] (a common noun, not a name)
- "keep": ["GATE OF THE ABYSS"] if it appears once and isn't reused

CRITICAL — "keep" entry format rules:
- Each entry MUST be a single word or short term (1-4 Chinese characters, or a single English/mixed word)
- NEVER use phrases, sentences, or descriptions as entries
- Break compound references into separate entries. Example: "波嫩的三儿子克拉克" → add "波嫩" and "克拉克" as separate entries
- No entry should contain 的 unless it is part of an official name
- Character names, place names, race/class names, rank labels, and core concept nouns should each be their own entry

Guidelines for "characters":
- Include all named characters that satisfy the validity rules above
- "chineseName" should be the standard Chinese translation of the name (e.g., "Charlie" → "查理")
- Each character should be a separate entry

Guidelines for "keep":
- Include the Chinese translations of all character names, place names, and important terms
- These Chinese terms will be enforced consistently throughout the translation
- Each entry should be 1-4 Chinese characters

Guidelines for "plot":
- One entry per ~${estimatedChunkSize}-page batch
- Focus on: who is present, what's happening, key events`
          : `FINAL — Produce the complete chapter analysis as JSON:
{
  "setting": "1-2 sentence genre/setting description",
  "characters": [{"name": "English Name", "chineseName": "中文名", "role": "brief description IN ENGLISH — shown to learners in the reader's dictionary popup"}],
  "plot": [{"pages": "1-20", "summary": "what happens"}],
  "keep": ["terms that should NOT be simplified"],
  "glosses": {"term": "a few-word English definition of this term"}
}

CRITICAL — Output language rules:
- "characters" role descriptions: ALWAYS in English (these are shown to English-speaking learners)
- "glosses" values: ALWAYS in English (short, few-word definitions)
- "characters" name: ALWAYS English — see the romanisation rules below
- "characters" chineseName: the Chinese form exactly as it appears in the chapter
- "setting", "plot" summaries and "keep" terms stay in Chinese

CRITICAL — Validity rules (apply to "keep" AND "characters"):
- Every entry MUST appear verbatim in the chapter text you have just read above. Before adding a term, verify you saw it in at least one page batch.
- Do NOT include game/UI labels (RPG, FPS, NPC, UI, AOS), race/class names the chapter doesn't actually use (HUMAN, DWARF, FAIRY, FURRY), or one-off English phrases that appeared only once.
- For "characters", entries must be proper names. Common nouns describing a role (大叔, 老师, 神官, 长老) are NOT characters.
- For "keep", a term should appear at least twice OR be unambiguously a proper name/place/concept. One-off terms that are not names should be excluded.

Examples of bad outputs to AVOID:
- "keep": ["HUMAN", "DWARF", "FAIRY"] when the chapter never mentions these races
- "characters": [{"name": "大叔"}] (a common noun, not a name)
- "keep": ["GATE OF THE ABYSS"] if it appears once and isn't reused

CRITICAL — "keep" entry format rules:
- Each entry MUST be a single word or short term (1-4 Chinese characters, or a single English/mixed word)
- NEVER use phrases, sentences, or descriptions as entries
- Break compound references into separate entries. Example: "波嫩的三儿子克拉克" → add "波嫩" and "克拉克" as separate entries
- No entry should contain 的 unless it is part of an official name
- Character names, place names, race/class names, rank labels, and core concept nouns should each be their own entry

Guidelines for "keep":
- Include ONLY nouns that either (a) are story-specific names/concepts, or (b) have
  NO simple HSK${levelNum} equivalent in Chinese. Be STRICT, not aggressive.
- ALWAYS include: character names, place names, race/class names, weapon/item names
  (锤子, 战斧), monster/creature names (刃狼, 尖耳怪), unique world concepts, game titles.
- Include concept nouns WITHOUT simple synonyms: 迷宫, 冒险家, 战士, 妖精, 蛮族, 怀表,
  传送门, 异世界, 主角 (protagonist — no clean HSK ${levelNum} equivalent).
- Do NOT include everyday nouns that have simple HSK${levelNum} alternatives:
  袋子 (use 包), 晚餐 (use 晚饭), 老公 (use 男人), 大叔 (use 男人), 表情 (use 样子),
  心情 (use 感觉), 氛围 (use 环境), 全名 (use 名字), 富裕 (use 有钱),
  疯子 (use 奇怪的人), 个性 (use 样子), 时代 (use 时候), 姓氏 (use 姓),
  熟人 (use 认识的人), 步骤 (use 事情).
- Slang and emotional expressions natural in speech (家伙, 直男) MAY be included if
  they have no HSK${levelNum} equivalent.
- Censored words (X蛋, X的, etc.) — include.
- Do NOT include verbs (打斗, 沉迷, 消失), adjectives (强悍, 敏捷), adverbs, or emotion
  words (羡慕, 惊讶).
- For ambiguous noun/verb words (消失, 和平), include only if the chapter uses them
  primarily as nouns AND they have no simple synonym.
- When unsure whether a term has a simple synonym, exclude it.

Guidelines for "characters":
- "chineseName" is the name as written in the chapter; "name" is its English form
- Many of these names are transliterations of Western names. Recover the original
  rather than romanising syllable by syllable: 比约恩 is Bjorn (not Biyue'en),
  克拉克 is Clark, 罗宾 is Robin, 洋德尔 is Yondel
- If the name is genuinely Chinese, use standard pinyin, capitalised, no tone marks
- Names must be consistent with each other: if one role description refers to
  another character, spell it the same way in both places
- Each character should be a separate entry with their own name

Guidelines for "plot":
- One entry per ~${estimatedChunkSize}-page batch
- Focus on: who is present, what's happening, key events`;

        messages.push({ role: 'user', content: finalMsg });

        const finalStart = Date.now();
        const finalResponse = await callAnalysisLLM(messages);
        const finalTime = ((Date.now() - finalStart) / 1000).toFixed(1);
        console.error(`[Simplifier] Chapter analysis turn ${chunkIdx + 2} (FINAL): ${finalTime}s`);

        const jsonStr = extractJsonFromResponse(finalResponse);
        if (jsonStr) {
          const parsed = JSON.parse(jsonStr);
          const finalKeep: string[] = parsed.keep || [];
          // Merge per-chunk candidates with FINAL keep (see analyzeChapter
          // for rationale + empirical effect).
          const mergedKeep = mergeChunkKeepCandidates(chunkKeepLists, finalKeep);
          const cleanedKeep = cleanKeepList(mergedKeep, allChapterText, inputLanguage);
          console.error(`[Simplifier] Setting: ${parsed.setting || '(none)'}`);
          console.error(`[Simplifier] Characters (${(parsed.characters || []).length}): ${(parsed.characters || []).map((c: any) => c.name).join(', ')}`);
          console.error(`[Simplifier] Plot sections: ${(parsed.plot || []).length}`);
          console.error(`[Simplifier] KEEP merged from ${chunkKeepLists.length} chunks + FINAL: ${finalKeep.length} → ${mergedKeep.length} pre-clean → ${cleanedKeep.length} final`);
          console.error(`[Simplifier] KEEP (${cleanedKeep.length}): ${cleanedKeep.join(', ')}`);

          const characters = parsed.characters || [];
          // glosses is optional + defensively typed — a malformed/missing field
          // degrades gracefully (KEEP terms fall back to CC-CEDICT or nothing).
          const glosses = (parsed.glosses && typeof parsed.glosses === 'object' && !Array.isArray(parsed.glosses))
            ? parsed.glosses : {};
          // concepts is EN-only. Merge with characters' name/chineseName pairs
          // so downstream Stage 1 has one unified EN→CN map. Malformed entries
          // are silently skipped; empty concepts is fine (CN input path).
          let concepts: { english: string; chinese: string }[] | undefined;
          if (isTranslationSource(inputLanguage)) {
            const map = new Map<string, string>();
            for (const c of characters) {
              const chineseName = (c as any).chineseName;
              if (c.name && chineseName) map.set(c.name.trim(), chineseName.trim());
            }
            if (Array.isArray(parsed.concepts)) {
              for (const c of parsed.concepts) {
                if (c && typeof c.english === 'string' && typeof c.chinese === 'string') {
                  map.set(c.english.trim(), c.chinese.trim());
                }
              }
            }
            concepts = [...map].map(([english, chinese]) => ({ english, chinese }));
          }
          const norm = await normalizeAnalysisToEnglish(characters, glosses);
          return {
            setting: parsed.setting || '',
            characters: norm.characters,
            plot: parsed.plot || [],
            keep: cleanedKeep,
            glosses: norm.glosses,
            definitions: buildChapterDefinitions(norm.characters, cleanedKeep, norm.glosses),
            concepts,
          };
        } else {
          console.error(`[Simplifier] No JSON found in final response (${finalResponse.length} chars):\n${finalResponse}`);
        }
      }

      messages.push({ role: 'assistant', content: response });
    } catch (e: any) {
      console.error(`[Simplifier] Chapter analysis turn ${chunkIdx + 1} FAILED: ${e.message}`);
      // A failed chunk contributes no candidates; FINAL turn can still produce
      // a usable analysis on its own.
      chunkKeepLists.push([]);
      messages.push({ role: 'assistant', content: '(error - continuing)' });
    }

    chunkIdx++;
  }

  console.error(`[Simplifier] Streaming chapter analysis: ${totalPages} total pages in ${chunkIdx} chunks`);
  return { setting: '', characters: [], plot: [], keep: [], definitions: new Map() };
}

/**
 * Parse sentinel-formatted LLM response back into Record<string, string>
 * Expected format: <<<B1>>>\nsimplified text\n<<<END>>>\n\n<<<B2>>>...
 */
// Permissive sentinel parsing:
//   Tokens are `<+TOKEN>+` (any number of < and > on each side) AND must sit on
//   their own line (newline or start/end-of-string on both sides). TOKEN is
//   restricted to `END` or digits — never freeform text — so Chinese content
//   that incidentally contains `<X>` can't false-match. The own-line rule
//   prevents matching mid-sentence markers entirely.
//
//   This tolerates the observed Fireworks failure mode where the closing
//   sentinel truncates by 1 character (`<<<END>>` vs `<<<END>>>`) without
//   loosening enough to false-positive on translated content.
function parseSentinelResponse(content: string, blocks: Record<string, string[]>): Record<string, string> {
  const result: Record<string, string> = {};
  for (const key in blocks) {
    // Start tag: `<+B${key}>+` on its own line
    const startRe = new RegExp(`(?:^|\\n)<+B${key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}>+(?:\\n|$)`);
    const startMatch = startRe.exec(content);
    if (!startMatch) {
      console.error(`[Simplifier] Pass 1: Missing sentinel block ${key}, using original`);
      result[key] = smartJoin(blocks[key]);
      continue;
    }
    const afterStart = startMatch.index + startMatch[0].length;
    const remainder = content.substring(afterStart);
    // End boundary: closest of (a) own-line <+END>+ or (b) next own-line <+B\d+>+
    const endMatch = remainder.match(/(?:^|\n)<+END>+(?:\n|$)/);
    const nextStartMatch = remainder.match(/(?:^|\n)<+B\d+>+(?:\n|$)/);
    let endRel = -1;
    if (endMatch?.index !== undefined && nextStartMatch?.index !== undefined) {
      endRel = Math.min(endMatch.index, nextStartMatch.index);
    } else if (endMatch?.index !== undefined) {
      endRel = endMatch.index;
    } else if (nextStartMatch?.index !== undefined) {
      endRel = nextStartMatch.index;
    }
    if (endRel < 0) {
      console.error(`[Simplifier] Pass 1: Missing END tag for block ${key}, using original`);
      result[key] = smartJoin(blocks[key]);
      continue;
    }
    result[key] = remainder.substring(0, endRel).trim();
  }
  return result;
}

/**
 * Parse batched sentinel response with sequential integer block markers.
 * Expected format: <<<1>>>\nsimplified text\n<<<END>>>\n\n<<<2>>>...
 * seqToLoc maps each sequential label back to its (pageNum, blockKey) — the
 * short integer labels cost fewer tokens than page/block composites.
 * Returns: Map<pageNum, Record<blockKey, simplifiedText>>
 */
function parseBatchedSentinelResponse(
  content: string,
  inputRecord: Map<number, Record<string, string[]>>,
  seqToLoc: Map<number, { pageNum: number; key: string }>
): Map<number, Record<string, string>> {
  // See parseSentinelResponse for the permissive own-line regex rationale.
  const results = new Map<number, Record<string, string>>();
  for (const pageNum of inputRecord.keys()) results.set(pageNum, {});
  for (const [seq, loc] of seqToLoc) {
    const pageResult = results.get(loc.pageNum)!;
    const startRe = new RegExp(`(?:^|\\n)<+${seq}>+(?:\\n|$)`);
    const startMatch = startRe.exec(content);
    if (!startMatch) {
      console.error(`[Simplifier] Batch: Missing sentinel <<<${seq}>>> (P${loc.pageNum} B${loc.key}), using original`);
      pageResult[loc.key] = smartJoin(inputRecord.get(loc.pageNum)![loc.key]);
      continue;
    }
    const afterStart = startMatch.index + startMatch[0].length;
    const remainder = content.substring(afterStart);
    const endMatch = remainder.match(/(?:^|\n)<+END>+(?:\n|$)/);
    const nextStartMatch = remainder.match(/(?:^|\n)<+\d+>+(?:\n|$)/);
    let endRel = -1;
    if (endMatch?.index !== undefined && nextStartMatch?.index !== undefined) {
      endRel = Math.min(endMatch.index, nextStartMatch.index);
    } else if (endMatch?.index !== undefined) {
      endRel = endMatch.index;
    } else if (nextStartMatch?.index !== undefined) {
      endRel = nextStartMatch.index;
    }
    if (endRel < 0) {
      console.error(`[Simplifier] Batch: Missing END for <<<${seq}>>> (P${loc.pageNum} B${loc.key}), using original`);
      pageResult[loc.key] = smartJoin(inputRecord.get(loc.pageNum)![loc.key]);
      continue;
    }
    pageResult[loc.key] = remainder.substring(0, endRel).trim();
  }
  return results;
}

/**
 * Batched simplification: simplify blocks from multiple pages in a single LLM call.
 * Groups blocks with sequential markers <<<1>>>...<<<END>>> (seqToLoc maps each
 * label back to its page/block) so the LLM processes all pages at once.
 */
/**
 * EN→CN pipeline (v4 architecture).
 *
 * Two LLM stages:
 *   Stage 1 (COMPOSE): tell the model to compose Chinese manga dialogue —
 *     not translate — using the concept map to lock in canonical Chinese
 *     terms for English concepts. No logit bias; temp 0.3 for variety.
 *   Stage 2 (STYLE REWRITE): rewrite Stage 1's Chinese in casual HSK 3
 *     manga register with few-shot examples showing translation → manga
 *     style transformations. Logit bias applied here. Preserves nouns.
 *
 * Then two deterministic passes:
 *   - KEEP repair: if Stage 2 dropped any concept-map Chinese term that
 *     Stage 1 had, splice it back at a numeric-measure anchor or fall
 *     back to Stage 1 for that block.
 *
 * Fragment rejoin (see fragmentRejoin) should be applied to region text
 * BEFORE calling this function.
 */

// Manga-style transformation examples — fabricated (not from any real
// chapter output) to avoid overfit. Show the model what "translation-style
// vs manga-style" looks like without teaching it specific chapter content.
const MANGA_STYLE_EXAMPLES: { formal: string; manga: string }[] = [
  { formal: '我认为这个方法可能是最好的选择。',      manga: '我觉得这样最好。' },
  { formal: '那个正在向我们走过来的人是谁？',        manga: '走过来的那个是谁？' },
  { formal: '这个消息被大家听到了。',              manga: '这消息大家都听到了。' },
  { formal: '因为下雨了，所以我们没有出去。',        manga: '下雨了，没出去。' },
  { formal: '我觉得我应该马上离开这里。',            manga: '得马上离开。' },
  { formal: '请你安静一点好吗？',                  manga: '小声点！' },
  { formal: '这真是令人难以相信！',                manga: '这太不敢相信了！' },
];

/**
 * KEEP repair: if any concept-map Chinese term was in Stage 1 but not in
 * Stage 2, either splice back at a [数字][量词] anchor or fall back to
 * Stage 1 for the whole block. Returns the repaired text and which action
 * fired (for stats logging).
 */
function repairKeepDrops(
  stage1Text: string,
  stage2Text: string,
  keepTerms: string[],
): { text: string; action: 'no-keep' | 'preserved' | 'spliced' | 'fallback-stage1'; dropped?: string[] } {
  const stage1Has = keepTerms.filter(k => stage1Text.includes(k));
  if (stage1Has.length === 0) return { text: stage2Text, action: 'no-keep' };
  const dropped = stage1Has.filter(k => !stage2Text.includes(k));
  if (dropped.length === 0) return { text: stage2Text, action: 'preserved' };

  let repaired = stage2Text;
  const stillDropped: string[] = [];
  for (const term of dropped) {
    // Look for [Arabic digits, optional commas][Chinese measure word] followed by `term` in Stage 1
    const anchorRe = /([0-9][0-9,]*)([个颗条只把件张块副支段道处场次])/g;
    let anchorInS1: { num: string; measure: string } | null = null;
    let m: RegExpExecArray | null;
    while ((m = anchorRe.exec(stage1Text)) !== null) {
      const afterAnchor = stage1Text.substring(m.index + m[0].length);
      if (afterAnchor.startsWith(term)) {
        anchorInS1 = { num: m[1], measure: m[2] };
        break;
      }
    }
    if (anchorInS1) {
      const targetRe = new RegExp(anchorInS1.num.replace(/[,.]/g, '\\$&') + anchorInS1.measure, 'g');
      const before = repaired;
      repaired = repaired.replace(targetRe, `${anchorInS1.num}${anchorInS1.measure}${term}`);
      if (repaired !== before) continue;
    }
    stillDropped.push(term);
  }
  if (stillDropped.length === 0) return { text: repaired, action: 'spliced' };
  // Fallback: use Stage 1 verbatim for this block. Loses Stage 2's rhythm
  // improvement, gains guaranteed concept preservation.
  return { text: stage1Text, action: 'fallback-stage1', dropped: stillDropped };
}

/**
 * Reassemble variant transliterations back to canonical. Chapter analysis
 * sometimes writes different transliterations across turns for the same
 * character; this collapses them to whatever the FINAL turn picked.
 * Currently a no-op — variant collection isn't threaded through the
 * streaming analyzer yet. Kept as a hook for future work.
 */
function normalizeVariants(text: string, variants?: Map<string, Set<string>>): string {
  if (!variants) return text;
  let out = text;
  for (const [canonical, variantSet] of variants) {
    for (const v of variantSet) {
      if (v === canonical || !v || v.length < 2) continue;
      if (out.includes(v)) out = out.split(v).join(canonical);
    }
  }
  return out;
}

/** Sentinel parser shared by Stages 1 and 2. Fills missing blocks with `（缺失）`. */
function parseEnSentinelResponse(
  content: string,
  seqToLoc: Map<number, { pageNum: number; blockIdx: number }>,
): Map<number, Map<number, string>> {
  const result = new Map<number, Map<number, string>>();
  for (const [, loc] of seqToLoc) {
    if (!result.has(loc.pageNum)) result.set(loc.pageNum, new Map());
  }
  let missing = 0;
  let empty = 0;
  for (const [seq, loc] of seqToLoc) {
    const startRe = new RegExp(`(?:^|\\n)<+${seq}>+(?:\\n|$)`);
    const startMatch = startRe.exec(content);
    if (!startMatch) {
      missing++;
      result.get(loc.pageNum)!.set(loc.blockIdx, '（缺失）');
      // DEBUG: dump surrounding context to understand what the LLM produced
      if (process.env.DEBUG_EMPTY_SENTINEL) {
        console.error(`[DEBUG_EMPTY_SENTINEL] MISSING sentinel <<<${seq}>>> for page ${loc.pageNum} block ${loc.blockIdx}`);
        // Look for near-matches: <<<seq>>> where seq is off-by-one, or with typos
        const nearMatch = content.match(new RegExp(`<+(${seq - 1}|${seq + 1})>+`, 'g'));
        if (nearMatch) console.error(`  Nearby sentinels found: ${nearMatch.slice(0, 3).join(', ')}`);
      }
      continue;
    }
    const afterStart = startMatch.index + startMatch[0].length;
    const remainder = content.substring(afterStart);
    const endMatch = remainder.match(/(?:^|\n)<+END>+(?:\n|$)/);
    const nextStartMatch = remainder.match(/(?:^|\n)<+\d+>+(?:\n|$)/);
    let endRel = -1;
    if (endMatch?.index !== undefined && nextStartMatch?.index !== undefined) endRel = Math.min(endMatch.index, nextStartMatch.index);
    else if (endMatch?.index !== undefined) endRel = endMatch.index;
    else if (nextStartMatch?.index !== undefined) endRel = nextStartMatch.index;
    if (endRel < 0) {
      missing++;
      result.get(loc.pageNum)!.set(loc.blockIdx, '（缺失）');
      continue;
    }
    const parsed = remainder.substring(0, endRel).trim();
    result.get(loc.pageNum)!.set(loc.blockIdx, parsed);
    if (parsed === '' && process.env.DEBUG_EMPTY_SENTINEL) {
      empty++;
      // Dump ±200 chars of context around the empty sentinel to see what the LLM actually emitted
      const startAbs = startMatch.index;
      const endAbs = startAbs + startMatch[0].length + endRel;
      const contextStart = Math.max(0, startAbs - 200);
      const contextEnd = Math.min(content.length, endAbs + 200);
      const before = content.substring(contextStart, startAbs);
      const inside = content.substring(startAbs, endAbs);
      const after = content.substring(endAbs, contextEnd);
      console.error(`[DEBUG_EMPTY_SENTINEL] EMPTY sentinel <<<${seq}>>> for page ${loc.pageNum} block ${loc.blockIdx}`);
      console.error(`  BEFORE: ...${JSON.stringify(before)}`);
      console.error(`  INSIDE: ${JSON.stringify(inside)}`);
      console.error(`  AFTER:  ${JSON.stringify(after)}...`);
    }
  }
  if (missing > 0) console.error(`[Simplifier] EN pipeline: ${missing}/${seqToLoc.size} blocks missing from response`);
  if (empty > 0) console.error(`[Simplifier] EN pipeline: ${empty}/${seqToLoc.size} blocks had EMPTY content (parsed to '')`);
  return result;
}

/**
 * Translation two-stage pipeline (EN/JA/KO → CN). Entry point when
 * simplifyPageBatch is called with a translation-source inputLanguage.
 * Returns Map<pageNum, Map<blockIdx, chineseText>>.
 */
export async function simplifyEnglishPageBatch(
  pageBlocks: Map<number, TextBlock[]>,
  level: HSKLevel,
  chapterContext: ChapterContext | undefined,
  plotSummary: string | undefined,
  temperature: number,
  sourceLanguage: InputLanguage = 'english',
): Promise<Map<number, Map<number, string>>> {
  const srcName = sourceLanguageName(sourceLanguage);
  // Build sentinel input. Apply fragmentRejoin per region so OCR splits
  // like "STINK ING" → "STINKING" before either stage sees the text.
  const blockEntries: string[] = [];
  const seqToLoc = new Map<number, { pageNum: number; blockIdx: number }>();
  let seq = 0;
  for (const [pageNum, blocks] of pageBlocks) {
    blocks.forEach((block, blockIdx) => {
      seq++;
      seqToLoc.set(seq, { pageNum, blockIdx });
      const rejoined = block.texts.map(t => fragmentRejoin(t));
      blockEntries.push(`<<<${seq}>>>\n${smartJoin(rejoined)}\n<<<END>>>`);
    });
  }
  if (blockEntries.length === 0) return new Map();

  // ── Stage 1: COMPOSE ────────────────────────────────────────────────
  const conceptMap = chapterContext?.concepts || [];
  const conceptLines = conceptMap.map(c => `  ${c.english} → ${c.chinese}`);
  const conceptSection = conceptLines.length > 0
    ? `\n\nUse these EXACT Chinese terms for these ${srcName} concepts (do not paraphrase):\n${conceptLines.join('\n')}\n\nWhen the ${srcName} mentions ANY of these, output MUST include the canonical Chinese.`
    : '';

  const charLine = (chapterContext?.characters || [])
    .map((c: any) => c.chineseName ? `${c.name} → ${c.chineseName} (${c.role || ''})` : `${c.name} (${c.role || ''})`)
    .join(', ');
  const settingLine = chapterContext?.setting ? `\nScene: ${chapterContext.setting}` : '';
  const plotLine = plotSummary ? `\nCurrent section: ${plotSummary}` : '';

  const stage1SystemPrompt = `You write Chinese manga dialogue for HSK ${level.replace('hsk', '')} readers.

Below are ${srcName} source lines from a manga scene. For each numbered block,
compose the Chinese dialogue a manga character would speak in that scene —
NOT a formal translation.

Style requirements:
- Natural spoken Chinese, colloquial and punchy
- HSK ${level.replace('hsk', '')} words, everyday vocabulary
- 儿子 not 之子; 第三个儿子 not 三子; 来 not 光临; 有 not 拿到

Content preservation:
- Preserve EVERY story-critical noun. Do not drop concepts.
- Preserve numeric formats exactly (24,476 stays 24,476).
- Preserve emotional punctuation (!! → ！！, ... → ……).

Character names: use the canonical transliterations below.${conceptSection}

Characters: ${charLine}${settingLine}${plotLine}

Copy each block label exactly.

Output format:
<<<1>>>
Chinese dialogue
<<<END>>>`;

  const sentinelInput = blockEntries.join('\n\n');
  const stage1UserPrompt = `Compose Chinese dialogue for each block:\n\n${sentinelInput}`;

  console.error(`[Simplifier] EN pipeline Stage 1 COMPOSE: ${blockEntries.length} blocks (no bias)...`);
  const t1 = Date.now();
  const stage1Content = await callLLMMultiTurn(
    [
      { role: 'system', content: stage1SystemPrompt },
      { role: 'user', content: stage1UserPrompt },
    ],
    temperature,
    240000,
    undefined,
    undefined,
    level,
    'chinese',
    false, // no logit bias in Stage 1 — needs freedom to translate
  );
  console.error(`[Simplifier] EN Stage 1 done in ${((Date.now() - t1) / 1000).toFixed(1)}s`);
  // DEBUG: dump raw LLM output to disk when DEBUG_EMPTY_SENTINEL is set
  if (process.env.DEBUG_EMPTY_SENTINEL) {
    const dumpPath = process.env.DEBUG_EMPTY_SENTINEL_FILE || 'outputs/debug-stage1-dump.txt';
    fs.appendFileSync(dumpPath, `\n===== Stage 1 output (${blockEntries.length} blocks, ${Date.now()}) =====\n${stage1Content}\n===== end Stage 1 =====\n`, 'utf-8');
  }
  const stage1Map = parseEnSentinelResponse(stage1Content, seqToLoc);

  // ── Stage 2: STYLE REWRITE ──────────────────────────────────────────
  const stage2BlockEntries: string[] = [];
  for (const [seqNum, loc] of seqToLoc) {
    const chinese = stage1Map.get(loc.pageNum)?.get(loc.blockIdx) ?? '（缺失）';
    stage2BlockEntries.push(`<<<${seqNum}>>>\n${chinese}\n<<<END>>>`);
  }

  const keepTerms = (chapterContext?.keep || []).slice();
  // Also include concept-map Chinese values (may overlap with keep; Set dedupes)
  for (const c of conceptMap) if (c.chinese) keepTerms.push(c.chinese);
  const uniqKeep = [...new Set(keepTerms)];
  const keepLine = uniqKeep.length > 0
    ? `\nPRESERVE LITERALLY (must appear verbatim if input has them):\n${uniqKeep.join('、')}\n`
    : '';

  const examplesBlock = MANGA_STYLE_EXAMPLES
    .map((ex, i) => `Example ${i + 1}:\nTranslation-style: ${ex.formal}\nManga-style:       ${ex.manga}`)
    .join('\n\n');

  const stage2SystemPrompt = `You rewrite Chinese manga dialogue in casual HSK ${level.replace('hsk', '')} register.

Each block contains Chinese text that may read as a translation — stiff word order, redundant subjects, formal phrasing. Rewrite as if a Chinese manga writer had written the block from scratch.

You MAY compress by:
- Dropping redundant subjects (我/你/他) when clear from context
- Dropping filler verbs (我觉得/我想/大概/其实)
- Combining short sentences with commas
- Reordering phrases for manga rhythm
- Adding manga particles (吧, 呢, 啊, 啦)
- Replacing formal 之所以/世世代代/庇佑/谢天谢地 with HSK ${level.replace('hsk', '')} equivalents

You MUST NOT change:
- Any noun — every noun in the input Chinese must appear in the output
- Any character or place name
- Numeric values or their measure words (24,476 颗 stays 24,476 颗 + noun)
- Negations (不/没/别 must be preserved)
- KEEP terms listed below
${keepLine}
Rewrite rules (always apply):
- 之子 → 的儿子; 三子/四子/次子 → 第三个/第四个/第二个儿子
- 拿到 → 得到/有; 光临 → 来; 之类 → 什么的
- 之所以 → 因为; 世世代代 → 好多年; 庇佑 → 保护
- 谢天谢地 → 太好了
- Avoid idioms above HSK ${level.replace('hsk', '')}

Preserve exactly:
- Numeric formats (24,476 stays 24,476)
- Punctuation style (！！ ？？ ……)

Here is what "translation-style vs manga-style" looks like — use as style reference only, do not copy content:

${examplesBlock}

Rewrite each block independently. Copy each label exactly.

Output format:
<<<1>>>
rewritten Chinese
<<<END>>>`;

  const stage2SentinelInput = stage2BlockEntries.join('\n\n');
  const stage2UserPrompt = `Rewrite each block in manga-style:\n\n${stage2SentinelInput}`;

  console.error(`[Simplifier] EN pipeline Stage 2 REWRITE: ${blockEntries.length} blocks (with bias)...`);
  const t2 = Date.now();
  // Marker + dump Stage 2 INPUT so we can compare LLM input/output
  if (process.env.DEBUG_EMPTY_SENTINEL) {
    const dumpPath = process.env.DEBUG_EMPTY_SENTINEL_FILE || 'outputs/debug-stage1-dump.txt';
    fs.appendFileSync(dumpPath, `\n===== Stage 2 INPUT (${blockEntries.length} blocks) =====\n${stage2SentinelInput}\n===== end Stage 2 INPUT =====\n`, 'utf-8');
  }
  const stage2Content = await callLLMMultiTurn(
    [
      { role: 'system', content: stage2SystemPrompt },
      { role: 'user', content: stage2UserPrompt },
    ],
    Math.max(0.1, temperature - 0.1), // slightly cooler for the constrained rewrite
    240000,
    undefined,
    undefined,
    level,
    'chinese',
    true, // apply logit bias — Stage 2 is the HSK-constrained pass
  );
  console.error(`[Simplifier] EN Stage 2 done in ${((Date.now() - t2) / 1000).toFixed(1)}s`);
  // Dump Stage 2 raw output for debugging misalignment
  if (process.env.DEBUG_EMPTY_SENTINEL) {
    const dumpPath = process.env.DEBUG_EMPTY_SENTINEL_FILE || 'outputs/debug-stage1-dump.txt';
    fs.appendFileSync(dumpPath, `\n===== Stage 2 OUTPUT (${blockEntries.length} blocks) =====\n${stage2Content}\n===== end Stage 2 OUTPUT =====\n\n`, 'utf-8');
  }
  const stage2Map = parseEnSentinelResponse(stage2Content, seqToLoc);

  // ── KEEP repair + variant normalization ─────────────────────────────
  const stats = { 'no-keep': 0, 'preserved': 0, 'spliced': 0, 'fallback-stage1': 0 };
  const fallbackBlocks: { pn: number; idx: number; dropped: string[] }[] = [];
  const result = new Map<number, Map<number, string>>();
  for (const [pn, blockMap] of stage2Map) {
    const pageResult = new Map<number, string>();
    for (const [idx, stage2Text] of blockMap) {
      const stage1Text = stage1Map.get(pn)?.get(idx) ?? '';
      const repaired = repairKeepDrops(stage1Text, stage2Text, uniqKeep);
      stats[repaired.action]++;
      if (repaired.action === 'fallback-stage1') fallbackBlocks.push({ pn, idx, dropped: repaired.dropped! });
      pageResult.set(idx, normalizeVariants(repaired.text));
    }
    result.set(pn, pageResult);
  }
  console.error(`[Simplifier] EN pipeline repair stats: no-keep=${stats['no-keep']} preserved=${stats.preserved} spliced=${stats.spliced} fallback=${stats['fallback-stage1']}`);
  if (fallbackBlocks.length > 0) {
    for (const b of fallbackBlocks.slice(0, 8)) {
      console.error(`  Stage-1 fallback: p${b.pn}.${b.idx} dropped [${b.dropped.join(', ')}]`);
    }
  }

  // Optional dump for A/B inspection (matches STAGE1_DUMP_FILE convention)
  const dumpFile = process.env.STAGE1_DUMP_FILE;
  if (dumpFile) {
    try {
      const lines: string[] = [];
      for (const [pageNum, blocks] of pageBlocks) {
        lines.push(`--- Page ${pageNum} ---`);
        blocks.forEach((block, blockIdx) => {
          const en = smartJoin(block.texts.map(t => fragmentRejoin(t)));
          const s1 = stage1Map.get(pageNum)?.get(blockIdx) ?? '（缺失）';
          const s2 = stage2Map.get(pageNum)?.get(blockIdx) ?? '（缺失）';
          const fin = result.get(pageNum)?.get(blockIdx) ?? '（缺失）';
          lines.push(`  EN:      ${en}`);
          lines.push(`  STAGE1:  ${s1}`);
          lines.push(`  STAGE2:  ${s2}`);
          lines.push(`  FINAL:   ${fin}`);
          lines.push('');
        });
      }
      fs.appendFileSync(dumpFile, lines.join('\n') + '\n', 'utf-8');
    } catch (e) {
      console.error(`[Simplifier] EN pipeline dump failed: ${e}`);
    }
  }

  return result;
}

export async function simplifyPageBatch(
  pageBlocks: Map<number, TextBlock[]>,  // pageNum → blocks for that page
  level: HSKLevel = 'hsk3',
  chapterContext?: ChapterContext,
  plotSummary?: string,
  inputLanguage: InputLanguage = 'chinese',
  temperature: number = 0.3,
  // Internal: the ORIGINAL input language before any recursion. The single-pass
  // EN flow re-enters with inputLanguage='chinese' to break the english-branch
  // recursion, but we want EN bias magnitudes preserved through. Callers
  // shouldn't pass this — it defaults to inputLanguage on the first call.
  originalInputLanguage?: InputLanguage,
): Promise<Map<number, Map<number, string>>> {
  const biasSourceLanguage = (originalInputLanguage || inputLanguage) as BiasInputLanguage;
  // EN input goes through the dedicated two-stage pipeline
  // (simplifyEnglishPageBatch). CN input continues below.
  if (isTranslationSource(inputLanguage)) {
    return simplifyEnglishPageBatch(pageBlocks, level, chapterContext, plotSummary, temperature, inputLanguage);
  }

  const levelNum = level === 'hsk1' ? 1 : level === 'hsk2' ? 2 : level === 'hsk3' ? 3
    : level === 'hsk4' ? 4 : level === 'hsk5' ? 5 : 6;

  // Detect whether the original input was English (single-pass EN recursed
  // into 'chinese' mode with biasSourceLanguage='english'). The Replace/flag
  // machinery and the "Keep English unchanged" rule are meaningless-to-harmful
  // for English input:
  //   - Flag computation on English text yields empty [Replace] (all English
  //     tokens pass isEnglish → marked isInHSK → not flagged), so the rule
  //     text describing [Replace] is a whole rule in the prompt with no
  //     referent. Wasted context tokens.
  //   - "Keep all English text unchanged" would tell the model NOT to
  //     translate — the exact opposite of what we want for EN input.
  // Both are gated off for EN via this flag.
  const isTranslationInput = isTranslationSource(originalInputLanguage);

  // Build sentinel-formatted input with per-block flagged words. Each block
  // gets a sequential integer label (<<<1>>>, <<<2>>>, …) — cheaper than a
  // page/block composite — and seqToLoc maps it back to its (pageNum, key).
  const blockEntries: string[] = [];
  const inputRecord = new Map<number, Record<string, string[]>>();
  const seqToLoc = new Map<number, { pageNum: number; key: string }>();
  const keepSet = chapterContext ? new Set(chapterContext.keep) : new Set<string>();
  const keepWords = chapterContext?.keep;
  let totalFlagged = 0;
  let seq = 0;

  for (const [pageNum, blocks] of pageBlocks) {
    const pageRecord: Record<string, string[]> = {};
    blocks.forEach((block, blockIdx) => {
      const key = String(blockIdx + 1);
      seq++;
      seqToLoc.set(seq, { pageNum, key });
      const text = smartJoin(block.texts);
      // Compute flagged words only for CN-originated input. For EN source,
      // getFlaggedWords on English text is either empty (all tokens caught
      // by isEnglish filter) or nonsensical (edge cases: URLs, decimals,
      // mixed-language OCR fragments). Skip entirely — matches the guard
      // that existed at 0f9d09f.
      let blockEntry = `<<<${seq}>>>\n${text}`;
      if (!isTranslationInput) {
        const blockFlagged = [...new Set(getFlaggedWords(text, level, keepWords))].filter(w => !isKeepOrSuffixed(w, keepSet));
        if (blockFlagged.length > 0) {
          blockEntry += `\n[Replace: ${blockFlagged.join('、')}]`;
          totalFlagged += blockFlagged.length;
        }
      }
      blockEntry += '\n<<<END>>>';
      blockEntries.push(blockEntry);
      pageRecord[key] = block.texts;
    });
    inputRecord.set(pageNum, pageRecord);
  }

  if (blockEntries.length === 0) {
    return new Map();
  }

  if (totalFlagged > 0) {
    console.error(`[Simplifier] Batch per-block flagging: ${totalFlagged} flagged words across ${blockEntries.length} blocks`);
  }

  const sentinelInput = blockEntries.join('\n\n');

  let flaggedInstruction = '';
  if (chapterContext && chapterContext.keep.length > 0) {
    flaggedInstruction += `\n- PROTECTED TERMS — do NOT simplify these, even if they are above HSK${levelNum}: ${chapterContext.keep.join('、')}`;
  }

  // Build context
  let contextSection = '';
  if (chapterContext) {
    contextSection = `\n\nStory context: ${chapterContext.setting || ''}`;
    if (plotSummary) {
      contextSection += `\nCurrent section: ${plotSummary}`;
    }
    const characters = chapterContext.characters
      .map(c => `${c.name} (${c.role})`).join(', ');
    if (characters) contextSection += `\nKey characters: ${characters}`;
  }

  // Static (rules + examples) first so it forms a stable cacheable prefix;
  // the variable per-chapter context goes at the end.
  //
  // "Keep all English text unchanged" is a CN-input rule — it preserves any
  // embedded English fragments (names, SFX) that appear inside otherwise-
  // Chinese source text. For EN source, we no longer reach this function
  // (EN routes to simplifyEnglishPageBatch), so isTranslationInput here means
  // "recursive re-entry from a legacy code path" — kept for backward
  // compatibility but shouldn't fire in practice.
  const englishRule = isTranslationInput ? '' : `- Keep all English text unchanged.`;

  // Replace rule + example: only for CN-source. For EN source, no [Replace: ...]
  // lines are injected per-block (see isTranslationInput guard above), so this
  // whole rule + its example set has no referent and just wastes context.
  const replaceRule = isTranslationInput ? '' : `- Each block may have a [Replace: ...] line listing words above HSK${levelNum}. EVERY word in that list MUST appear NOWHERE in your output for that block. If a direct synonym is awkward, restructure — split the sentence, change subject, drop redundancy. Returning the input unchanged is wrong.\n`;

  const outputInstruction = isTranslationInput
    ? `Copy each block label exactly (format: <<<N>>>), then the translated Chinese text, then <<<END>>> on its own line. Example:

<<<1>>>
translated chinese text
<<<END>>>`
    : `Copy each block label exactly (format: <<<N>>>). Output ONLY the simplified text, not the [Replace: ...] line.`;

  const taskLine = isTranslationInput
    ? `Translate each block into HSK${levelNum} Chinese.\nRead all blocks for context. Translate each block separately — do not combine or split blocks.`
    : `Simplify each block to HSK${levelNum} Chinese.\nRead all blocks for context. Simplify each block separately — do not combine or split blocks.`;

  const headerLine = isTranslationInput
    ? `You translate ${sourceLanguageName(originalInputLanguage)} manga dialogue into Chinese.`
    : `You simplify Chinese manhua text.`;

  const examplesBlock = isTranslationInput ? '' : `\n
Examples — note how each Output replaces flagged words and restructures when a direct swap isn't natural:

Input:
<<<1>>>
她感到非常惭愧。
[Replace: 惭愧]
<<<END>>>
`;

  // Layout: contextSection follows the header up front (primes the model with
  // story + character context BEFORE the rules). flaggedInstruction sits at
  // the tail of the Rules block as an additional bullet, alongside the other
  // constraints, giving PROTECTED TERMS equal weight to other rules.
  const systemPrompt = `${noThinkPrefix()}${headerLine}${contextSection}

${taskLine}

Rules:
${replaceRule}- Write in subject-verb-object order. Every sentence needs an explicit subject — never rely on context for who is doing what. No topic-drop.
- Keep sentences short: one clause per sentence. If the input has two clauses joined by 而/但是/因为/所以/虽然, split into two sentences.
- Replace 成语 (4-character idioms) with a plain description. Any 成语 not in HSK${levelNum} must be expanded into literal language.
- If you genuinely cannot remove a flagged word without losing meaning (only true for technical terms with no synonym), you may keep it — but restructure the rest of the sentence to HSK${levelNum}.
- Preserve the original meaning, emotion, and tone. Do NOT weaken strong statements or remove metaphors.
- Keep character names unchanged.
${englishRule}${flaggedInstruction}

${outputInstruction}${isTranslationInput ? '' : `

Examples — note how each Output replaces flagged words and restructures when a direct swap isn't natural:

Input:
<<<1>>>
她感到非常惭愧。
[Replace: 惭愧]
<<<END>>>

<<<2>>>
我无法原谅他的背叛。
[Replace: 无法、原谅、背叛]
<<<END>>>

<<<3>>>
他气得浑身发抖。
[Replace: 浑身、发抖]
<<<END>>>

<<<4>>>
反正我也不打算玩太久，就随便选一个吧。
[Replace: 反正、随便]
<<<END>>>

Output:
<<<1>>>
她觉得很不好意思。
<<<END>>>

<<<2>>>
他做了对不起我的事，我不能忘。
<<<END>>>

<<<3>>>
他很生气，身体都在动。
<<<END>>>

<<<4>>>
我本来就不会玩太久，选一个就行。
<<<END>>>`}`;

  // Translation task for EN source; simplification task for CN source.
  const userPrompt = isTranslationInput
    ? `Translate to HSK${levelNum} Chinese:\n\n${sentinelInput}`
    : `Simplify to HSK${levelNum}:\n\n${sentinelInput}`;

  console.error(`[Simplifier] Batch simplifying ${blockEntries.length} blocks across ${pageBlocks.size} pages (temp=${temperature})...`);
  const content = await callQwenChat(systemPrompt, userPrompt, temperature, level, inputLanguage, chapterContext?.keepTokenIds, biasSourceLanguage);
  const parsed = parseBatchedSentinelResponse(content, inputRecord, seqToLoc);

  // Normalize punctuation, restore swapped KEEP terms, convert to Map
  const keepTermsList = chapterContext ? chapterContext.keep : [];
  const resultMap = new Map<number, Map<number, string>>();
  for (const [pageNum, pageResult] of parsed) {
    const blockMap = new Map<number, string>();
    const origBlocks = inputRecord.get(pageNum);
    for (const key in pageResult) {
      let text = normalizePunctuation(pageResult[key]);
      // Re-apply English-spacing fix on simplifier output. The LLM sometimes
      // normalizes "DUNGEON AND STONE" → "DUNGEONANDSTONE" because chapter
      // analysis put the no-space form into the KEEP list, and KEEP terms get
      // protected. Running fixEnglishSpacing here recovers the spaces using
      // the dictionary-DP splitter — same tool we use at OCR time.
      text = fixEnglishSpacing(text);
      // Restore any KEEP terms the LLM swapped with different KEEP terms
      if (origBlocks && keepTermsList.length > 0) {
        const origText = smartJoin(origBlocks[key] || []);
        for (const term of keepTermsList) {
          if (origText.includes(term) && !text.includes(term)) {
            // Find which wrong KEEP term replaced it
            for (const wrongTerm of keepTermsList) {
              if (wrongTerm !== term && text.includes(wrongTerm) && !origText.includes(wrongTerm)) {
                console.error(`[Simplifier] Restoring swapped KEEP term: "${wrongTerm}" → "${term}" in P${pageNum}_B${key}`);
                text = text.replace(wrongTerm, term);
                break;
              }
            }
          }
        }
      }
      blockMap.set(parseInt(key) - 1, text);
    }
    resultMap.set(pageNum, blockMap);
  }

  return resultMap;
}

/**
 * Get the plot summary for a given page number from chapter context
 */
function getPlotForPage(chapterContext: ChapterContext, pageNum: number): string {
  for (const section of chapterContext.plot) {
    const match = section.pages.match(/(\d+)-(\d+)/);
    if (match) {
      const start = parseInt(match[1]);
      const end = parseInt(match[2]);
      if (pageNum >= start && pageNum <= end) {
        return `Pages ${section.pages}: ${section.summary}`;
      }
    }
  }
  return '';
}

/**
 * Pass 1: HSK Semantic Simplifier
 * Simplifies each block to a single string with no line constraints.
 * Uses sentinel block markers (<<<B1>>>...<<<END>>>) instead of JSON
 * to prevent token-level continuation bias across block boundaries.
 *
 * When chapterContext is provided (Approach 4), injects story context,
 * characters, current plot section, and PROTECTED TERMS into the prompt.
 */
async function callQwenPass1(
  blocks: Record<string, string[]>,
  level: HSKLevel,
  _flaggedWords?: string[],
  chapterContext?: ChapterContext,
  pageNum?: number,
  inputLanguage: InputLanguage = 'chinese'
): Promise<Record<string, string>> {
  const levelNum = level === 'hsk1' ? 1 : level === 'hsk2' ? 2 : level === 'hsk3' ? 3
    : level === 'hsk4' ? 4 : level === 'hsk5' ? 5 : 6;

  // Build sentinel-formatted input with per-block flagged words
  const keepSet = chapterContext ? new Set(chapterContext.keep) : new Set<string>();
  const keepWords = chapterContext?.keep;
  const blockEntries: string[] = [];
  for (const key in blocks) {
    const text = smartJoin(blocks[key]);
    let entry = `<<<B${key}>>>\n${text}`;
    if (!isTranslationSource(inputLanguage)) {
      const blockFlagged = [...new Set(getFlaggedWords(text, level, keepWords))].filter(w => !isKeepOrSuffixed(w, keepSet));
      if (blockFlagged.length > 0) {
        entry += `\n[Replace: ${blockFlagged.join('、')}]`;
      }
    }
    entry += '\n<<<END>>>';
    blockEntries.push(entry);
  }
  const sentinelInput = blockEntries.join('\n\n');

  let systemPrompt: string;

  if (isTranslationSource(inputLanguage)) {
    // English → Chinese translation mode
    let contextSection = '';
    if (chapterContext) {
      contextSection = `\n\nStory context: ${chapterContext.setting || ''}`;
      if (pageNum !== undefined) {
        const plotContext = getPlotForPage(chapterContext, pageNum);
        if (plotContext) contextSection += `\nCurrent section: ${plotContext}`;
      }
      const characters = chapterContext.characters
        .map((c: any) => c.chineseName ? `${c.name} → ${c.chineseName} (${c.role})` : `${c.name} (${c.role})`).join(', ');
      if (characters) contextSection += `\nCharacter name translations: ${characters}`;
    }

    let keepInstruction = '';
    if (chapterContext && chapterContext.keep.length > 0) {
      keepInstruction = `\n- Use these Chinese terms consistently: ${chapterContext.keep.join('、')}`;
    }

    systemPrompt = `${noThinkPrefix()}You translate ${sourceLanguageName(inputLanguage)} manga dialogue into Chinese.${contextSection}

Translate each block into Chinese using only HSK${levelNum} vocabulary.
Read all blocks for context. Translate each block separately — do not combine or split blocks.

Rules:
- Use simple, natural Chinese that an HSK${levelNum} learner can read
- Preserve the tone and emotion of the original (excitement, anger, sarcasm, etc.)
- Keep character names consistent with the translations provided above
- For concepts with no simple Chinese equivalent, use the simplest possible expression
- Preserve punctuation style (!! → ！！, ... → ……, ? → ？)
- Do NOT add explanations or notes — output only the translated Chinese text
- Do NOT use idioms (成语) unless they are in HSK${levelNum}${keepInstruction}

Copy each block label exactly. Output format:

<<<B1>>>
translated chinese text
<<<END>>>`;
  } else {
    // Chinese simplification mode
    let keepInstruction = '';
    if (chapterContext && chapterContext.keep.length > 0) {
      keepInstruction = `\n- PROTECTED TERMS — do NOT simplify these, even if they are above HSK${levelNum}: ${chapterContext.keep.join('、')}`;
    }

    // Build context section from chapter analysis
    let contextSection = '';
    if (chapterContext) {
      contextSection = `\n\nStory context: ${chapterContext.setting || ''}`;
      if (pageNum !== undefined) {
        const plotContext = getPlotForPage(chapterContext, pageNum);
        if (plotContext) contextSection += `\nCurrent section: ${plotContext}`;
      }
      const characters = chapterContext.characters
        .map(c => `${c.name} (${c.role})`).join(', ');
      if (characters) contextSection += `\nKey characters: ${characters}`;
    }

    systemPrompt = `${noThinkPrefix()}You simplify Chinese manhua text.${contextSection}

Simplify each block to HSK${levelNum} Chinese.
Read all blocks for context. Simplify each block separately — do not combine or split blocks.

Rules:
- Each block may have a [Replace: ...] line listing words above HSK${levelNum}. EVERY word in that list MUST appear NOWHERE in your output for that block. If a direct synonym is awkward, restructure — split the sentence, change subject, drop redundancy. Returning the input unchanged is wrong.
- Write in subject-verb-object order. Every sentence needs an explicit subject — never rely on context for who is doing what. No topic-drop.
- Keep sentences short: one clause per sentence. If the input has two clauses joined by 而/但是/因为/所以/虽然, split into two sentences.
- Replace 成语 (4-character idioms) with a plain description. Any 成语 not in HSK${levelNum} must be expanded into literal language.
- If you genuinely cannot remove a flagged word without losing meaning (only true for technical terms with no synonym), you may keep it — but restructure the rest of the sentence to HSK${levelNum}.
- Preserve the original meaning, emotion, and tone. Do NOT weaken strong statements or remove metaphors.
- Keep character names unchanged.
- Keep all English text unchanged.${keepInstruction}

Copy each block label exactly. Output ONLY the simplified text, not the [Replace: ...] line.

Examples — note how each Output replaces flagged words and restructures when a direct swap isn't natural:

Input:
<<<B1>>>
她感到非常惭愧。
[Replace: 惭愧]
<<<END>>>

<<<B2>>>
我无法原谅他的背叛。
[Replace: 无法、原谅、背叛]
<<<END>>>

<<<B3>>>
他气得浑身发抖。
[Replace: 浑身、发抖]
<<<END>>>

<<<B4>>>
反正我也不打算玩太久，就随便选一个吧。
[Replace: 反正、随便]
<<<END>>>

Output:
<<<B1>>>
她觉得很不好意思。
<<<END>>>

<<<B2>>>
他做了对不起我的事，我不能忘。
<<<END>>>

<<<B3>>>
他很生气，身体都在动。
<<<END>>>

<<<B4>>>
我本来就不会玩太久，选一个就行。
<<<END>>>`;
  }

  const userPrompt = isTranslationSource(inputLanguage)
    ? `Translate to HSK${levelNum} Chinese:\n\n${sentinelInput}`
    : `Simplify to HSK${levelNum}:\n\n${sentinelInput}`;

  const content = await callQwenChat(systemPrompt, userPrompt, 0.3, level, inputLanguage);
  const result = parseSentinelResponse(content, blocks);
  // Normalize punctuation and restore swapped KEEP terms for Chinese output
  if (!isTranslationSource(inputLanguage)) {
    const keepTermsList = chapterContext ? chapterContext.keep : [];
    for (const key in result) {
      result[key] = normalizePunctuation(result[key]);
      // Restore any KEEP terms the LLM swapped
      if (keepTermsList.length > 0) {
        const origText = smartJoin(blocks[key]);
        for (const term of keepTermsList) {
          if (origText.includes(term) && !result[key].includes(term)) {
            for (const wrongTerm of keepTermsList) {
              if (wrongTerm !== term && result[key].includes(wrongTerm) && !origText.includes(wrongTerm)) {
                console.error(`[Simplifier] Restoring swapped KEEP term: "${wrongTerm}" → "${term}" in B${key}`);
                result[key] = result[key].replace(wrongTerm, term);
                break;
              }
            }
          }
        }
      }
    }
  }
  return result;
}

/**
 * Code-based post-processing for structural rules.
 * More reliable than LLM for deterministic rules.
 */
function applyPostProcessing(simplified: string, original: string): string {
  // Strip added punctuation if original had none
  const endsWithPunct = /[。！？，、；：…—.!?]$/;
  if (!endsWithPunct.test(original) && endsWithPunct.test(simplified)) {
    simplified = simplified.replace(/[。！？，、；：…—.!?]+$/, '');
  }

  return simplified;
}

/**
 * KEEP enforcement: re-run blocks where KEEP terms were accidentally simplified.
 * Chains fixes so multiple violations in the same block accumulate correctly.
 *
 * @param blocks Original text blocks
 * @param results Current simplification results (mutated in place)
 * @param keepList KEEP terms to enforce
 * @returns Updated results map
 */
/**
 * Restore English text that the LLM may have translated to Chinese.
 * For each block, extracts English segments from the original and checks
 * if they survive in the simplified output. If not, calls the LLM to fix.
 */
async function restoreEnglishText(
  blocks: TextBlock[],
  results: Map<number, string>
): Promise<Map<number, string>> {
  const details: Array<{ blockIdx: number; english: string; before: string; after: string }> = [];

  for (let blockIdx = 0; blockIdx < blocks.length; blockIdx++) {
    const originalText = smartJoin(blocks[blockIdx].texts);
    const simplified = results.get(blockIdx);
    if (!simplified) continue;

    // Extract English segments (2+ consecutive letters, may include spaces between words)
    const englishSegments = originalText.match(/[A-Za-z][A-Za-z0-9\s]*[A-Za-z0-9]/g) || [];
    const missing = englishSegments.filter(seg => seg.trim().length >= 2 && !simplified.includes(seg));
    if (missing.length === 0) continue;

    // Chain fixes for multiple missing segments
    for (const segment of missing) {
      const current = results.get(blockIdx)!;
      if (current.includes(segment)) continue; // Fixed by prior iteration

      const fixPrompt = `${noThinkPrefix()}The text below was simplified, but the English word "${segment}" must not be translated.\n\nOriginal: ${originalText}\nSimplified: ${current}\n\nOutput the simplified text with "${segment}" restored exactly. Output ONLY the corrected text, nothing else.`;

      try {
        const fixed = await callQwenChat(fixPrompt, current, 0.1);
        const fixedClean = fixed.trim().replace(/^["']|["']$/g, '');
        if (fixedClean.includes(segment)) {
          details.push({ blockIdx, english: segment, before: current, after: fixedClean });
          results.set(blockIdx, fixedClean);
        } else {
          details.push({ blockIdx, english: segment, before: current, after: `(fix failed: ${fixedClean.substring(0, 50)})` });
        }
      } catch (e: any) {
        details.push({ blockIdx, english: segment, before: current, after: `(error: ${e.message})` });
      }
    }
  }

  if (details.length > 0) {
    const fixes = details.filter(d => !d.after.startsWith('(')).length;
    console.error(`[Simplifier] English restoration: ${details.length} violation(s) found, ${fixes} fixed`);
    for (const d of details) {
      console.error(`  Block ${d.blockIdx}: "${d.english}" — ${d.before.substring(0, 40)}... → ${d.after.substring(0, 40)}...`);
    }
  }

  return results;
}

async function enforceKeepTerms(
  blocks: TextBlock[],
  results: Map<number, string>,
  keepList: string[]
): Promise<Map<number, string>> {
  if (keepList.length === 0) return results;

  const details: Array<{ blockIdx: number; term: string; before: string; after: string }> = [];

  for (let blockIdx = 0; blockIdx < blocks.length; blockIdx++) {
    const originalText = smartJoin(blocks[blockIdx].texts);
    const simplified = results.get(blockIdx);
    if (!simplified) continue;

    // Check each KEEP term — chain fixes so multiple violations accumulate
    for (const term of keepList) {
      if (term.length < 2) continue;
      const current = results.get(blockIdx)!; // re-read after each prior fix
      if (originalText.includes(term) && !current.includes(term)) {
        // Violation: term was in original but got simplified away
        const fixPrompt = `${noThinkPrefix()}The following sentence was simplified but the term "${term}" should have been kept exactly as written. Replace whatever was substituted for "${term}" and output ONLY the corrected sentence. Do not add any explanation.`;

        try {
          const fixed = await callQwenChat(fixPrompt, current, 0.1);
          const fixedClean = fixed.trim().replace(/^["']|["']$/g, '');
          if (fixedClean.includes(term)) {
            details.push({ blockIdx, term, before: current, after: fixedClean });
            results.set(blockIdx, fixedClean);
          } else {
            details.push({ blockIdx, term, before: current, after: `(fix failed: ${fixedClean.substring(0, 50)})` });
          }
        } catch (e: any) {
          details.push({ blockIdx, term, before: current, after: `(error: ${e.message})` });
        }
      }
    }
  }

  if (details.length > 0) {
    const fixes = details.filter(d => !d.after.startsWith('(')).length;
    console.error(`[Simplifier] KEEP enforcement: ${details.length} violation(s) found, ${fixes} fixed`);
    for (const d of details) {
      console.error(`  Block ${d.blockIdx}: "${d.term}" — ${d.before.substring(0, 40)}... → ${d.after.substring(0, 40)}...`);
    }
  }

  return results;
}

/**
 * Simplify text blocks (grouped speech bubble lines) with full context.
 * When chapterContext is provided, uses Approach 4: story context + KEEP enforcement.
 *
 * @param blocks Array of text blocks to simplify
 * @param level Target HSK level
 * @param chapterContext Optional chapter-level context from analyzeChapter()
 * @param pageNum Optional page number (for plot section lookup)
 * @returns Map of block index to single simplified string per block
 */
export async function simplifyBlocks(
  blocks: TextBlock[],
  level: HSKLevel = 'hsk2',
  chapterContext?: ChapterContext,
  pageNum?: number,
  inputLanguage: InputLanguage = 'chinese'
): Promise<Map<number, string>> {
  const provider = detectProvider();

  if (blocks.length === 0) {
    return new Map<number, string>();
  }

  // For Anthropic using its native SDK, fall back to batch approach
  if (provider === 'anthropic') {
    const allTexts = blocks.flatMap(b => b.texts);
    const results = await simplifyTextBatch(allTexts, level);

    const resultMap = new Map<number, string>();
    let offset = 0;
    for (let i = 0; i < blocks.length; i++) {
      const blockSize = blocks[i].texts.length;
      const blockResults = results.slice(offset, offset + blockSize).map(r => r.simplified);
      resultMap.set(i, smartJoin(blockResults));
      offset += blockSize;
    }
    return resultMap;
  }

  // Build input JSON: { "1": ["line1", "line2"], "2": ["line1"] }
  const inputObject: Record<string, string[]> = {};
  blocks.forEach((block, i) => {
    inputObject[String(i + 1)] = block.texts;
  });

  // Pre-detection: find words above target level in original text
  // Skip for English input — input is English, not Chinese
  const allOriginalText = blocks.map(b => smartJoin(b.texts)).join('');
  const flaggedWords = !isTranslationSource(inputLanguage) ? getFlaggedWords(allOriginalText, level, chapterContext?.keep) : [];
  if (flaggedWords.length > 0) {
    const unique = [...new Set(flaggedWords)];
    console.error(`[Simplifier] Pre-detection: ${unique.length} flagged words above ${level}: ${unique.join('、')}`);
  }

  try {
    // Pass 1: Semantic simplification with flagged words hint
    console.error(`[Simplifier] Simplifying ${blocks.length} blocks...`);
    const uniqueFlagged = [...new Set(flaggedWords)];
    const pass1Result = await callQwenPass1(
      inputObject, level,
      uniqueFlagged.length > 0 ? uniqueFlagged : undefined,
      chapterContext, pageNum, inputLanguage
    );
    console.error(`[Simplifier] ${isTranslationSource(inputLanguage) ? 'Translation' : 'Simplification'} complete`);

    // Debug: write original → simplified for each block to a file
    if (process.env.DEBUG_SIMPLIFY) {
      const fs = await import('fs');
      const debugPath = 'debug-simplify-only.txt';
      const lines = blocks.map((block, i) => {
        const key = String(i + 1);
        const original = smartJoin(block.texts);
        const simplified = pass1Result[key] || '(missing)';
        return `Block ${key}: "${original}" -> "${simplified}"`;
      });
      fs.appendFileSync(debugPath, lines.join('\n') + '\n---\n', 'utf8');
      console.error(`[Simplifier] Debug: wrote ${lines.length} blocks to ${debugPath}`);
    }

    // Post-processing and validation
    const resultMap = new Map<number, string>();
    // const blocksToRetry = new Map<string, string[]>(); // Retry disabled

    blocks.forEach((block, i) => {
      const key = String(i + 1);
      let simplified = pass1Result[key] || smartJoin(block.texts);
      const originalText = smartJoin(block.texts);
      simplified = applyPostProcessing(simplified, originalText);

      // Length validation: reject suspiciously short output (likely block spillover)
      // For English→Chinese translation, Chinese text is naturally much shorter than English,
      // so use a lower threshold (15% instead of 40%)
      // Translation output (CN) is naturally much shorter than EN/JA/KO source
      const minLengthRatio = isTranslationSource(inputLanguage) ? 0.15 : 0.4;
      if (simplified.length < originalText.length * minLengthRatio) {
        console.error(`[Simplifier] Block ${key} output too short (${simplified.length}/${originalText.length} chars, min ${(minLengthRatio * 100).toFixed(0)}%), using original`);
        simplified = originalText;
      }

      resultMap.set(i, simplified);

      // Post-validation: check if output still has above-level words
      const outputFlagged = getFlaggedWords(simplified, level, chapterContext?.keep);
      if (outputFlagged.length > 0) {
        const unique = [...new Set(outputFlagged)];
        console.error(`[Simplifier] Block ${key} still has ${unique.length} flagged words: ${unique.join('、')}`);
        // blocksToRetry.set(key, unique);  // Retry disabled — see Step 3
      }
    });

    // KEEP enforcement: fix blocks where KEEP terms were accidentally simplified
    if (chapterContext && chapterContext.keep.length > 0) {
      await enforceKeepTerms(blocks, resultMap, chapterContext.keep);
    }

    // English preservation: restore English text that the LLM translated to Chinese
    // Skip for translation input — output should be all Chinese
    if (!isTranslationSource(inputLanguage)) {
      await restoreEnglishText(blocks, resultMap);
    }

    return resultMap;
  } catch (e) {
    console.error('Simplification failed, using originals:', e);
    const resultMap = new Map<number, string>();
    blocks.forEach((block, i) => {
      resultMap.set(i, smartJoin(block.texts));
    });
    return resultMap;
  }
}

/**
 * Multi-level simplification: simplifies blocks at multiple HSK levels
 * (and optionally English) in a single LLM call.
 * Returns Map<blockIdx, Map<levelKey, text>> where levelKey is e.g. "hsk2", "hsk3", "english"
 */
export async function simplifyBlocksMultiLevel(
  blocks: TextBlock[],
  levels: HSKLevel[],
  includeEnglish: boolean = false,
  chapterContext?: ChapterContext,
  pageNum?: number,
  inputLanguage: InputLanguage = 'chinese'
): Promise<Map<number, Map<string, string>>> {
  // For English→Chinese translation or single level without English, delegate to simplifyBlocks
  if ((levels.length === 1 && !includeEnglish) || isTranslationSource(inputLanguage)) {
    const targetLevel = levels[0];
    const singleResult = await simplifyBlocks(blocks, targetLevel, chapterContext, pageNum, inputLanguage);
    const multiResult = new Map<number, Map<string, string>>();
    singleResult.forEach((text, blockIdx) => {
      multiResult.set(blockIdx, new Map([[levels[0], text]]));
    });
    return multiResult;
  }

  if (blocks.length === 0) {
    return new Map();
  }

  // Build sentinel input
  const inputObject: Record<string, string[]> = {};
  blocks.forEach((block, i) => {
    inputObject[String(i + 1)] = block.texts;
  });

  const blockEntries: string[] = [];
  for (const key in inputObject) {
    const text = smartJoin(inputObject[key]);
    blockEntries.push(`<<<B${key}>>>\n${text}\n<<<END>>>`);
  }
  const sentinelInput = blockEntries.join('\n\n');

  // Build output keys
  const outputKeys: string[] = [...levels];
  if (includeEnglish) outputKeys.push('english');

  // Build level descriptions
  const levelDescriptions = levels.map(l => {
    const num = l === 'hsk1' ? 1 : l === 'hsk2' ? 2 : l === 'hsk3' ? 3 : 4;
    return `HSK${num}`;
  });

  // Pre-detection with lowest level
  const lowestLevel = levels.reduce((a, b) => {
    const aNum = a === 'hsk1' ? 1 : a === 'hsk2' ? 2 : a === 'hsk3' ? 3 : 4;
    const bNum = b === 'hsk1' ? 1 : b === 'hsk2' ? 2 : b === 'hsk3' ? 3 : 4;
    return aNum < bNum ? a : b;
  });
  const allOriginalText = blocks.map(b => smartJoin(b.texts)).join('');
  const flaggedWords = getFlaggedWords(allOriginalText, lowestLevel, chapterContext?.keep);
  if (flaggedWords.length > 0) {
    const unique = [...new Set(flaggedWords)];
    console.error(`[Simplifier] Pre-detection: ${unique.length} flagged words above ${lowestLevel}: ${unique.join('、')}`);
  }

  // Build context section
  const keepSet = chapterContext ? new Set(chapterContext.keep) : new Set<string>();
  let contextSection = '';
  if (chapterContext) {
    contextSection = `\n\nStory context: ${chapterContext.setting || ''}`;
    if (pageNum !== undefined) {
      const plotContext = getPlotForPage(chapterContext, pageNum);
      if (plotContext) contextSection += `\nCurrent section: ${plotContext}`;
    }
    const characters = chapterContext.characters
      .map(c => `${c.name} (${c.role})`).join(', ');
    if (characters) contextSection += `\nKey characters: ${characters}`;
  }

  let flaggedInstruction = '';
  const filteredFlagged = chapterContext && flaggedWords
    ? flaggedWords.filter(w => !keepSet.has(w))
    : flaggedWords;
  if (filteredFlagged && filteredFlagged.length > 0) {
    const unique = [...new Set(filteredFlagged)];
    flaggedInstruction = `\n- These words need simplification: ${unique.join('、')}`;
  }
  if (chapterContext && chapterContext.keep.length > 0) {
    flaggedInstruction += `\n- PROTECTED TERMS — do NOT simplify these: ${chapterContext.keep.join('、')}`;
  }

  const levelList = levelDescriptions.join(' and ');

  // Build prompt with a concrete example tailored to the exact output keys
  // Each combination gets its own example so the LLM sees the exact pattern
  const exampleSentences = [
    { cn: '她感到非常惭愧。', hsk1: '她觉得很不好。', hsk2: '她觉得很不好意思。', hsk3: '她觉得很不好意思。', hsk4: '她感到很惭愧。', hsk5: '她感到非常不好意思。', hsk6: '她感到非常不好意思。', en: 'She felt very ashamed.' },
    { cn: '我无法原谅他的背叛。', hsk1: '他骗了我，我不能不生气。', hsk2: '他骗了我，我不能原谅他。', hsk3: '我不能原谅他的背叛。', hsk4: '我无法原谅他的背叛。', hsk5: '我无法原谅他做的事。', hsk6: '我无法原谅他做的事。', en: "I can't forgive his betrayal." },
  ];

  // Build example input (2 blocks)
  const exampleInput = exampleSentences.map((s, i) =>
    `<<<B${i + 1}>>>\n${s.cn}\n<<<END>>>`
  ).join('\n\n');

  // Build example output (2 blocks × all output keys)
  const exampleOutput = exampleSentences.map((s, i) =>
    outputKeys.map(key => {
      const levelMap: Record<string, string> = { hsk1: s.hsk1, hsk2: s.hsk2, hsk3: s.hsk3, hsk4: s.hsk4, hsk5: s.hsk5, hsk6: s.hsk6, english: s.en };
      return `<<<B${i + 1}:${key}>>>\n${levelMap[key] || s.cn}\n<<<END>>>`;
    }).join('\n\n')
  ).join('\n\n');

  let taskDescription: string;
  if (includeEnglish && levels.length > 1) {
    taskDescription = `For each block, provide simplified versions at ${levelList} levels, plus an English translation.`;
  } else if (includeEnglish) {
    taskDescription = `For each block, provide a simplified version at ${levelList} level and an English translation.`;
  } else {
    taskDescription = `For each block, provide simplified versions at ${levelList} levels.`;
  }

  const systemPrompt = `${noThinkPrefix()}You simplify Chinese manhua text.${contextSection}

${taskDescription}
Read all blocks for context. Simplify each block separately — do not combine or split blocks.
You MUST output ALL blocks and ALL levels. Do not skip any block.

Rules:
- Only replace words that are above the target level. If a sentence is already at level, keep it unchanged.
- Preserve the original meaning, emotion, and natural tone.
- Keep character names unchanged.
- Keep all English text in the input unchanged. Do NOT introduce any new English words in Chinese output.
- Do NOT use idioms (成语) unless they are within the target level.
- In English translations, keep PROTECTED TERMS in their original Chinese form (do not translate them).${flaggedInstruction}

Copy each block label exactly. Output every B# for every level.

Example:
Input:
${exampleInput}

Output:
${exampleOutput}

Input:`;

  const userPrompt = sentinelInput;

  try {
    console.error(`[Simplifier] Multi-level simplifying ${blocks.length} blocks at ${outputKeys.join(', ')}...`);
    // Multi-level produces several HSK outputs in one call. Bias toward the
    // HIGHEST requested level (largest vocab) so the LLM has the full
    // dictionary it needs for all output levels — per-level constraints are
    // enforced via prompt rules, not bias.
    const highestLevel = levels.reduce<HSKLevel>((acc, l) => (l > acc ? l : acc), 'hsk1');
    const content = await callQwenChat(systemPrompt, userPrompt, 0.3, highestLevel);
    console.error(`[Simplifier] Multi-level simplification complete`);

    // Parse multi-level sentinel response
    const multiResult = new Map<number, Map<string, string>>();
    const missingEnglishBlocks: number[] = [];

    blocks.forEach((block, i) => {
      const key = String(i + 1);
      const originalText = smartJoin(block.texts);
      const levelMap = new Map<string, string>();

      for (const levelKey of outputKeys) {
        const startTag = `<<<B${key}:${levelKey}>>>`;
        const endTag = `<<<END>>>`;
        const startIdx = content.indexOf(startTag);
        if (startIdx === -1) {
          console.error(`[Simplifier] Missing ${startTag}, using original`);
          levelMap.set(levelKey, originalText);
          if (levelKey === 'english') missingEnglishBlocks.push(i);
          continue;
        }
        const afterStart = startIdx + startTag.length;
        const endIdx = content.indexOf(endTag, afterStart);
        if (endIdx === -1) {
          console.error(`[Simplifier] Missing END tag for ${startTag}, using original`);
          levelMap.set(levelKey, originalText);
          if (levelKey === 'english') missingEnglishBlocks.push(i);
          continue;
        }
        let text = content.substring(afterStart, endIdx).trim();
        if (levelKey !== 'english') {
          text = applyPostProcessing(text, originalText);
          if (text.length < originalText.length * 0.4) {
            console.error(`[Simplifier] Block ${key}:${levelKey} too short, using original`);
            text = originalText;
          }
        }
        levelMap.set(levelKey, text);
      }

      multiResult.set(i, levelMap);
    });

    // Retry missing English translations with a focused translate-only call
    if (includeEnglish && missingEnglishBlocks.length > 0) {
      console.error(`[Simplifier] Retrying English for ${missingEnglishBlocks.length} block(s): ${missingEnglishBlocks.map(i => i + 1).join(', ')}`);
      try {
        const retryInput = missingEnglishBlocks.map(i => {
          const text = smartJoin(blocks[i].texts);
          return `<<<B${i + 1}>>>\n${text}\n<<<END>>>`;
        }).join('\n\n');

        let keepInstruction = '';
        if (chapterContext && chapterContext.keep.length > 0) {
          keepInstruction = `\nKeep these terms in their original Chinese form (do not translate): ${chapterContext.keep.join('、')}`;
        }

        let retryContext = '';
        if (chapterContext) {
          retryContext = `\n\nStory context: ${chapterContext.setting || ''}`;
          const characters = chapterContext.characters
            .map(c => `${c.name} (${c.role})`).join(', ');
          if (characters) retryContext += `\nKey characters: ${characters}`;
        }

        const retrySystem = `${noThinkPrefix()}Translate Chinese manhua dialogue to natural English.${retryContext}${keepInstruction}

Output format — for each block:
<<<B#>>>
English translation
<<<END>>>

Example:
Input:
<<<B1>>>
她感到非常惭愧。
<<<END>>>

Output:
<<<B1>>>
She felt very ashamed.
<<<END>>>

Input:`;

        const retryContent = await callQwenChat(retrySystem, retryInput, 0.3);

        // Parse retry results using simple B# tags (no :english suffix)
        for (const i of missingEnglishBlocks) {
          const key = String(i + 1);
          const startTag = `<<<B${key}>>>`;
          const endTag = `<<<END>>>`;
          const startIdx = retryContent.indexOf(startTag);
          if (startIdx === -1) continue;
          const afterStart = startIdx + startTag.length;
          const endIdx = retryContent.indexOf(endTag, afterStart);
          if (endIdx === -1) continue;
          const text = retryContent.substring(afterStart, endIdx).trim();
          if (text.length > 0) {
            multiResult.get(i)?.set('english', text);
            console.error(`[Simplifier] Retry recovered English for block ${key}`);
          }
        }
      } catch (retryErr) {
        console.error('[Simplifier] English retry failed:', retryErr);
      }
    }

    return multiResult;
  } catch (e) {
    console.error('Multi-level simplification failed, using originals:', e);
    const multiResult = new Map<number, Map<string, string>>();
    blocks.forEach((block, i) => {
      const originalText = smartJoin(block.texts);
      const levelMap = new Map<string, string>();
      for (const levelKey of outputKeys) {
        levelMap.set(levelKey, originalText);
      }
      multiResult.set(i, levelMap);
    });
    return multiResult;
  }
}
