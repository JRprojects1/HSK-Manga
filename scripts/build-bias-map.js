/**
 * Build a unified logit_bias map for the DeepSeek-V4-Flash tokenizer.
 *
 * Supports four modes via the MODE env var:
 *
 *   (unset / default) — production logic:
 *       Negative-bias (-NEG) if decoded text contains ANY HSK 4+ multi-char word.
 *       Positive-bias (+POS) if all Chinese chars in decoded text are HSK 1-N.
 *       Negative wins overlap. Output: hsk-token-bias-{level}.deepseek-v4.json
 *
 *   MODE=b1 — "naive remove negative":
 *       NO negative entries at all. Char-based positive bias applies to all
 *       tokens with all-HSK-1-N chars — INCLUDING compound HSK 4+ words built
 *       from those chars (好学 etc.). They get POSITIVE bias.
 *
 *   MODE=b2 — "suppress positive on compound HSK 4+ words":
 *       NO negative entries. Substring check still runs, but instead of
 *       assigning negative, it just SUPPRESSES the positive bias for that
 *       token (it stays neutral). Compound HSK 4+ words get no push either way.
 *
 *   MODE=b3 — "exact-word positive only":
 *       NO negative entries. NO char-based positive. Positive bias is only
 *       assigned to tokens whose decoded text IS a HSK 1-N dictionary word.
 *       Strictest of the three.
 *
 *   MODE=hybrid — B2 on overlap + A's negative on pure HSK 4+ tokens:
 *       Splits the substring-match HSK 4+ negative into two cases. If the
 *       token's chars are ALL HSK 1-N (overlap zone, e.g. 好学 made from
 *       HSK 1 chars but is itself HSK 4+) → SUPPRESS (no bias). If the
 *       token has any HSK 4+ char (pure HSK 4+ zone, e.g. 经济 where 济 is
 *       HSK 7) → assign NEGATIVE. Tests "don't penalize tokens the model
 *       could rearrange into HSK 1-N words, but penalize tokens that
 *       encode genuinely hard vocab the learner can't read."
 *
 *   MODE=graded — per-level positive bias (for the tester URL):
 *       Same inclusion rule as production (all Chinese chars HSK 1-N, no
 *       compound HSK 4+ words). But instead of writing a single +POS value,
 *       each positive entry stores the MAX HSK level among the token's
 *       Chinese chars (1, 2, 3, ...). Negative entries store -1 (sign marker).
 *       The runtime converts these markers to actual magnitudes per-level
 *       using config the tester sets in the UI. Format: `level_map` instead
 *       of `bias_map`. File header has `format: 'graded'`.
 *
 * GPT-2-style byte-to-unicode mapping is used to decode BPE token strings
 * back to UTF-8 bytes, then UTF-8 → JS string.
 *
 * Usage:
 *   node build-bias-map.js              # production maps
 *   MODE=b1 node build-bias-map.js      # variant B1
 *   MODE=b2 node build-bias-map.js      # variant B2
 *   MODE=b3 node build-bias-map.js      # variant B3
 *   MODE=graded node build-bias-map.js  # per-level positive markers
 */
const fs = require('fs');
const path = require('path');

const TOKENIZER_PATH = 'data/dsv4-tokenizer.json';
const OUT_DIR = 'data';

// Per-level positive/negative bias values (mirrors the values production uses today).
// Positive is the level-specific BIAS we already use.
// Negative should be a gentle nudge, symmetric in magnitude with positive — NOT a
// hard ban. -50 would multiply likelihood by e^-50 ≈ 10^-22 (effectively
// forbidden); -2.5 matches the positive's +2.5 and lets the model still pick
// HSK 4+ words when context strongly demands them (e.g. chapter-defining
// concepts like 猎人).
// Override via env: POS_BIAS=2.5 NEG_BIAS=-2.5 node build-bias-map.js
const POS_BIAS_DEFAULT = parseFloat(process.env.POS_BIAS || '2.5');
const NEG_BIAS = parseFloat(process.env.NEG_BIAS || '-2.5');
const MODE = (process.env.MODE || 'current').toLowerCase();
if (!['current', 'b1', 'b2', 'b3', 'hybrid', 'graded'].includes(MODE)) {
  console.error(`Unknown MODE=${MODE}. Valid: current, b1, b2, b3, hybrid, graded.`);
  process.exit(1);
}
console.log(`Build mode: ${MODE.toUpperCase()}`);

// ------- GPT-2 byte-to-unicode encoding (the encoding used by the tokenizer) -------
//   Reverses byte-pair-encoding's byte-to-printable map.
function buildByteToUnicode() {
  const bs = [];
  for (let b = 33; b <= 126; b++) bs.push(b);           // !..~
  for (let b = 161; b <= 172; b++) bs.push(b);          // ¡..¬
  for (let b = 174; b <= 255; b++) bs.push(b);          // ®..ÿ
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  const byteToChar = new Map();   // byte → unicode code point
  const charToByte = new Map();   // unicode char → byte (for decoding)
  for (let i = 0; i < bs.length; i++) {
    byteToChar.set(bs[i], cs[i]);
    charToByte.set(String.fromCodePoint(cs[i]), bs[i]);
  }
  return { byteToChar, charToByte };
}

// Decode a single BPE token string back to UTF-8 string.
function decodeBPE(tokenStr, charToByte) {
  const bytes = [];
  for (const ch of tokenStr) {
    const b = charToByte.get(ch);
    if (b === undefined) {
      // Some special tokens use literal chars (e.g. <|begin▁of▁sentence|>) — not byte-encoded.
      return null;
    }
    bytes.push(b);
  }
  try {
    return Buffer.from(bytes).toString('utf-8');
  } catch (e) {
    return null;
  }
}

// ------- HSK loaders -------
function loadHSK(level) {
  const file = `data/hsk${level}.json`;
  const d = JSON.parse(fs.readFileSync(file, 'utf-8'));
  return Array.isArray(d) ? d : Object.keys(d);
}

// ------- Build a substring-matcher for fast HSK 4+ word detection -------
//
// For each token string, check if any HSK 4+ word appears in it. Naive approach
// is O(|tokens| × |words|) but words are short and tokens are short, so practical.
// We use a trie for efficient multi-substring matching.
class SubstringSet {
  constructor(words) {
    this.byFirstChar = new Map(); // first-char → sorted list of words
    for (const w of words) {
      if (!w || w.length === 0) continue;
      const c = w[0];
      if (!this.byFirstChar.has(c)) this.byFirstChar.set(c, []);
      this.byFirstChar.get(c).push(w);
    }
  }
  // Returns the FIRST HSK 4+ word found in `text`, or null.
  findFirstMatch(text) {
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      const cands = this.byFirstChar.get(c);
      if (!cands) continue;
      for (const w of cands) {
        if (text.startsWith(w, i)) return w;
      }
    }
    return null;
  }
}

// ------- Main build -------
(() => {
  console.log('Loading tokenizer.json…');
  const t = JSON.parse(fs.readFileSync(TOKENIZER_PATH, 'utf-8'));
  const vocab = t.model.vocab; // { tokenStr: tokenId }

  const { charToByte } = buildByteToUnicode();
  console.log(`Vocab size: ${Object.keys(vocab).length} tokens`);

  // Decode every token to its UTF-8 string
  console.log('Decoding BPE token strings…');
  const tokens = []; // { id, decoded }
  let skipped = 0;
  for (const [tokenStr, idAny] of Object.entries(vocab)) {
    const id = +idAny;
    const decoded = decodeBPE(tokenStr, charToByte);
    if (decoded === null) { skipped++; continue; }
    tokens.push({ id, decoded });
  }
  console.log(`Decoded: ${tokens.length}, skipped (special/unmappable): ${skipped}`);

  // Load HSK word lists
  console.log('Loading HSK word lists…');
  const hskLevels = {};
  for (const lvl of [1, 2, 3, 4, 5, 6, 7]) hskLevels[lvl] = loadHSK(lvl);
  for (const lvl of [1, 2, 3, 4, 5, 6, 7]) console.log(`  HSK${lvl}: ${hskLevels[lvl].length} words`);

  // Char sets per level (for the "all chars are HSK 1-N" check)
  function charsAtOrBelow(levelMax) {
    const s = new Set();
    for (let lvl = 1; lvl <= levelMax; lvl++) {
      for (const w of hskLevels[lvl]) {
        for (const c of w) s.add(c);
      }
    }
    return s;
  }
  function wordsAtOrBelow(levelMax) {
    const s = new Set();
    for (let lvl = 1; lvl <= levelMax; lvl++) {
      for (const w of hskLevels[lvl]) s.add(w);
    }
    return s;
  }

  // Build the negative-substring matcher for tokens whose decoded text contains
  // an HSK 4+ word. We build it once per target level since the "above target"
  // word set differs.
  function negativeSubstringSet(targetLevel) {
    const words = [];
    for (let lvl = targetLevel + 1; lvl <= 7; lvl++) {
      for (const w of hskLevels[lvl]) {
        // Skip single-character HSK entries — they overlap heavily with the
        // positive char set and cause false positives ("把" is HSK4 but also
        // a grammar particle that almost every sentence uses). Multi-char
        // entries are the high-value targets.
        if (w.length >= 2) words.push(w);
      }
    }
    return new SubstringSet(words);
  }

  // Per-char HSK level lookup (lowest level a char appears in). Used by graded
  // mode to assign each token its MAX char level.
  function buildCharLevelMap() {
    const m = new Map(); // char → lowest HSK level it appears in
    for (let lvl = 1; lvl <= 7; lvl++) {
      for (const w of hskLevels[lvl]) {
        for (const c of w) {
          if (!m.has(c)) m.set(c, lvl); // earliest level wins (lowest number)
        }
      }
    }
    return m;
  }
  const charLevelMap = buildCharLevelMap();

  // Generate the bias map per target level
  function buildForLevel(targetLevel) {
    const positiveChars = charsAtOrBelow(targetLevel);
    const positiveWords = wordsAtOrBelow(targetLevel);
    const negSubstring = negativeSubstringSet(targetLevel);

    const biasMap = {}; // id → bias value (or, in graded mode, level marker)
    const stats = { positive: 0, negative: 0, suppressed: 0, neutral: 0, overrides: 0, byLevel: {} };
    const sampleNeg = [];
    const sampleGraded = []; // for graded mode: shows a few examples per level

    // Char-based positive helper (used by current, b1, b2)
    function charBasedPositive(decoded) {
      let allCharsInPos = true;
      let hasChineseChar = false;
      for (const c of decoded) {
        const cp = c.codePointAt(0);
        if (cp >= 0x4E00 && cp <= 0x9FFF) {
          hasChineseChar = true;
          if (!positiveChars.has(c)) { allCharsInPos = false; break; }
        }
      }
      return hasChineseChar && allCharsInPos;
    }

    // Graded helper: returns max HSK level among Chinese chars in decoded, or
    // null if not eligible (no Chinese chars, or any char outside HSK 1-N).
    function maxCharLevelIfEligible(decoded) {
      let max = 0;
      let hasChineseChar = false;
      for (const c of decoded) {
        const cp = c.codePointAt(0);
        if (cp >= 0x4E00 && cp <= 0x9FFF) {
          hasChineseChar = true;
          const lvl = charLevelMap.get(c);
          if (!lvl || lvl > targetLevel) return null;
          if (lvl > max) max = lvl;
        }
      }
      return hasChineseChar ? max : null;
    }

    for (const { id, decoded } of tokens) {
      if (decoded.length === 0) { stats.neutral++; continue; }

      if (MODE === 'current') {
        const negMatch = negSubstring.findFirstMatch(decoded);
        if (negMatch !== null) {
          biasMap[id] = NEG_BIAS;
          stats.negative++;
          if (sampleNeg.length < 30) sampleNeg.push({ id, decoded, match: negMatch });
          continue;
        }
        if (charBasedPositive(decoded)) { biasMap[id] = POS_BIAS_DEFAULT; stats.positive++; }
        else { stats.neutral++; }

      } else if (MODE === 'b1') {
        // No negative; char-based positive only. Compound HSK 4+ words built
        // from HSK 1-N chars get POSITIVE bias here (the case the user
        // specifically flagged — 好学, etc. fall through to positive).
        if (charBasedPositive(decoded)) { biasMap[id] = POS_BIAS_DEFAULT; stats.positive++; }
        else { stats.neutral++; }

      } else if (MODE === 'b2') {
        // Substring check still runs but only SUPPRESSES positive; no negative
        // is assigned. Compound HSK 4+ words → neutral. Other tokens with all
        // HSK 1-N chars → positive.
        const negMatch = negSubstring.findFirstMatch(decoded);
        if (negMatch !== null) {
          stats.suppressed++;
          if (sampleNeg.length < 30) sampleNeg.push({ id, decoded, match: negMatch });
          continue;
        }
        if (charBasedPositive(decoded)) { biasMap[id] = POS_BIAS_DEFAULT; stats.positive++; }
        else { stats.neutral++; }

      } else if (MODE === 'b3') {
        // Exact dictionary-word positive only; no negative; no char-based positive.
        if (positiveWords.has(decoded)) { biasMap[id] = POS_BIAS_DEFAULT; stats.positive++; }
        else { stats.neutral++; }

      } else if (MODE === 'hybrid') {
        // Substring check runs. If matches:
        //   - All chars HSK 1-N (overlap zone, like 好学) → SUPPRESS (no bias)
        //   - Any char HSK 4+ (pure HSK 4+, like 经济) → NEGATIVE
        // Otherwise, char-based positive on all-HSK-1-N tokens.
        const negMatch = negSubstring.findFirstMatch(decoded);
        if (negMatch !== null) {
          if (charBasedPositive(decoded)) {
            // Overlap zone — model could rearrange these into HSK 1-N words
            stats.suppressed++;
            if (sampleNeg.length < 30) sampleNeg.push({ id, decoded, match: negMatch });
            continue;
          } else {
            // Pure HSK 4+ — chars themselves encode hard vocab, penalize
            biasMap[id] = NEG_BIAS;
            stats.negative++;
            if (sampleNeg.length < 30) sampleNeg.push({ id, decoded, match: negMatch });
            continue;
          }
        }
        if (charBasedPositive(decoded)) { biasMap[id] = POS_BIAS_DEFAULT; stats.positive++; }
        else { stats.neutral++; }

      } else if (MODE === 'graded') {
        // Same inclusion rule as production: HSK 4+ compound words → -1 marker,
        // otherwise max char level (1..targetLevel) for positive tokens.
        const negMatch = negSubstring.findFirstMatch(decoded);
        if (negMatch !== null) {
          biasMap[id] = -1; // sign marker; runtime maps to user-set negative magnitude
          stats.negative++;
          if (sampleNeg.length < 30) sampleNeg.push({ id, decoded, match: negMatch });
          continue;
        }
        const lvl = maxCharLevelIfEligible(decoded);
        if (lvl !== null) {
          biasMap[id] = lvl;
          stats.positive++;
          stats.byLevel[lvl] = (stats.byLevel[lvl] || 0) + 1;
          if (!sampleGraded.find(s => s.level === lvl) || sampleGraded.filter(s => s.level === lvl).length < 4) {
            sampleGraded.push({ id, decoded, level: lvl });
          }
        } else {
          stats.neutral++;
        }
      }
    }

    // Compare with existing positive map to count "overrides" (tokens that
    // would have been positive under old logic but are now negative).
    const oldPath = `data/hsk-token-ids-${targetLevel === 1 ? '1' : `1to${targetLevel}`}.deepseek-v4.json`;
    if (fs.existsSync(oldPath)) {
      const oldData = JSON.parse(fs.readFileSync(oldPath, 'utf-8'));
      const oldPos = new Set(oldData.all_unique_token_ids || []);
      let overrides = 0;
      for (const id of Object.keys(biasMap)) {
        if (biasMap[+id] === NEG_BIAS && oldPos.has(+id)) overrides++;
      }
      stats.overrides = overrides;
    }

    return { biasMap, stats, sampleNeg, sampleGraded };
  }

  const modeSuffix = MODE === 'current' ? '' : `.${MODE}`;
  // LEVELS env selects which target levels to (re)build, e.g. LEVELS=5,6.
  // Default reproduces the original 1-4 set.
  const targetLevels = (process.env.LEVELS || '1,2,3,4').split(',').map(Number);
  for (const targetLevel of targetLevels) {
    console.log(`\n=== Building for HSK${targetLevel} target (mode=${MODE}) ===`);
    const { biasMap, stats, sampleNeg, sampleGraded } = buildForLevel(targetLevel);
    console.log(`  positive entries: ${stats.positive}`);
    console.log(`  negative entries: ${stats.negative}`);
    if (stats.suppressed) console.log(`  suppressed (was-positive-now-neutral via HSK 4+ substring): ${stats.suppressed}`);
    console.log(`  neutral (no bias): ${stats.neutral}`);
    if (stats.overrides) console.log(`  overrides (was-positive-now-negative): ${stats.overrides}`);
    if (MODE === 'graded') {
      for (const lvl of Object.keys(stats.byLevel).sort((a, b) => +a - +b)) {
        console.log(`    HSK${lvl}: ${stats.byLevel[lvl]} tokens`);
      }
    }

    const outName = `hsk-token-bias-${targetLevel === 1 ? '1' : `1to${targetLevel}`}.deepseek-v4${modeSuffix}.json`;
    const outPath = path.join(OUT_DIR, outName);
    const isGraded = MODE === 'graded';
    fs.writeFileSync(outPath, JSON.stringify({
      tokenizer_repo: t._tokenizer_repo || 'deepseek-ai/DeepSeek-V4-Flash',
      target_level: targetLevel,
      mode: MODE,
      format: isGraded ? 'graded' : 'flat',
      ...(isGraded
        ? {
            // Graded markers: positive entries store HSK level (1..target),
            // negative entries store -1. Runtime maps these to user-set
            // magnitudes per level.
            level_map_legend: {
              '1': 'token containing HSK 1 chars only — gets HSK 1 magnitude',
              '2': 'token max char level = HSK 2 — gets HSK 2 magnitude',
              '3': 'token max char level = HSK 3 — gets HSK 3 magnitude',
              '4': 'token max char level = HSK 4 — gets HSK 4 magnitude',
              '-1': 'token contains HSK 4+ compound word — gets negative magnitude',
            },
            positive_count_by_level: stats.byLevel,
          }
        : {
            positive_bias: POS_BIAS_DEFAULT,
            negative_bias: MODE === 'current' ? NEG_BIAS : 0,
          }),
      total_entries: Object.keys(biasMap).length,
      positive_count: stats.positive,
      negative_count: stats.negative,
      suppressed_count: stats.suppressed,
      overrides: stats.overrides,
      [isGraded ? 'level_map' : 'bias_map']: biasMap,
    }, null, 0), 'utf-8');
    console.log(`  written: ${outPath}`);

    if (sampleNeg.length) {
      const label = MODE === 'current' || MODE === 'graded' ? 'negative' : 'suppressed';
      console.log(`  sample ${label} classifications:`);
      for (const s of sampleNeg.slice(0, 8)) {
        console.log(`    id=${s.id}  "${s.decoded}"  matched HSK4+ word: "${s.match}"`);
      }
    }
    if (MODE === 'graded' && sampleGraded.length) {
      console.log(`  sample positive classifications by level:`);
      const byLvl = {};
      for (const s of sampleGraded) {
        (byLvl[s.level] ||= []).push(s);
      }
      for (const lvl of Object.keys(byLvl).sort((a, b) => +a - +b)) {
        const samples = byLvl[lvl].slice(0, 4).map(s => `"${s.decoded}"`).join(', ');
        console.log(`    HSK${lvl}: ${samples}`);
      }
    }
  }
})();
