// Chinese tokenizer backed by @node-rs/jieba (DAG + DP + HMM).
// Replaces the previous segmentit-based implementation, which had a broken
// default particle frequency table and a non-standard multi-criteria scoring
// algorithm that produced unpredictable behavior under user-dict injection.

// eslint-disable-next-line @typescript-eslint/no-var-requires
const { Jieba } = require('@node-rs/jieba');
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { dict: defaultDict } = require('@node-rs/jieba/dict');

const jieba = Jieba.withDict(defaultDict);

// Track KEEP words already injected into jieba's dictionary so we don't
// redundantly reload them across calls. jieba.loadDict appends, so
// accumulating is safe; this just avoids wasted Buffer allocation.
const injectedKeep = new Set<string>();

// A token that is purely a number — Arabic/fullwidth digits, or a run of
// Chinese numeral characters.
const NUMERIC_TOKEN = /^(?:[\d０-９]+|[一二三四五六七八九十百千万亿零两兩〇○]+)$/;

/**
 * Fix the one measure-word mis-tokenization that has real consequences:
 * jieba glues 个人 into a single token when a number precedes it via an Arabic
 * numeral (4个人 → [4, 个人]) or after 二/零 (二个人 → [二, 个人]) — because 4个 /
 * 二个 are not dictionary words while 个人 ("individual") is. In that position
 * 个人 is really 个 (measure word) + 人 (noun), both HSK1, but the glued token
 * mis-flags as the HSK5 word 个人. Splitting it is gated on a preceding number
 * so the standalone word 个人 ("personal", e.g. 我的个人意见) is never touched.
 *
 * A dictionary-wide scan confirmed 个人 is the only classifier+noun fusion that
 * both flags above-level and has low-level constituents — so this is a single
 * named special case, not a measure-word list (which mis-splits 分钟/点钟/把手).
 */
function fixMeasurePerson(tokens: string[]): string[] {
  const out: string[] = [];
  for (let i = 0; i < tokens.length; i++) {
    if (tokens[i] === '个人' && i > 0 && NUMERIC_TOKEN.test(tokens[i - 1])) {
      out.push('个', '人');
    } else {
      out.push(tokens[i]);
    }
  }
  return out;
}

/**
 * Tokenize Chinese text into individual words.
 *
 * @param text         Chinese text to tokenize
 * @param targetLevel  Kept in signature for backward compat; ignored (jieba
 *                     does not need per-level dictionary tuning — HSK filtering
 *                     happens downstream in hsk-checker).
 * @param keepWords    Chapter-specific terms that must be preserved as single
 *                     tokens. Injected into jieba's dictionary at very high
 *                     frequency.
 */
export function tokenize(text: string, targetLevel?: number, keepWords?: string[]): string[] {
  // Inject any newly-seen KEEP words. High freq biases jieba toward picking
  // them as single tokens; where two KEEP terms overlap (e.g. 高手 inside
  // 测验高手), the longer term naturally wins via DAG path probability, which
  // is the desired behavior.
  if (keepWords && keepWords.length > 0) {
    const fresh: string[] = [];
    for (const word of keepWords) {
      if (word.length < 2 || injectedKeep.has(word)) continue;
      // jieba's user-dict format is `word freq pos` separated by single spaces.
      // Whitespace inside `word` (e.g. "Jung Yoon-kang") makes the native loader
      // see 4 fields instead of 3 and throw "Load dict failed". Skip those.
      if (/\s/.test(word)) {
        injectedKeep.add(word);
        continue;
      }
      fresh.push(word);
      injectedKeep.add(word);
    }
    if (fresh.length > 0) {
      const entries = fresh.map((w) => `${w} 1000000 n`).join('\n');
      jieba.loadDict(Buffer.from(entries, 'utf8'));
    }
  }

  return fixMeasurePerson(jieba.cut(text, true) as string[]);
}

/**
 * Tokenize with part-of-speech tagging. Returns jieba-style POS tags encoded
 * as a short numeric placeholder (0) — no current consumer uses the tag value
 * meaningfully, and segmentit's bit-flag POS scheme is not portable to jieba.
 */
export function tokenizeWithPOS(text: string): Array<{ word: string; tag: number }> {
  const result = jieba.tag(text, true) as Array<{ tag: string; word: string }>;
  return result.map((item) => ({ word: item.word, tag: 0 }));
}

/**
 * Split text into sentences based on Chinese punctuation.
 */
export function splitSentences(text: string): string[] {
  const sentenceEnders = /([。！？；\n])/;
  const parts = text.split(sentenceEnders);

  const sentences: string[] = [];
  let current = '';

  for (const part of parts) {
    if (sentenceEnders.test(part)) {
      current += part;
      if (current.trim()) {
        sentences.push(current.trim());
      }
      current = '';
    } else {
      current += part;
    }
  }

  if (current.trim()) {
    sentences.push(current.trim());
  }

  return sentences;
}

export function isPunctuation(text: string): boolean {
  const punctuationRegex = /^[\s\p{P}\p{S}]+$/u;
  return punctuationRegex.test(text);
}

export function isNumber(text: string): boolean {
  const numberRegex = /^[\d\uFF10-\uFF19]+$/;
  return numberRegex.test(text);
}

export function isEnglish(text: string): boolean {
  // Alphanumeric tokens (letters, optionally with digits) count as English.
  // Must contain at least one letter — pure-digit tokens are handled by isNumber.
  // This covers AI, RPG, NPC (pure letters) AND 2D, 3D, 4K, H264 (letter+digit).
  // Without the letter+digit case, mixed alphanumeric tokens fall through to the
  // HSK classifier, get flagged as above-level, and end up in [Replace: ...]
  // instructing the LLM to remove them — dropping standalone terms like "2D".
  return /^[a-zA-Z0-9]+( [a-zA-Z0-9]+)*$/.test(text) && /[a-zA-Z]/.test(text);
}
