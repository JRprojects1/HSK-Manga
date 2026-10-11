import * as fs from 'fs';
import * as path from 'path';
import { HSKLevel, TokenInfo } from './types';
import { tokenize, isPunctuation, isNumber, isEnglish } from './tokenizer';

// Load HSK vocabulary lists
const dataDir = path.join(__dirname, '..', 'data');

const hsk1Words: Set<string> = new Set(
  JSON.parse(fs.readFileSync(path.join(dataDir, 'hsk1.json'), 'utf-8'))
);

const hsk2Words: Set<string> = new Set(
  JSON.parse(fs.readFileSync(path.join(dataDir, 'hsk2.json'), 'utf-8'))
);

const hsk3Words: Set<string> = new Set(
  JSON.parse(fs.readFileSync(path.join(dataDir, 'hsk3.json'), 'utf-8'))
);

const hsk4Words: Set<string> = new Set(
  JSON.parse(fs.readFileSync(path.join(dataDir, 'hsk4.json'), 'utf-8'))
);

const hsk5Words: Set<string> = new Set(
  JSON.parse(fs.readFileSync(path.join(dataDir, 'hsk5.json'), 'utf-8'))
);

const hsk6Words: Set<string> = new Set(
  JSON.parse(fs.readFileSync(path.join(dataDir, 'hsk6.json'), 'utf-8'))
);

const hsk7Words: Set<string> = new Set(
  fs.existsSync(path.join(dataDir, 'hsk7.json'))
    ? JSON.parse(fs.readFileSync(path.join(dataDir, 'hsk7.json'), 'utf-8'))
    : []
);

// Build character-level sets for fallback checking
// Each level includes characters from all words at that level and below
function buildCharacterSet(wordSets: Set<string>[]): Set<string> {
  const chars = new Set<string>();
  for (const wordSet of wordSets) {
    for (const word of wordSet) {
      for (const char of word) {
        chars.add(char);
      }
    }
  }
  return chars;
}

const hsk1Chars = buildCharacterSet([hsk1Words]);
const hsk2Chars = buildCharacterSet([hsk1Words, hsk2Words]);
const hsk3Chars = buildCharacterSet([hsk1Words, hsk2Words, hsk3Words]);
const hsk4Chars = buildCharacterSet([hsk1Words, hsk2Words, hsk3Words, hsk4Words]);
const hsk5Chars = buildCharacterSet([hsk1Words, hsk2Words, hsk3Words, hsk4Words, hsk5Words]);
const hsk6Chars = buildCharacterSet([hsk1Words, hsk2Words, hsk3Words, hsk4Words, hsk5Words, hsk6Words]);

/**
 * Get the HSK level of a character
 * Returns the lowest level where this character appears
 */
function getCharacterLevel(char: string): number | null {
  if (hsk1Chars.has(char)) {
    // Check if it's exclusively HSK1 or also in lower sets
    // Since hsk1Chars only has HSK1, if found here it's level 1
    // But we need to check the original word sets to find the actual level
    for (const word of hsk1Words) {
      if (word.includes(char)) return 1;
    }
  }
  if (hsk2Chars.has(char) && !hsk1Chars.has(char)) {
    return 2;
  }
  if (hsk3Chars.has(char) && !hsk2Chars.has(char)) {
    return 3;
  }
  if (hsk4Chars.has(char) && !hsk3Chars.has(char)) {
    return 4;
  }
  if (hsk5Chars.has(char) && !hsk4Chars.has(char)) {
    return 5;
  }
  if (hsk6Chars.has(char) && !hsk5Chars.has(char)) {
    return 6;
  }
  // Character is in a lower level set
  if (hsk1Chars.has(char)) return 1;
  if (hsk2Chars.has(char)) return 2;
  if (hsk3Chars.has(char)) return 3;
  if (hsk4Chars.has(char)) return 4;
  if (hsk5Chars.has(char)) return 5;
  if (hsk6Chars.has(char)) return 6;
  return null;
}

/**
 * Check if all characters in a word are within the target HSK level
 * Returns the highest character level if all chars are valid, null otherwise
 */
function getCharacterBasedLevel(word: string, targetLevelNum: number): number | null {
  const targetChars = targetLevelNum === 1 ? hsk1Chars :
                      targetLevelNum === 2 ? hsk2Chars :
                      targetLevelNum === 3 ? hsk3Chars :
                      targetLevelNum === 4 ? hsk4Chars :
                      targetLevelNum === 5 ? hsk5Chars : hsk6Chars;

  let maxLevel = 0;
  for (const char of word) {
    if (!targetChars.has(char)) {
      return null; // Character not in target level
    }
    const charLevel = getCharacterLevel(char);
    if (charLevel !== null && charLevel > maxLevel) {
      maxLevel = charLevel;
    }
  }
  return maxLevel > 0 ? maxLevel : null;
}

/**
 * Try to split a word into known HSK words using forward max matching.
 * Returns the highest component level, or null if the word can't be fully split.
 * e.g. 法律责任 → 法律 (HSK4) + 责任 (HSK3) → returns 4
 */
function getWordSplitLevel(word: string): number | null {
  const allSets: [Set<string>, number][] = [
    [hsk1Words, 1], [hsk2Words, 2], [hsk3Words, 3], [hsk4Words, 4],
    [hsk5Words, 5], [hsk6Words, 6], [hsk7Words, 7],
  ];

  function getLevel(w: string): number | null {
    for (const [set, level] of allSets) {
      if (set.has(w)) return level;
    }
    return null;
  }

  let i = 0;
  let maxLevel = 0;
  let hasMultiChar = false;

  while (i < word.length) {
    let matched = false;
    for (let len = Math.min(4, word.length - i); len >= 1; len--) {
      const sub = word.substring(i, i + len);
      const level = getLevel(sub);
      if (level !== null) {
        if (sub.length >= 2) hasMultiChar = true;
        if (level > maxLevel) maxLevel = level;
        i += len;
        matched = true;
        break;
      }
    }
    if (!matched) return null; // Can't split fully into HSK words
  }

  // Require at least one multi-char component to avoid matching arbitrary char sequences
  return hasMultiChar && maxLevel > 0 ? maxLevel : null;
}

/**
 * Get the HSK level of a word
 * Returns null if word is not in HSK 1-4
 * Uses character-level fallback for compound words not in vocabulary
 */
export function getHSKLevel(word: string, targetLevel?: HSKLevel): number | null {
  // First try exact word match
  if (hsk1Words.has(word)) return 1;
  if (hsk2Words.has(word)) return 2;
  if (hsk3Words.has(word)) return 3;
  if (hsk4Words.has(word)) return 4;
  if (hsk5Words.has(word)) return 5;
  if (hsk6Words.has(word)) return 6;
  if (hsk7Words.has(word)) return 7;

  // Word-level fallback: try splitting into known HSK words (forward max matching)
  // e.g. 法律责任 → 法律 (HSK4) + 责任 (HSK3) → HSK 4
  if (word.length >= 3) {
    const level = getWordSplitLevel(word);
    if (level !== null) return level;
  }

  // Single-character fallback: if the character appears in any HSK word
  // at or below the target level, treat it as known (a learner who knows
  // 将来 has already seen 将)
  if (targetLevel && word.length === 1) {
    const targetNum = targetLevel === 'hsk1' ? 1 :
                      targetLevel === 'hsk2' ? 2 :
                      targetLevel === 'hsk3' ? 3 :
                      targetLevel === 'hsk4' ? 4 :
                      targetLevel === 'hsk5' ? 5 : 6;
    const levelSets = [hsk1Words, hsk2Words, hsk3Words, hsk4Words, hsk5Words, hsk6Words];
    for (let lvl = 0; lvl < targetNum; lvl++) {
      for (const w of levelSets[lvl]) {
        if (w.includes(word)) return lvl + 1;
      }
    }
  }

  // Multi-character fallback: check if all characters are within target level
  if (targetLevel && word.length > 1) {
    const targetNum = targetLevel === 'hsk1' ? 1 :
                      targetLevel === 'hsk2' ? 2 :
                      targetLevel === 'hsk3' ? 3 :
                      targetLevel === 'hsk4' ? 4 :
                      targetLevel === 'hsk5' ? 5 : 6;
    return getCharacterBasedLevel(word, targetNum);
  }

  return null;
}

/**
 * Check if a word is within the target HSK level
 */
export function isWithinLevel(word: string, targetLevel: HSKLevel): boolean {
  const level = getHSKLevel(word, targetLevel);
  if (level === null) return false;

  if (targetLevel === 'hsk1') {
    return level === 1;
  }
  if (targetLevel === 'hsk2') {
    return level <= 2;
  }
  if (targetLevel === 'hsk3') {
    return level <= 3;
  }
  if (targetLevel === 'hsk4') {
    return level <= 4;
  }
  if (targetLevel === 'hsk5') {
    return level <= 5;
  }
  return level <= 6; // hsk6 includes everything below it
}

/**
 * Analyze text and return token information
 */
export function analyzeText(text: string, targetLevel: HSKLevel, keepWords?: string[]): TokenInfo[] {
  const tokens = tokenize(text, undefined, keepWords);

  return tokens.map((word) => {
    // Skip punctuation, numbers, and English text - they don't need HSK checking
    if (isPunctuation(word) || isNumber(word) || isEnglish(word)) {
      return {
        word,
        isInHSK: true, // Treat as valid
        hskLevel: null,
      };
    }

    const level = getHSKLevel(word, targetLevel);
    const isInHSK = isWithinLevel(word, targetLevel);

    return {
      word,
      isInHSK,
      hskLevel: level,
    };
  });
}

/**
 * Get all words that are above the target HSK level
 */
export function getFlaggedWords(text: string, targetLevel: HSKLevel, keepWords?: string[]): string[] {
  const analysis = analyzeText(text, targetLevel, keepWords);

  return analysis
    .filter((token) => !token.isInHSK && !isPunctuation(token.word) && !isNumber(token.word))
    .map((token) => token.word);
}

/**
 * Calculate what percentage of words are within target HSK level
 */
export function calculateHSKCoverage(text: string, targetLevel: HSKLevel, keepWords?: string[]): number {
  const analysis = analyzeText(text, targetLevel, keepWords);

  const relevantTokens = analysis.filter(
    (token) => !isPunctuation(token.word) && !isNumber(token.word)
  );

  if (relevantTokens.length === 0) return 100;

  const inHSKCount = relevantTokens.filter((token) => token.isInHSK).length;
  return (inHSKCount / relevantTokens.length) * 100;
}

/**
 * Get vocabulary stats for a text
 */
export function getVocabularyStats(text: string, targetLevel: HSKLevel, keepWords?: string[]) {
  const analysis = analyzeText(text, targetLevel, keepWords);

  const relevantTokens = analysis.filter(
    (token) => !isPunctuation(token.word) && !isNumber(token.word)
  );

  const hsk1Count = relevantTokens.filter((t) => t.hskLevel === 1).length;
  const hsk2Count = relevantTokens.filter((t) => t.hskLevel === 2).length;
  const hsk3Count = relevantTokens.filter((t) => t.hskLevel === 3).length;
  const hsk4Count = relevantTokens.filter((t) => t.hskLevel === 4).length;
  const hsk5Count = relevantTokens.filter((t) => t.hskLevel === 5).length;
  const hsk6Count = relevantTokens.filter((t) => t.hskLevel === 6).length;
  const outsideHSKCount = relevantTokens.filter((t) => t.hskLevel === null || t.hskLevel === 7).length;

  return {
    total: relevantTokens.length,
    hsk1: hsk1Count,
    hsk2: hsk2Count,
    hsk3: hsk3Count,
    hsk4: hsk4Count,
    hsk5: hsk5Count,
    hsk6: hsk6Count,
    outsideHSK: outsideHSKCount,
    coverage: calculateHSKCoverage(text, targetLevel, keepWords),
  };
}
