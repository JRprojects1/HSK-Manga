/**
 * Build the simplification thesaurus.
 *
 * Inputs (data/raw/):
 *   - chinese-ls-annotations.csv  Chinese-LS human-verified complex→simple pairs
 *   - cilin-extended.txt          HIT Tongyici Cilin Extended (UTF-8)
 *   - chi-syn.txt                 g2384/chi-syn synonym groups
 *
 * HSK filter:
 *   - Load data/hsk1.json, hsk2.json, hsk3.json (the target band)
 *
 * Output:
 *   - data/simplification-thesaurus.json   {word: [hint1, hint2, hint3] | null}
 *   - data/thesaurus-build-report.json     coverage stats + sample entries
 *
 * Run: node scripts/build-thesaurus.js
 */

const fs = require('fs');
const path = require('path');

const RAW_DIR = path.join(__dirname, '..', 'data', 'raw');
const OUT_DIR = path.join(__dirname, '..', 'data');
const OUT_JSON = path.join(OUT_DIR, 'simplification-thesaurus.json');
const OUT_REPORT = path.join(OUT_DIR, 'thesaurus-build-report.json');
const MAX_HINTS_PER_WORD = 3;

// --- HSK sets --------------------------------------------------------------

function loadHsk() {
  const hsk1 = require(path.join(OUT_DIR, 'hsk1.json'));
  const hsk2 = require(path.join(OUT_DIR, 'hsk2.json'));
  const hsk3 = require(path.join(OUT_DIR, 'hsk3.json'));
  const level = new Map();
  for (const w of hsk1) level.set(w, 1);
  for (const w of hsk2) if (!level.has(w)) level.set(w, 2);
  for (const w of hsk3) if (!level.has(w)) level.set(w, 3);
  return level; // Map<word, 1|2|3>
}

// --- Source parsers --------------------------------------------------------

function parseChineseLS(file) {
  // TSV: sentence \t word \t POS \t count \t alternative1 alternative2 ...
  const text = fs.readFileSync(file, 'utf-8');
  const map = new Map(); // word → Set<alternatives>
  for (const raw of text.split(/\r?\n/)) {
    if (!raw.trim()) continue;
    const parts = raw.split('\t');
    if (parts.length < 5) continue;
    const word = parts[1].trim();
    const alts = parts[4].trim().split(/\s+/).filter(Boolean);
    if (!map.has(word)) map.set(word, new Set());
    for (const a of alts) map.get(word).add(a);
  }
  return map;
}

function parseChls(file) {
  // CHLS train/trail format: ID \t sentence \t position \t word \t sub1 count1;sub2 count2;... [\t extra]
  // Vote count 0 means annotator-rejected; we keep only votes >= 1.
  const text = fs.readFileSync(file, 'utf-8');
  const map = new Map(); // word → Map<substitute, cumulative_votes>
  for (const raw of text.split(/\r?\n/)) {
    if (!raw.trim()) continue;
    const parts = raw.split('\t');
    if (parts.length < 5) continue;
    const word = parts[3].trim();
    if (!word) continue;
    const subsField = parts[4].trim();
    if (!map.has(word)) map.set(word, new Map());
    const scoreMap = map.get(word);
    // substitutes are 'subA votesA;subB votesB;...'
    for (const pair of subsField.split(';')) {
      const m = pair.trim().match(/^(.+?)\s+(\d+)$/);
      if (!m) continue;
      const sub = m[1].trim();
      const votes = parseInt(m[2], 10);
      if (votes < 1) continue;  // annotator-rejected
      if (!sub || sub === word) continue;
      scoreMap.set(sub, (scoreMap.get(sub) || 0) + votes);
    }
    // Extra extra-column substitutes (some rows have a 6th col with bonus suggestions)
    if (parts.length >= 6 && parts[5].trim()) {
      for (const pair of parts[5].trim().split(';')) {
        const m = pair.trim().match(/^(.+?)\s+(\d+)$/);
        if (!m) continue;
        const sub = m[1].trim();
        const votes = parseInt(m[2], 10);
        if (votes < 1 || !sub || sub === word) continue;
        scoreMap.set(sub, (scoreMap.get(sub) || 0) + votes);
      }
    }
  }
  return map;
}

function parseChlsAll(dir) {
  // Merge test + trail + train
  const merged = new Map();
  for (const fname of ['all_train_gold.txt', 'all_trail_gold.txt', 'all_test_gold.txt']) {
    const partial = parseChls(path.join(dir, fname));
    for (const [word, subMap] of partial) {
      if (!merged.has(word)) merged.set(word, new Map());
      const target = merged.get(word);
      for (const [sub, votes] of subMap) {
        target.set(sub, (target.get(sub) || 0) + votes);
      }
    }
  }
  return merged;
}

function parseCilin(file) {
  // Lines like "Aa01A01= word1 word2 word3"
  const text = fs.readFileSync(file, 'utf-8');
  const map = new Map(); // word → Set<synonyms>
  for (const raw of text.split(/\r?\n/)) {
    const m = raw.match(/^\S+=\s*(.+)$/);
    if (!m) continue;
    const words = m[1].trim().split(/\s+/).filter(Boolean);
    if (words.length < 2) continue;
    for (const w of words) {
      if (!map.has(w)) map.set(w, new Set());
      for (const other of words) {
        if (other !== w) map.get(w).add(other);
      }
    }
  }
  return map;
}

function parseChisyn(file) {
  // Lines: word1\tword2\tword3...  (polysemy: word1 or word1 with digit suffix like 不合1)
  const text = fs.readFileSync(file, 'utf-8');
  const map = new Map();
  for (const raw of text.split(/\r?\n/)) {
    if (!raw.trim()) continue;
    const words = raw.split(/\t+/).map(w => w.trim()).filter(Boolean);
    if (words.length < 2) continue;
    for (const w of words) {
      const base = w.replace(/\d+$/, '');
      if (!map.has(base)) map.set(base, new Set());
      for (const other of words) {
        const otherBase = other.replace(/\d+$/, '');
        if (otherBase !== base) map.get(base).add(otherBase);
      }
    }
  }
  return map;
}

// --- HSK-aware filtering ---------------------------------------------------

// Single characters that are HSK1-3 but don't work as synonym substitutes.
// These are particles, auxiliaries, sentence markers, or too-generic function words.
// Cilin groups sometimes include them because of archaic/compound senses
// (e.g., 了 in the "completeness" group because of 了结 / 完了).
const BAD_SINGLE_CHAR_HINTS = new Set([
  '了', '着', '过', '的', '地', '得',  // aspect/structural particles
  '吧', '啊', '吗', '呢', '嘛', '哦', '呀',  // sentence-final particles
  '在', '是', '有', '会', '要', '能',  // high-freq aux/copula
  '我', '你', '他', '她', '它', '们',  // pronouns
  '和', '与', '或', '而', '也',        // conjunctions
  '不', '没',                          // negation (would break meaning)
  '就', '才', '都', '还', '再', '又',  // tone adverbs (ambiguous)
]);

function rankCandidates(word, allCandidates, hskLevel) {
  // allCandidates: array of { word, source, votes? }
  //   source ∈ {chls, gold, cilin, chisyn}
  //   votes present for chls entries (higher = stronger annotator agreement)
  // Filter to HSK1-3 candidates (with single-char fallback, excluding particles).
  // Rank by: source priority, votes (desc), HSK level, length, alphabetical.

  const scored = [];
  const seen = new Set();
  const sourceRank = { chls: 0, gold: 1, cilin: 2, chisyn: 3 };

  for (const c of allCandidates) {
    if (c.word === word) continue;            // no self-suggestions
    if (seen.has(c.word)) continue;           // dedupe; first encounter wins (already ordered by source)
    seen.add(c.word);

    let lvl = hskLevel.get(c.word);
    let passType = 'direct';

    // Single-char fallback: include if single char and in HSK1-3, EXCEPT blocklist of particles.
    if (lvl === undefined && c.word.length === 1 && hskLevel.has(c.word)) {
      if (BAD_SINGLE_CHAR_HINTS.has(c.word)) continue;
      lvl = hskLevel.get(c.word);
      passType = 'single-char';
    }

    // Additional direct-HSK-word filter: even multi-char HSK1-3 words can be bad hints
    // if they're in the blocklist (unlikely but safety).
    if (lvl !== undefined && BAD_SINGLE_CHAR_HINTS.has(c.word)) continue;

    if (lvl === undefined) continue; // not in HSK — skip

    scored.push({
      word: c.word,
      source: c.source,
      sourceRank: sourceRank[c.source],
      votes: c.votes || 0,
      hskLevel: lvl,
      length: c.word.length,
      passType,
    });
  }

  scored.sort((a, b) => {
    if (a.sourceRank !== b.sourceRank) return a.sourceRank - b.sourceRank;
    if (b.votes !== a.votes) return b.votes - a.votes;            // higher votes first
    if (a.hskLevel !== b.hskLevel) return a.hskLevel - b.hskLevel;
    if (a.length !== b.length) return a.length - b.length;
    return a.word.localeCompare(b.word, 'zh');
  });

  return scored.slice(0, MAX_HINTS_PER_WORD).map(s => s.word);
}

// --- Main ------------------------------------------------------------------

function main() {
  console.log('[thesaurus] Loading HSK data...');
  const hskLevel = loadHsk();
  console.log(`  HSK1-3: ${hskLevel.size} words`);

  console.log('[thesaurus] Parsing CHLS (EMNLP 2023)...');
  const chls = parseChlsAll(path.join(RAW_DIR, 'chls'));
  console.log(`  ${chls.size} CHLS entries (vote-weighted, novel/news/wiki domains)`);

  console.log('[thesaurus] Parsing Chinese-LS...');
  const gold = parseChineseLS(path.join(RAW_DIR, 'chinese-ls-annotations.csv'));
  console.log(`  ${gold.size} Chinese-LS gold entries`);

  console.log('[thesaurus] Parsing Cilin Extended...');
  const cilin = parseCilin(path.join(RAW_DIR, 'cilin-extended.txt'));
  console.log(`  ${cilin.size} Cilin entries`);

  console.log('[thesaurus] Parsing chi-syn...');
  const chisyn = parseChisyn(path.join(RAW_DIR, 'chi-syn.txt'));
  console.log(`  ${chisyn.size} chi-syn entries`);

  // Union of all input words (words that might be "complex" worth simplifying)
  const allWords = new Set([...chls.keys(), ...gold.keys(), ...cilin.keys(), ...chisyn.keys()]);
  console.log(`[thesaurus] Unified ${allWords.size} unique words across sources`);

  const thesaurus = {};
  const stats = {
    totalWords: 0,
    withHints: 0,
    hopeless: 0,
    bySourceFirstHit: { chls: 0, gold: 0, cilin: 0, chisyn: 0 },
    hintCount: { 1: 0, 2: 0, 3: 0 },
  };

  for (const word of allWords) {
    // Only build entries for words that are NOT themselves in HSK1-3
    if (hskLevel.has(word)) continue;

    stats.totalWords++;

    const candidates = [];
    // CHLS carries vote counts (Map<sub, votes>)
    const chlsEntry = chls.get(word);
    if (chlsEntry) {
      for (const [sub, votes] of chlsEntry) {
        candidates.push({ word: sub, source: 'chls', votes });
      }
    }
    for (const c of gold.get(word) || []) candidates.push({ word: c, source: 'gold' });
    for (const c of cilin.get(word) || []) candidates.push({ word: c, source: 'cilin' });
    for (const c of chisyn.get(word) || []) candidates.push({ word: c, source: 'chisyn' });

    const hints = rankCandidates(word, candidates, hskLevel);

    if (hints.length > 0) {
      thesaurus[word] = hints;
      stats.withHints++;
      stats.hintCount[hints.length] = (stats.hintCount[hints.length] || 0) + 1;
      const top = candidates.find(c => c.word === hints[0]);
      if (top) stats.bySourceFirstHit[top.source]++;
    } else if (chls.has(word) || gold.has(word)) {
      // Word is in a human-annotated source but no alternative passes HSK filter
      thesaurus[word] = null;
      stats.hopeless++;
    }
  }

  console.log('[thesaurus] Writing output...');
  fs.writeFileSync(OUT_JSON, JSON.stringify(thesaurus, null, 2) + '\n');
  console.log(`  ${OUT_JSON}`);

  // Write report
  const report = {
    generated: new Date().toISOString(),
    sources: {
      gold: { size: gold.size, file: 'chinese-ls-annotations.csv' },
      cilin: { size: cilin.size, file: 'cilin-extended.txt' },
      chisyn: { size: chisyn.size, file: 'chi-syn.txt' },
    },
    hskLevels: { size: hskLevel.size, files: ['hsk1.json', 'hsk2.json', 'hsk3.json'] },
    stats,
    samples: {},
  };

  // Sample entries for manual review
  const sampleWords = ['完全', '之前', '控制', '敌人', '战争', '杀手', '家族', '指挥官', '撤退', '惩戒', '突然', '整顿', '追究', '厉害', '反而', '之后', '以前', '朋友', '开始', '举动'];
  for (const w of sampleWords) {
    report.samples[w] = thesaurus[w] !== undefined ? thesaurus[w] : 'NOT IN THESAURUS';
  }

  fs.writeFileSync(OUT_REPORT, JSON.stringify(report, null, 2) + '\n');
  console.log(`  ${OUT_REPORT}`);

  console.log('');
  console.log('=== Build summary ===');
  console.log(`Words considered:        ${stats.totalWords}`);
  console.log(`Words with hints:        ${stats.withHints}  (${(100*stats.withHints/stats.totalWords).toFixed(1)}%)`);
  console.log(`Words marked hopeless:   ${stats.hopeless}  (gold-verified "no HSK3 alt")`);
  console.log(`Top-hint source:         chls=${stats.bySourceFirstHit.chls} gold=${stats.bySourceFirstHit.gold} cilin=${stats.bySourceFirstHit.cilin} chisyn=${stats.bySourceFirstHit.chisyn}`);
  console.log(`Hint-count distribution: 1=${stats.hintCount[1]||0} 2=${stats.hintCount[2]||0} 3=${stats.hintCount[3]||0}`);
  console.log('');
  console.log('Sample lookups:');
  for (const w of sampleWords.slice(0, 12)) {
    const v = report.samples[w];
    const display = v === 'NOT IN THESAURUS' ? '(not complex or unknown)' :
                    v === null ? '(hopeless — no HSK3 alt)' :
                    v.join(' / ');
    console.log(`  ${w.padEnd(6)} → ${display}`);
  }
}

main();
