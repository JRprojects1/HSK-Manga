import * as fs from 'fs';
import * as path from 'path';

export interface DictionaryEntry {
  traditional: string;
  simplified: string;
  /** CC-CEDICT numbered pinyin, e.g. "di4 fang5". */
  pinyin: string;
  definitions: string[];
}

/**
 * CC-CEDICT stores one entry PER PRONUNCIATION. 地方 has two (di4 fang1
 * "regional", di4 fang5 "place"), 强 has seven, 过 has three.
 *
 * These used to be merged into a single entry that kept the FIRST entry's
 * pinyin and concatenated every reading's definitions. That one decision caused:
 *   - pinyin and definition being sourced from different readings (地方 showed
 *     "dì fāng" with entry 2's definition)
 *   - 48 occurrences in the sample chapter whose cedictDefinition belonged to a
 *     reading other than the one displayed (看 shown as kàn carrying kān)
 *   - isCedictProperNoun('强') === true, because 强's surname entry sorts first
 *   - 吗 defined as "used in 嗎啡|吗啡" (the má/morphine reading), 吧 as "bar
 *     (loanword)", 打 as "(loanword) dozen" — all HSK 1-2 words
 *
 * Entries are now kept per reading and one is chosen deliberately. See
 * resolveEntry().
 */
let dictMap: Map<string, DictionaryEntry[]> | null = null;

// ---------------------------------------------------------------- pinyin

const TONE_MARKS: Record<string, string[]> = {
  a: ['ā', 'á', 'ǎ', 'à', 'a'], e: ['ē', 'é', 'ě', 'è', 'e'],
  i: ['ī', 'í', 'ǐ', 'ì', 'i'], o: ['ō', 'ó', 'ǒ', 'ò', 'o'],
  u: ['ū', 'ú', 'ǔ', 'ù', 'u'], 'ü': ['ǖ', 'ǘ', 'ǚ', 'ǜ', 'ü'],
};

/** "di4 fang5" → "dì fang". Tone 5 (neutral) renders unmarked. */
function syllableToToneMark(syllable: string): string {
  const m = syllable.match(/^([a-zA-Zü:]+)([1-5])$/);
  if (!m) return syllable.toLowerCase();
  const body = m[1].toLowerCase().replace(/u:/g, 'ü').replace(/v/g, 'ü');
  const tone = parseInt(m[2], 10);
  if (tone === 5) return body;
  // Standard placement: a/e win; in "ou" the o takes it; otherwise the last vowel.
  let idx = body.indexOf('a');
  if (idx === -1) idx = body.indexOf('e');
  if (idx === -1 && body.includes('ou')) idx = body.indexOf('o');
  if (idx === -1) {
    for (let i = body.length - 1; i >= 0; i--) {
      if ('aeiouü'.includes(body[i])) { idx = i; break; }
    }
  }
  if (idx === -1) return body;
  const marked = TONE_MARKS[body[idx]] ? TONE_MARKS[body[idx]][tone - 1] : body[idx];
  return body.slice(0, idx) + marked + body.slice(idx + 1);
}

export function numberedToToneMarks(numbered: string): string {
  return numbered.trim().split(/\s+/).map(syllableToToneMark).join(' ');
}

const normNumbered = (p: string) => p.trim().toLowerCase().replace(/\s+/g, ' ');
const toneless = (p: string) => normNumbered(p).replace(/[1-5]/g, '');

/** Tone-marked pinyin → CC-CEDICT numbered form, for comparing a caller's hint. */
function toneMarkToNumbered(marked: string): string {
  return marked.trim().toLowerCase().split(/\s+/).map(syl => {
    let tone = 5;
    let out = '';
    for (const ch of syl) {
      let found = false;
      for (const [base, marks] of Object.entries(TONE_MARKS)) {
        const i = marks.indexOf(ch);
        if (i !== -1 && i < 4) { tone = i + 1; out += base; found = true; break; }
        if (i === 4) { out += base; found = true; break; }
      }
      if (!found) out += ch;
    }
    return out + tone;
  }).join(' ');
}

// ------------------------------------------------------------- HSK pinyin

/**
 * HSK 3.0 gives a curated, authoritative pinyin for exactly the vocabulary a
 * learner tool cares about, which CC-CEDICT alone cannot disambiguate. Where it
 * lists a single reading it is treated as the answer (没 → mei2, 强 → qiang2,
 * 吗 → ma5). Where it lists several (了 le5/liao3, 行 xing2/hang2) it abstains.
 */
let hskPinyinMap: Map<string, Set<string>> | null = null;

function loadHskPinyin(): Map<string, Set<string>> {
  const map = new Map<string, Set<string>>();
  const csv = path.join(__dirname, '..', 'data', 'hsk30.csv');
  if (!fs.existsSync(csv)) return map;
  try {
    const rows = fs.readFileSync(csv, 'utf-8').split(/\r?\n/).slice(1);
    for (const row of rows) {
      const cols = row.split(',');
      if (cols.length < 11) continue;
      const word = cols[1]?.trim();
      const m = cols[10]?.match(/\[([^\]]+)\]/);
      if (!word || !m) continue;
      if (!map.has(word)) map.set(word, new Set());
      map.get(word)!.add(normNumbered(m[1]));
    }
  } catch {
    // Optional data — absence just means this signal is unavailable.
  }
  return map;
}

function hskPinyin(word: string): string | undefined {
  if (!hskPinyinMap) hskPinyinMap = loadHskPinyin();
  const set = hskPinyinMap.get(word);
  return set && set.size === 1 ? [...set][0] : undefined;
}

// ---------------------------------------------------------------- loading

function parseLine(line: string): DictionaryEntry | null {
  const match = line.match(/^(\S+)\s+(\S+)\s+\[([^\]]+)\]\s+\/(.+)\/\s*$/);
  if (!match) return null;
  const [, traditional, simplified, pinyinRaw, defsRaw] = match;
  // Split on '/' ONLY. In CC-CEDICT '/' separates SENSES while ';' separates
  // SYNONYMS inside one sense — "intelligent; clever; bright" is a single meaning
  // phrased three ways, and "(bound form) to walk; to go; to travel" is three
  // bound readings of one sense, not one bound and two free.
  //
  // Splitting on ';' as well turned synonyms into rival candidates. That wasted
  // the display budget restating sense 1 (花 spent two of six slots on
  // "flower" and "blossom") and pushed real meanings off the end — it is why
  // "to spend (money, time)" and 成's "to become" needed the cap raised to find
  // them at all. Keeping senses whole fits more distinct meanings in less space
  // and preserves the grouping a learner most wants to see.
  const definitions = defsRaw
    .split('/')
    .map(d => d.trim())
    .filter(d => d.length > 0);
  return { traditional, simplified, pinyin: pinyinRaw, definitions };
}

function loadDictionary(): Map<string, DictionaryEntry[]> {
  const dictPath = path.join(__dirname, '..', 'data', 'cedict_ts.u8');

  if (!fs.existsSync(dictPath)) {
    console.error('[Dictionary] CC-CEDICT not found at', dictPath);
    console.error('[Dictionary] Download from: https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz');
    return new Map();
  }

  const start = Date.now();
  const lines = fs.readFileSync(dictPath, 'utf-8').split('\n');
  const map = new Map<string, DictionaryEntry[]>();

  for (const line of lines) {
    if (line.startsWith('#') || line.trim().length === 0) continue;
    const entry = parseLine(line);
    if (!entry) continue;
    // One bucket per headword, entries kept separate — never merged.
    const list = map.get(entry.simplified);
    if (list) list.push(entry);
    else map.set(entry.simplified, [entry]);
  }

  console.error(`[Dictionary] Loaded ${map.size} headwords in ${Date.now() - start}ms`);
  return map;
}

function ensureLoaded(): Map<string, DictionaryEntry[]> {
  if (!dictMap) dictMap = loadDictionary();
  return dictMap;
}

// ------------------------------------------------------------- sense filter

/**
 * Senses that are pointers or register notes rather than meanings. 'used in'
 * matters more than it looks: without it 单 resolves to "used in 單于|单于" and
 * 上 to "used in 上聲|上声".
 */
const SKIP_PREFIXES = [
  'variant of', 'old variant', 'archaic variant', 'erhua variant',
  'see ', 'used in ', 'abbr. for', 'CL:', 'surname ',
  'taiwan pr.', 'also pr.', 'also written', '(old)', '(Note:',
];

/**
 * Real senses, but rarely what a learner reading a comic needs. Ranked last.
 *
 * `Taiwan pr.` was in this list and did not belong: it annotates how a sense is
 * PRONOUNCED in Taiwan, not what register it belongs to. Because the regex
 * matches anywhere in the sense, 和's primary meaning — "(joining two nouns)
 * and; together with; with (Taiwan pr. [han4])" — took the archaic penalty,
 * sorted last, and was then dropped by the character budget. The most common
 * word in the chapter displayed "(math.) sum; to make peace; (sports) to draw;
 * to tie; harmonious; Japan; Japanese" and no "and" at all.
 *
 * 36 senses across CC-CEDICT carry the footnote and 7 of those are their
 * entry's primary sense (和 着 纵 幢 辙 圳 钴). The footnote is now stripped as
 * metadata in cleanDefinition, like CL: and (Note:, so it can neither mislead
 * the ranking nor spend the display budget.
 */
const ARCHAIC = /\((old|archaic|literary|classical|dialect|onom\.)[^)]*\)/i;
const BOUND = /\(bound form\)/i;

/**
 * How many senses a popup shows, applied to the ranked and filtered list.
 *
 * Measured on the sample chapter: the sense a sentence actually needs sits at
 * raw index 2-6 for common polysemous words (在 "located at" at 2, 或 "or" at 4,
 * 明白 "to understand" at 5, 花 "to spend" and 成 "to become" at 6). A cap of 4
 * cut several of them off entirely. Six covers them without turning the popup
 * into a wall of text; anything past that is the long tail CC-CEDICT keeps for
 * completeness.
 */
const SENSE_CAP = 6;

function cleanDefinition(def: string): string {
  return def
    .replace(/\s*\(CL:[^)]*\)/g, '')
    .replace(/\s*\bCL:\S+/g, '')
    .replace(/\s*\(Note:[^)]*\)/g, '')
    // Pronunciation footnotes are metadata, not meaning. Stripping them keeps
    // them from spending the character budget of the sense they annotate.
    //
    // Matched by SHAPE, not by a list of known wordings: a parenthetical that
    // mentions "pr." AND cites a reading in brackets. CC-CEDICT writes these 52
    // different ways — "(Taiwan pr. [x])", "(from Taiwanese, Tai-lo pr. [x])",
    // "(colloquial pr. [x])", "(in this sense, also pr. [x])" and a long tail —
    // and an earlier version of this rule listed two of them and missed 那 and
    // 这. 113 senses carry one; 78 of those are their entry's primary sense.
    //
    // The bracketed-reading requirement is what keeps ordinary grammatical
    // parentheticals safe: "(pronoun)" has no period after "pr", and "(old)",
    // "(math.)", "(bound form)" cite no reading. Verified against all 120,666
    // headwords with zero false positives.
    .replace(/\s*\([^)]*\bpr\.\s*\[[^\]]*\][^)]*\)/gi, '')
    .replace(/\bsth\b/g, 'something')
    .replace(/\bsb\b/g, 'somebody')
    .trim();
}

/** Senses with their CC-CEDICT register markers still attached, for ranking. */
function usableSenses(entry: DictionaryEntry): string[] {
  const kept = entry.definitions
    .filter(d => !SKIP_PREFIXES.some(p => d.toLowerCase().startsWith(p.toLowerCase())))
    .map(d => d.trim())
    .filter(d => d.length > 0);
  return kept;
}

/** usableSenses, but never empty — falls back to the raw senses. */
function displayableSenses(entry: DictionaryEntry): string[] {
  const kept = usableSenses(entry);
  if (kept.length > 0) return kept;
  return entry.definitions.map(d => d.trim()).filter(Boolean);
}

/**
 * Demote archaic and bound-form senses without dropping them. CC-CEDICT orders
 * senses by tradition, not usefulness — 行 lists nine bound senses ("to walk",
 * "to go", "temporary", "current"…) before the one a reader of 就行 needs
 * ("all right; OK; will do").
 *
 * The penalties must exceed the total order spread or they do nothing: a 15-sense
 * entry accumulates 2.1 of order weight, so a BOUND penalty of 1 would still lose
 * to position. ARCHAIC > BOUND > any amount of ordering.
 */
function rankSenses(senses: string[]): string[] {
  const ORDER_WEIGHT = 0.15;
  const BOUND_PENALTY = 4;
  const ARCHAIC_PENALTY = 8;
  return senses
    .map((s, i) => ({
      s,
      key: (ARCHAIC.test(s) ? ARCHAIC_PENALTY : 0)
         + (BOUND.test(s) ? BOUND_PENALTY : 0)
         + i * ORDER_WEIGHT,
    }))
    .sort((a, b) => a.key - b.key)
    .map(x => x.s);
}

/**
 * Re-attaching register markers per sense (see parseLine) is right for ranking
 * but repetitive to read: 这 came out as "(pronoun) this; (pronoun) these".
 * Keep the marker on its first appearance in a run and drop it from the rest.
 */
function collapseRepeatedMarkers(senses: string[]): string[] {
  let previous: string | null = null;
  return senses.map(sense => {
    const m = sense.match(/^(\([^)]*\))\s*(.+)$/);
    if (!m) { previous = null; return sense; }
    const [, marker, rest] = m;
    if (marker === previous) return rest;
    previous = marker;
    return sense;
  });
}

/**
 * "(bound form)" is a note about grammar, not meaning, and repeating it on every
 * sense makes a popup unreadable. It has already done its work in rankSenses by
 * this point, so drop it for display.
 */
function forDisplay(sense: string): string {
  return cleanDefinition(sense.replace(/\(bound form\)\s*/gi, ''));
}

// -------------------------------------------------------------- hand tables

/**
 * Entries CC-CEDICT states in a way that is accurate but useless in a popup.
 * 打's real entry opens "a semantically light, transitive verb that is combined
 * with a great variety of objects…", which is a lexicographer's note, not a
 * definition.
 */
const OVERRIDES: Record<string, string> = {
  '打': 'to hit; to strike; to play (a game)',
  '不想': 'to not want to; to be unwilling to',
  '招': 'a move; a trick; to beckon',
  '错': 'wrong; mistake; bad',
  '游': 'to swim; to tour; to roam',
  '版': 'version; edition',
  '一次': 'once; one time',
  '老大': 'eldest; boss; leader',
  '本': 'this; current; root; origin',
  // CC-CEDICT buries "don't ...!" at sense 11, behind "to leave", "to pin" and
  // several bound forms. In dialogue the prohibitive is overwhelmingly the one
  // meant (别急, 别动, 别说话).
  '别': "don't ...!; other; another; to leave; to part from",
};

/**
 * Closed-class grammar words. CC-CEDICT describes these for a linguist (跟 leads
 * with "heel", 吧 with "bar (loanword)"), and no ranking heuristic rescues them
 * because the unhelpful senses are neither archaic nor bound.
 *
 * `pinyin` names the reading the note describes, in CC-CEDICT numbered form. It
 * does double duty: it steers resolveEntry() to that reading, and it gates the
 * note so it is NOT shown when the word was resolved to a different reading —
 * an initial 啊 (ā, an interjection) must not display "sentence-final particle".
 */
const FUNCTION_WORDS: Record<string, { note: string; pinyin?: string }> = {
  '的': { note: 'possessive / descriptive particle', pinyin: 'de5' },
  '地': { note: 'adverbial particle — turns a description into "how"', pinyin: 'de5' },
  '得': { note: 'complement particle — how well an action is performed', pinyin: 'de5' },
  '了': { note: 'aspect particle — completed action or change of state', pinyin: 'le5' },
  '着': { note: 'aspect particle — ongoing state', pinyin: 'zhe5' },
  '过': { note: 'aspect particle — past experience', pinyin: 'guo5' },
  '没': { note: 'negation — did not; not have', pinyin: 'mei2' },
  '吗': { note: 'sentence-final particle — yes/no question', pinyin: 'ma5' },
  '呢': { note: 'sentence-final particle — softens a question', pinyin: 'ne5' },
  '吧': { note: 'sentence-final particle — suggestion', pinyin: 'ba5' },
  '啊': { note: 'sentence-final particle — emotion or emphasis', pinyin: 'a5' },
  '被': { note: 'passive marker — the subject receives the action' },
  '把': { note: 'disposal marker — what is done to the object' },
  '让': { note: 'causative — to let or make someone do something' },
  '从': { note: 'from — marks a starting point' },
  '跟': { note: 'with; together with — marks a companion or comparison' },
  '就': { note: 'adverb — then; as soon as' },
  '才': { note: 'adverb — only then; not until' },
  '也': { note: 'adverb — also; too' },
  '又': { note: 'adverb — again (repeated)' },
  '再': { note: 'adverb — again (a further time)' },
  '很': { note: 'adverb — very; neutral link before an adjective' },
  '不': { note: 'negation — not' },
  '而': { note: 'conjunction — and; yet; but' },
  '之': { note: 'classical possessive particle' },
  '所': { note: 'classical particle — forms a passive or nominalised phrase' },
};

// ------------------------------------------------------------- entry choice

const isProperEntry = (e: DictionaryEntry) => /^[A-Z]/.test(e.pinyin);
const isNeutralEntry = (e: DictionaryEntry) => /5(\s|$)/.test(e.pinyin);

/**
 * Pick which reading of `word` is meant.
 *
 * Order: HSK 3.0 authority → caller's pinyin hint (single chars) → neutral-tone
 * sibling (multi-char) → richest remaining. Proper-noun entries are demoted
 * throughout, and among equally-matching entries the richest wins rather than
 * the first — 后 [hou4] exists twice, as "empress" and as "back; behind", and
 * taking the first hands the caller a menu with no correct option on it.
 *
 * @param hintPinyin  tone-marked pinyin the caller already believes, if any
 */
export function resolveEntry(word: string, hintPinyin?: string): DictionaryEntry | undefined {
  const entries = ensureLoaded().get(word);
  if (!entries || entries.length === 0) return undefined;

  const ranked = [...entries]
    .sort((a, b) => (isProperEntry(a) ? 1 : 0) - (isProperEntry(b) ? 1 : 0));
  // A handful of headwords have NOTHING but pointer/register senses — 姆 is
  // "used in 姆媽", "(old) female tutor", "used in 保姆" and nothing else. Filtering
  // those away used to drop the word entirely, so the reader showed a blank
  // popup. A filtered sense beats no sense.
  const real = ranked.filter(e => usableSenses(e).length > 0);
  const pool = real.length > 0 ? real : ranked;
  if (pool.length === 0) return undefined;
  if (pool.length === 1) return pool[0];

  // Richness, used to break ties.
  //
  // Counts MEANINGS, not senses. Senses are kept whole for display, but one can
  // hold several synonyms — 地方's neutral entry is two senses containing seven
  // meanings ("place; location; area; space; territory" + "aspect; part") while
  // its full-tone sibling is three senses holding three. Counting senses made the
  // sparse entry look richer and cost 地方 its neutral reading, and 行 its xíng.
  //
  // Falls back to meaning-bearing senses when everything was filtered — otherwise
  // all candidates score 0 and the first wins arbitrarily, handing 姆 the
  // "used in 姆媽" entry over the one that at least says "female tutor".
  const meanings = (senses: string[]) =>
    senses.reduce((n, d) => n + d.split(';').filter(x => x.trim()).length, 0);
  const live = (e: DictionaryEntry) => {
    const kept = usableSenses(e).filter(d => !ARCHAIC.test(d));
    if (kept.length > 0) return meanings(kept);
    return meanings(displayableSenses(e).filter(d => !/^(used in |see )/i.test(d)));
  };
  const richest = (es: DictionaryEntry[]) => [...es].sort((a, b) => live(b) - live(a))[0];

  // The caller's hint comes FIRST. It is derived from sentence context (see
  // pinyin-context.ts), so it outranks HSK's static default and the grammar-word
  // table — both of which describe the word in isolation. Without this a
  // sentence-initial 啊 would be forced back to the a5 particle by HSK, and 过 in
  // 过了9年 back to the aspect particle by the function-word table.
  //
  // Single characters only: at that granularity a reading maps cleanly onto an
  // entry. For multi-character words the hint is usually a citation-tone guess
  // (pinyin-pro renders 地方 as "dì fāng"), so trusting it would defeat the
  // neutral-tone rule below.
  if ([...word].length === 1 && hintPinyin) {
    const target = toneMarkToNumbered(hintPinyin);
    const hits = pool.filter(e => normNumbered(e.pinyin) === target);
    if (hits.length) return richest(hits);
  }

  const hsk = hskPinyin(word);
  if (hsk) {
    const hits = pool.filter(e => normNumbered(e.pinyin) === hsk);
    if (hits.length) return richest(hits);
  }

  // Closed-class grammar words name their own reading. Without this 了 resolves
  // to liǎo "to finish" (the richer entry) instead of the aspect particle.
  const fwPinyin = FUNCTION_WORDS[word]?.pinyin;
  if (fwPinyin) {
    const hits = pool.filter(e => normNumbered(e.pinyin) === fwPinyin);
    if (hits.length) return richest(hits);
  }

  // Multi-character words usually differ only by a neutral tone, and the neutral
  // reading is the modern everyday word (故事 gu4 shi5, 东西 dong1 xi5,
  // 地方 di4 fang5). Only prefer it when it is at least as rich as its sibling —
  // 不是 [bu2 shi5] "fault; blame" is the rare reading, not the common one.
  if ([...word].length > 1) {
    const groups = new Map<string, DictionaryEntry[]>();
    for (const e of pool) {
      const k = toneless(e.pinyin);
      groups.set(k, [...(groups.get(k) || []), e]);
    }
    for (const list of groups.values()) {
      if (list.length < 2) continue;
      const neutral = list.find(isNeutralEntry);
      if (neutral && list.filter(o => o !== neutral).every(o => live(neutral) >= live(o))) {
        return neutral;
      }
    }
  }

  return richest(pool);
}

// ------------------------------------------------------------------- public

/** Every reading CC-CEDICT lists for this headword, unmerged. */
export function getEntries(word: string): DictionaryEntry[] {
  return ensureLoaded().get(word) ?? [];
}

/** The chosen reading. `hintPinyin` is tone-marked, e.g. "dì fang". */
export function getEntry(word: string, hintPinyin?: string): DictionaryEntry | undefined {
  return resolveEntry(word, hintPinyin);
}

/**
 * Learner-friendly definition for the chosen reading. Senses are filtered,
 * ranked and capped — and they now come from the SAME entry as getPinyin(),
 * so the two can no longer disagree.
 */
export function getDefinition(word: string, hintPinyin?: string): string | undefined {
  if (OVERRIDES[word]) return OVERRIDES[word];

  const entry = resolveEntry(word, hintPinyin);

  // The grammar-word note applies only when the word actually resolved to the
  // reading it describes. A sentence-initial 啊 resolves to ā and must show the
  // interjection sense, not "sentence-final particle".
  const fw = FUNCTION_WORDS[word];
  if (fw && (!fw.pinyin || !entry || normNumbered(entry.pinyin) === fw.pinyin)) {
    return fw.note;
  }

  if (!entry) return undefined;
  const senses = rankSenses(displayableSenses(entry)).map(forDisplay).filter(Boolean);
  if (senses.length === 0) return undefined;
  return withinCharBudget(collapseRepeatedMarkers(senses.slice(0, SENSE_CAP))).join('; ');
}

/**
 * A popup has a width budget as well as a sense budget.
 *
 * Sense count alone does not bound length — a few CC-CEDICT senses are sprawling
 * usage notes, so 只有 reached 508 characters while still inside the 6-sense cap,
 * and 厉害 343. Senses are dropped whole, never truncated mid-sense, and at least
 * one is always kept so a word whose single sense exceeds the budget still shows
 * something.
 *
 * Measured on the sample chapter: 120 shortens 36 of 385 words and loses none of
 * the senses their sentences actually need. It also lands 只有 on exactly the
 * text the per-sentence LLM pass chose for it, which is the outcome to aim for.
 *
 * Deliberately NOT applied to getCedictDefinition — a flashcard is reviewed with
 * no sentence around it and wants the full set.
 */
const MAX_DEFINITION_CHARS = 120;

function withinCharBudget(senses: string[]): string[] {
  const kept: string[] = [];
  let length = 0;
  for (const sense of senses) {
    const added = (kept.length ? 2 : 0) + sense.length;
    if (kept.length && length + added > MAX_DEFINITION_CHARS) break;
    kept.push(sense);
    length += added;
  }
  return kept;
}

/**
 * The chosen reading's senses straight from CC-CEDICT, skipping the OVERRIDES
 * and FUNCTION_WORDS tables.
 *
 * getDefinition() is what a learner should READ, so it prefers a hand-written
 * note where CC-CEDICT is unusable (吧 "bar (loanword)"). This is what the
 * reader stores as `cedictDefinition` to back the ⭐ flashcard button, and that
 * field means "CC-CEDICT had an entry" — its presence is what enables the
 * button. Letting a hand note leak into it would make the field lie about
 * whether the word is in the dictionary at all.
 */
export function getCedictDefinition(word: string, hintPinyin?: string): string | undefined {
  const entry = resolveEntry(word, hintPinyin);
  if (!entry) return undefined;
  const senses = rankSenses(displayableSenses(entry)).map(forDisplay).filter(Boolean);
  if (senses.length === 0) return undefined;
  // Uncapped, unlike the popup. A flashcard is reviewed with no sentence around
  // it, so the sense that mattered in the chapter may not be the first one —
  // showing only the top few would hide it. Pointer and register junk
  // ("variant of", "CL:", archaic) is still filtered out; this is the full set
  // of real senses for the resolved reading, best-first.
  return collapseRepeatedMarkers(senses).join('; ');
}

/** Tone-marked pinyin for the chosen reading, or undefined if not in CC-CEDICT. */
export function getPinyin(word: string, hintPinyin?: string): string | undefined {
  const entry = resolveEntry(word, hintPinyin);
  return entry ? numberedToToneMarks(entry.pinyin) : undefined;
}

/**
 * Heuristic proper-noun test using CC-CEDICT's romanization convention: proper
 * nouns are written with a capitalized pinyin (北京 → "Bei3 jing1"), ordinary
 * words lowercase. It is noisy (it also capitalizes God, languages, ethnic
 * groups, dynasties), so callers still layer their own filters on top.
 *
 * Now tested against the RESOLVED entry rather than whichever entry happened to
 * be first, which is what made isCedictProperNoun('强') return true — 强's
 * surname entry sorted ahead of "strong; powerful".
 */
export function isCedictProperNoun(word: string): boolean {
  const entry = resolveEntry(word);
  return !!entry && isProperEntry(entry);
}
