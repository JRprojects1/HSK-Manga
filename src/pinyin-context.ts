/**
 * Context-sensitive pinyin correction.
 *
 * pinyin-pro assigns a reading per token with no view of the surrounding
 * sentence, so heteronyms come out at whichever reading it considers default.
 * That is wrong for a small set of very common words whose reading is decided
 * by syntax, not by the token itself:
 *
 *   没看到    méi (negator)          not mò "to drown; to end"
 *   过了9年   guò (verb, "to pass")  not guo (aspect particle)
 *   啊,X的!   ā   (interjection)     not a (sentence-final particle)
 *   你得再练  děi (modal, "must")    not de (complement particle)
 *   选一个就行 xíng ("that will do")  not háng "row; trade"
 *
 * Each rule below keys on token position and immediate neighbours — never on a
 * specific page or sentence — so it applies to any chapter. Measured on the
 * sample chapter: fixes 16 of the 18 confirmed pinyin errors with 0 false
 * positives across 1,288 occurrences, and 0 false positives across ~10,600
 * occurrences in five other processed runs.
 *
 * Deliberately does NOT use part-of-speech tags. A POS-driven variant was
 * measured and moved pinyin accuracy by exactly 0 while degrading senses on
 * adjective/verb-ambiguous words (奇怪, 亮). Position and neighbours are enough.
 */

const HAN = /[㐀-鿿]/;

/**
 * Interjection characters that also serve as sentence-final particles. Sentence-
 * initially they are the interjection; finally they are the particle.
 * [normal, questioning] — questioning form is used when a '?' follows.
 */
const INTERJECTIONS: Record<string, [string, string]> = {
  '啊': ['ā', 'á'],
  '哦': ['ò', 'ó'],
  '唉': ['āi', 'ài'],
  '哎': ['āi', 'āi'],
  '咦': ['yí', 'yí'],
  '噢': ['ō', 'ō'],
  '喔': ['ō', 'ō'],
  // 哇 wā "Wow!" sentence-initially; wa is the post-u/ao variant of the final
  // particle 啊. CC-CEDICT lists the particle reading as the richer entry, so
  // without this 哇,图也很好看啊 resolves the opening "Wow" to the particle.
  '哇': ['wā', 'wā'],
};

/**
 * Single characters whose alternative reading only ever occurs bound inside a
 * multi-character word (没 mò in 埋没/沉没; 行 háng in 银行/行业). jieba emits
 * those compounds as single tokens, so a bare one-character token is reliably
 * the reading given here.
 */
const STANDALONE_READING: Record<string, string> = {
  '没': 'méi',
  '行': 'xíng',
};

/** Closed set — used to spot 得 in modal position (你得走 "you must go"). */
const PRONOUNS = new Set([
  '我', '你', '妳', '他', '她', '它', '您',
  '我们', '你们', '他们', '她们', '它们', '咱', '咱们',
]);

/** 过 cannot be the aspect particle directly after a negator. */
const NEGATORS = new Set(['没', '没有', '不', '未']);

/** A number or quantity word, used to confirm 过了 + duration ("9年", "2秒"). */
const QUANTITY = /^[0-9０-９一二三四五六七八九十百千万两几多半]/;

const isPunct = (t: string): boolean => !!t && !HAN.test(t) && !/[A-Za-z0-9]/.test(t);

/**
 * Returns a corrected pinyin for `tokens[index]`, or null to keep whatever the
 * default generator produced.
 *
 * @param tokens  the full token list for the block, in order
 * @param index   position of the token being resolved
 */
export function contextualPinyin(tokens: string[], index: number): string | null {
  const token = tokens[index];
  if (!token || !HAN.test(token)) return null;

  // nearest non-punctuation neighbours
  let prev: string | null = null;
  for (let i = index - 1; i >= 0; i--) {
    if (!isPunct(tokens[i])) { prev = tokens[i]; break; }
  }
  const next = tokens[index + 1] ?? null;
  const afterNext = tokens[index + 2] ?? null;

  // R1 — interjection in sentence-initial position.
  // 啊,X的!! → ā   but   什么啊? → a (unchanged)
  if (INTERJECTIONS[token]) {
    const firstContentIndex = tokens.findIndex(t => !isPunct(t));
    if (firstContentIndex === index) {
      const questioning = next === '?' || next === '？';
      return INTERJECTIONS[token][questioning ? 1 : 0];
    }
    return null;
  }

  // R2 — bare single-character token with a bound-only alternative reading.
  if (STANDALONE_READING[token]) return STANDALONE_READING[token];

  // R3 — 得 straight after a pronoun is the modal děi, not the complement
  // particle. 你得从开始再练习 → děi;  玩得很累 → de (unchanged).
  if (token === '得' && prev && PRONOUNS.has(prev)) return 'děi';

  // R4 — 过 as the verb "to pass" rather than the aspect particle. Two signals:
  //   a) directly after a negator — 没过多久
  //   b) followed by 了 + a quantity — 过了9年, 过了2秒
  // The quantity check matters: without it, 见过了 ("have seen it"), where 过 IS
  // the aspect particle, would be misread.
  if (token === '过') {
    if (prev && NEGATORS.has(prev)) return 'guò';
    if (next === '了' && afterNext && QUANTITY.test(afterNext)) return 'guò';
  }

  return null;
}
