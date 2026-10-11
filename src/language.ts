/**
 * Shared input-language model.
 *
 * The pipeline distinguishes two orthogonal properties of a source language:
 *
 * 1. SEMANTIC: is the source translated into Chinese (english/japanese/korean)
 *    or simplified in-language (chinese)? → isTranslationSource()
 * 2. GEOMETRY/DENSITY: do the hand-tuned box thresholds treat text as
 *    CJK-dense (chinese/japanese) or Latin-like (english/korean)?
 *    → usesLatinDensity()
 *
 * Japanese deliberately splits the two: it translates like English but its
 * boxes are CJK-shaped. Korean is provisionally on the Latin side of density
 * (hangul runs longer than han for the same content) — revisit against real
 * Korean chapters.
 */
export type InputLanguage = 'chinese' | 'english' | 'japanese' | 'korean' | 'auto';

export const VALID_INPUT_LANGUAGES: InputLanguage[] = ['chinese', 'english', 'japanese', 'korean', 'auto'];

/** Source is translated into Chinese (vs Chinese input, which is simplified). */
export function isTranslationSource(lang: InputLanguage | string | undefined): boolean {
  return lang === 'english' || lang === 'japanese' || lang === 'korean';
}

/** English name of the source language, for LLM prompt text. */
export function sourceLanguageName(lang: InputLanguage | string | undefined): string {
  switch (lang) {
    case 'japanese': return 'Japanese';
    case 'korean': return 'Korean';
    case 'chinese': return 'Chinese';
    default: return 'English';
  }
}

/**
 * Which side of the tuned geometry/density thresholds a language takes
 * (gap ratios, height-similarity bounds, long-block char thresholds,
 * small-region filters). Chinese/Japanese = CJK side; English/Korean = Latin
 * side. See file header for the Korean caveat.
 */
export function usesLatinDensity(lang: InputLanguage | string | undefined): boolean {
  return lang === 'english' || lang === 'korean';
}

/**
 * Bias maps only exist in two flavors (Chinese-input vs translation-input).
 * Japanese/Korean ride the English-side bias configuration.
 */
export function toBiasLanguage(lang: InputLanguage | string | undefined): 'chinese' | 'english' | 'auto' {
  if (lang === 'english' || lang === 'japanese' || lang === 'korean') return 'english';
  if (lang === 'auto') return 'auto';
  return 'chinese';
}

// U+30FC (ー, prolonged sound mark) is Script=Common, so the Katakana property
// alone misses it — it appears constantly in real kana text (ラーメン).
const HAN_RE = /\p{Script=Han}/u;
const KANA_RE = /[\p{Script=Hiragana}\p{Script=Katakana}ー]/u;
const HANGUL_RE = /\p{Script=Hangul}/u;

export function isHanChar(ch: string): boolean { return HAN_RE.test(ch); }
export function isKanaChar(ch: string): boolean { return KANA_RE.test(ch); }
export function isHangulChar(ch: string): boolean { return HANGUL_RE.test(ch); }

/**
 * Count characters that constitute source-language CONTENT for a block/region.
 * Used by the incidental-text drop: a block with zero content chars is a game
 * title, watermark, SFX, or UI label — never dialogue to translate/simplify.
 *
 * chinese → Han. japanese → Han + kana (a pure-kana bubble is real dialogue).
 * korean → Hangul + Han (rare hanja). english → any Latin letter (the drop is
 * not applied to English input, but the definition keeps callers total).
 */
export function countSourceContentChars(text: string, lang: InputLanguage | string | undefined): number {
  let count = 0;
  for (const ch of text) {
    if (HAN_RE.test(ch)) {
      if (lang !== 'korean') { count++; continue; }
      // Korean: han (hanja) counts too — it is content, just rare.
      count++;
      continue;
    }
    if (lang === 'japanese' && KANA_RE.test(ch)) count++;
    else if (lang === 'korean' && HANGUL_RE.test(ch)) count++;
    else if (lang === 'english' && /\p{Script=Latin}/u.test(ch)) count++;
  }
  return count;
}
