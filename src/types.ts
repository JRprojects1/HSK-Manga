export type HSKLevel = 'hsk1' | 'hsk2' | 'hsk3' | 'hsk4' | 'hsk5' | 'hsk6';

export interface SimplificationConfig {
  targetLevel: HSKLevel;
  strictMode: boolean;
  preserveNames: boolean;
  maxSentenceLength: number;
}

export interface WordMapping {
  original: string;
  simplified: string;
  reason: 'vocabulary' | 'idiom' | 'grammar' | 'preserved';
}

export interface SimplificationResult {
  original: string;
  simplified: string;
  mappings: WordMapping[];
  preservedAdvanced: string[];
  confidence: number;
}

export interface HSKVocabulary {
  word: string;
  pinyin?: string;
  level: number;
}

export interface TokenInfo {
  word: string;
  isInHSK: boolean;
  hskLevel: number | null;
}
