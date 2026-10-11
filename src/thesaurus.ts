import * as fs from 'fs';
import * as path from 'path';

const DISABLED = process.env.THESAURUS_DISABLED === 'true';

const THESAURUS: Record<string, string[] | null> = DISABLED
  ? {}
  : JSON.parse(
      fs.readFileSync(
        path.join(__dirname, '..', 'data', 'simplification-thesaurus.json'),
        'utf-8'
      )
    );

export function getHints(word: string): string[] {
  const v = THESAURUS[word];
  return Array.isArray(v) ? v : [];
}

export function isKnownHopeless(word: string): boolean {
  return THESAURUS[word] === null;
}
