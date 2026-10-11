#!/usr/bin/env node
/**
 * Seeds the shared word-audio library with citation clips for HSK vocabulary.
 *
 *   node scripts/seed-word-library.js          # HSK 1-4 (default)
 *   node scripts/seed-word-library.js --hsk6   # HSK 1-6
 *
 * One-time (per voice): a word's isolated pronunciation never changes, so
 * this is the last time most of the vocabulary is ever synthesized. Chapter
 * builds add their story-specific tail to the same library lazily.
 * Re-running is safe and cheap — existing entries are skipped.
 *
 * Requires a build (`npm run build`) and `pip install "edge-tts>=7"`.
 */
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const { synthesizeWordsToLibrary, defaultWordLibraryDir } =
  require(path.join(ROOT, 'dist', 'audio-generator.js'));

const levels = process.argv.includes('--hsk6') ? [1, 2, 3, 4, 5, 6] : [1, 2, 3, 4];
const words = new Set();
for (const lvl of levels) {
  const file = path.join(ROOT, 'data', `hsk${lvl}.json`);
  if (!fs.existsSync(file)) {
    console.error(`missing ${file}, skipping`);
    continue;
  }
  for (const w of JSON.parse(fs.readFileSync(file, 'utf8'))) words.add(w);
}

const voice = process.env.TTS_VOICE || 'zh-CN-XiaoyiNeural';
const libraryDir = defaultWordLibraryDir();
console.log(`seeding ${words.size} HSK ${levels[0]}-${levels[levels.length - 1]} words`
  + ` -> ${libraryDir} (voice ${voice})`);

const t0 = Date.now();
synthesizeWordsToLibrary([...words], { libraryDir, voice, onLog: (m) => console.log('  ' + m) })
  .then((entries) => {
    const index = JSON.parse(fs.readFileSync(path.join(libraryDir, 'index.json'), 'utf8'));
    console.log(`done in ${((Date.now() - t0) / 1000).toFixed(1)}s: `
      + `${Object.keys(entries).length}/${words.size} requested words available, `
      + `library now holds ${Object.keys(index.words).length} words`);
  })
  .catch((err) => {
    console.error('seed failed:', err.message || err);
    process.exit(1);
  });
