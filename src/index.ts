#!/usr/bin/env node

import * as dotenv from 'dotenv';
import * as path from 'path';

// Load .env file from project root
dotenv.config({ path: path.join(__dirname, '..', '.env') });

import { Command } from 'commander';
import * as fs from 'fs';
import * as readline from 'readline';
import { HSKLevel } from './types';
import { simplifyText } from './simplifier';
import { formatResult, formatResultJSON, formatStats } from './formatter';
import { getVocabularyStats } from './hsk-checker';
import {
  processMangaPage,
  processMangaPages,
  analyzeMangaPage,
  formatAnalysisResult,
  formatProcessingResult,
} from './manga-pipeline';
import { isSupportedImage, isPDF, extractPagesFromPDF } from './image-processor';
import { VisionProvider } from './text-extractor';
import { InputLanguage, isTranslationSource, VALID_INPUT_LANGUAGES } from './language';
import { processHTMLFile, saveProcessedHTML } from './html-processor';
import { downloadWebtoonChapter } from './webtoon-downloader';

const program = new Command();

program
  .name('chinese-simplify')
  .description('Simplify Chinese text to HSK-level vocabulary for learners')
  .version('0.1.0')
  // The root (default) command defines -l for plain-text input, which used to
  // swallow the -l flag of subcommands (`manga dir -l hsk6` silently ran at
  // the manga default). Positional option parsing scopes options written after
  // the subcommand name to that subcommand.
  .enablePositionalOptions();

program
  .argument('[input]', 'Input file path or Chinese text directly')
  .option('-l, --level <level>', 'Target HSK level (hsk1, hsk2, hsk3, or hsk4)', 'hsk2')
  .option('-j, --json', 'Output as JSON', false)
  .option('-s, --stats', 'Show vocabulary statistics only (no simplification)', false)
  .option('-o, --output <file>', 'Write output to file')
  .action(async (input: string | undefined, options) => {
    try {
      const level = options.level as HSKLevel;

      // Validate level
      if (!['hsk1', 'hsk2', 'hsk3', 'hsk4', 'hsk5', 'hsk6'].includes(level)) {
        console.error('Error: Level must be one of "hsk1"…"hsk6"');
        process.exit(1);
      }

      let text: string;

      if (input) {
        // Check if input is a file path or direct text
        if (fs.existsSync(input)) {
          text = fs.readFileSync(input, 'utf-8').trim();
        } else {
          // Treat as direct text input
          text = input;
        }
      } else {
        // Read from stdin
        text = await readStdin();
      }

      if (!text) {
        console.error('Error: No input text provided');
        console.error('Usage: chinese-simplify <text or file> [options]');
        console.error('   or: echo "中文文本" | chinese-simplify');
        process.exit(1);
      }

      // Stats-only mode
      if (options.stats) {
        const stats = getVocabularyStats(text, level);
        console.log(formatStats(stats));
        return;
      }

      // Simplify the text
      console.error('Simplifying text...\n');
      const startTime = Date.now();
      const result = await simplifyText(text, level);
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);

      // Format output
      let output: string;
      if (options.json) {
        output = formatResultJSON(result);
      } else {
        output = formatResult(result);
      }

      // Write to file or stdout
      if (options.output) {
        fs.writeFileSync(options.output, output);
        console.log(`Output written to ${options.output}`);
      } else {
        console.log(output);
      }

      // Show timing
      console.log(`\n⏱️ Completed in ${elapsed} seconds`);
    } catch (error) {
      if (error instanceof Error) {
        console.error('Error:', error.message);
      } else {
        console.error('An unexpected error occurred');
      }
      process.exit(1);
    }
  });

/**
 * Read text from stdin
 */
async function readStdin(): Promise<string> {
  return new Promise((resolve) => {
    // Check if stdin is a TTY (interactive terminal)
    if (process.stdin.isTTY) {
      resolve('');
      return;
    }

    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      terminal: false,
    });

    const lines: string[] = [];

    rl.on('line', (line) => {
      lines.push(line);
    });

    rl.on('close', () => {
      resolve(lines.join('\n').trim());
    });
  });
}

// Manga subcommand
program
  .command('manga <input>')
  .description('Process manga/manhua images - extract, simplify, and replace text')
  .option('-l, --level <levels>', 'Target HSK level(s), comma-separated (e.g. hsk2,hsk3)', 'hsk3')
  .option('-e, --english', 'Include English translation', false)
  .option('-o, --output <dir>', 'Output directory for processed images')
  .option('-a, --analyze', 'Analyze only - show text extraction and HSK stats without processing', false)
  .option('--no-convert', 'Do not convert Traditional to Simplified Chinese')
  .option('-p, --provider <provider>', 'Vision API provider: gemini, claude, paddleocr, or manga-ocr', 'gemini')
  .option('-s, --separate', 'Output separate files per page instead of combined scrollable image', false)
  .option('--input-language <lang>', 'Input language: auto, chinese, or english (default: auto)', 'auto')
  .action(async (input: string, options) => {
    try {
      // Parse multiple levels
      const validLevels = ['hsk1', 'hsk2', 'hsk3', 'hsk4', 'hsk5', 'hsk6'];
      const levelStrs = (options.level as string).split(',').map(s => s.trim());
      for (const l of levelStrs) {
        if (!validLevels.includes(l)) {
          console.error(`Error: Level "${l}" must be one of: ${validLevels.join(', ')}`);
          process.exit(1);
        }
      }
      let levels = levelStrs as HSKLevel[];
      // HSK_LEVEL env var override, kept for scripts written around the old
      // root-vs-subcommand -l conflict (fixed via enablePositionalOptions).
      const envOverride = process.env.HSK_LEVEL as HSKLevel | undefined;
      if (envOverride && ['hsk1','hsk2','hsk3','hsk4','hsk5','hsk6'].includes(envOverride)) {
        levels = [envOverride];
      }
      const level = levels[0]; // Primary level for simplification pipeline

      // Validate provider
      const provider = options.provider as VisionProvider;
      if (provider !== 'gemini' && provider !== 'claude' && provider !== 'paddleocr' && provider !== 'manga-ocr') {
        console.error('Error: Provider must be "gemini", "claude", "paddleocr", or "manga-ocr"');
        process.exit(1);
      }

      // Check if input exists
      if (!fs.existsSync(input)) {
        console.error(`Error: File not found: ${input}`);
        process.exit(1);
      }

      const stat = fs.statSync(input);

      // Handle directory, PDF, or image input
      let imagePaths: string[];
      if (stat.isDirectory()) {
        const files = fs.readdirSync(input);
        imagePaths = files
          .filter((f) => isSupportedImage(f))
          .sort((a, b) => {
            // Extract leading numbers from filenames for numeric comparison
            const numA = parseInt(a.match(/^(\d+)/)?.[1] || '0', 10);
            const numB = parseInt(b.match(/^(\d+)/)?.[1] || '0', 10);
            if (numA !== numB) return numA - numB;
            // Fall back to locale-aware comparison for non-numeric or equal prefixes
            return a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });
          })
          .map((f) => path.join(input, f));

        if (imagePaths.length === 0) {
          console.error('Error: No supported images found in directory');
          console.error('Supported formats: .png, .jpg, .jpeg, .webp, .gif, .pdf');
          process.exit(1);
        }
      } else if (isPDF(input)) {
        // Convert PDF to images
        console.error('Processing PDF file...');
        const pages = await extractPagesFromPDF(input);
        imagePaths = pages.map((p) => p.imagePath);
        console.error(`Found ${imagePaths.length} pages in PDF`);
      } else if (isSupportedImage(input)) {
        imagePaths = [input];
      } else {
        console.error('Error: Unsupported file format');
        console.error('Supported formats: .png, .jpg, .jpeg, .webp, .gif, .pdf');
        process.exit(1);
      }

      console.error(`Processing ${imagePaths.length} image(s)...`);
      console.error(`Target level${levels.length > 1 ? 's' : ''}: ${levels.map(l => l.toUpperCase()).join(', ')}`);
      if (options.english) console.error('English translation: enabled');
      console.error(`Vision provider: ${provider}`);
      console.error('');

      // Analyze mode
      if (options.analyze) {
        for (const imagePath of imagePaths) {
          console.error(`Analyzing: ${path.basename(imagePath)}`);
          const analysis = await analyzeMangaPage(imagePath, level, provider);
          console.log(formatAnalysisResult(analysis));
          console.log('');
        }
        return;
      }

      // Process mode
      const outputDir = options.output || path.join(path.dirname(input), 'simplified');
      const validLangs: string[] = VALID_INPUT_LANGUAGES;
      const inputLang = validLangs.includes(options.inputLanguage) ? options.inputLanguage : 'auto';
      const result = await processMangaPages(imagePaths, {
        targetLevel: level,
        targetLevels: levels,
        includeEnglish: options.english || false,
        inputLanguage: inputLang as InputLanguage,
        outputDir,
        convertTraditional: isTranslationSource(inputLang) ? false : (options.convert !== false ? undefined : false),
        visionProvider: provider,
        separateFiles: options.separate,
      });

      console.log(formatProcessingResult(result));
    } catch (error) {
      if (error instanceof Error) {
        console.error('Error:', error.message);
      } else {
        console.error('An unexpected error occurred');
      }
      process.exit(1);
    }
  });

// HTML subcommand
program
  .command('html <input>')
  .description('Process downloaded manhua HTML files - extract images, simplify text, and output new HTML')
  .option('-l, --level <level>', 'Target HSK level (hsk1, hsk2, hsk3, or hsk4)', 'hsk3')
  .option('-o, --output <file>', 'Output HTML file path')
  .option('-p, --provider <provider>', 'Vision API provider: gemini, claude, paddleocr, or manga-ocr', 'paddleocr')
  .option('--no-convert', 'Do not convert Traditional to Simplified Chinese')
  .action(async (input: string, options) => {
    try {
      const level = options.level as HSKLevel;

      // Validate level
      if (!['hsk1', 'hsk2', 'hsk3', 'hsk4', 'hsk5', 'hsk6'].includes(level)) {
        console.error('Error: Level must be one of "hsk1"…"hsk6"');
        process.exit(1);
      }

      // Validate provider
      const provider = options.provider as VisionProvider;
      if (provider !== 'gemini' && provider !== 'claude' && provider !== 'paddleocr' && provider !== 'manga-ocr') {
        console.error('Error: Provider must be "gemini", "claude", "paddleocr", or "manga-ocr"');
        process.exit(1);
      }

      // Check if input exists
      if (!fs.existsSync(input)) {
        console.error(`Error: File not found: ${input}`);
        process.exit(1);
      }

      // Check if it's an HTML file
      if (!input.toLowerCase().endsWith('.html') && !input.toLowerCase().endsWith('.htm')) {
        console.error('Error: Input must be an HTML file (.html or .htm)');
        process.exit(1);
      }

      console.error(`Processing HTML file: ${input}`);
      console.error(`Target level: ${level.toUpperCase()}`);
      console.error(`Vision provider: ${provider}`);
      console.error('');

      const startTime = Date.now();

      // Process the HTML file
      const processedHTML = await processHTMLFile(input, {
        level,
        provider,
        convertTraditional: options.convert !== false,
      });

      // Determine output path
      const outputPath = options.output || input.replace(/\.html?$/i, '_simplified.html');

      // Save processed HTML
      saveProcessedHTML(processedHTML, outputPath);

      const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
      console.error(`\n⏱️ Completed in ${elapsed} seconds`);

    } catch (error) {
      if (error instanceof Error) {
        console.error('Error:', error.message);
      } else {
        console.error('An unexpected error occurred');
      }
      process.exit(1);
    }
  });

// Webtoon subcommand
program
  .command('webtoon <url>')
  .description('Download and process a webtoon/manhua chapter from a URL')
  .option('-l, --level <level>', 'Target HSK level (hsk1, hsk2, hsk3, or hsk4)', 'hsk3')
  .option('-o, --output <dir>', 'Output directory for processed images')
  .option('-p, --provider <provider>', 'Vision API provider: gemini, claude, paddleocr, or manga-ocr', 'paddleocr')
  .option('--no-convert', 'Do not convert Traditional to Simplified Chinese')
  .option('-d, --download-only', 'Only download images, do not process', false)
  .action(async (url: string, options) => {
    try {
      const level = options.level as HSKLevel;

      // Validate level
      if (!['hsk1', 'hsk2', 'hsk3', 'hsk4', 'hsk5', 'hsk6'].includes(level)) {
        console.error('Error: Level must be one of "hsk1"…"hsk6"');
        process.exit(1);
      }

      // Validate provider
      const provider = options.provider as VisionProvider;
      if (provider !== 'gemini' && provider !== 'claude' && provider !== 'paddleocr' && provider !== 'manga-ocr') {
        console.error('Error: Provider must be "gemini", "claude", "paddleocr", or "manga-ocr"');
        process.exit(1);
      }

      // Validate URL
      if (!url.startsWith('http://') && !url.startsWith('https://')) {
        console.error('Error: URL must start with http:// or https://');
        process.exit(1);
      }

      const outputDir = options.output || './webtoon-downloads';

      console.error(`Downloading webtoon chapter from: ${url}`);
      console.error(`Output directory: ${outputDir}`);
      console.error('');

      const startTime = Date.now();

      // Download the chapter
      const downloadResult = await downloadWebtoonChapter(url, outputDir);

      if (downloadResult.downloadedImages === 0) {
        console.error('Error: No images were downloaded');
        process.exit(1);
      }

      // If download-only mode, stop here
      if (options.downloadOnly) {
        console.error(`\n✓ Downloaded ${downloadResult.downloadedImages} images to: ${downloadResult.outputDir}`);
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
        console.error(`⏱️ Completed in ${elapsed} seconds`);
        return;
      }

      // Process the downloaded images
      console.error(`\nProcessing ${downloadResult.imagePaths.length} images...`);
      console.error(`Target level: ${level.toUpperCase()}`);
      console.error(`Vision provider: ${provider}`);
      console.error('');

      const simplifiedDir = path.join(downloadResult.outputDir, 'simplified');
      const result = await processMangaPages(downloadResult.imagePaths, {
        targetLevel: level,
        outputDir: simplifiedDir,
        convertTraditional: options.convert !== false,
        visionProvider: provider,
        separateFiles: false, // Combine into scrollable HTML
      });

      console.log(formatProcessingResult(result));

      const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
      console.error(`\n⏱️ Total time: ${elapsed} seconds`);

    } catch (error) {
      if (error instanceof Error) {
        console.error('Error:', error.message);
      } else {
        console.error('An unexpected error occurred');
      }
      process.exit(1);
    }
  });

program.parse();
