import chalk from 'chalk';
import { SimplificationResult, WordMapping } from './types';
import { analyzeText } from './hsk-checker';
import { tokenize, isPunctuation, isNumber } from './tokenizer';

/**
 * Highlight advanced words in text that are still above HSK 1-2
 */
function highlightAdvancedWords(text: string): { highlighted: string; advancedWords: string[] } {
  const tokens = tokenize(text);
  const analysis = analyzeText(text, 'hsk2');
  const advancedWords: string[] = [];

  let highlighted = '';
  let tokenIndex = 0;
  let textIndex = 0;

  while (textIndex < text.length) {
    if (tokenIndex < tokens.length) {
      const token = tokens[tokenIndex];
      const tokenStart = text.indexOf(token, textIndex);

      // Add any characters before this token (spaces, etc.)
      if (tokenStart > textIndex) {
        highlighted += text.substring(textIndex, tokenStart);
      }

      const tokenInfo = analysis[tokenIndex];
      if (tokenInfo && !tokenInfo.isInHSK && !isPunctuation(token) && !isNumber(token) && token.trim()) {
        // Highlight advanced word in red with underline
        highlighted += chalk.red.underline(token);
        if (!advancedWords.includes(token)) {
          advancedWords.push(token);
        }
      } else {
        highlighted += token;
      }

      textIndex = tokenStart + token.length;
      tokenIndex++;
    } else {
      // Add remaining text
      highlighted += text.substring(textIndex);
      break;
    }
  }

  return { highlighted, advancedWords };
}

/**
 * Format a single simplification result for terminal output
 */
export function formatResult(result: SimplificationResult): string {
  const lines: string[] = [];

  // Header separator
  const separator = chalk.gray('═'.repeat(60));
  lines.push(separator);

  // Original text
  lines.push(chalk.bold('原文: ') + result.original);

  // Analyze and highlight advanced words in simplified text
  const { highlighted, advancedWords } = highlightAdvancedWords(result.simplified);

  // Simplified text with highlighting
  lines.push(chalk.bold.green('简化: ') + highlighted);

  lines.push(separator);

  // Show stats for simplified output
  const simplifiedAnalysis = analyzeText(result.simplified, 'hsk2');
  const relevantTokens = simplifiedAnalysis.filter(t => !isPunctuation(t.word) && !isNumber(t.word) && t.word.trim());
  const inHSK = relevantTokens.filter(t => t.isInHSK).length;
  const coverage = relevantTokens.length > 0 ? (inHSK / relevantTokens.length * 100).toFixed(1) : '100';

  lines.push('');
  lines.push(chalk.bold('简化后HSK覆盖率: ') + chalk.cyan(coverage + '%') +
    (parseFloat(coverage) < 80 ? chalk.yellow(' (仍有较多高级词汇)') : chalk.green(' ✓')));

  // List advanced words still in simplified text
  if (advancedWords.length > 0) {
    lines.push('');
    lines.push(chalk.bold.red('⚠️ 简化文本中仍有的高级词汇 (红色下划线):'));
    lines.push('  ' + advancedWords.join(', '));
  }

  // Word mappings
  if (result.mappings.length > 0) {
    lines.push('');
    lines.push(chalk.bold('词汇变化:'));

    for (const mapping of result.mappings) {
      const arrow = chalk.yellow(' → ');
      const reasonTag = formatReason(mapping.reason);
      lines.push(`  ${mapping.original}${arrow}${mapping.simplified} ${reasonTag}`);
    }
  }

  // Preserved advanced words
  if (result.preservedAdvanced.length > 0) {
    lines.push('');
    lines.push(
      chalk.bold.yellow('⚠️ 保留的高级词汇: ') +
        result.preservedAdvanced.join(', ')
    );
  } else {
    lines.push('');
    lines.push(chalk.dim('⚠️ 保留的高级词汇: (无)'));
  }

  lines.push('');

  return lines.join('\n');
}

/**
 * Format reason tag with color
 */
function formatReason(reason: WordMapping['reason']): string {
  switch (reason) {
    case 'vocabulary':
      return chalk.cyan('[词汇]');
    case 'idiom':
      return chalk.magenta('[成语]');
    case 'grammar':
      return chalk.blue('[语法]');
    case 'preserved':
      return chalk.yellow('[保留]');
    default:
      return '';
  }
}

/**
 * Format result as JSON
 */
export function formatResultJSON(result: SimplificationResult): string {
  return JSON.stringify(result, null, 2);
}

/**
 * Format multiple results
 */
export function formatResults(results: SimplificationResult[], asJSON: boolean = false): string {
  if (asJSON) {
    return JSON.stringify(results, null, 2);
  }

  return results.map((r) => formatResult(r)).join('\n');
}

/**
 * Format a simple side-by-side comparison
 */
export function formatComparison(original: string, simplified: string): string {
  const lines: string[] = [];

  lines.push(chalk.bold.white('┌─ 原文 ─────────────────────────────────────────────┐'));
  lines.push(chalk.white('│ ') + original);
  lines.push(chalk.bold.white('├─ 简化 ─────────────────────────────────────────────┤'));
  lines.push(chalk.green('│ ') + simplified);
  lines.push(chalk.bold.white('└───────────────────────────────────────────────────┘'));

  return lines.join('\n');
}

/**
 * Format vocabulary statistics
 */
export function formatStats(stats: {
  total: number;
  hsk1: number;
  hsk2: number;
  hsk3: number;
  hsk4: number;
  hsk5?: number;
  hsk6?: number;
  outsideHSK: number;
  coverage: number;
}): string {
  const lines: string[] = [];

  lines.push(chalk.bold('词汇统计:'));
  lines.push(`  总词数: ${stats.total}`);
  lines.push(`  HSK1 词汇: ${chalk.green(stats.hsk1.toString())}`);
  lines.push(`  HSK2 词汇: ${chalk.cyan(stats.hsk2.toString())}`);
  lines.push(`  HSK3 词汇: ${chalk.magenta(stats.hsk3.toString())}`);
  lines.push(`  HSK4 词汇: ${chalk.yellow(stats.hsk4.toString())}`);
  if (stats.hsk5 !== undefined) lines.push(`  HSK5 词汇: ${chalk.blue(stats.hsk5.toString())}`);
  if (stats.hsk6 !== undefined) lines.push(`  HSK6 词汇: ${chalk.blueBright(stats.hsk6.toString())}`);
  lines.push(`  超纲词汇: ${chalk.red(stats.outsideHSK.toString())}`);
  lines.push(`  覆盖率: ${chalk.bold(stats.coverage.toFixed(1) + '%')}`);

  return lines.join('\n');
}
