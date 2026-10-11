import * as fs from 'fs';
import * as path from 'path';
import pLimit from 'p-limit';
import { pinyin } from 'pinyin-pro';
import sharp from 'sharp';
import { loadImage, saveImage, pickImageExtension, PageImage, isSupportedImage, generateScrollableHTML, generateInteractiveReaderHTML, ReaderData, ReaderPageData, ReaderBlockData } from './image-processor';
import {
  extractTextFromImage,
  convertExtractionToSimplified,
  ExtractionResult,
  TextRegion,
  traditionalToSimplified,
  simplifiedToTraditionalInContext,
  VisionProvider,
  getOCRServerCount,
  hasKoreanOCRPool,
} from './text-extractor';
import { replaceTextInImage, inpaintImageRegions, ReplacementRegion, RenderedBlockInfo, groupNearbyRegions, detectSuspiciousGray, sampleCornerBrightness, GrayDetectionResult, compactPunctuation, compactAdjacentPunctuation, computeRenderBoxPct } from './text-replacer';
import { verifyGrayRegionsWithGemini, checkLeadingPunctuation } from './gemini-verifier';
import { simplifyText, simplifyTextBatch, simplifyBlocks, simplifyBlocksMultiLevel, simplifyPageBatch, TextBlock, ChapterContext, analyzeChapter, analyzeChapterStreaming, detectProvider, smartJoin, resetLLMUsage, getKeepNegBiasIds } from './simplifier';
import { calculateBackgroundVariance } from './inpainting';
import { HSKLevel, SimplificationResult } from './types';
import { tokenize, isPunctuation, isNumber, isEnglish } from './tokenizer';
import { contextualPinyin } from './pinyin-context';
import { generateBubbleGlosses, GlossTarget } from './bubble-gloss';
import { getHSKLevel, isWithinLevel } from './hsk-checker';
import { getVocabularyStats } from './hsk-checker';
import { getDefinition, getCedictDefinition, getPinyin, isCedictProperNoun } from './dictionary';
import { InputLanguage, isTranslationSource, usesLatinDensity, sourceLanguageName, countSourceContentChars, isHanChar, isKanaChar, isHangulChar } from './language';

/**
 * Check if text pixels in a bounding box region have high color saturation.
 * Used to detect art-overlay text (colorful text drawn on artwork) vs normal
 * speech bubble text (black/white/gray).
 *
 * Approach: Use Otsu's method to binarize the region and isolate actual text
 * characters (the minority class). Then measure chroma of only those text pixels.
 * This avoids false positives from artwork bleeding through translucent boxes.
 */
// Synchronous variant — caller decodes pageImage.buffer to raw pixels ONCE and
// passes the shared buffer in. Avoids the prior per-region sharp.extract which
// re-decoded the full page each call (~37ms × 15 regions × 113 pages = ~63s).
function checkTextColorSaturation(
  pageData: Buffer,
  pageWidth: number,
  pageHeight: number,
  pageChannels: number,
  boundingBox: { x: number; y: number; width: number; height: number },
): { isColorful: boolean; avgChroma: number; textPixelCount: number; hasSolidBackground: boolean; bgStdDev: number } {
  // Convert percentage bbox to pixels
  const pixelX = Math.max(0, Math.round((boundingBox.x / 100) * pageWidth));
  const pixelY = Math.max(0, Math.round((boundingBox.y / 100) * pageHeight));
  let pixelW = Math.round((boundingBox.width / 100) * pageWidth);
  let pixelH = Math.round((boundingBox.height / 100) * pageHeight);

  // Clamp to image bounds
  pixelW = Math.min(pixelW, pageWidth - pixelX);
  pixelH = Math.min(pixelH, pageHeight - pixelY);
  if (pixelW <= 2 || pixelH <= 2) return { isColorful: false, avgChroma: 0, textPixelCount: 0, hasSolidBackground: false, bgStdDev: 0 };

  // Sample pixels directly from the shared raw buffer (no sharp.extract).
  // Limit to ~1000 samples for compute cost.
  const totalPixels = pixelW * pixelH;
  const step = Math.max(1, Math.floor(Math.sqrt(totalPixels / 1000)));
  const stride = pageWidth * pageChannels;

  const sampledPixels: Array<{ r: number; g: number; b: number; brightness: number }> = [];
  for (let y = 0; y < pixelH; y += step) {
    const rowOff = (pixelY + y) * stride + pixelX * pageChannels;
    for (let x = 0; x < pixelW; x += step) {
      const idx = rowOff + x * pageChannels;
      if (idx + 2 >= pageData.length) continue;
      const r = pageData[idx];
      const g = pageData[idx + 1];
      const b = pageData[idx + 2];
      const brightness = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
      sampledPixels.push({ r, g, b, brightness });
    }
  }

  if (sampledPixels.length < 10) return { isColorful: false, avgChroma: 0, textPixelCount: 0, hasSolidBackground: false, bgStdDev: 0 };

  // Otsu's method: find brightness threshold that best separates text from background
  const histogram = new Array(256).fill(0);
  for (const p of sampledPixels) {
    histogram[p.brightness]++;
  }

  const total = sampledPixels.length;
  let sumTotal = 0;
  for (let i = 0; i < 256; i++) {
    sumTotal += i * histogram[i];
  }

  let bestThreshold = 0;
  let bestVariance = 0;
  let sumBg = 0;
  let weightBg = 0;

  for (let t = 0; t < 256; t++) {
    weightBg += histogram[t];
    if (weightBg === 0) continue;
    const weightFg = total - weightBg;
    if (weightFg === 0) break;
    sumBg += t * histogram[t];
    const meanBg = sumBg / weightBg;
    const meanFg = (sumTotal - sumBg) / weightFg;
    const variance = weightBg * weightFg * (meanBg - meanFg) ** 2;
    if (variance > bestVariance) {
      bestVariance = variance;
      bestThreshold = t;
    }
  }

  // Text = minority class (text characters occupy less area than background)
  const darkPixels = sampledPixels.filter(p => p.brightness <= bestThreshold);
  const lightPixels = sampledPixels.filter(p => p.brightness > bestThreshold);
  const textPixels = darkPixels.length <= lightPixels.length ? darkPixels : lightPixels;

  if (textPixels.length < 5) return { isColorful: false, avgChroma: 0, textPixelCount: textPixels.length, hasSolidBackground: false, bgStdDev: 0 };

  // Measure chroma of text pixels only
  let totalChroma = 0;
  for (const p of textPixels) {
    const chroma = Math.max(p.r, p.g, p.b) - Math.min(p.r, p.g, p.b);
    totalChroma += chroma;
  }

  const avgChroma = totalChroma / textPixels.length;
  // Threshold: 15 — Otsu isolates actual text characters.
  // Black/white text: chroma ≈ 0. Colored text (brown, red): chroma 30+.
  // JPEG artifacts: chroma 2-5. Threshold of 15 gives safe margin.

  // Background uniformity check: speech bubbles have solid fill (low brightness variance),
  // while SFX on manga art has textured backgrounds (high variance).
  const bgPixels = darkPixels.length <= lightPixels.length ? lightPixels : darkPixels;
  let bgSum = 0;
  for (const p of bgPixels) bgSum += p.brightness;
  const bgMean = bgSum / bgPixels.length;
  let bgVariance = 0;
  for (const p of bgPixels) bgVariance += (p.brightness - bgMean) ** 2;
  bgVariance /= bgPixels.length;
  const bgStdDev = Math.sqrt(bgVariance);
  // Solid fill: stddev < 15 (uniform white/black/gray). Manga art: stddev 30+.
  const hasSolidBackground = bgStdDev < 15;

  return { isColorful: avgChroma > 15, avgChroma, textPixelCount: textPixels.length, hasSolidBackground, bgStdDev };
}

/**
 * Merge adjacent text regions on the same horizontal line.
 * This handles Traditional Chinese OCR fragmenting English+Chinese into separate boxes.
 *
 * Example: "LINE", "WEBTOON", "致力於" as separate boxes → "LINE WEBTOON致力於" as one box
 *
 * @param regions - Array of text regions from OCR
 * @returns Merged regions where horizontally adjacent text is combined
 */
/**
 * Get the effective height of a text region for filtering purposes,
 * correcting for rotation-inflated axis-aligned bounding boxes.
 *
 * When text is rotated, the AABB height includes the width's contribution:
 *   AABB_h_px = actualWidth_px × |sin(θ)| + actualHeight_px × cos(θ)
 * For wide rotated text (e.g., comment section text at 7°), this significantly
 * inflates the reported height, causing small text to pass the height filter.
 *
 * Since boundingBox uses percentages with different bases (width% of imgW,
 * height% of imgH), we need the image aspect ratio to convert to common units.
 *
 * @param imageAspectRatio - image height / image width (e.g., 7321/690 = 10.6)
 */
function getEffectiveHeight(region: TextRegion, imageAspectRatio: number): number {
  const rotation = region.rotation ?? 0;
  const absRot = Math.abs(rotation);

  // No correction needed for near-zero rotation or missing data
  if (absRot < 2) return region.boundingBox.height;

  // Don't correct for very high rotation (formula becomes unstable near 45°)
  if (absRot > 30) return region.boundingBox.height;

  const theta = absRot * (Math.PI / 180);
  const cosT = Math.cos(theta);
  const sinT = Math.sin(theta);
  const cos2T = Math.cos(2 * theta);

  const aabbH = region.boundingBox.height;
  // Convert width% to same scale as height% (both relative to image height)
  const aabbW = region.boundingBox.width / imageAspectRatio;

  const corrected = (aabbH * cosT - aabbW * sinT) / cos2T;

  // Clamp: don't let correction reduce below 25% of AABB height (safety)
  return Math.max(corrected, aabbH * 0.25);
}

/**
 * Median cap for the minimum-height filter.
 *
 * The base threshold is a fixed fraction of image WIDTH (baseline × 2.6 ≈
 * 2.86% of width for English, 2.47% for Chinese), tuned on webtoon typography
 * where dialogue letters at ~3.5–5% of width. Print-manga scans letter at
 * ~2.5–3.2% of width — at or under the cutoff — so the fixed threshold
 * deletes most of their dialogue (2026-08-12: 77% of regions deleted across
 * a 9-chapter upload batch, user-reported on job f754a969).
 *
 * Text is the only object in a page whose physical size is roughly invariant
 * across media (legibility pins it to ~2.5–3.5mm), so the page's own
 * line-height distribution is the right ruler: cap the threshold at
 * K × median line height. The cap can only LOWER the threshold, so pages
 * whose text clears the base threshold are untouched.
 *
 * Heights are pooled as height ÷ width (scale-invariant) so pages of
 * different pixel widths — double-page spreads served at 2× — compare
 * correctly. The FLOOR keeps 10–14px scanlator credit walls dead.
 *
 * Validated on 16 job folders / 531 pages: byte-identical on every webtoon
 * chapter (EN+CN), rescues ~90% of deleted dialogue on print scans.
 * Kill switch: HEIGHT_MEDIAN_CAP=false restores the fixed threshold.
 */
const HEIGHT_CAP_K = 0.75;          // cap = K × median line height
const HEIGHT_CAP_FLOOR_W = 1.6;     // never cap below this % of image width
const HEIGHT_CAP_MIN_REGIONS = 6;   // median is noise below this
function computeMedianHeightCap(
  regions: TextRegion[],
  imageAspectRatio: number,
  baseThresholdPct: number
): { thresholdPct: number; capped: boolean; medianWPct: number } {
  const base = { thresholdPct: baseThresholdPct, capped: false, medianWPct: 0 };
  if (process.env.HEIGHT_MEDIAN_CAP === 'false') return base;
  if (regions.length < HEIGHT_CAP_MIN_REGIONS) return base;

  // Line heights in % of image WIDTH (heightPct is % of height; × aspect converts)
  const hw = regions
    .map(r => getEffectiveHeight(r, imageAspectRatio) * imageAspectRatio)
    .sort((a, b) => a - b);
  const mid = hw.length >> 1;
  const medianW = hw.length % 2 ? hw[mid] : (hw[mid - 1] + hw[mid]) / 2;

  const baseW = baseThresholdPct * imageAspectRatio;
  const capW = Math.max(HEIGHT_CAP_FLOOR_W, Math.min(baseW, HEIGHT_CAP_K * medianW));
  if (capW >= baseW) return base;
  return { thresholdPct: capW / imageAspectRatio, capped: true, medianWPct: medianW };
}

/**
 * Calculate rotation threshold based on character count, bounding box shape, and height.
 * Short text (1-2 chars) with square bounding boxes get unreliable rotation from PaddleOCR.
 * Large short text (>5% height) is likely a sound effect — use strict threshold.
 * For English, the threshold decays 5x slower per char (English letters are smaller units than Chinese chars).
 */
function getRotationThreshold(text: string, boundingBox: { width: number; height: number }, isEnglishInput: boolean): number {
  const strippedText = text.replace(/[\s\p{P}]/gu, '');
  const charCount = strippedText.length;
  const aspectRatio = boundingBox.width / boundingBox.height;
  const heightPct = boundingBox.height;

  if (charCount <= 2) {
    if (heightPct > 5) {
      return 3.5;
    }
    if (aspectRatio >= 0.6 && aspectRatio <= 1.6) {
      return 90;
    }
    return 17.5;
  }

  const punctCount = text.length - strippedText.length;
  const effectiveChars = charCount + punctCount * 0.5;
  const decayRate = isEnglishInput ? 0.065 : 0.325;
  return Math.max(1.5, 17.5 - decayRate * (effectiveChars - 3));
}

/**
 * Site watermarks: strip URL-shaped tokens from OCR text. Aggregator rips stamp
 * "www.dmzj.com", "lai man hua88.com" etc. on every page, and OCR sometimes
 * glues the stamp onto real dialogue ("的存在悄然诞imanhua88.com"). Only the
 * matched characters are removed, so glued dialogue survives; a region left
 * with nothing meaningful is dropped by the existing punctuation/single-char
 * filters. Up to three preceding Latin words are taken with the domain so
 * "lai man hua88.com" goes as a whole.
 */
const URL_TOKEN_RE = /(?:https?:\/\/\S+|(?<![@\w])(?:[a-z0-9-]+\s+){0,3}(?:www[.:])?[a-z0-9-]+(?:[.:][a-z0-9-]+)*[.:](?:com|net|org|cc|co|me|xyz|top|club|tv|io|vip|cn|cm|dom)(?:\/\S*)?|www[.:][a-z0-9.-]+)/gi;

export function stripUrlTokens(text: string): string {
  return text.replace(URL_TOKEN_RE, ' ').replace(/\s{2,}/g, ' ').trim();
}

function stripUrlTokensFromExtraction<T extends { regions: Array<{ text: string }>; fullText: string }>(extraction: T): T {
  let changed = 0;
  for (const region of extraction.regions) {
    const cleaned = stripUrlTokens(region.text);
    if (cleaned !== region.text) {
      console.error(`  [Watermark] Stripped URL token: "${region.text}" -> "${cleaned}"`);
      region.text = cleaned;
      changed++;
    }
  }
  if (changed > 0) extraction.fullText = extraction.regions.map(r => r.text).join('\n');
  return extraction;
}

/**
 * Page image saves are fire-and-forget for throughput, but a save that fails
 * leaves the reader pointing at a file that does not exist (observed: JPEG
 * encoder refusing a 68,515 px unit — the error was logged and the job still
 * completed). Track every save and check them all before the reader is built.
 */
const pendingImageSaves: Array<Promise<void>> = [];
const failedImageSaves: string[] = [];
function trackImageSave(promise: Promise<void>, label: string): void {
  pendingImageSaves.push(promise.catch(err => { failedImageSaves.push(`${label}: ${err}`); }));
}
function resetImageSaveTracking(): void { pendingImageSaves.length = 0; failedImageSaves.length = 0; }
async function assertAllImageSavesSucceeded(): Promise<void> {
  await Promise.all(pendingImageSaves);
  if (failedImageSaves.length > 0) {
    const err: any = new Error(`${failedImageSaves.length} page image(s) could not be saved: ${failedImageSaves.slice(0, 3).join('; ')}`);
    err.code = 'IMAGE_SAVE_FAILED';
    throw err;
  }
}

/**
 * Filter OCR regions before chapter analysis — removes obvious noise using the same
 * checks as processMangaPageFromBuffer (rotation, max height, emoji, repeated 1s, punctuation-only).
 * Does NOT apply chroma/saturation (needs image pixels) or comment detection (needs clustering).
 */
function filterRegionsForAnalysis(regions: TextRegion[], isEnglishInput: boolean, imageAspectRatio: number): TextRegion[] {
  const baselineAspect = 2.6;
  const scaledMaxHeight = 12 * (baselineAspect / imageAspectRatio);

  return regions.filter(region => {
    const text = region.text.trim();

    // Rotation filter (same as processMangaPageFromBuffer, minus saturationMap override)
    const rotation = Math.abs(region.rotation || 0);
    const rotationThreshold = getRotationThreshold(text, region.boundingBox, isEnglishInput);
    if (rotation > rotationThreshold) {
      const charCount = text.replace(/[^\u4e00-\u9fff\u3400-\u4dbf]/g, '').length;
      if (!(region.confidence >= 0.90 && charCount >= 3)) return false;
    }

    // Max height filter (same formula as line 1279 / merged path)
    if (region.boundingBox.height > scaledMaxHeight) return false;

    // Emoji-only filter
    const strippedEmoji = text.replace(/[\p{Emoji_Presentation}\p{Extended_Pictographic}]/gu, '').trim();
    if (strippedEmoji.length === 0 && text.length > 0) return false;

    // Repeated 1s filter (OCR misreads "!!!" as "111")
    if (/^1{2,}$/.test(text)) return false;

    // Punctuation-only filter
    const nonPunctuation = text.replace(/[，。！？、；：""''「」『』【】（）—…·～《》〈〉\s\.,!?;:'"()\-]/g, '');
    if (nonPunctuation.length === 0) return false;

    return true;
  });
}

function mergeAdjacentRegions(regions: TextRegion[]): TextRegion[] {
  if (regions.length <= 1) return regions;

  // Group regions by horizontal line (similar Y center within tolerance)
  const lineGroups: TextRegion[][] = [];
  const used = new Set<number>();

  for (let i = 0; i < regions.length; i++) {
    if (used.has(i)) continue;

    const region = regions[i];
    const centerY = region.boundingBox.y + region.boundingBox.height / 2;
    // Y tolerance: regions on same line if Y centers within 50% of their height
    const yTolerance = region.boundingBox.height * 0.5;

    // Start a new line group
    const lineGroup: TextRegion[] = [region];
    used.add(i);

    // Find all other regions on the same line
    for (let j = i + 1; j < regions.length; j++) {
      if (used.has(j)) continue;

      const other = regions[j];
      const otherCenterY = other.boundingBox.y + other.boundingBox.height / 2;
      const otherTolerance = other.boundingBox.height * 0.5;

      // Check if on same line (Y overlap with mutual tolerance)
      if (Math.abs(centerY - otherCenterY) < Math.max(yTolerance, otherTolerance)) {
        lineGroup.push(other);
        used.add(j);
      }
    }

    lineGroups.push(lineGroup);
  }

  // Within each line group, sort by X and merge adjacent regions
  const merged: TextRegion[] = [];
  const GAP_THRESHOLD = 5; // 5% of image width - merge if gap is smaller than this

  for (const lineGroup of lineGroups) {
    if (lineGroup.length === 1) {
      merged.push(lineGroup[0]);
      continue;
    }

    // Sort by X (left to right)
    lineGroup.sort((a, b) => a.boundingBox.x - b.boundingBox.x);

    let current = { ...lineGroup[0], boundingBox: { ...lineGroup[0].boundingBox } };

    for (let i = 1; i < lineGroup.length; i++) {
      const next = lineGroup[i];
      const gap = next.boundingBox.x - (current.boundingBox.x + current.boundingBox.width);

      if (gap < GAP_THRESHOLD) {
        // Merge: combine bounding boxes and concatenate text
        const newX = Math.min(current.boundingBox.x, next.boundingBox.x);
        const newY = Math.min(current.boundingBox.y, next.boundingBox.y);
        const newRight = Math.max(
          current.boundingBox.x + current.boundingBox.width,
          next.boundingBox.x + next.boundingBox.width
        );
        const newBottom = Math.max(
          current.boundingBox.y + current.boundingBox.height,
          next.boundingBox.y + next.boundingBox.height
        );

        // Concatenate text - no space for Chinese, but add space between English words
        const needsSpace = /[a-zA-Z]$/.test(current.text) && /^[a-zA-Z]/.test(next.text);
        const separator = needsSpace ? ' ' : '';

        console.error(`  [Merge] Combining: "${current.text}" + "${next.text}" (gap=${gap.toFixed(1)}%)`);

        current = {
          text: current.text + separator + next.text,
          boundingBox: {
            x: newX,
            y: newY,
            width: newRight - newX,
            height: newBottom - newY,
          },
          confidence: Math.min(current.confidence, next.confidence), // Use lower confidence
          isVertical: current.isVertical && next.isVertical,
          rotation: 0, // Reset rotation for merged region
          polygon: undefined, // Clear polygon for merged region
        };
      } else {
        // Gap too large, save current and start new
        merged.push(current);
        current = { ...next, boundingBox: { ...next.boundingBox } };
      }
    }

    merged.push(current);
  }

  const mergeCount = regions.length - merged.length;
  if (mergeCount > 0) {
    console.error(`  [Merge] Combined ${regions.length} regions → ${merged.length} (merged ${mergeCount})`);
  }

  return merged;
}

/**
 * Classify the dominant script of sampled OCR text.
 *
 * Counting design: tally each supported script family separately (Han, Latin,
 * kana, hangul); anything else that is a letter (Thai, Cyrillic, Arabic, ...)
 * accumulates in `other` and yields 'unsupported' when it dominates.
 *
 * DETECT_SCRIPT_FRACTION is deliberately low: Japanese dialogue mixes kanji
 * (counted as Han here) with kana, so kana is often only 30-60% of the
 * letters — while OCR misread noise on a genuine zh/EN chapter stays in the
 * low single digits across a multi-page sample. Hangul is checked before
 * kana: real Korean text contains no kana, so any hangul dominance is
 * unambiguous.
 *
 * Korean caveat: the CN PP-OCRv5/v6 models cannot output hangul, so on the
 * paddleocr provider this function alone NEVER returns 'korean' — Korean
 * chapters sample as 'empty' (hangul-only pages) or 'english' (incidental
 * latin). The caller's Korean probe re-samples through the Korean OCR pool
 * in both cases and flips the result when hangul comes back; gemini/claude
 * read hangul directly and detect 'korean' here without the probe.
 *
 * CJK weight of 3× compensates for the semantic density mismatch: one Han
 * char carries roughly the meaning of one English word (~3-5 letters), so
 * without the weight, mixed content over-detects as english.
 *
 * 'empty' (no letters at all) is the caller's decision: URL-sourced jobs
 * fail fast, uploads and the CLI fall back to chinese (historic default).
 */
const CJK_WEIGHT = 3;
const DETECT_SCRIPT_FRACTION = 0.2;
const HAN_CHAR = /\p{Script=Han}/u;
const LATIN_CHAR = /\p{Script=Latin}/u;
const ANY_LETTER = /\p{L}/u;
export type DetectedInputLanguage = 'chinese' | 'english' | 'japanese' | 'korean' | 'unsupported' | 'empty';
export function detectLanguageFromText(texts: string[]): DetectedInputLanguage {
  let han = 0;
  let latin = 0;
  let kana = 0;
  let hangul = 0;
  let other = 0;
  for (const text of texts) {
    for (const char of text) {
      if (HAN_CHAR.test(char)) han++;
      else if (isKanaChar(char)) kana++;
      else if (isHangulChar(char)) hangul++;
      else if (LATIN_CHAR.test(char)) latin++;
      else if (ANY_LETTER.test(char)) other++;
    }
  }
  const lettered = han + latin + kana + hangul + other;
  if (lettered === 0) return 'empty';
  if (hangul / lettered >= DETECT_SCRIPT_FRACTION) return 'korean';
  if (kana / lettered >= DETECT_SCRIPT_FRACTION) return 'japanese';
  if (other / lettered >= DETECT_SCRIPT_FRACTION) return 'unsupported';
  return latin > han * CJK_WEIGHT ? 'english' : 'chinese';
}

/**
 * Group text regions into blocks based on proximity and visual similarity.
 * Regions in the same speech bubble should be grouped together for context-aware simplification.
 *
 * Block detection criteria:
 * 1. Vertical gap < adjacency threshold (close together)
 * 2. Horizontal overlap > 50% (same column/bubble)
 * 3. Height within 25% (same font size)
 *
 * @param regions - Array of text regions to group
 * @param adjacencyThreshold - Maximum vertical gap (in % of image) to be considered same block
 * @returns Array of TextBlocks with region indices and texts
 */
/**
 * Identify post-merge blocks whose combined text is too long to plausibly be
 * comic dialogue. On the reference chapter (Barbarian Ep1), legitimate merged
 * Chinese dialogue tops out at ~60 chars; noise blocks (forum-post
 * screenshots, long copyright text, review articles) run 200+ chars. A 100-char
 * threshold has comfortable margin on both sides.
 *
 * CN-INPUT ONLY: English dialogue is much less character-dense (100 chars in
 * English ≈ 1 sentence; 100 chars in Chinese ≈ a paragraph), so the same
 * threshold would drop legitimate English monologues. English chapters skip
 * this filter entirely.
 *
 * Set `LOG_ONLY_FILTER=1` to log what WOULD be dropped without actually
 * excluding it — use this to sanity-check the threshold on new content before
 * enabling the filter for real.
 *
 * Returns a Set of block indices to exclude (empty in log-only mode + English).
 */
function identifyLongNonDialogueBlocks(
  blocks: TextBlock[],
  inputLanguage: InputLanguage = 'chinese',
): Set<number> {
  const excluded = new Set<number>();
  // Chinese chars carry ~3-5× the information density of English chars.
  // A 100-char Chinese block is a paragraph; a 100-char English block is one
  // sentence. Use separate defaults to avoid dropping normal English dialogue.
  // Per-language defaults. CN=100 is tuned (Barbarian Ep1: dialogue ≤60,
  // noise 200+). EN=500 accounts for ~5 letters/word. JA=200 and KO=350 are
  // PROVISIONAL (kana/hangul dilute density vs han) — validate with
  // LOG_ONLY_FILTER=1 on real chapters before trusting them.
  const LONG_BLOCK_DEFAULTS: Record<string, string> = { chinese: '100', japanese: '200', korean: '350', english: '500' };
  const defaultThreshold = LONG_BLOCK_DEFAULTS[inputLanguage] ?? '100';
  const threshold = parseInt(process.env.LONG_BLOCK_THRESHOLD || defaultThreshold, 10);
  const logOnly = process.env.LOG_ONLY_FILTER === '1';
  const droppedForLog: Array<{ idx: number; len: number; preview: string }> = [];
  blocks.forEach((block, idx) => {
    const combinedLen = (block.texts || []).reduce((s, t) => s + (t || '').length, 0);
    if (combinedLen >= threshold) {
      const preview = (block.texts || []).join(' ').slice(0, 60);
      droppedForLog.push({ idx, len: combinedLen, preview });
      if (!logOnly) excluded.add(idx);
    }
  });
  if (droppedForLog.length > 0) {
    const action = logOnly ? 'WOULD DROP (log-only)' : 'DROPPED';
    console.error(`  [LongBlockFilter] ${action} ${droppedForLog.length} block(s) with combined text >= ${threshold} chars (CN input):`);
    for (const d of droppedForLog) {
      console.error(`    block[${d.idx}] len=${d.len}: ${d.preview}${d.len > 60 ? '...' : ''}`);
    }
  }
  return excluded;
}

// [CLUSTER_FILTER] Given the text blocks of a detected garbage band (a reading-app UI
// list, comment/review section, or game status window), in top-to-bottom order, return
// the indices that are actually STORY and must be kept. Every other block in the band is
// dropped to original art. A block is only ever dropped if it's INSIDE a detected band
// AND the model didn't mark it story, so the LLM can't delete arbitrary content. No-op if
// the key is missing.
/**
 * Memo for identifyStoryBlocks, keyed on the band's exact text.
 *
 * This is load-bearing for correctness, not a speed optimization. The batched
 * pipeline runs every page through processMangaPageFromBuffer THREE times —
 * Phase 4a (build blocks for the LLM), 4b (inpaint), 4c (render) — and each
 * pass re-ran this filter with its own independent LLM call. When the model
 * answered differently across passes, the three passes ended up with different
 * region sets, so the block list 4c rendered was a different length from the
 * one 4a numbered. Because Phase 4b→4c hands off simplified text BY INDEX, every
 * block past the divergence then received a neighbour's sentence.
 *
 * Measured before this memo: 2 of 5 identical runs mis-assigned text on page 6
 * of the sample chapter (24 vs 26 blocks, shift of exactly 2 = the number of
 * extra blocks). The only clean run was the one where all three passes happened
 * to agree.
 *
 * Band detection itself is deterministic (union-find over box geometry), so the
 * same band yields a byte-identical `texts` array on all three passes and this
 * key is stable. Caching inside the function rather than threading the result
 * through options covers both call sites (batched and non-batched) at once.
 *
 * NOTE: this makes the three passes agree; it does not make separate chapter
 * runs agree. Which comment boxes get dropped still varies run to run — that is
 * ordinary LLM variance and is harmless, because it is now consistent within a
 * run. The underlying fragility (index-keyed handoff between 4b and 4c) is
 * untouched; any other cross-pass divergence would fail the same silent way.
 */
const storyBlockCache = new Map<string, Set<number>>();

/** Cleared per job — see resetTimingStats(). Bands are page-local, so entries
 *  have no value across chapters and would only grow the map. */
export function resetStoryBlockCache(): void {
  storyBlockCache.clear();
}

async function identifyStoryBlocks(texts: string[]): Promise<Set<number>> {
  const cacheKey = texts.join('\u0000');
  const cached = storyBlockCache.get(cacheKey);
  if (cached) return cached;

  const keep = new Set<number>();
  const KEY = process.env.FIREWORKS_API_KEY;
  if (!KEY || texts.length === 0) return keep;
  const MODEL = process.env.SIMPLIFY_MODEL || 'accounts/fireworks/models/deepseek-v4-flash';
  const SYS = `These text blocks form a dense "junk band" detected on a webtoon/manhua page — a reading-app UI list, a comment/review section, or a game status window. MOST are non-story chrome: view/like/comment counts, episode/chapter lists, usernames, timestamps, ratings, game stats, ad/promo text.
Your job: find the blocks that are ACTUALLY STORY and must be kept — dialogue, narration, character thoughts, meaningful signs. Blocks that together read as ONE utterance must all be kept. Leave genuine app-UI / comments / stats / promo out. When a block is clearly readable story text, keep it.
Reply ONLY with a JSON array of the indices to KEEP, e.g. [2,3]. If none are story, reply []. No other text.`;
  const BATCH = 400;
  for (let s = 0; s < texts.length; s += BATCH) {
    const chunk = texts.slice(s, s + BATCH);
    const user = chunk.map((t, i) => `${i}: ${t}`).join('\n');
    try {
      const res = await fetch('https://api.fireworks.ai/inference/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${KEY}` },
        body: JSON.stringify({ model: MODEL, messages: [{ role: 'system', content: SYS }, { role: 'user', content: user }], temperature: 0, reasoning_effort: 'none', max_tokens: 4000 }),
      });
      if (!res.ok) { console.error(`  [ClusterFilter] LLM ${res.status}`); continue; }
      const content = JSON.parse(await res.text()).choices[0].message.content;
      const m = content.match(/\[[\s\S]*?\]/);
      if (!m) continue;
      for (const n of JSON.parse(m[0])) {
        if (typeof n === 'number' && n >= 0 && n < chunk.length) keep.add(s + n);
      }
    } catch (e: any) {
      console.error(`  [ClusterFilter] batch error: ${e.message}`);
    }
  }
  storyBlockCache.set(cacheKey, keep);
  return keep;
}

// [OVERLAP_DROP] Change 2: after merging, find blocks whose TRUE (oriented) render
// boxes heavily overlap each other and leave the whole cluster as ORIGINAL artwork
// (caller folds them into chromaExcludedBlocks — skipped by both the inpaint and
// render loops, so nothing is typeset over the pile). Uses computeRenderBoxPct (the
// un-inflated oriented box) so it does NOT re-flag the tilted-line false-overlap that
// Change 1 fixed — only genuine collisions (e.g. a large SFX character sitting on top
// of narration). Overlap is oriented-rectangle coverage (point-sampled in pixel
// space), never the axis-aligned bbox. Only drops a HOMOGENEOUS cluster (all rotated
// OR all upright) so a rotated SFX overlapping a real upright bubble doesn't take the
// bubble with it. Off unless OVERLAP_DROP=true.
function overlappingClustersToDrop(
  blocks: TextBlock[],
  regionsToSimplify: TextRegion[],
  width: number,
  height: number,
  isExcluded: (blockIdx: number) => boolean,
): Set<number> {
  const ROT = 5;         // |rotation| > this ⇒ "rotated"
  const COVERAGE = 0.5;  // one box covers >50% of the other ⇒ heavy overlap
  const GRID = 5;        // GRID×GRID sample points per box

  type OBox = { cx: number; cy: number; w: number; h: number; rot: number };
  const idxs: number[] = [];
  const boxes: OBox[] = [];
  blocks.forEach((b, i) => {
    if (isExcluded(i) || b.regionIndices.length === 0) return;
    const box = computeRenderBoxPct(b.regionIndices.map(ri => ({ original: regionsToSimplify[ri] })), width, height);
    const rot = b.regionIndices.reduce((s, ri) => s + (regionsToSimplify[ri].rotation || 0), 0) / b.regionIndices.length;
    idxs.push(i);
    boxes.push({
      cx: ((box.x + box.width / 2) / 100) * width,
      cy: ((box.y + box.height / 2) / 100) * height,
      w: (box.width / 100) * width,
      h: (box.height / 100) * height,
      rot,
    });
  });
  if (boxes.length < 2) return new Set();

  // Point-in-oriented-rectangle: rotate the point into the box's local frame.
  const inside = (px: number, py: number, o: OBox): boolean => {
    const t = -o.rot * Math.PI / 180, c = Math.cos(t), s = Math.sin(t);
    const dx = px - o.cx, dy = py - o.cy;
    return Math.abs(dx * c - dy * s) <= o.w / 2 && Math.abs(dx * s + dy * c) <= o.h / 2;
  };
  // Fraction of B's (sampled) area that lies inside A.
  const coverage = (A: OBox, B: OBox): number => {
    const t = (B.rot * Math.PI) / 180, c = Math.cos(t), s = Math.sin(t);
    let inCount = 0, n = 0;
    for (let gi = 0; gi < GRID; gi++) {
      for (let gj = 0; gj < GRID; gj++) {
        const u = (gi / (GRID - 1) - 0.5) * B.w;
        const v = (gj / (GRID - 1) - 0.5) * B.h;
        n++;
        if (inside(B.cx + u * c - v * s, B.cy + u * s + v * c, A)) inCount++;
      }
    }
    return n ? inCount / n : 0;
  };

  // Union-find over heavy-overlap pairs.
  const parent = boxes.map((_, i) => i);
  const find = (x: number): number => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      if (Math.max(coverage(boxes[i], boxes[j]), coverage(boxes[j], boxes[i])) > COVERAGE) parent[find(i)] = find(j);
    }
  }
  const comps: Record<number, number[]> = {};
  boxes.forEach((_, i) => { const r = find(i); (comps[r] = comps[r] || []).push(i); });

  const drop = new Set<number>();
  for (const members of Object.values(comps)) {
    if (members.length < 2) continue;
    const rotated = members.map(m => Math.abs(boxes[m].rot) > ROT);
    if (!rotated.every(Boolean) && !rotated.every(r => !r)) continue; // mixed rotated+upright → keep (protect a bubble under an SFX)
    members.forEach(m => drop.add(idxs[m]));
    console.error(`  [OverlapDrop] ${members.length} overlapping ${rotated[0] ? 'rotated' : 'upright'} blocks → original art: ${members.map(m => JSON.stringify(blocks[idxs[m]].texts.join('').slice(0, 14))).join(', ')}`);
  }
  return drop;
}

export function groupRegionsIntoBlocks(
  regions: TextRegion[],
  adjacencyThreshold: number,
  inputLanguage: InputLanguage = 'chinese',
  // imageAspectRatio = imageHeight / imageWidth. Used inside Pass 2 horizontal
  // merging to correct a unit mismatch: hGap (% of width) is compared against
  // minH (% of height) × threshold-multiplier. Without the aspect-ratio
  // correction, the comparison is only valid on square images; on a 800×3391
  // page (aspect 4.2), the effective threshold becomes ~0.19× the intended
  // value in pixel terms, causing dialogue words like "COMPLETELY" and
  // "SHIT." to fail to merge even when their pixel gap is well within
  // single-line distance. Default 2.5 matches the canonical webtoon shape
  // for backward compatibility if a caller doesn't pass anything.
  imageAspectRatio: number = 2.5,
  // When supplied, Pass 4 (rotated-overlap filter) populates this Set with the
  // region indices of any block it drops. Callers can then filter those
  // regions out of extraction.regions so they're never inpainted/rendered.
  outDroppedByOverlap?: Set<number>,
): TextBlock[] {
  const _bgStart = Date.now();
  try {
    return _groupRegionsIntoBlocksImpl(regions, adjacencyThreshold, inputLanguage, imageAspectRatio, outDroppedByOverlap);
  } finally {
    globalTimingStats.blockGrouping += Date.now() - _bgStart;
  }
}
function _groupRegionsIntoBlocksImpl(
  regions: TextRegion[],
  adjacencyThreshold: number,
  inputLanguage: InputLanguage = 'chinese',
  imageAspectRatio: number = 2.5,
  outDroppedByOverlap?: Set<number>,
): TextBlock[] {
  if (regions.length === 0) return [];

  // Helper: check if two regions can be in the same block (vertical stacking)
  function regionsMatch(blockRegion: TextRegion, region: TextRegion): boolean {
    // Rotation compatibility: regions must have similar rotation to merge
    const rotA = blockRegion.rotation || 0;
    const rotB = region.rotation || 0;
    const rotDiff = Math.abs(rotA - rotB);
    const minChars = Math.min(blockRegion.text.length, region.text.length);
    const rotTolerance = minChars <= 2 ? 10 : 3;
    if (rotDiff > rotTolerance) return false;
    // Prevent axis-aligned (< 2°) from merging with significantly rotated (> 5°)
    if (Math.min(Math.abs(rotA), Math.abs(rotB)) < 2 && Math.max(Math.abs(rotA), Math.abs(rotB)) > 5) return false;

    // Compute vertical gap using polygon edge midpoints when available (accurate for rotated text).
    // Axis-aligned bounding boxes inflate for rotated text, causing fake vertical overlap.
    // Using min/max Y of polygon corners also fails for tilted text because opposite corners
    // extend far beyond the actual text edges. Edge midpoints give the true gap.
    // PaddleOCR polygon vertices: [top-left, top-right, bottom-right, bottom-left]
    let verticalGap: number;
    if (blockRegion.polygon && blockRegion.polygon.length === 4 && region.polygon && region.polygon.length === 4) {
      // Bottom edge of block = midpoint of vertices 2 (bottom-right) and 3 (bottom-left)
      const blockBottomMidY = (blockRegion.polygon[2][1] + blockRegion.polygon[3][1]) / 2;
      // Top edge of region = midpoint of vertices 0 (top-left) and 1 (top-right)
      const regionTopMidY = (region.polygon[0][1] + region.polygon[1][1]) / 2;
      verticalGap = regionTopMidY - blockBottomMidY;
    } else {
      // Fallback to axis-aligned bounding box
      const blockBottom = blockRegion.boundingBox.y + blockRegion.boundingBox.height;
      verticalGap = region.boundingBox.y - blockBottom;
    }

    if (verticalGap >= adjacencyThreshold || verticalGap < -adjacencyThreshold) return false;

    // Scale gap tolerance by text box size — lines within the same bubble have gaps < text height
    // Separate bubbles have gaps roughly equal to or larger than the text height
    // English uses tighter ratio (0.6) because multi-paragraph bubbles have visible gaps.
    // Chinese was historically 0.9 (untuned default); tightened to 0.7 to better split
    // vertically-stacked separate bubbles while still merging tight in-bubble line spacing.
    const minHeight = Math.min(blockRegion.boundingBox.height, region.boundingBox.height);
    // [CLUSTER_FILTER] tighten Chinese merge 0.7 → 0.3. In-bubble line spacing is ~0.05 so
    // bubbles still cohere, but UI/comment rows (~0.37) no longer fuse into a bubble or into
    // one another — which is what structurally prevents Bug 1 (garbled overflow) and Bug 2
    // (bubble buried in a long UI block, then dropped by the LongBlockFilter). Production
    // (flag off) unchanged; the only cost is a rare loose bubble splitting into two boxes.
    const maxGapRatio = inputLanguage === 'english' ? 0.6 : (process.env.CLUSTER_FILTER === 'true' ? 0.3 : 0.7);
    if (minHeight > 0 && verticalGap > 0 && verticalGap / minHeight > maxGapRatio) return false;
    // Horizontal overlap check — skip only for heavily rotated text (20°+) where
    // axis-aligned x-coordinates are unreliable
    const skipOverlapCheck = Math.min(Math.abs(rotA), Math.abs(rotB)) > 20;
    if (!skipOverlapCheck) {
      const overlapStart = Math.max(blockRegion.boundingBox.x, region.boundingBox.x);
      const overlapEnd = Math.min(
        blockRegion.boundingBox.x + blockRegion.boundingBox.width,
        region.boundingBox.x + region.boundingBox.width
      );
      const overlapWidth = Math.max(0, overlapEnd - overlapStart);
      const minWidth = Math.min(blockRegion.boundingBox.width, region.boundingBox.width);
      const overlapRatio = minWidth > 0 ? overlapWidth / minWidth : 0;
      if (overlapRatio <= 0.5) return false;
    }

    // Check height similarity — English manga fonts vary more in size (emphasized words)
    const heightRatio = blockRegion.boundingBox.height > 0
      ? region.boundingBox.height / blockRegion.boundingBox.height
      : 1;
    const minRatio = inputLanguage === 'english' ? 0.55 : 0.75;
    const maxRatio = inputLanguage === 'english' ? 1.45 : 1.25;
    const heightOk = heightRatio >= minRatio && heightRatio <= maxRatio;
    if (heightOk && process.env.MERGE_DEBUG === 'true') {
      const oS = Math.max(blockRegion.boundingBox.x, region.boundingBox.x);
      const oE = Math.min(blockRegion.boundingBox.x + blockRegion.boundingBox.width, region.boundingBox.x + region.boundingBox.width);
      const mW = Math.min(blockRegion.boundingBox.width, region.boundingBox.width);
      const oR = mW > 0 ? Math.max(0, oE - oS) / mW : 0;
      console.error(`  [MergeDbg] "${blockRegion.text.slice(0, 8)}"+"${region.text.slice(0, 8)}" vGap=${verticalGap.toFixed(2)} gapR=${(minHeight > 0 ? verticalGap / minHeight : 0).toFixed(2)} hOv=${oR.toFixed(2)} hR=${heightRatio.toFixed(2)} rot=${rotA.toFixed(0)}/${rotB.toFixed(0)}`);
    }
    return heightOk;
  }

  // Helper: check if two regions are horizontally adjacent on the same line
  function regionsHorizontallyAdjacent(a: TextRegion, b: TextRegion): boolean {
    // Must be on the same line: vertical overlap > 50% of the shorter region
    const overlapTop = Math.max(a.boundingBox.y, b.boundingBox.y);
    const overlapBottom = Math.min(
      a.boundingBox.y + a.boundingBox.height,
      b.boundingBox.y + b.boundingBox.height
    );
    const vOverlap = Math.max(0, overlapBottom - overlapTop);
    const minHeight = Math.min(a.boundingBox.height, b.boundingBox.height);
    if (minHeight <= 0 || vOverlap / minHeight < 0.5) return false;

    // Horizontal gap must be small (< adjacency threshold).
    // Note: text-height-based merging happens in Pass 2 (with proper
    // aspect-ratio conversion). This helper is only called in one
    // colorful-text adjacency check elsewhere, where the original flat
    // threshold is appropriate.
    const aRight = a.boundingBox.x + a.boundingBox.width;
    const bRight = b.boundingBox.x + b.boundingBox.width;
    const hGap = Math.max(0, Math.max(a.boundingBox.x, b.boundingBox.x) - Math.min(aRight, bRight));
    return hGap < adjacencyThreshold;
  }

  // Create indexed list and sort by Y coordinate
  const indexed = regions.map((r, i) => ({ region: r, index: i }));
  indexed.sort((a, b) => a.region.boundingBox.y - b.region.boundingBox.y);

  // Pass 1: Group by vertical stacking
  // Uses open blocks approach: multiple blocks can be "open" simultaneously.
  // When a region doesn't match the current block, check recently closed blocks
  // in case interleaved regions from other speech bubbles broke the chain.
  // e.g. right bubble line 1, left bubble line 1, right bubble line 2 — the two
  // right bubble lines should merge even though the left bubble line is between them.
  const openBlocks: { items: { region: TextRegion; index: number }[]; lastY: number }[] = [];
  const closedBlocks: { items: { region: TextRegion; index: number }[] }[] = [];

  for (const item of indexed) {
    const y = item.region.boundingBox.y;

    // Try to find a matching open block (check most recent first)
    let matched = false;
    for (let i = openBlocks.length - 1; i >= 0; i--) {
      const block = openBlocks[i];
      const matchesBlock = block.items.some(blockItem => regionsMatch(blockItem.region, item.region));
      if (matchesBlock) {
        block.items.push(item);
        block.lastY = y;
        matched = true;
        break;
      }
    }

    if (!matched) {
      openBlocks.push({ items: [item], lastY: y });
    }

    // Prune blocks that are too far above to ever match again
    // (their bottom edge + adjacency threshold is above current y)
    // Save pruned blocks to closedBlocks so they're not lost
    const pruneThreshold = adjacencyThreshold * 2;
    while (openBlocks.length > 0) {
      const oldest = openBlocks[0];
      const oldestBottom = Math.max(...oldest.items.map(it =>
        it.region.boundingBox.y + it.region.boundingBox.height));
      if (y - oldestBottom > pruneThreshold) {
        closedBlocks.push(openBlocks.shift()!);
      } else {
        break;
      }
    }
  }

  const blocks: TextBlock[] = [...closedBlocks, ...openBlocks].map(ob => ({
    regionIndices: ob.items.map(it => it.index),
    texts: ob.items.map(it => it.region.text)
  }));

  // Pass 2: Merge horizontally adjacent same-line blocks
  // PaddleOCR splits English text at word boundaries (e.g., "YOU'VE", "ANSWERED", "'YES" on one line).
  // These sit side-by-side with no horizontal overlap, so Pass 1 misses them.
  // Requirements are tight to avoid merging separate speech bubbles:
  //   - Y-top alignment within 0.5%
  //   - Height ratio within 15% (0.85-1.15)
  //   - Horizontal gap < 80% of the shorter region's height (proportional to font size)
  //
  // The hGap-vs-height check has a unit-mismatch problem on non-square
  // images: hGap is in % of width, minH is in % of height, so a flat
  // comparison only works when aspect is 1. Multiplying by imageAspectRatio
  // converts the height-percentage into width-equivalent terms so the
  // intended "0.8× character height in pixel space" check actually
  // evaluates that.
  let mergedHorizontal = true;
  while (mergedHorizontal) {
    mergedHorizontal = false;
    for (let i = 0; i < blocks.length; i++) {
      if (blocks[i].texts.length === 0) continue;
      for (let j = i + 1; j < blocks.length; j++) {
        if (blocks[j].texts.length === 0) continue;
        const shouldMerge = blocks[i].regionIndices.some(ri =>
          blocks[j].regionIndices.some(rj => {
            const a = regions[ri].boundingBox;
            const b = regions[rj].boundingBox;
            // Y-top alignment ("same line"). A flat 1% is looser than a line
            // height on tightly-spaced Chinese (~0.7%), so the line stacked
            // directly BELOW wrongly passes as same-line and Pass 2 fuses a
            // speech bubble into the UI panel beneath it. Tie the tolerance to
            // text height: same-line fragments share a top edge (small fraction
            // of a line height); the next line down sits ~1 full line away.
            // 0.4x still absorbs leading-punctuation top-edge shifts (e.g. a
            // "[USER" bracket lifts the box ~40% of its height) for genuinely
            // same-line English fragments. Height ratio stays 0.55-1.45.
            const yTol = Math.min(a.height, b.height) * 0.4;
            if (Math.abs(a.y - b.y) > yTol) return false;
            // Height must be similar
            const heightRatio = a.height > 0 ? b.height / a.height : 1;
            if (heightRatio < 0.55 || heightRatio > 1.45) return false;
            // Horizontal gap must be small relative to text height (in pixel
            // terms — aspect ratio converts height-percentage to width-equivalent)
            const aRight = a.x + a.width;
            const bRight = b.x + b.width;
            const hGap = Math.max(0, Math.max(a.x, b.x) - Math.min(aRight, bRight));
            const minH = Math.min(a.height, b.height);
            return hGap < minH * 0.8 * imageAspectRatio;
          })
        );
        if (shouldMerge) {
          // Bucket by Y-line then sort by X within each line so the merged
          // block reads top-to-bottom, left-to-right.
          //
          // The previous "sort by X only" produced scrambled multi-line
          // blocks: e.g. "STRONG RAY OF / LIGHT POURED OUT," whose two
          // lines have OCR regions at similar (but not identical) Y values
          // interleaved when sorted purely by X — emitted as "LIGHT STRONG
          // POURED RAY OUT, OF". Y-bucketing first keeps each line's
          // regions together, then X-sorts within each line.
          //
          // Y-tolerance is half the median text height: same-line regions
          // differ by a few pixels; different-line regions differ by ~one
          // full line-height or more, so half-height cleanly partitions
          // them.
          const allIndices = [...blocks[i].regionIndices, ...blocks[j].regionIndices];
          const allTexts = [...blocks[i].texts, ...blocks[j].texts];
          const items = allIndices.map((ri, idx) => ({
            ri,
            text: allTexts[idx],
            x: regions[ri].boundingBox.x,
            y: regions[ri].boundingBox.y,
            h: regions[ri].boundingBox.height,
          }));
          const sortedHeights = items.map(it => it.h).sort((p, q) => p - q);
          const medianH = sortedHeights[Math.floor(sortedHeights.length / 2)] || 0;
          const yTolerance = medianH * 0.5;
          items.sort((a, b) => a.y - b.y);
          const buckets: typeof items[] = [];
          for (const item of items) {
            const last = buckets[buckets.length - 1];
            if (last && item.y - last[last.length - 1].y < yTolerance) {
              last.push(item);
            } else {
              buckets.push([item]);
            }
          }
          for (const bucket of buckets) {
            bucket.sort((a, b) => a.x - b.x);
          }
          const sorted = buckets.flat();
          blocks[i].regionIndices = sorted.map(s => s.ri);
          blocks[i].texts = sorted.map(s => s.text);
          blocks[j].regionIndices = [];
          blocks[j].texts = [];
          mergedHorizontal = true;
          console.error(`  [Blocks] Merged ${buckets.length}-line block: "${blocks[i].texts.join(' ')}"`);
        }
      }
    }
  }

  // Pass 3: Merge standalone colorful regions that are horizontally adjacent to a block
  const standaloneColorful: number[] = [];
  blocks.forEach((block, blockIdx) => {
    if (block.texts.length === 1) {
      const regionIdx = block.regionIndices[0];
      if (regions[regionIdx].isColorful) {
        standaloneColorful.push(blockIdx);
      }
    }
  });

  for (const colorfulBlockIdx of standaloneColorful) {
    const colorfulRegion = regions[blocks[colorfulBlockIdx].regionIndices[0]];
    let bestTarget = -1;

    for (let i = 0; i < blocks.length; i++) {
      if (i === colorfulBlockIdx || blocks[i].texts.length === 0) continue;
      // SFX guard: a "standalone colorful" region that is much taller than the
      // target block's dialogue text is almost certainly stylized SFX (BADUM,
      // CLENCH, CRASH, etc.) sitting on top of or next to an unrelated speech
      // bubble — not part of the same dialogue. regionsHorizontallyAdjacent
      // doesn't catch this because tall SFX bounding boxes spuriously overlap
      // with nearby small dialogue boxes. Block merge would then produce
      // garbage like "WHAT'S THE MEANING OF THIS | BADUM".
      const colorfulH = colorfulRegion.boundingBox.height;
      const targetMaxH = Math.max(...blocks[i].regionIndices.map(ri => regions[ri].boundingBox.height));
      if (targetMaxH > 0 && colorfulH / targetMaxH > 2.0) continue;
      // Check if any region in the target block is horizontally adjacent
      const hasAdjacent = blocks[i].regionIndices.some(ri =>
        regionsHorizontallyAdjacent(regions[ri], colorfulRegion)
      );
      if (hasAdjacent) {
        bestTarget = i;
        break;
      }
    }

    if (bestTarget >= 0) {
      console.error(`  [Blocks] Merging colorful standalone "${colorfulRegion.text}" into block with "${blocks[bestTarget].texts.join(' / ')}"`);
      blocks[bestTarget].regionIndices.push(...blocks[colorfulBlockIdx].regionIndices);
      blocks[bestTarget].texts.push(...blocks[colorfulBlockIdx].texts);
      // Mark as empty for removal
      blocks[colorfulBlockIdx].regionIndices = [];
      blocks[colorfulBlockIdx].texts = [];
    }
  }

  // Pass 4: drop rotated blocks that have a non-rotated neighbor.
  // A rotated block (max region rotation > 5°) usually represents stylized SFX
  // or game-UI text. When a non-rotated block (real dialogue, OCR jitter aside)
  // sits within 2% edge-to-edge gap of it, render conflicts surface — the
  // rotated AABB's translation/inpaint visually collides with the adjacent
  // bubble. Drop the rotated block in that case. Standalone rotated blocks
  // (no close non-rotated neighbor) are kept since they're likely the only
  // text on their panel.
  const ROT_THRESHOLD = 5;
  const GAP_THRESHOLD = 2;
  const blockBoxes = blocks.map((b, idx) => {
    if (b.regionIndices.length === 0) return null;
    const xs = b.regionIndices.flatMap(i => [regions[i].boundingBox.x, regions[i].boundingBox.x + regions[i].boundingBox.width]);
    const ys = b.regionIndices.flatMap(i => [regions[i].boundingBox.y, regions[i].boundingBox.y + regions[i].boundingBox.height]);
    const maxRot = Math.max(0, ...b.regionIndices.map(i => Math.abs(regions[i].rotation || 0)));
    return { idx, maxRot, x_min: Math.min(...xs), x_max: Math.max(...xs), y_min: Math.min(...ys), y_max: Math.max(...ys) };
  });
  for (let i = 0; i < blocks.length; i++) {
    const a = blockBoxes[i];
    if (!a || a.maxRot <= ROT_THRESHOLD) continue;
    let nearestGap = Infinity;
    let nearestIdx = -1;
    for (let j = 0; j < blocks.length; j++) {
      if (i === j) continue;
      const bMeta = blockBoxes[j];
      if (!bMeta || bMeta.maxRot > ROT_THRESHOLD) continue;
      const xGap = Math.max(0, Math.max(a.x_min, bMeta.x_min) - Math.min(a.x_max, bMeta.x_max));
      const yGap = Math.max(0, Math.max(a.y_min, bMeta.y_min) - Math.min(a.y_max, bMeta.y_max));
      const gap =
        xGap === 0 && yGap === 0 ? 0 :
        xGap === 0 ? yGap :
        yGap === 0 ? xGap :
        Math.sqrt(xGap * xGap + yGap * yGap);
      if (gap < nearestGap) { nearestGap = gap; nearestIdx = j; }
    }
    if (nearestIdx >= 0 && nearestGap <= GAP_THRESHOLD) {
      console.error(`  [Blocks] Dropping rotated block "${blocks[i].texts.join(' / ').slice(0, 50)}" (rot=${a.maxRot.toFixed(1)}°) — non-rotated neighbor "${blocks[nearestIdx].texts.join(' / ').slice(0, 50)}" at gap=${nearestGap.toFixed(2)}%`);
      // Track which underlying region indices were dropped so the caller can
      // also remove them from extraction.regions (so nothing downstream tries
      // to inpaint/render them — see issue: rotated banner left as English).
      if (outDroppedByOverlap) {
        for (const ri of blocks[i].regionIndices) outDroppedByOverlap.add(ri);
      }
      blocks[i].regionIndices = [];
      blocks[i].texts = [];
    }
  }

  // Remove empty blocks
  let finalBlocks = blocks.filter(b => b.texts.length > 0);

  // Drop blocks whose merged text contains no letters or CJK characters — pure
  // punctuation/digits/symbols. These are OCR artifacts (e.g. "888:", "!!.!.",
  // "***", timestamps, score displays overlaid on manga panels) that the LLM
  // otherwise wastes tokens on (and produced the p1 "888:" misalignment where
  // the numeric bubble threw off Stage 1's sentinel alignment).
  //
  // Applied AFTER merging so a fragment like "!!" that would attach to a
  // real bubble doesn't get dropped on its own — only isolated punct-only
  // blocks are removed.
  const hasLettersOrCJK = (s: string): boolean => /[A-Za-z一-鿿㐀-䶿]/.test(s);
  const beforeGarbageFilter = finalBlocks.length;
  const droppedGarbage: string[] = [];
  finalBlocks = finalBlocks.filter(b => {
    const joined = b.texts.join('');
    if (hasLettersOrCJK(joined)) return true;
    droppedGarbage.push(joined);
    // Report the dropped regions so the caller also excludes them from inpaint
    // + render (via outDroppedByOverlap → chromaExcludedIndices). Without this,
    // the physically-removed block's regions are orphaned: they get no blockId,
    // fall through to the standalone-region path, and are inpainted over the
    // artwork then re-typeset with their original junk text (e.g. "888:").
    // Mirrors the rotated-overlap drop at the top of this pass (814-816).
    if (outDroppedByOverlap) {
      for (const ri of b.regionIndices) outDroppedByOverlap.add(ri);
    }
    return false;
  });
  if (droppedGarbage.length > 0) {
    console.error(`  [Blocks] Dropped ${droppedGarbage.length} punct/number-only blocks (out of ${beforeGarbageFilter}): ${droppedGarbage.map(t => JSON.stringify(t.slice(0, 30))).join(', ')}`);
  }

  // Log block grouping with coordinates
  const multiLineBlocks = finalBlocks.filter(b => b.texts.length > 1);
  console.error(`  [Blocks] Grouped ${regions.length} regions into ${finalBlocks.length} blocks (${multiLineBlocks.length} multi-line)`);
  finalBlocks.forEach((block, blockIdx) => {
    const regionCount = block.texts.length;
    console.error(`  [Blocks] Block #${blockIdx} (${regionCount} region${regionCount > 1 ? 's' : ''}):`);
    block.regionIndices.forEach((ri, i) => {
      const r = regions[ri].boundingBox;
      console.error(`    ${i}: x=${r.x.toFixed(1)}-${(r.x + r.width).toFixed(1)}% y=${r.y.toFixed(1)}-${(r.y + r.height).toFixed(1)}% "${block.texts[i]}"`);
    });
  });

  return finalBlocks;
}

/**
 * Convert Chinese text to pinyin with tone marks
 */
function toPinyin(text: string): string {
  return pinyin(text, { toneType: 'symbol', type: 'string' });
}

/**
 * Phrase data for the interactive reader — one entry per tokenized word/phrase
 */
export interface PhraseData {
  text: string;
  pinyin: string;
  isAboveLevel: boolean;
  hskLevel: number | null;
  definition?: string;
  // Phase 3.2: separate CC-CEDICT definition for the ⭐ save button.
  // `definition` above shows whatever's most useful in-context (chapter
  // override beats CC-CEDICT). `cedictDefinition` is ONLY set when CC-CEDICT
  // had an entry — the reader's save button uses it to decide whether to
  // enable the save action and which text to save. Story-specific names
  // (e.g. character names invented for the manga) have no CC-CEDICT entry,
  // so this field is undefined and the button is disabled.
  cedictDefinition?: string;
  // True when CC-CEDICT romanizes this token as a proper noun (capitalized
  // pinyin). Baked at build time because the reader can't reach CC-CEDICT at
  // display time. Drives the "show names in English" toggle's dictionary path
  // (see resolveEnglishName). Only set when true, to keep the payload small.
  properNoun?: boolean;
  // Traditional (Taiwan-standard) form of `text`, for the reader's script
  // toggle. Baked here rather than converted in the browser because the
  // Simplified→Traditional OpenCC dictionary is ~1MB — far too much to ship
  // per reader page for a display toggle. Set ONLY when it differs from `text`
  // (most characters are identical in both scripts), so the payload cost is
  // limited to the words that actually change.
  //
  // `text` stays canonical Simplified everywhere: it is the flashcard storage
  // key and the HSK lookup key. This field is display-only.
  t?: string;
}

/**
 * Build phrase-level data for a block's text: tokenize, generate pinyin, check HSK level, look up definitions
 */
/**
 * Split a multi-char jieba token into sub-tokens that have dictionary definitions.
 * Used as a fallback when jieba returned a compound that isn't in CC-CEDICT or
 * the chapter's definition map.
 *
 * Uses backward maximum matching (BMM): scan right-to-left, taking longest
 * dictionary match. BMM handles Chinese head-final compounds correctly:
 *   小游戏 → 小 + 游戏  (BMM)   vs   小游 + 戏  (FMM, wrong: leaves "戏" with HSK 5 "drama")
 *
 * The caller only invokes this for tokens that have NO definition (and are not
 * KEEP terms — those are excluded upstream). So there is nothing to "protect"
 * by returning a token whole: a definition-less token returned whole renders as
 * an empty popup in the reader. Every token therefore decomposes down to pieces
 * that have definitions, with single characters as the floor — a single char
 * always has a CC-CEDICT entry. This sacrifices the compound meaning of a
 * genuine-but-CEDICT-missing word, but such words are KEEP terms (excluded
 * here) or rare; per-character definitions are the honest fallback for the rest
 * (jieba mis-groupings, garbled-translation artifacts like 人动).
 */
function splitIntoDefinedWords(token: string, chapterDefinitions?: Map<string, string>): string[] {
  // Single characters can't be split further; everything longer is decomposed.
  if (token.length <= 1) return [token];

  const result: string[] = [];
  let i = token.length;
  while (i > 0) {
    let matched = false;
    for (let len = Math.min(4, i); len >= 2; len--) {
      const sub = token.substring(i - len, i);
      if (chapterDefinitions?.has(sub) || getDefinition(sub)) {
        result.unshift(sub);
        i -= len;
        matched = true;
        break;
      }
    }
    if (!matched) {
      result.unshift(token[i - 1]);
      i--;
    }
  }

  // Return the decomposition even when it is all single characters: the token
  // had no definition of its own, so per-character definitions beat an empty popup.
  return result;
}

export function buildPhraseData(text: string, targetLevel: HSKLevel, chapterDefinitions?: Map<string, string>, keepWords?: string[]): PhraseData[] {
  const targetNum = targetLevel === 'hsk1' ? 1 :
                    targetLevel === 'hsk2' ? 2 :
                    targetLevel === 'hsk3' ? 3 : 4;
  const keepSet = keepWords ? new Set(keepWords) : null;
  // Strip newlines before tokenization — jieba treats \n as a hard boundary
  // and produces wrong segmentation across line wraps (e.g. "四\n个人" → ["那四","个人"]
  // instead of the correct ["那","四个","人"]). Newlines are visual layout, not meaning.
  const cleanedText = text.replace(/[\r\n]+/g, '');
  const rawTokens = tokenize(cleanedText, targetNum, keepWords);

  // Split punctuation from tokens — handles leading, trailing, AND embedded punctuation
  // e.g. "吧。" → ["吧", "。"], "東､西､南" → ["東", "､", "西", "､", "南"]
  const tokens: string[] = [];
  for (const token of rawTokens) {
    let current = '';
    for (const char of token) {
      if (isPunctuation(char)) {
        if (current) { tokens.push(current); current = ''; }
        tokens.push(char);
      } else {
        current += char;
      }
    }
    if (current) tokens.push(current);
  }

  // Drop space tokens that sit between two English tokens. jieba returns
  // these as separate single-char punct tokens, and without this pre-pass the
  // merge below fails (isEnglish(' ') is false), leaving each English word as
  // its own phrase span. In the reader those space spans render as ~0-width
  // inline-blocks, visually collapsing "DUNGEON AND STONE" to "DUNGEONANDSTONE".
  const compactedTokens: string[] = [];
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i];
    if (t === ' ' && i > 0 && i < tokens.length - 1 &&
        isEnglish(tokens[i - 1]) && isEnglish(tokens[i + 1])) {
      continue;
    }
    compactedTokens.push(t);
  }

  // Merge consecutive English tokens with spaces: ["DUNGEON", "AND", "STONE"] → ["DUNGEON AND STONE"]
  const mergedTokens: string[] = [];
  for (const token of compactedTokens) {
    if (isEnglish(token) && mergedTokens.length > 0 && isEnglish(mergedTokens[mergedTokens.length - 1])) {
      mergedTokens[mergedTokens.length - 1] += ' ' + token;
    } else {
      mergedTokens.push(token);
    }
  }

  const phrases: PhraseData[] = [];

  for (let tokenIndex = 0; tokenIndex < mergedTokens.length; tokenIndex++) {
    const token = mergedTokens[tokenIndex];
    // Skip non-Chinese tokens — they don't need pinyin/HSK analysis
    if (isPunctuation(token) || isNumber(token) || isEnglish(token)) {
      // Still include punctuation as a phrase so the reader can render it inline
      phrases.push({
        text: token,
        pinyin: '',
        isAboveLevel: false,
        hskLevel: null,
      });
      continue;
    }

    // Look up definition: chapter context first (for popup display), then CC-CEDICT.
    // We track the CC-CEDICT result separately so the ⭐ save button can use it
    // (or know to disable itself when there's no dictionary entry).
    // Two different things, deliberately: `definition` is what the learner reads
    // (may be a hand-written note where CC-CEDICT is unusable), `cedictDef` is
    // the dictionary text behind the ⭐ button. Do not collapse them.
    const cedictDef = getCedictDefinition(token);
    const definition = chapterDefinitions?.get(token) || getDefinition(token);

    // Skip the dictionary-split fallback for KEEP terms. Chapter analysis flagged
    // these as story-specific (names, made-up words like 恶灵), so jieba already
    // kept them whole — splitting into single chars would lose the compound meaning
    // even when individual chars happen to have CC-CEDICT entries.
    const isKeep = keepSet?.has(token) || false;

    if (!definition && token.length > 1 && !isKeep) {
      // Fallback: split into sub-tokens that have definitions
      // Try splitting into known dictionary words first, then individual characters
      const subTokens = splitIntoDefinedWords(token, chapterDefinitions);
      for (const sub of subTokens) {
        if (isPunctuation(sub) || isNumber(sub) || isEnglish(sub)) {
          phrases.push({
            text: sub,
            pinyin: '',
            isAboveLevel: false,
            hskLevel: null,
          });
          continue;
        }
        const subPinyin = getPinyin(sub) ?? pinyin(sub, { toneType: 'symbol' });
        const subHskLevel = getHSKLevel(sub, targetLevel);
        const subIsAbove = !isWithinLevel(sub, targetLevel);
        const subCedictDef = getCedictDefinition(sub);
        const subDef = chapterDefinitions?.get(sub) || getDefinition(sub);
        phrases.push({
          text: sub,
          pinyin: subPinyin,
          isAboveLevel: subIsAbove,
          hskLevel: subHskLevel,
          definition: subDef,
          cedictDefinition: subCedictDef || undefined,
          properNoun: isCedictProperNoun(sub) || undefined,
        });
      }
      continue;
    }

    // Pinyin comes from the SAME CC-CEDICT entry the definition came from, so
    // the two cannot disagree (see dictionary.resolveEntry). pinyin-pro remains
    // the fallback for anything CC-CEDICT does not list — names, invented terms.
    const tokenPinyin = getPinyin(token) ?? pinyin(token, { toneType: 'symbol' });

    // Check HSK level (pass targetLevel so character-level fallback works for compounds)
    const hskLevel = getHSKLevel(token, targetLevel);
    const isAbove = !isWithinLevel(token, targetLevel);

    phrases.push({
      text: token,
      pinyin: tokenPinyin,
      isAboveLevel: isAbove,
      hskLevel,
      definition,
      cedictDefinition: cedictDef || undefined,
      properNoun: isCedictProperNoun(token) || undefined,
    });
  }

  // Context pass for syntax-decided heteronyms (没/过/啊/得/行 — see
  // src/pinyin-context.ts). Runs over the FINAL phrase list rather than jieba's
  // raw tokens: jieba emits 很久没 / 没过多久 / 就行 as single tokens, which
  // splitIntoDefinedWords then breaks apart, so the standalone 没/过/行 the
  // reader actually displays only exist at this stage. Running here also means
  // the rules see the true rendered neighbours.
  const phraseTexts = phrases.map(p => p.text);
  for (let i = 0; i < phrases.length; i++) {
    const ph = phrases[i];
    if (!ph.pinyin) continue; // punctuation
    const corrected = contextualPinyin(phraseTexts, i);
    if (!corrected || corrected === ph.pinyin) continue;
    ph.pinyin = corrected;
    // Changing the reading changes which CC-CEDICT entry applies, so the
    // definition has to follow it. Passing the corrected pinyin as the hint makes
    // resolveEntry pick that entry — otherwise an initial 啊 would read "ā" while
    // still displaying "sentence-final particle". Chapter-specific definitions
    // are story truth and are never overwritten.
    if (chapterDefinitions?.has(ph.text)) continue;
    const reDef = getDefinition(ph.text, corrected);
    if (reDef) ph.definition = reDef;
    // Kept separate: cedictDefinition must stay actual dictionary text.
    const reCedict = getCedictDefinition(ph.text, corrected);
    ph.cedictDefinition = reCedict || undefined;
  }

  // Traditional forms for the reader's script toggle. Runs last, over the final
  // phrase list, for the same reason the pinyin pass does: these are the spans
  // the reader actually renders, and each token's neighbours here are its true
  // rendered neighbours — which is exactly the context OpenCC needs to resolve
  // 发→發/髮 and friends. Punctuation and English are skipped (nothing to
  // convert) but still contribute to their neighbours' context.
  for (let i = 0; i < phrases.length; i++) {
    const ph = phrases[i];
    if (!ph.pinyin) continue; // punctuation / English / numbers
    const trad = simplifiedToTraditionalInContext(
      ph.text,
      phraseTexts[i - 1] || '',
      phraseTexts[i + 1] || '',
    );
    if (trad !== ph.text) ph.t = trad;
  }

  // NOTE: an equivalent context pass for DEFINITIONS was built and deliberately
  // removed (2026-07-27). Frame rules like 跟…一样 → "compared with" and
  // 没有 + verb → "did not" look decidable but are not: they misfired on 我跟你说
  // (跟 = "to speak to"), 跟我来 (跟 = "to follow"), 没有希望 / 没有工作 (possessive,
  // not perfective) and 有没有 (A-not-A question form). Guarding each mode needs
  // clause structure the tokenizer does not provide. Two further candidates were
  // measured and rejected outright: 都…了 → "already" (17% precision) and
  // 的话 → conditional (85%, breaks conditionals lacking 如果).
  //
  // 跟's real defect is upstream anyway: filterDefinitions() takes CC-CEDICT's
  // first two senses blind, and 跟 is ordered "heel / to follow closely / ... /
  // compared with". Fix sense ranking there rather than re-adding frame rules.
  return phrases;
}

/**
 * Merged bubble region for the interactive reader
 * Combines multiple text regions from the same speech bubble into one clickable area
 */
interface MergedBubble {
  bbox: { x: number; y: number; width: number; height: number };
  originalText: string;
  simplifiedText: string;
  originalPinyin: string;
  simplifiedPinyin: string;
}

/**
 * Merge text regions that belong to the same bubble into single clickable areas
 * Uses groupNearbyRegions to identify which regions belong together
 */
function mergeBubbleRegions(
  simplifiedRegions: ReplacementRegion[],
  imageWidth: number,
  imageHeight: number,
  renderedBlockBboxes?: Map<number, { x: number; y: number; width: number; height: number }>
): MergedBubble[] {
  if (simplifiedRegions.length === 0) return [];

  // Get group assignments for all regions
  const groups = groupNearbyRegions(simplifiedRegions, imageWidth, imageHeight);

  // Group regions by their group ID
  const groupMap = new Map<number, ReplacementRegion[]>();
  simplifiedRegions.forEach((region, i) => {
    const group = groups[i];
    if (!groupMap.has(group)) groupMap.set(group, []);
    groupMap.get(group)!.push(region);
  });

  // Merge each group into a single bubble
  const mergedBubbles: MergedBubble[] = [];
  for (const groupRegions of groupMap.values()) {
    // Sort by Y coordinate (top-to-bottom) for natural reading order
    groupRegions.sort((a, b) => a.original.boundingBox.y - b.original.boundingBox.y);

    // Determine bounding box: use rendered bbox only when ALL regions map to a single blockId.
    // When mixed (some regions have no blockId, e.g. short text like "这是"), fall back to
    // OCR union bbox so the overlay covers the entire speech bubble.
    const blockIds = new Set(groupRegions.map(r => r.blockId).filter((id): id is number => id !== undefined));
    const allHaveBlocks = groupRegions.every(r => r.blockId !== undefined);
    let bbox: { x: number; y: number; width: number; height: number };

    if (blockIds.size === 1 && renderedBlockBboxes && allHaveBlocks) {
      const blockId = [...blockIds][0];
      const renderedBbox = renderedBlockBboxes.get(blockId);
      if (renderedBbox) {
        bbox = renderedBbox;
      } else {
        bbox = computeOCRBbox(groupRegions);
      }
    } else {
      bbox = computeOCRBbox(groupRegions);
    }

    // For block regions: use block-level text (deduplicated by blockId)
    // Also include standalone regions (no blockId) that share this bubble (e.g. short text)
    let originalText: string;
    let simplifiedText: string;
    if (blockIds.size > 0) {
      const seenBlockIds = new Set<number>();
      const origTexts: string[] = [];
      const simpTexts: string[] = [];
      for (const r of groupRegions) {
        if (r.blockId !== undefined && !seenBlockIds.has(r.blockId)) {
          seenBlockIds.add(r.blockId);
          if (r.originalBlockText) origTexts.push(r.originalBlockText);
          if (r.blockText) simpTexts.push(r.blockText);
        } else if (r.blockId === undefined) {
          // Standalone short-text region sharing this bubble — include its text
          origTexts.push(r.original.text);
          simpTexts.push(r.simplifiedText);
        }
      }
      originalText = origTexts.join('\n');
      simplifiedText = simpTexts.join('\n');
    } else {
      originalText = groupRegions.map(r => r.original.text).join('\n');
      simplifiedText = groupRegions.map(r => r.simplifiedText).join('\n');
    }

    mergedBubbles.push({
      bbox,
      originalText,
      simplifiedText,
      originalPinyin: toPinyin(originalText),
      simplifiedPinyin: toPinyin(simplifiedText),
    });
  }

  // Sort bubbles by Y position (top-to-bottom reading order)
  mergedBubbles.sort((a, b) => a.bbox.y - b.bbox.y);

  return mergedBubbles;
}

function computeOCRBbox(regions: ReplacementRegion[]): { x: number; y: number; width: number; height: number } {
  const minX = Math.min(...regions.map(r => r.original.boundingBox.x));
  const minY = Math.min(...regions.map(r => r.original.boundingBox.y));
  const maxX = Math.max(...regions.map(r => r.original.boundingBox.x + r.original.boundingBox.width));
  const maxY = Math.max(...regions.map(r => r.original.boundingBox.y + r.original.boundingBox.height));
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
}

export interface MangaProcessingOptions {
  targetLevel: HSKLevel;
  targetLevels?: HSKLevel[]; // Multiple levels for multi-level reader (e.g. ['hsk2', 'hsk3'])
  includeEnglish?: boolean;  // Include English translation in reader
  /**
   * Per-bubble English glosses (default on). Measured on a 190-bubble chapter:
   * 5 concurrent LLM calls, 7.7s, $0.0014. Set false to skip.
   */
  includeGlosses?: boolean;
  inputLanguage?: InputLanguage; // Source language: 'auto' (default) detects automatically, or force a specific language
  outputDir?: string;
  // Browser-tab title for the reader (<title>). Set by jobService from the
  // resolved chapter title; when absent, falls back to a filename-derived name.
  readerTitle?: string;
  analyzeOnly?: boolean;
  convertTraditional?: boolean;
  visionProvider?: VisionProvider;
  separateFiles?: boolean; // If true, keep individual page files; otherwise combine into single scrollable image
  onProgress?: (current: number, total: number, message: string) => void; // Progress callback for web API
  chapterContext?: ChapterContext; // Chapter-level context from multi-turn analysis (Approach 4)
  skipSimplification?: boolean; // Skip LLM simplification — return blocks without simplifying (for batched mode)
  precomputedBlockResults?: Map<number, string>; // Pre-computed block simplifications (blockIdx → text), skips LLM call
  inpaintOnly?: boolean; // Run only the inpaint step (no LLM, no metadata) — used to overlap inpainting with LLM
  precomputedCleanedImage?: Buffer; // Pre-inpainted image buffer; if set, replaceTextInImage skips its inpaint step
  // Cache hit support: when set, Phase 4b LLM is skipped — cached simplifications
  // are bbox-matched to fresh blocks instead. Phase 3 chapter analysis is also
  // skipped (no LLM = no need for chapter context).
  cachedSimplifications?: { pages: Array<{ pageNumber: number; blocks: Array<{ bbox: { x: number; y: number; width: number; height: number }; simplifiedText: string }> }> };
  skipChapterAnalysis?: boolean; // Skip Phase 3 chapter analysis entirely (set automatically on cache hit)
  // URL-sourced jobs only: reject the chapter when the language-detection
  // sample finds zero readable text across pages spread through the whole
  // chapter (textless chapter or unsupported language, e.g. Korean — which
  // the CN OCR model cannot read at all). Uploads leave this unset and keep
  // the historic default-to-chinese behavior.
  failOnNoText?: boolean;
  // Internal: misdetection-guard tallies, armed by auto-detection when it
  // resolves to chinese. Shared BY REFERENCE across per-page option spreads —
  // page processing increments it, processMangaPages checks it at the end.
  _langStats?: { totalBlocks: number; allEnglishDropped: number };
  // Internal: set on the one-time forced-english rerun to prevent loops.
  _langRerun?: boolean;
}

export interface PageProcessingResult {
  pageNumber: number;
  originalText: string;
  simplifiedText: string;
  regions: TextRegion[];
  simplifiedRegions: ReplacementRegion[];
  originalImagePath?: string; // Path to the original (unmodified) image
  cleanedImagePath?: string;  // Path to inpainted image (text removed, no new text)
  imageWidth?: number;         // Native pixel width of the page image
  imageHeight?: number;        // Native pixel height; both propagate to reader data
                               // so <img> can reserve correct vertical space upfront
  renderedBlocks?: RenderedBlockInfo[]; // Actual bounding boxes used for rendering
  multiLevelBlockResults?: Map<number, Map<string, string>>; // blockIdx → levelKey → text
  blocksToSimplify?: TextBlock[]; // Blocks ready for simplification (used in batched mode)
  blocksToSimplifyBboxes?: Array<{ x: number; y: number; width: number; height: number }>; // Bbox per simplify-block (parallel to blocksToSimplify), used for cache writes/lookups
  simplifyBlockToOrigBlock?: number[]; // Mapping from simplify index to original block index
  cleanedImageBuffer?: Buffer; // Inpainted image buffer (set when inpaintOnly: true)
  inpaintMs?: number;          // Time spent inpainting (set when inpaintOnly: true)
}

export interface MangaProcessingResult {
  inputPath: string;
  pages: PageProcessingResult[];
  totalRegions: number;
  processingTime: number;
  combinedOutputPath?: string; // Path to the combined scrollable image
  // Text-only cache data (bbox + simplifiedText per block per page) for jobService
  // to write to disk after a successful run. Empty when nothing useful to cache.
  cacheBlocks?: Array<{ pageNumber: number; blocks: Array<{ bbox: { x: number; y: number; width: number; height: number }; simplifiedText: string }> }>;
  // Chapter analysis output (characters, keep, definitions) — surfaced so
  // jobService can persist popup-restoring metadata alongside the cache.
  // Undefined when chapter analysis didn't run (single-page chapters,
  // analyze-only mode, or cache hit on a v1 cache without these fields).
  chapterContext?: ChapterContext;
}

// Timing statistics for performance analysis
export interface TimingStats {
  ocr: number;
  llm: number;
  inpaintServer: number;
  textRendering: number;
  total: number;
  chapterAnalysis: number;
  loading: number;
  stitching: number;
  ocrScan: number;  // Phase 3 OCR scan (separate from per-page OCR)
  // Diagnostic buckets — previously untracked work
  saveImage: number;       // sharp jpeg encode for original/cleaned saves
  blockGrouping: number;   // groupRegionsIntoBlocks calls (cumulative across all passes)
  filterPasses: number;    // chroma/structureless/punct/etc filter loops
  colorSampling: number;   // sampleRegionColor for fill+text colors
  metaBuild: number;       // renderedBlocks metadata construction (font calc, pinyin, etc.)
  saturation: number;      // checkTextColorSaturation per region (one sharp.extract each)
  decodePage: number;      // decodePage calls in runInpaintStep + replaceTextInImage
}

const EMPTY_TIMING: TimingStats = {
  ocr: 0, llm: 0, inpaintServer: 0, textRendering: 0, total: 0,
  chapterAnalysis: 0, loading: 0, stitching: 0, ocrScan: 0,
  saveImage: 0, blockGrouping: 0, filterPasses: 0, colorSampling: 0, metaBuild: 0,
  saturation: 0, decodePage: 0,
};

// Global timing accumulator for the current job
let globalTimingStats: TimingStats = { ...EMPTY_TIMING };
let processedPageCount = 0;
// Page-dimension accumulator (pixels) — summed as images load, averaged for the
// cost dashboard.
let pageWidthSum = 0;
let pageHeightSum = 0;
let dimSampleCount = 0;

export function resetTimingStats(): void {
  globalTimingStats = { ...EMPTY_TIMING };
  processedPageCount = 0;
  pageWidthSum = 0;
  pageHeightSum = 0;
  dimSampleCount = 0;
}

export function getTimingStats(): TimingStats & {
  pageCount: number;
  avgPageWidth: number;
  avgPageHeight: number;
} {
  return {
    ...globalTimingStats,
    pageCount: processedPageCount,
    avgPageWidth: dimSampleCount > 0 ? Math.round(pageWidthSum / dimSampleCount) : 0,
    avgPageHeight: dimSampleCount > 0 ? Math.round(pageHeightSum / dimSampleCount) : 0,
  };
}

/**
 * Pre-processing: split source images that are taller than the threshold into
 * smaller chunks BEFORE the merge / OCR logic runs.
 *
 * Why: PaddleOCR's PP-OCRv5 server downscales any image whose longest side
 * exceeds 4000 px, degrading detection on small text. Above ~32,767 px it
 * fails entirely (OpenCV's `cv::remap` asserts on SHRT_MAX dimensions). Some
 * sites (asurascans, scrolltoon-style hosts) serve a whole chapter as one
 * 16-48k-px-tall image, which lands solidly in both failure zones.
 *
 * Splitting source images into ~2000 px chunks keeps every chunk in
 * PaddleOCR's sweet spot. Downstream `determineMergeGroups` then checks every
 * chunk boundary for spanning text and stitches chunks back where dialogue
 * crosses the cut — so we don't have to do any bbox-aware splitting here. The
 * existing merge logic handles all the spanning-text edge cases for free.
 *
 * Worst case (every chunk boundary has spanning text → all chunks remerge
 * into the original size): behavior is identical to today, no regression.
 */
const INPUT_SPLIT_THRESHOLD_PX = parseInt(process.env.INPUT_SPLIT_THRESHOLD_PX || '2500', 10);
const INPUT_SPLIT_TARGET_PX = parseInt(process.env.INPUT_SPLIT_TARGET_PX || '2000', 10);

async function splitTallImages(imageBuffers: Buffer[]): Promise<Buffer[]> {
  // Per-source work: decode metadata once, clone the sharp pipeline per chunk
  // (avoids re-decoding the full image for each extract), encode each chunk as
  // lossless webp (2-3x faster than default PNG). Sources run in parallel with
  // bounded concurrency — sharp/libvips is multi-threaded per op so 4-way
  // concurrent sources saturates the CPU without thrashing memory.
  const SOURCE_CONCURRENCY = 4;
  const limit = pLimit(SOURCE_CONCURRENCY);
  let splitSourceCount = 0;

  const perSource = await Promise.all(imageBuffers.map((buffer, srcIdx) =>
    limit(async (): Promise<Buffer[]> => {
      const src = sharp(buffer);
      const meta = await src.metadata();
      const height = meta.height || 0;
      const width = meta.width || 0;

      if (height <= INPUT_SPLIT_THRESHOLD_PX || width <= 0) {
        return [buffer];
      }

      const numChunks = Math.ceil(height / INPUT_SPLIT_TARGET_PX);
      const chunkHeight = Math.floor(height / numChunks);

      console.error(`  [Split] src#${srcIdx} ${width}x${height} -> ${numChunks} chunks of ~${chunkHeight}px`);

      const chunkSpecs: Array<{ top: number; height: number }> = [];
      for (let i = 0; i < numChunks; i++) {
        const top = i * chunkHeight;
        // Last chunk absorbs any remainder pixels so the chunks sum to `height` exactly
        const thisChunkHeight = (i === numChunks - 1) ? (height - top) : chunkHeight;
        chunkSpecs.push({ top, height: thisChunkHeight });
      }

      const chunks = await Promise.all(chunkSpecs.map(spec =>
        src.clone()
          .extract({ left: 0, top: spec.top, width, height: spec.height })
          .toBuffer()
      ));
      splitSourceCount++;
      return chunks;
    })
  ));

  const result = perSource.flat();
  if (splitSourceCount > 0) {
    console.error(`  [Split] Split ${splitSourceCount} oversized source image(s); page count: ${imageBuffers.length} -> ${result.length}`);
  }

  return result;
}

/**
 * Check if two adjacent pages have content spanning their boundary.
 * Extracts strips from the bottom of page N and top of page N+1,
 * stitches them together, and runs OCR to detect any text.
 *
 * @param pageNBuffer - Image buffer for page N
 * @param pageN1Buffer - Image buffer for page N+1
 * @param stripPercent - Percentage of page height to extract (default 3%)
 * @returns true if text is detected spanning the boundary
 */
export async function checkBoundaryForText(
  pageNBuffer: Buffer,
  pageN1Buffer: Buffer,
  stripPercent: number = 3
): Promise<{ hasText: boolean; texts: string[] }> {
  try {
    // Get dimensions of both pages
    const metaN = await sharp(pageNBuffer).metadata();
    const metaN1 = await sharp(pageN1Buffer).metadata();

    console.error(`  [Boundary Debug] Page N: ${metaN.width}x${metaN.height}, Page N+1: ${metaN1.width}x${metaN1.height}`);

    if (!metaN.height || !metaN.width || !metaN1.height || !metaN1.width) {
      console.error(`  [Boundary Debug] Missing dimensions, skipping boundary check`);
      return { hasText: false, texts: [] };
    }

    // Calculate strip heights (percentage of each page)
    const stripHeightN = Math.round(metaN.height * (stripPercent / 100));
    const stripHeightN1 = Math.round(metaN1.height * (stripPercent / 100));

    console.error(`  [Boundary Debug] Strip heights: N=${stripHeightN}px (${stripPercent}% of ${metaN.height}), N+1=${stripHeightN1}px`);

    // Ensure strip heights are at least 1 pixel
    if (stripHeightN < 1 || stripHeightN1 < 1) {
      console.error(`  [Boundary Debug] Strip height too small, skipping`);
      return { hasText: false, texts: [] };
    }

    // Extract bottom strip from page N
    const bottomStrip = await sharp(pageNBuffer)
      .extract({
        left: 0,
        top: metaN.height - stripHeightN,
        width: metaN.width,
        height: stripHeightN
      })
      .toBuffer();

    // Extract top strip from page N+1
    const topStrip = await sharp(pageN1Buffer)
      .extract({
        left: 0,
        top: 0,
        width: metaN1.width,
        height: stripHeightN1
      })
      .toBuffer();

    // Stitch the strips together vertically
    // First, resize to same width if needed
    const targetWidth = Math.min(metaN.width, metaN1.width);

    const resizedBottom = await sharp(bottomStrip)
      .resize(targetWidth, stripHeightN)
      .toBuffer();

    const resizedTop = await sharp(topStrip)
      .resize(targetWidth, stripHeightN1)
      .toBuffer();

    // Combine into one image
    const combinedStrip = await sharp({
      create: {
        width: targetWidth,
        height: stripHeightN + stripHeightN1,
        channels: 3,
        background: { r: 255, g: 255, b: 255 }
      }
    })
      .composite([
        { input: resizedBottom, top: 0, left: 0 },
        { input: resizedTop, top: stripHeightN, left: 0 }
      ])
      .png()
      .toBuffer();

    // Run OCR on the combined strip
    const extraction = await extractTextFromImage(combinedStrip, 0, 'paddleocr');

    // Merge only when a text box STRADDLES the seam line. "Any text in the
    // strip" was the old test, and it fires on every page of a site that stamps
    // its watermark near the bottom edge (measured 2026-09-04 on a 51-page
    // dmzj upload: 49/50 seams merged into a 68,515 px unit that OCR could not
    // read, while 0/50 seams actually crossed a text box). A bubble split
    // across two pages shows up here as one box starting above the seam and
    // ending below it; a watermark or a line sitting near the edge does not.
    const seamPct = (stripHeightN / (stripHeightN + stripHeightN1)) * 100;
    const tolerancePct = (2 / (stripHeightN + stripHeightN1)) * 100; // ~2 px either side
    const straddling = extraction.regions.filter(r =>
      r.boundingBox.y < seamPct - tolerancePct &&
      r.boundingBox.y + r.boundingBox.height > seamPct + tolerancePct
    );
    const hasText = straddling.length > 0;
    const texts = extraction.regions.map(r => r.text);

    if (hasText) {
      console.error(`  [Boundary] ${straddling.length} text region(s) straddle the seam: ${straddling.map(r => `"${r.text}"`).join(', ')}`);
    } else if (extraction.regions.length > 0) {
      console.error(`  [Boundary] ${extraction.regions.length} region(s) near the seam but none cross it — not merging`);
    }

    return { hasText, texts };
  } catch (err) {
    console.error(`  [Boundary] Error checking boundary: ${err}`);
    return { hasText: false, texts: [] };
  }
}

/**
 * Determine which pages should be merged based on boundary text detection.
 * Returns an array of merge groups, where each group is an array of page indices
 * that should be merged together.
 *
 * Example: [[0], [1, 2, 3], [4], [5, 6]] means:
 * - Page 0 stays alone
 * - Pages 1, 2, 3 merge into one
 * - Page 4 stays alone
 * - Pages 5, 6 merge into one
 */
async function determineMergeGroups(
  imageBuffers: Buffer[],
  onProgress?: (current: number, total: number, message: string) => void
): Promise<{ groups: number[][]; boundaryTexts: string[] }> {
  if (imageBuffers.length <= 1) {
    return { groups: imageBuffers.map((_, i) => [i]), boundaryTexts: [] };
  }

  console.error(`  [Boundary] Checking ${imageBuffers.length - 1} page boundaries for text...`);

  // Check each boundary in parallel, collecting detected text for language detection
  const allBoundaryTexts: string[] = [];
  const limit = pLimit(5); // Limit concurrent boundary checks
  const boundaryChecks = [];

  for (let i = 0; i < imageBuffers.length - 1; i++) {
    boundaryChecks.push(
      limit(async () => {
        const result = await checkBoundaryForText(imageBuffers[i], imageBuffers[i + 1]);
        allBoundaryTexts.push(...result.texts);
        return { index: i, needsMerge: result.hasText };
      })
    );
  }

  const results = await Promise.all(boundaryChecks);

  // Sort by index to ensure order
  results.sort((a, b) => a.index - b.index);

  // Build merge groups based on which boundaries have text
  const mergeGroups: number[][] = [];
  let currentGroup: number[] = [0];

  for (const { index, needsMerge } of results) {
    if (needsMerge) {
      // Boundary has text, add next page to current group
      currentGroup.push(index + 1);
    } else {
      // No text at boundary, finalize current group and start new one
      mergeGroups.push(currentGroup);
      currentGroup = [index + 1];
    }
  }

  // Don't forget the last group
  mergeGroups.push(currentGroup);

  // Log merge groups
  const mergedCount = mergeGroups.filter(g => g.length > 1).length;
  console.error(`  [Boundary] ${mergedCount} merge group(s) detected`);
  mergeGroups.filter(g => g.length > 1).forEach(g => {
    console.error(`    Pages ${g.map(i => i + 1).join(' + ')} will be merged`);
  });

  return { groups: mergeGroups, boundaryTexts: allBoundaryTexts };
}

/**
 * Stitch result containing the merged image and metadata
 */
interface StitchResult {
  buffer: Buffer;
  stitchYPositions: number[]; // Y positions (as %) where pages meet
  pageHeights: number[]; // Original page heights in pixels
  totalHeight: number;
}

/**
 * Stitch multiple page images together vertically.
 * Returns the stitched buffer plus metadata about stitch positions.
 */
async function stitchPages(imageBuffers: Buffer[]): Promise<StitchResult> {
  if (imageBuffers.length === 1) {
    const meta = await sharp(imageBuffers[0]).metadata();
    return {
      buffer: imageBuffers[0],
      stitchYPositions: [],
      pageHeights: [meta.height || 0],
      totalHeight: meta.height || 0,
    };
  }

  // Get dimensions of all pages
  const metadatas = await Promise.all(
    imageBuffers.map(buf => sharp(buf).metadata())
  );

  // Use the minimum width (resize others to match)
  const targetWidth = Math.min(...metadatas.map(m => m.width || 0));
  const heights = metadatas.map(m => m.height || 0);
  const totalHeight = heights.reduce((sum, h) => sum + h, 0);

  // Calculate stitch Y positions (where each page boundary is, as percentage)
  const stitchYPositions: number[] = [];
  let cumHeight = 0;
  for (let i = 0; i < heights.length - 1; i++) {
    cumHeight += heights[i];
    stitchYPositions.push((cumHeight / totalHeight) * 100);
  }

  // Resize all images to target width
  const resizedBuffers = await Promise.all(
    imageBuffers.map(async (buf, i) => {
      const meta = metadatas[i];
      if (meta.width !== targetWidth) {
        return sharp(buf).resize(targetWidth).toBuffer();
      }
      return buf;
    })
  );

  // Build composite operations
  let currentTop = 0;
  const composites = resizedBuffers.map((buf, i) => {
    const op = { input: buf, top: currentTop, left: 0 };
    currentTop += heights[i];
    return op;
  });

  // Create the stitched image
  const stitched = await sharp({
    create: {
      width: targetWidth,
      height: totalHeight,
      channels: 3,
      background: { r: 255, g: 255, b: 255 }
    }
  })
    .composite(composites)
    .png()
    .toBuffer();

  console.error(`  [Stitch] Merged ${imageBuffers.length} pages into ${targetWidth}x${totalHeight} image`);
  console.error(`  [Stitch] Stitch Y positions: ${stitchYPositions.map(y => y.toFixed(1) + '%').join(', ')}`);

  return {
    buffer: stitched,
    stitchYPositions,
    pageHeights: heights,
    totalHeight,
  };
}

/**
 * Process a single manga page
 */
export async function processMangaPage(
  imagePath: string,
  options: MangaProcessingOptions
): Promise<PageProcessingResult> {
  const pageStartTime = Date.now();
  console.error(`Processing: ${path.basename(imagePath)}`);

  // Load image
  const pageImage = await loadImage(imagePath);

  // Extract text from image (OCR)
  console.error('  Extracting text...');
  const ocrStartTime = Date.now();
  let extraction = await extractTextFromImage(
    pageImage.buffer,
    pageImage.pageNumber,
    options.visionProvider || 'gemini',
    options.inputLanguage
  );
  const ocrTime = Date.now() - ocrStartTime;
  console.error(`  [TIMING] OCR: ${(ocrTime / 1000).toFixed(2)}s`);
  globalTimingStats.ocr += ocrTime;

  // Convert Traditional to Simplified if needed
  if (options.convertTraditional !== false) {
    extraction = convertExtractionToSimplified(extraction);
    extraction = stripUrlTokensFromExtraction(extraction);
  }

  // Filter out likely sound effects (backup filters in case OCR still extracts them)
  const originalCount = extraction.regions.length;

  // Aspect ratio is needed inside the filter callback below; declare it here
  // so it's available when the filter runs (was previously declared after,
  // causing a temporal-dead-zone error in the english-input + skipSimplification path).
  const imageAspectRatio = pageImage.height / pageImage.width;

  // Helper: Calculate rotation threshold based on character count, bounding box shape, and height
  // Short text (1-2 chars) with square bounding boxes get unreliable rotation from PaddleOCR
  // But large short text (>5% height) is likely a sound effect - use strict threshold
  // For English, the threshold decays 5x slower per char (English letters are smaller units than Chinese chars)
  const isEnglishInput = options.inputLanguage === 'english';

  // Pre-compute text color saturation for all regions. Decode the page ONCE
  // and KEEP the raw buffer alive — we'll reuse it later in replaceTextInImage
  // for the classifier + bbox-expansion checks (Fix #2: share decode across phases).
  const saturationMap = new Map<number, { isColorful: boolean; avgChroma: number; textPixelCount: number; hasSolidBackground: boolean; bgStdDev: number }>();
  const _satStart1 = Date.now();
  let sharedPageData: { data: Buffer; width: number; height: number; channels: number } | undefined;
  try {
    const { data: satPageData, info: satPageInfo } = await sharp(pageImage.buffer).raw().toBuffer({ resolveWithObject: true });
    sharedPageData = { data: satPageData, width: satPageInfo.width, height: satPageInfo.height, channels: satPageInfo.channels };
    for (let i = 0; i < extraction.regions.length; i++) {
      try {
        const result = checkTextColorSaturation(satPageData, satPageInfo.width, satPageInfo.height, satPageInfo.channels, extraction.regions[i].boundingBox);
        saturationMap.set(i, result);
      } catch (e) { /* skip */ }
    }
  } catch (e) {
    console.error(`  [Saturation] Page decode failed: ${e}`);
  }
  globalTimingStats.saturation += Date.now() - _satStart1;

  extraction.regions = extraction.regions.filter((region, i) => {
    const text = region.text.trim();

    // Filter by rotation angle - threshold varies by character count and bounding box shape
    // PaddleOCR rotation is unreliable for short text with square bounding boxes
    const rotation = Math.abs(region.rotation || 0);
    let rotationThreshold = getRotationThreshold(text, region.boundingBox, isEnglishInput);
    // Large 1-2 char text with punctuation on a solid background is likely dialogue (e.g. "呃！!")
    // Override the strict 3.5° threshold to lenient 17.5°
    if (rotationThreshold === 3.5) {
      const hasPunct = /[！？。，!?,.]/.test(text);
      const solidBg = saturationMap.get(i)?.hasSolidBackground ?? false;
      if (hasPunct && solidBg) {
        rotationThreshold = 17.5;
      }
    }
    if (rotation > rotationThreshold) {
      // High-confidence multi-char text bypasses rotation filter — likely rotated dialogue, not SFX
      const charCount = text.replace(/[^\u4e00-\u9fff\u3400-\u4dbf]/g, '').length;
      if (region.confidence >= 0.90 && charCount >= 3) {
        console.error(`  [Rotation] Bypassing filter (${rotation.toFixed(1)}° > ${rotationThreshold}°, conf=${region.confidence.toFixed(2)}, ${charCount} chars): "${text}"`);
      } else {
        console.error(`  Filtering rotated text (${rotation.toFixed(1)}° > ${rotationThreshold}°): "${text}"`);
        return false;
      }
    }

    // Height filters are deferred until after non-height filters run,
    // so we can use region density to pick the right baseline.
    // Max height filter still runs here since oversized text is always wrong.
    const baselineAspect = 2.6;
    const scaledMaxHeight = 12 * (baselineAspect / imageAspectRatio);
    const heightPct = region.boundingBox.height;
    if (heightPct > scaledMaxHeight) {
      console.error(`  Filtering oversized text (h=${heightPct.toFixed(1)}% > ${scaledMaxHeight.toFixed(1)}%): "${text}"`);
      return false;
    }

    // NOTE: Aspect ratio filter REMOVED - it was catching legitimate vertical dialogue text
    // Vertical Chinese text naturally has high aspect ratios and should not be filtered

    // Filter emoji-only text (OCR artifacts from decorative elements)
    const strippedEmoji = text.replace(/[\p{Emoji_Presentation}\p{Extended_Pictographic}]/gu, '').trim();
    if (strippedEmoji.length === 0 && text.length > 0) {
      console.error(`  Filtering emoji: "${text}"`);
      return false;
    }

    // Filter "111", "1111" etc. — OCR misreads "!!!" exclamation bubbles as repeated 1s
    if (/^1{2,}$/.test(text)) {
      console.error(`  Filtering misread exclamation marks: "${text}"`);
      return false;
    }


    // Log saturation info; chroma filtering is deferred until after block grouping
    // so that multi-line colored UI boxes (e.g. system notifications) can merge first
    const satResult = saturationMap.get(i);
    if (satResult) {
      console.error(`  [Saturation] "${text}" → chroma=${satResult.avgChroma.toFixed(0)}, textPixels=${satResult.textPixelCount}, colorful=${satResult.isColorful}`);
      if (satResult.isColorful) {
        region.isColorful = true;
        region.avgChroma = satResult.avgChroma;
      }
    }

    return true;
  });

  if (extraction.regions.length < originalCount) {
    console.error(`  Filtered ${originalCount - extraction.regions.length} regions (rotation/height/sound)`);
  }

  // Comment section detection for single pages (same logic as merged path)
  const baseAdjacencyDistanceForComment = parseFloat(process.env.NEIGHBOR_DISTANCE || '15');
  const imageAspectRatioForComment = pageImage.height / pageImage.width;
  const adjacencyDistanceForComment = baseAdjacencyDistanceForComment / Math.max(1, imageAspectRatioForComment);
  const isEnglishInputSingle = options.inputLanguage === 'english';
  let heightBaseline = isEnglishInputSingle ? 1.1 : 0.95;
  if (extraction.regions.length >= 20) {
    const prelimBlocks = groupRegionsIntoBlocks(extraction.regions, adjacencyDistanceForComment, 'chinese', imageAspectRatioForComment);
    const nonEmptyBlocks = prelimBlocks.filter(b => b.regionIndices.length > 0);
    const regionsPerBlock = nonEmptyBlocks.length > 0 ? extraction.regions.length / nonEmptyBlocks.length : 999;
    console.error(`  [Comment detection] ${extraction.regions.length} regions, ${nonEmptyBlocks.length} blocks, ratio=${regionsPerBlock.toFixed(2)}`);
    // Chinese-natural ratio is 1.5-2.0 (each char/line is its own OCR region,
    // bubbles don't merge into multi-region blocks). English/Japanese natural
    // ratio is 4+. Different thresholds prevent dialogue from being mis-flagged.
    // Granularity follows line structure, not glyph shape: PaddleOCR returns
    // EN/JA/KO text as full-line regions (natural regions/block ratio 4+),
    // Chinese as char/short-run regions (1.5-2) — so translation sources share
    // the English threshold here.
    const commentRatioThreshold = isTranslationSource(options.inputLanguage) ? 2.75 : 1.3;
    if (regionsPerBlock < commentRatioThreshold) {
      heightBaseline = isEnglishInputSingle ? 1.9 : 2.4;
      console.error(`  [Comment detection] Low regions/block ratio — raising height baseline to ${heightBaseline}`);
    }
  }

  const baselineAspect = 2.6;
  const imageAspectRatio_height = pageImage.height / pageImage.width;
  let scaledMinHeight = heightBaseline * (baselineAspect / imageAspectRatio_height);
  // Median cap — only when comment detection did NOT raise the baseline
  // (a raised baseline means the page looks like UI/comment chrome and the
  // strict threshold is intentional).
  const singleBaselineRaised = heightBaseline > (isEnglishInputSingle ? 1.1 : 0.95);
  if (!singleBaselineRaised) {
    const heightCapSingle = computeMedianHeightCap(extraction.regions, imageAspectRatio_height, scaledMinHeight);
    if (heightCapSingle.capped) {
      console.error(`  [HeightCap] median cap active: threshold ${scaledMinHeight.toFixed(2)}% -> ${heightCapSingle.thresholdPct.toFixed(2)}% of height (median line = ${heightCapSingle.medianWPct.toFixed(2)}% of width, ${extraction.regions.length} regions pooled)`);
      scaledMinHeight = heightCapSingle.thresholdPct;
    }
  }
  const beforeHeightFilter = extraction.regions.length;
  extraction.regions = extraction.regions.filter((region) => {
    const heightPct = getEffectiveHeight(region, imageAspectRatio_height);
    if (heightPct < scaledMinHeight) {
      const rawH = region.boundingBox.height;
      const rotNote = (rawH !== heightPct) ? ` (raw=${rawH.toFixed(1)}%, rot-corrected)` : '';
      console.error(`  Filtering short text (h=${heightPct.toFixed(1)}% < ${scaledMinHeight.toFixed(1)}%, baseline=${heightBaseline})${rotNote}: "${region.text.trim()}"`);
      return false;
    }
    return true;
  });
  if (extraction.regions.length < beforeHeightFilter) {
    console.error(`  Height filter (baseline=${heightBaseline}, ${beforeHeightFilter} regions pre-filter): removed ${beforeHeightFilter - extraction.regions.length} regions`);
  }

  // Filter: Skip non-Chinese text (English, numbers-only, etc.) UNLESS adjacent to Chinese text
  // Pure English adjacent to Chinese = likely part of same speech bubble (e.g., game names)
  // Pure English isolated = likely watermark/UI (e.g., "LINE WEBTOON")
  // Skip entirely when input is English — all text is expected to be English
  const beforeNonChineseFilter = extraction.regions.length;
  const baseAdjacencyDistance = parseFloat(process.env.NEIGHBOR_DISTANCE || '15');  // % of image size

  // Scale threshold by aspect ratio - tall images (webtoons) need tighter threshold
  // because percentage-based distance doesn't account for extreme aspect ratios
  // e.g., 25% on a 10:1 aspect ratio image = 1800+ pixels in Y direction
  // (imageAspectRatio is declared earlier in this function)
  const adjacencyDistance = baseAdjacencyDistance / Math.max(1, imageAspectRatio);
  console.error(`  [DEBUG] Image ${pageImage.width}x${pageImage.height} (aspect ${imageAspectRatio.toFixed(1)}:1), adjacency threshold: ${adjacencyDistance.toFixed(1)}%`);

  // Helper: count Chinese characters in text
  const countChineseChars = (text: string): number => {
    let count = 0;
    for (const char of text) {
      const code = char.charCodeAt(0);
      if ((code >= 0x4e00 && code <= 0x9fff) || (code >= 0x3400 && code <= 0x4dbf)) {
        count++;
      }
    }
    return count;
  };

  // Helper: check if region has any Chinese characters
  const hasChinese = (region: TextRegion): boolean => countChineseChars(region.text) > 0;

  // Helper: calculate distance between two regions (edge-to-edge gap, in % coordinates)
  // Returns 0 if boxes touch or overlap, otherwise the gap between closest edges
  const regionDistance = (a: TextRegion, b: TextRegion): number => {
    const gapX = Math.max(0,
      Math.max(a.boundingBox.x, b.boundingBox.x) -
      Math.min(a.boundingBox.x + a.boundingBox.width, b.boundingBox.x + b.boundingBox.width)
    );
    const gapY = Math.max(0,
      Math.max(a.boundingBox.y, b.boundingBox.y) -
      Math.min(a.boundingBox.y + a.boundingBox.height, b.boundingBox.y + b.boundingBox.height)
    );
    return Math.sqrt(gapX ** 2 + gapY ** 2);
  };

  // English-input mode only: drop "structureless" English blocks. CN input no
  // longer filters English per-region here — all-English blocks are dropped
  // post-merge (see the all-English block drop below), so isolated English just
  // flows through to block grouping.
  if (options.inputLanguage === 'english') {
    // English-input mode: drop blocks whose joined text has no sentence
    // punctuation. "No punctuation = filter" is the only check — spaces
    // are not a reliable dialogue signal because (a) "Then..." is
    // dialogue but has no space, and (b) OCR's English-spacing-fix
    // inserts artificial spaces into misread UI text ("BARBAPTAN" →
    // "BARB APT AN", "RMATION" → "RMA TION").
    //
    // "Punctuation" here is deliberately narrow — only actual sentence-
    // terminal / clause-separator marks: . , ! ? ; : ( ) … (English) and
    // their Chinese equivalents. Square brackets, curly braces, angle
    // brackets, quotes, apostrophes, hyphens, dashes, slashes, pipes,
    // etc. do NOT count. This excludes stylized title-card text like
    // "[BARBARIAN]" (brackets aren't real punctuation), bracketed
    // tags, and other decorative typography that would otherwise slip
    // through. \p{P} (all-Unicode-punctuation) is too permissive — it
    // matches brackets and lots of glyphs that exist on UI labels but
    // never in real dialogue.
    //
    // What catches:
    //   - Single-word UI: "BARBARIAN", "INFORMATION", "BOOM"
    //   - UI grids: "HUMAN BARBARIAN FAIRY FURRY"
    //   - Spacing-fixed UI: "BARB APT AN", "RMA TION"
    //   - Bracketed UI titles: "[BARBARIAN]"
    //
    // What passes:
    //   - "Then...", "What?", "Hello!" (terminal punctuation)
    //   - "WAS COMPLETELY SHIT." (joined multi-region, has period)
    //   - "POW!", "BAM!" (punctuated SFX)
    //
    // Cost: "Hello there" / "I am" type unpunctuated dialogue gets
    // filtered. In manga, real dialogue almost always has terminal
    // punctuation, so this is rare.
    // Bypass rule: text with 2+ whitespace characters (3+ tokens) bypasses
    // the punctuation check. "THAT WAS WHY" type unpunctuated 3-word
    // dialogue is rare-but-real; treating "3+ tokens" as a strong dialogue
    // signal lets it through. Trade-off: 3+ token UI grids ("HUMAN
    // BARBARIAN FAIRY FURRY") and 3+ token misread-then-spacing-fixed UI
    // ("BARB APT AN") also bypass. Empirically these are rarer than the
    // dialogue case, especially the 3-token unpunctuated dialogue case
    // which appears once per chapter at most.
    const SENTENCE_PUNCT = /[.,!?;:()…。，！？；：（）]/u;
    // A trailing dash is comics' interruption mark ("THEN AGAIN--", "WAIT-")
    // — real dialogue by definition, so it counts as sentence punctuation.
    // Checked on the JOINED block text so only the block's final line needs
    // the dash. Known accepted leak: dash-suffixed SFX/garble ("B22-",
    // "BZZ-") now passes too — deliberate, per the err-toward-keep asymmetry
    // (dropped dialogue like Ep120 p35's "THEN AGAIN--" is the worse failure).
    const TRAILING_DASH = /[-–—]$/;
    const beforeStructureFilter = extraction.regions.length;
    const prelimBlocks = groupRegionsIntoBlocks(extraction.regions, adjacencyDistance, 'english', imageAspectRatio);
    const structurelessIndices = new Set<number>();
    for (const block of prelimBlocks) {
      if (block.regionIndices.length === 0) continue;
      const joinedText = block.regionIndices
        .map(i => extraction.regions[i].text.trim())
        .filter(t => t.length > 0)
        .join(' ');
      if (joinedText.length === 0) {
        for (const i of block.regionIndices) structurelessIndices.add(i);
        continue;
      }
      const spaceCount = (joinedText.match(/\s/g) || []).length;
      if (spaceCount >= 2) continue; // bypass punctuation check
      if (!SENTENCE_PUNCT.test(joinedText) && !TRAILING_DASH.test(joinedText)) {
        for (const i of block.regionIndices) structurelessIndices.add(i);
      }
    }
    let filteredStructureless = 0;
    extraction.regions = extraction.regions.filter((region, i) => {
      if (structurelessIndices.has(i)) {
        filteredStructureless++;
        console.error(`  Filtering structureless block (no sentence punct): "${region.text.trim().substring(0, 30)}"`);
        return false;
      }
      return true;
    });
    if (filteredStructureless > 0) {
      console.error(`  Structure filter (EN mode): removed ${filteredStructureless} structureless regions of ${beforeStructureFilter}`);
    }
  }

  // Filter: Remove punctuation-only regions (no actual characters)
  // These can be orphaned punctuation marks left after nearby text was filtered
  const beforePunctFilter = extraction.regions.length;
  extraction.regions = extraction.regions.filter(region => {
    const stripped = region.text.replace(/[\s\p{P}\p{S}]/gu, '');
    if (stripped.length === 0) {
      console.error(`  Filtering punctuation-only: "${region.text}"`);
      return false;
    }
    return true;
  });
  if (extraction.regions.length < beforePunctFilter) {
    console.error(`  Punctuation filter: removed ${beforePunctFilter - extraction.regions.length} regions`);
  }

  // Filter: Proximity-based clustering + size filter
  // Forum/UI text has many nearby regions AND small font size; speech bubbles are isolated or have large text
  if (process.env.ENABLE_CLUSTER_FILTER !== 'false' && extraction.regions.length > 0) {
    const neighborThreshold = parseInt(process.env.NEIGHBOR_THRESHOLD || '4', 10);
    const distancePercent = parseFloat(process.env.NEIGHBOR_DISTANCE || '15');  // % of image size
    const minTextHeight = parseFloat(process.env.MIN_TEXT_HEIGHT || '1.5');  // % of image height
    const largeTextHeight = parseFloat(process.env.LARGE_TEXT_HEIGHT || '2.5');  // % of image height

    // Cap the height denominator at FILTER_HEIGHT_CAP_PX (default 2000). The
    // size/distance thresholds are calibrated for webtoon-shape pages (~2000 px
    // tall). On long-strip images (e.g. an asurascans chapter served as one
    // 16000 px file) bbox.height collapses to a tiny fraction of a huge
    // denominator — normal 60 px text becomes 0.375% of image height and gets
    // dropped as "small + has neighbor." The same denominator inflation makes
    // distant bubbles look adjacent: a 500 px gap reads as 3.125% < the 5%
    // threshold, so they cluster. Scaling y-coords and heights by
    // (image_height / min(image_height, cap)) treats any image taller than the
    // cap as if it were exactly `cap` tall for filter math. No-op for
    // h <= cap (heightScale = 1) — webtoons behave identically to before.
    const filterHeightCap = parseInt(process.env.FILTER_HEIGHT_CAP_PX || '2000', 10);
    const heightScale = pageImage.height / Math.min(pageImage.height, filterHeightCap);

    const beforeFilter = extraction.regions.length;

    // For each region, count neighbors within distance threshold
    const neighborCounts = extraction.regions.map((region, i) => {
      let neighbors = 0;
      const cx = region.boundingBox.x + region.boundingBox.width / 2;
      const cy = (region.boundingBox.y + region.boundingBox.height / 2) * heightScale;

      for (let j = 0; j < extraction.regions.length; j++) {
        if (i === j) continue;
        const other = extraction.regions[j];
        const ox = other.boundingBox.x + other.boundingBox.width / 2;
        const oy = (other.boundingBox.y + other.boundingBox.height / 2) * heightScale;

        // Distance in percentage coordinates
        const dist = Math.sqrt((cx - ox) ** 2 + (cy - oy) ** 2);
        if (dist < distancePercent) {
          neighbors++;
        }
      }
      return neighbors;
    });

    // Filter out regions with too many neighbors OR small text with any neighbors
    // But large text always passes (clearly dialogue)
    extraction.regions = extraction.regions.filter((region, i) => {
      const textHeight = region.boundingBox.height * heightScale;
      const isSmallText = textHeight < minTextHeight;
      const isLargeText = textHeight >= largeTextHeight;

      // Large text = clearly dialogue, always keep regardless of neighbors
      if (isLargeText) {
        return true;
      }

      // Large text with many neighbors = filter (crowded area)
      if (neighborCounts[i] > neighborThreshold) {
        console.error(`  Filtering crowded region (${neighborCounts[i]} neighbors): "${region.text.substring(0, 20)}..."`);
        return false;
      }

      // Small text with ANY neighbors = likely forum/UI text, filter
      if (isSmallText && neighborCounts[i] > 0) {
        console.error(`  Filtering small crowded text (h=${textHeight.toFixed(1)}%, ${neighborCounts[i]} neighbors): "${region.text.substring(0, 20)}..."`);
        return false;
      }

      return true;
    });

    if (extraction.regions.length < beforeFilter) {
      console.error(`  Proximity+size filter: removed ${beforeFilter - extraction.regions.length} regions`);
    }
  }

  // Filter: Skip single-character regions (interjections like 啊, 嗯, etc.)
  // These are too short to simplify and look cluttered when rendered in a text box
  // Leave them completely untouched in the original image (no inpainting, no rendering)
  // Skip for English input — single English words are valid
  if (options.inputLanguage !== 'english') {
  const beforeSingleCharFilter = extraction.regions.length;
  extraction.regions = extraction.regions.filter(region => {
    if (region.text.trim().length <= 1) {
      console.error(`  Filtering single character: "${region.text}"`);
      return false;
    }
    return true;
  });
  if (extraction.regions.length < beforeSingleCharFilter) {
    console.error(`  Single-char filter: removed ${beforeSingleCharFilter - extraction.regions.length} regions`);
  }
  }

  if (extraction.regions.length === 0) {
    console.error('  No text found on this page');

    // Still save the original image so it appears in HTML output
    let originalImagePath: string | undefined;
    if (options.outputDir) {
      const baseName = path.basename(imagePath, path.extname(imagePath));

      const originalDir = path.join(options.outputDir, 'original');
      if (!fs.existsSync(originalDir)) fs.mkdirSync(originalDir, { recursive: true });

      originalImagePath = path.join(originalDir, `${baseName}${pickImageExtension(pageImage.width, pageImage.height)}`);
      // Fire-and-forget: don't block the per-page critical path on disk I/O.
      // The save is tracked and checked before the reader is generated.
      const _saveStart = Date.now();
      trackImageSave(saveImage(pageImage.buffer, originalImagePath)
        .then(() => { globalTimingStats.saveImage += Date.now() - _saveStart; }), originalImagePath);
    }

    return {
      pageNumber: pageImage.pageNumber,
      originalText: '',
      simplifiedText: '',
      regions: [],
      simplifiedRegions: [],
      originalImagePath,
      imageWidth: pageImage.width,
      imageHeight: pageImage.height,
    };
  }

  console.error(`  Found ${extraction.regions.length} text regions`);

  // Check for leading punctuation missed by OCR (e.g. "..." in manwha inner monologue)
  // Detects gray solid fill regions and asks Gemini if there's punctuation before the text
  // Enabled by ENABLE_GEMINI_VERIFY env var (default: disabled for speed)
  const grayCorrectedRegionIndices = new Set<number>();
  if (process.env.ENABLE_GEMINI_VERIFY === 'true' && extraction.regions.length > 0) {
    try {
      const metadata = await sharp(pageImage.buffer).metadata();
      const imgWidth = metadata.width || 1;
      const imgHeight = metadata.height || 1;

      // Check each region for suspicious gray background
      const grayChecks: Array<{ regionIndex: number; region: TextRegion }> = [];

      for (let rIdx = 0; rIdx < extraction.regions.length; rIdx++) {
        const region = extraction.regions[rIdx];

        // Sample color from expanded area in reading direction
        const expandedBox = region.isVertical
          ? { x: region.boundingBox.x, y: Math.max(0, region.boundingBox.y - 20),
              width: region.boundingBox.width, height: region.boundingBox.height + 20 }
          : { x: Math.max(0, region.boundingBox.x - 20), y: region.boundingBox.y,
              width: region.boundingBox.width + 20, height: region.boundingBox.height };

        const pixelX = Math.round((expandedBox.x / 100) * imgWidth);
        const pixelY = Math.round((expandedBox.y / 100) * imgHeight);
        const pixelW = Math.round((expandedBox.width / 100) * imgWidth);
        const pixelH = Math.round((expandedBox.height / 100) * imgHeight);

        const left = Math.max(0, pixelX);
        const top = Math.max(0, pixelY);
        const width = Math.min(pixelW, imgWidth - left);
        const height = Math.min(pixelH, imgHeight - top);

        if (width > 0 && height > 0) {
          const { data, info } = await sharp(pageImage.buffer)
            .extract({ left, top, width, height })
            .raw()
            .toBuffer({ resolveWithObject: true });

          let totalR = 0, totalG = 0, totalB = 0;
          const step = Math.max(1, Math.floor(data.length / info.channels / 100));
          let samples = 0;
          for (let i = 0; i < data.length; i += info.channels * step) {
            totalR += data[i];
            totalG += data[i + 1];
            totalB += data[i + 2];
            samples++;
          }
          const sampledColor = {
            r: Math.round(totalR / samples),
            g: Math.round(totalG / samples),
            b: Math.round(totalB / samples)
          };

          const grayResult = await detectSuspiciousGray(sampledColor, pageImage.buffer, imgWidth, imgHeight);
          if (grayResult.isGray) {
            grayChecks.push({ regionIndex: rIdx, region });
          }
        }
      }

      // For each gray region, check for leading punctuation with Gemini
      if (grayChecks.length > 0) {
        console.error(`  [GeminiVerify] Found ${grayChecks.length} suspicious gray regions, checking for leading punctuation...`);
        const verifyStartTime = Date.now();

        for (const { regionIndex, region } of grayChecks) {
          const result = await checkLeadingPunctuation(
            pageImage.buffer, imgWidth, imgHeight, region, regionIndex
          );

          if (result.wasModified) {
            extraction.regions[regionIndex] = {
              ...extraction.regions[regionIndex],
              text: result.foundPunctuation + extraction.regions[regionIndex].text,
              boundingBox: result.expandedBoundingBox
            };
            grayCorrectedRegionIndices.add(regionIndex);
            console.error(`  [GeminiVerify] Prepended "${result.foundPunctuation}" → "${extraction.regions[regionIndex].text}"`);
          }
        }

        const verifyTime = Date.now() - verifyStartTime;
        console.error(`  [TIMING] Punctuation check: ${(verifyTime / 1000).toFixed(2)}s`);
      }
    } catch (error: any) {
      console.error(`  [GeminiVerify] Error: ${error.message}`);
    }
  }

  // If analyze only, just return the extraction
  if (options.analyzeOnly) {
    return {
      pageNumber: pageImage.pageNumber,
      originalText: extraction.fullText,
      simplifiedText: extraction.fullText,
      regions: extraction.regions,
      simplifiedRegions: [],
      imageWidth: pageImage.width,
      imageHeight: pageImage.height,
    };
  }

  // Simplify text regions using block-based approach for context
  console.error('  Simplifying text...');
  const llmStartTime = Date.now();
  const simplifiedRegions: ReplacementRegion[] = [];

  // [CLUSTER_FILTER — gated] Pre-merge garbage-band removal. Group the raw regions by LOOSE
  // vertical adjacency (gap/height < 0.7 + horizontal overlap); a band whose combined text is
  // >= MINLEN chars is a UI list / comment wall / status window. Dialogue never chains this
  // far — separate bubbles live in separate columns, so the horizontal-overlap requirement
  // stops them linking (measured region-level ceiling ~43 chars across merged Tower-of-God
  // pages). Hand each band to the LLM, keep the regions it marks story, drop the rest to
  // original art. Running PRE-merge is essential: it removes the UI before Pass 1/2 can fuse a
  // UI cell into a bubble line, so surviving bubble regions merge cleanly. Cheap geometry
  // gates a rare LLM call; if it fails/off, the band just renders (no bug). Off unless
  // CLUSTER_FILTER=true.
  // CLUSTER_FILTER is a Chinese-tuned experiment (off in prod) — scope it to
  // Chinese only rather than letting it fire on JA/KO with untested thresholds.
  if (process.env.CLUSTER_FILTER === 'true' && (options.inputLanguage || 'chinese') === 'chinese' && extraction.regions.length > 2) {
    const Rg = extraction.regions;
    const LOOSE = 0.7, HOVL = 0.5, MINLEN = 50, M = Rg.length;
    const adj = (i: number, j: number) => {
      const a = Rg[i].boundingBox, b = Rg[j].boundingBox;
      const ov = Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x));
      const mw = Math.min(a.width, b.width);
      if (mw <= 0 || ov / mw <= HOVL) return false;                    // need horizontal overlap
      const up = a.y <= b.y ? a : b, lo2 = a.y <= b.y ? b : a;
      const mh = Math.min(a.height, b.height);
      return mh > 0 && (lo2.y - (up.y + up.height)) / mh < LOOSE;      // loose vertical adjacency
    };
    const parent = Array.from({ length: M }, (_, i) => i);
    const find = (x: number): number => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
    for (let i = 0; i < M; i++) for (let j = i + 1; j < M; j++) if (adj(i, j)) parent[find(i)] = find(j);
    const comps: Record<number, number[]> = {};
    Rg.forEach((_, i) => { const r = find(i); (comps[r] = comps[r] || []).push(i); });
    const cleanLen = (s: string) => (s || '').replace(/\s/g, '').length;
    const drop = new Array(M).fill(false);
    for (const idxs of Object.values(comps)) {
      const totalLen = idxs.reduce((s, i) => s + cleanLen(Rg[i].text), 0);
      if (totalLen < MINLEN) continue;                                 // not a garbage band
      const ord = idxs.slice().sort((a, b) => Rg[a].boundingBox.y - Rg[b].boundingBox.y);
      const keep = await identifyStoryBlocks(ord.map(i => Rg[i].text || ''));
      ord.forEach((ri, k) => { if (!keep.has(k)) drop[ri] = true; });
      console.error(`  [ClusterFilter] band ${idxs.length} regions / ${totalLen} chars: kept ${keep.size}, dropped ${idxs.length - keep.size}`);
      if (process.env.CLUSTER_DEBUG === 'true') ord.forEach((ri, k) => console.error(`      [${keep.has(k) ? 'KEEP' : 'DROP'}] ${(Rg[ri].text || '').slice(0, 40)}`));
    }
    const before = M;
    extraction.regions = Rg.filter((_, i) => !drop[i]);
    if (extraction.regions.length < before) console.error(`  [ClusterFilter] total dropped ${before - extraction.regions.length}/${before} regions`);
  }

  // No regions are pre-skipped — preserveAsIs regions (adjacent English) now participate
  // in block grouping so mixed Chinese+English lines merge and render correctly.
  // The simplifier handles English text within blocks (preserves it verbatim).
  const skipRegionIndices = new Set<number>();

  // Filter regions for block grouping (includes short text so Pass 2 can merge them)
  const regionsToSimplify = extraction.regions.filter((_, i) => !skipRegionIndices.has(i));

  // Build mapping from regionsToSimplify index → original extraction.regions index
  // This must be done BEFORE the short text filter modifies skipRegionIndices,
  // because block.regionIndices are indices into regionsToSimplify.
  const simplifyToOriginalIdx: number[] = [];
  extraction.regions.forEach((_, i) => {
    if (!skipRegionIndices.has(i)) {
      simplifyToOriginalIdx.push(i);
    }
  });

  // Group regions into blocks for context-aware simplification.
  // outDroppedByOverlap collects regionsToSimplify indices that Pass 4
  // (rotated-overlap filter) drops; we add the corresponding original region
  // indices to chromaExcludedIndices below so neither inpaint nor render
  // touches them — area is left as original artwork.
  const droppedByOverlap = new Set<number>();
  const blocks = groupRegionsIntoBlocks(regionsToSimplify, adjacencyDistance, options.inputLanguage || 'chinese', imageAspectRatio, droppedByOverlap);

  // Now apply short text filter: skip short regions that are still standalone after grouping
  const mergedRegionIndices = new Set<number>();
  blocks.forEach(block => {
    if (block.regionIndices.length > 1) {
      block.regionIndices.forEach(ri => mergedRegionIndices.add(ri));
    }
  });
  regionsToSimplify.forEach((region, simplifyIdx) => {
    if (options.inputLanguage !== 'english' && region.text.length <= 2 && !mergedRegionIndices.has(simplifyIdx)) {
      const origIdx = simplifyToOriginalIdx[simplifyIdx];
      skipRegionIndices.add(origIdx);
    }
  });

  // Deferred chroma filter: now that blocks are formed, filter colorful regions
  // that are standalone with few Chinese chars (art overlays / SFX).
  // Multi-line colored UI boxes (system notifications) survive by merging into blocks first.
  // Use chromaExcludedIndices (not skipRegionIndices) so these regions are fully excluded
  // from inpainting, not just kept with original text.
  const chromaExcludedIndices = new Set<number>();
  // Seed exclusion list with any regions dropped by Pass 4 (rotated overlap).
  // These should never be inpainted or rendered — leave original artwork.
  for (const idx of droppedByOverlap) {
    const origIdx = simplifyToOriginalIdx[idx];
    if (origIdx !== undefined) chromaExcludedIndices.add(origIdx);
  }
  const chromaExcludedBlocks = new Set<number>();
  blocks.forEach((block, blockIdx) => {
    // Count total Chinese characters in the block
    let blockChineseChars = 0;
    for (const text of block.texts) {
      for (const char of text) {
        const code = char.charCodeAt(0);
        if ((code >= 0x4e00 && code <= 0x9fff) || (code >= 0x3400 && code <= 0x4dbf)) {
          blockChineseChars++;
        }
      }
    }

    // Check if any regions in the block are colorful
    const colorfulIndices = block.regionIndices.filter(ri => regionsToSimplify[ri]?.isColorful);
    if (colorfulIndices.length === 0) return;

    // Mixed colorful/non-colorful block = styled dialogue, keep it
    if (colorfulIndices.length < block.regionIndices.length) {
      console.error(`  [Chroma] Keeping mixed colorful/non-colorful block #${blockIdx}: "${block.texts.join(' | ')}"`);
      return;
    }

    // Chroma drop DISABLED — was dropping legitimate stylized UI text (level-up
    // boxes, name cards, system messages) in gamer-manhwa categories. Other
    // filters (alignment, size, overlap) still catch most SFX shapes.
    let blockMaxChroma = 0;
    for (const ri of colorfulIndices) {
      const chroma = regionsToSimplify[ri]?.avgChroma || 0;
      if (chroma > blockMaxChroma) blockMaxChroma = chroma;
    }
    console.error(`  [Chroma] Keeping colorful block #${blockIdx} (filter disabled, chroma=${blockMaxChroma.toFixed(0)}): "${block.texts.join(' | ')}"`);
  });

  // English mode: filter blocks that are only numbers (OCR hallucinations or page numbers)
  if (options.inputLanguage === 'english') {
    blocks.forEach((block, blockIdx) => {
      if (chromaExcludedBlocks.has(blockIdx)) return;
      const combined = block.texts.join(' ').trim();
      if (/^[\d\s.,-]+$/.test(combined)) {
        console.error(`  [English] Filtering numbers-only block: "${combined}"`);
        chromaExcludedBlocks.add(blockIdx);
        block.regionIndices.forEach(ri => {
          chromaExcludedIndices.add(simplifyToOriginalIdx[ri]);
        });
      }
    });
  }

  // English mode: fully exclude standalone single-word blocks with no punctuation.
  // These are garbled OCR (GMLP, DIMG, EHARF) or game UI/SFX baked into artwork.
  // Legitimate standalone dialogue always has punctuation (NEXT!, NO!!, H-HUH?).
  // Only . , ! ? and a TRAILING dash count as punctuation. The trailing dash is
  // comics' interruption mark ("WAIT-", "THEN-") — real dialogue. Accepted
  // leak: dash-suffixed SFX ("B22-" ← mangled "BZZ-") passes too; dropped
  // dialogue is the worse failure. Mid-word hyphens/apostrophes still don't
  // protect ("X-4", "GMLP").
  if (options.inputLanguage === 'english') {
    blocks.forEach((block, blockIdx) => {
      if (chromaExcludedBlocks.has(blockIdx)) return;
      if (block.regionIndices.length !== 1) return;
      const text = block.texts[0].trim();
      const stripped = text.replace(/[^a-zA-Z]/g, '');
      if (stripped.length === 0) return;
      const hasSpace = /\s/.test(text);
      const hasPunctuation = /[.!?,]/.test(text) || /[-–—]$/.test(text);
      if (!hasSpace && !hasPunctuation) {
        console.error(`  [English] Filtering standalone single-word without punctuation: "${text}"`);
        chromaExcludedBlocks.add(blockIdx);
        block.regionIndices.forEach(ri => {
          chromaExcludedIndices.add(simplifyToOriginalIdx[ri]);
        });
      }
    });
  }

  // Alignment-based UI filter. Multi-region blocks where regions all start at the
  // same left edge (low left-edge variance) but have varied centers (high center
  // variance) are paragraph-style text — review cards, forum posts, in-comic
  // platform UI mockups. Speech bubbles are the opposite: text is centered, so
  // left edges vary but centers cluster.
  //
  // Threshold 0.05 (configurable). Measured separation in a sample chapter:
  // max filtered ar = 0.019, min kept dialogue ar = 0.132 (gap ~7x, with an
  // empty band between 0.019 and 0.132). 0.05 sits in that empty band and
  // gives the closest real dialogue case (~0.132) a 2.6x safety margin while
  // still catching every currently-flagged UI block. Asymmetry matters:
  // wrongly filtering dialogue is worse than missing UI, so err toward keep.
  // Chinese dialogue bubbles legitimately have precise left-alignment with
  // varying line widths, which the alignment filter mistakes for UI blocks.
  // The same-width override doesn't catch them because real dialogue has
  // mixed-length lines. Skip entirely for Chinese; accept the rare risk of
  // a UI mockup inside a panel getting simplified.
  if (!isEnglishInputSingle) {
    console.error(`  [Alignment] Skipping alignment filter for Chinese input`);
  } else {
    const alignmentThreshold = parseFloat(process.env.UI_ALIGNMENT_THRESHOLD || '0.05');
    const alignmentSameSizeRatio = parseFloat(process.env.UI_ALIGNMENT_SAMESIZE_RATIO || '0.90');
    blocks.forEach((block, blockIdx) => {
      if (chromaExcludedBlocks.has(blockIdx)) return;
      if (block.regionIndices.length < 2) return;
      const blockRegions = block.regionIndices.map(ri => regionsToSimplify[ri]).filter(Boolean);
      if (blockRegions.length < 2) return;
      // Same-width override: wrapped speech-bubble lines tend to have widths that
      // closely match (within 10% by default). Skip the alignment drop for these
      // since they're almost always dialogue, not UI menus.
      const widths = blockRegions.map(r => r.boundingBox.width);
      const maxW = Math.max(...widths);
      const minW = Math.min(...widths);
      if (maxW > 0 && minW / maxW >= alignmentSameSizeRatio) return;
      const lefts = blockRegions.map(r => r.boundingBox.x);
      const centers = blockRegions.map(r => r.boundingBox.x + r.boundingBox.width / 2);
      const meanLeft = lefts.reduce((s, x) => s + x, 0) / lefts.length;
      const meanCenter = centers.reduce((s, x) => s + x, 0) / centers.length;
      const leftVar = lefts.reduce((s, x) => s + (x - meanLeft) ** 2, 0) / lefts.length;
      const centerVar = centers.reduce((s, x) => s + (x - meanCenter) ** 2, 0) / centers.length;
      // Perfectly aligned centers (rare, but happens when text is monospaced) → dialogue, keep
      if (centerVar === 0) return;
      const ratio = leftVar / centerVar;
      if (ratio < alignmentThreshold) {
        console.error(`  [Alignment] Filtering left-aligned UI block (ratio=${ratio.toFixed(3)}): "${block.texts.join(' | ').slice(0, 80)}"`);
        chromaExcludedBlocks.add(blockIdx);
        block.regionIndices.forEach(ri => {
          chromaExcludedIndices.add(simplifyToOriginalIdx[ri]);
        });
      }
    });
  }

  // Filter out chroma-excluded blocks + non-dialogue long blocks (CN-only)
  const longBlocksToDrop = identifyLongNonDialogueBlocks(blocks, options.inputLanguage || 'chinese');
  // A long non-dialogue block is dropped ENTIRELY, not merely skipped for
  // simplification. Fold it into the chroma-exclusion sets so both the render
  // loop (skips chromaExcludedBlocks) and the inpaint loop (skips
  // chromaExcludedIndices) leave it untouched. Without this, the block falls
  // back to smartJoin(original) and gets re-rendered with its original,
  // unsimplified text — the bug that let OCR'd comment sections survive.
  longBlocksToDrop.forEach((blockIdx) => {
    chromaExcludedBlocks.add(blockIdx);
    blocks[blockIdx].regionIndices.forEach((ri) => {
      const origIdx = simplifyToOriginalIdx[ri];
      if (origIdx !== undefined) chromaExcludedIndices.add(origIdx);
    });
  });

  // CN/JA/KO input: any block with zero source-content chars is incidental — a game
  // title, UI label, SFX, sign, or watermark. It's never content to translate
  // in CN→CN, so drop it (same fold as longBlocksToDrop → skipped by both the
  // inpaint and render loops, leaving the original pixels untouched). English
  // that belongs to a Chinese line already merged INTO that line's block during
  // grouping, so it survives via the has-Chinese test here.
  if (options.inputLanguage !== 'english') {
    blocks.forEach((block, blockIdx) => {
      if (chromaExcludedBlocks.has(blockIdx)) return;
      // Misdetection-guard tally. Pages may pass through here more than once
      // (inpaint/LLM/render passes each regroup) — the drop RATIO is
      // pass-invariant, which is all the guard reads.
      if (options._langStats) options._langStats.totalBlocks++;
      // Language-aware content test: Han for CN, Han+kana for JA, Hangul+Han
      // for KO. A pure-kana or pure-hangul bubble is real dialogue, not
      // incidental English — counting Han only would silently erase it.
      const hasSourceContent = block.texts.some(t => countSourceContentChars(t, options.inputLanguage || 'chinese') > 0);
      if (!hasSourceContent) {
        if (options._langStats) options._langStats.allEnglishDropped++;
        console.error(`  [NonChinese] Dropping no-source-content block #${blockIdx}: "${block.texts.join(' | ').slice(0, 60)}"`);
        chromaExcludedBlocks.add(blockIdx);
        block.regionIndices.forEach((ri) => {
          const origIdx = simplifyToOriginalIdx[ri];
          if (origIdx !== undefined) chromaExcludedIndices.add(origIdx);
        });
      }
    });
  }
  // [OVERLAP_DROP] Change 2 — leave heavily-overlapping homogeneous clusters as
  // original art (folded into chromaExcludedBlocks, so neither inpainted nor
  // re-typeset). Runs post-merge on the corrected oriented boxes. Off unless
  // OVERLAP_DROP=true.
  if (process.env.OVERLAP_DROP === 'true') {
    const overlapDrop = overlappingClustersToDrop(blocks, regionsToSimplify, pageImage.width, pageImage.height,
      (i) => chromaExcludedBlocks.has(i) || longBlocksToDrop.has(i));
    overlapDrop.forEach((blockIdx) => {
      chromaExcludedBlocks.add(blockIdx);
      blocks[blockIdx].regionIndices.forEach((ri) => {
        const origIdx = simplifyToOriginalIdx[ri];
        if (origIdx !== undefined) chromaExcludedIndices.add(origIdx);
      });
    });
  }
  const blocksToSimplify = blocks.filter((_, idx) => !chromaExcludedBlocks.has(idx) && !longBlocksToDrop.has(idx));
  // Map from blocksToSimplify index back to original blocks index
  const simplifyBlockToOrigBlock: number[] = [];
  blocks.forEach((_, idx) => {
    if (!chromaExcludedBlocks.has(idx) && !longBlocksToDrop.has(idx)) simplifyBlockToOrigBlock.push(idx);
  });

  // Simplify blocks
  const useMultiLevel = (options.targetLevels && options.targetLevels.length > 1) || options.includeEnglish;
  let blockResults = new Map<number, string>();  // keyed by original block index
  let multiLevelBlockResults: Map<number, Map<string, string>> | undefined;
  if (blocksToSimplify.length > 0) {
    try {
      if (useMultiLevel && options.targetLevels) {
        const rawResults = await simplifyBlocksMultiLevel(
          blocksToSimplify, options.targetLevels, options.includeEnglish || false,
          undefined, undefined, options.inputLanguage
        );
        multiLevelBlockResults = new Map();
        const primaryLevel = options.targetLevel;
        rawResults.forEach((levelMap, simplifyIdx) => {
          const origIdx = simplifyBlockToOrigBlock[simplifyIdx];
          multiLevelBlockResults!.set(origIdx, levelMap);
          blockResults.set(origIdx, levelMap.get(primaryLevel) || blocks[origIdx].texts.join(''));
        });
      } else {
        const rawResults = await simplifyBlocks(blocksToSimplify, options.targetLevel, undefined, undefined, options.inputLanguage);
        rawResults.forEach((text, simplifyIdx) => {
          const origIdx = simplifyBlockToOrigBlock[simplifyIdx];
          blockResults.set(origIdx, text);
        });
      }
    } catch (e) {
      console.error('  Block simplification failed, using originals');
    }
  }
  const llmTime = Date.now() - llmStartTime;
  console.error(`  [TIMING] LLM: ${(llmTime / 1000).toFixed(2)}s (${regionsToSimplify.length} texts in ${blocksToSimplify.length} blocks)`);
  globalTimingStats.llm += llmTime;

  // Log simplifier output for each block
  blockResults.forEach((text, blockIdx) => {
    const original = blocks[blockIdx]?.texts.join(' | ');
    console.error(`  [Simplifier] Block ${blockIdx}: "${original}" → "${text}"`);
  });

  // Build maps for block-based rendering: original region index → blockId and blockText
  const blockIdMap = new Map<number, number>();
  const blockTextMap = new Map<number, string>();
  const originalBlockTextMap = new Map<number, string>();

  blocks.forEach((block, blockIdx) => {
    if (chromaExcludedBlocks.has(blockIdx)) return;
    const blockText = blockResults.get(blockIdx) || smartJoin(block.texts);
    blockTextMap.set(blockIdx, blockText);
    originalBlockTextMap.set(blockIdx, smartJoin(block.texts));
    block.regionIndices.forEach((regIdx) => {
      const originalIdx = simplifyToOriginalIdx[regIdx];
      blockIdMap.set(originalIdx, blockIdx);
    });
  });

  // Reconstruct regions in original order with block data for rendering
  for (let i = 0; i < extraction.regions.length; i++) {
    const region = extraction.regions[i];

    // Chroma-excluded regions are fully skipped — no inpainting, no rendering
    if (chromaExcludedIndices.has(i)) continue;

    // Detect top-edge regions (y < 3%) - these are likely continuations from previous page
    const isTopEdge = region.boundingBox.y < 3;

    // Force white fill for gray-corrected regions
    const forceFill = grayCorrectedRegionIndices.has(i) ? { forceFillColor: { r: 255, g: 255, b: 255 } } : {};

    if (skipRegionIndices.has(i)) {
      // Short text - keep original, no block
      simplifiedRegions.push({
        original: region,
        simplifiedText: region.text,
        isCrossPageContinuation: isTopEdge,
        ...forceFill,
      });
    } else {
      // Assign block data for block-based rendering
      const blockIdx = blockIdMap.get(i);
      const blockText = blockIdx !== undefined ? blockTextMap.get(blockIdx) : undefined;
      const originalBlockText = blockIdx !== undefined ? originalBlockTextMap.get(blockIdx) : undefined;
      simplifiedRegions.push({
        original: region,
        simplifiedText: region.text,
        isCrossPageContinuation: isTopEdge,
        blockId: blockIdx,
        blockText: blockText,
        originalBlockText: originalBlockText,
        ...forceFill,
      });
    }
  }

  // Replace text in image (inpainting + metadata only, no text rendering)
  console.error('  Replacing text in image...');
  let cleanedBuffer: Buffer = pageImage.buffer;
  let renderedBlocks: RenderedBlockInfo[] = [];
  try {
    const replaceResult = await replaceTextInImage(pageImage.buffer, simplifiedRegions, { inputLanguage: options.inputLanguage }, undefined, sharedPageData);
    cleanedBuffer = replaceResult.cleanedImageBuffer;
    renderedBlocks = replaceResult.renderedBlocks;
    globalTimingStats.inpaintServer += replaceResult.timing.inpaintServer;
    globalTimingStats.textRendering += replaceResult.timing.textRendering;
    globalTimingStats.colorSampling += replaceResult.timing.colorSampling;
    globalTimingStats.metaBuild += replaceResult.timing.metaBuild;
    globalTimingStats.decodePage += replaceResult.timing.decodePage;
    console.error(`  [TIMING] Inpaint server: ${(replaceResult.timing.inpaintServer / 1000).toFixed(2)}s, Metadata: ${(replaceResult.timing.textRendering / 1000).toFixed(2)}s`);
  } catch (err) {
    console.error(`  Warning: Text replacement failed, using original image: ${err}`);
  }

  // Save output
  let originalImagePath: string | undefined;
  let cleanedImagePath: string | undefined;
  if (options.outputDir) {
    const baseName = path.basename(imagePath, path.extname(imagePath));

    const originalDir = path.join(options.outputDir, 'original');
    const cleanedDir = path.join(options.outputDir, 'cleaned');
    if (!fs.existsSync(originalDir)) fs.mkdirSync(originalDir, { recursive: true });
    if (!fs.existsSync(cleanedDir)) fs.mkdirSync(cleanedDir, { recursive: true });

    const imageExt = pickImageExtension(pageImage.width, pageImage.height);
    originalImagePath = path.join(originalDir, `${baseName}${imageExt}`);
    cleanedImagePath = path.join(cleanedDir, `${baseName}${imageExt}`);
    // Fire-and-forget: launch both saves, don't await. Disk I/O happens in
    // parallel with the next page's CPU work. Both are tracked and checked
    // before the reader is generated.
    const _saveStart1 = Date.now();
    trackImageSave(saveImage(pageImage.buffer, originalImagePath)
      .then(() => { globalTimingStats.saveImage += Date.now() - _saveStart1; }), originalImagePath);
    const _saveStart2 = Date.now();
    trackImageSave(saveImage(cleanedBuffer, cleanedImagePath)
      .then(() => { globalTimingStats.saveImage += Date.now() - _saveStart2; }), cleanedImagePath);
  }

  const simplifiedFullText = simplifiedRegions.map((r) => r.simplifiedText).join('\n');

  // Track page timing
  const pageTotalTime = Date.now() - pageStartTime;
  globalTimingStats.total += pageTotalTime;
  processedPageCount++;
  console.error(`  [TIMING] Page total: ${(pageTotalTime / 1000).toFixed(2)}s`);

  return {
    pageNumber: pageImage.pageNumber,
    originalText: extraction.fullText,
    simplifiedText: simplifiedFullText,
    regions: extraction.regions,
    simplifiedRegions,
    originalImagePath,
    cleanedImagePath,
    imageWidth: pageImage.width,
    imageHeight: pageImage.height,
    renderedBlocks,
    multiLevelBlockResults,
    blocksToSimplify,
    simplifyBlockToOrigBlock,
  };
}

/**
 * Process a single manga page from a buffer (used for merged pages)
 */
export async function processMangaPageFromBuffer(
  imageBuffer: Buffer,
  pageNumber: number,
  originalPageIndices: number[], // Which original pages this represents
  options: MangaProcessingOptions,
  originalPageCount: number = 1, // How many original pages were merged (for filter scaling)
  stitchYPositions: number[] = [], // Y positions (as %) where pages were stitched
  cachedExtraction?: ExtractionResult // Reuse OCR from chapter analysis scan
): Promise<PageProcessingResult> {
  const pageStartTime = Date.now();
  const isMerged = originalPageIndices.length > 1;
  const pageLabel = isMerged
    ? `Merged pages ${originalPageIndices.map(i => i + 1).join('+')}`
    : `Page ${pageNumber}`;
  console.error(`Processing: ${pageLabel}`);

  // Log buffer size for merged pages to help debug OCR issues
  if (isMerged) {
    console.error(`  [Merged] Buffer size: ${(imageBuffer.length / 1024).toFixed(0)}KB, mergeRatio=${originalPageCount}`);
  }

  // Create a PageImage-like object
  const pageImage = {
    buffer: imageBuffer,
    pageNumber,
    width: 0,
    height: 0,
  };

  // Get dimensions
  const metadata = await sharp(imageBuffer).metadata();
  pageImage.width = metadata.width || 0;
  pageImage.height = metadata.height || 0;

  // Extract text from image (OCR) — reuse cached extraction from chapter analysis if available
  console.error('  Extracting text...');
  const ocrStartTime = Date.now();
  let extraction: ExtractionResult;
  if (cachedExtraction) {
    // Deep-clone so this phase's filter mutations don't leak back into the shared cache.
    // Three phases (4a scan, 4b inpaint, 4c render) each call this function per unit;
    // without cloning, each phase re-runs the structureless filter on the previous phase's
    // mutated state, causing block indices to drift between LLM input (4a) and render (4c).
    extraction = structuredClone(cachedExtraction);
    console.error(`  Using cached OCR (${extraction.regions.length} regions)`);
  } else {
    extraction = await extractTextFromImage(
      pageImage.buffer,
      pageImage.pageNumber,
      options.visionProvider || 'paddleocr',
      options.inputLanguage
    );
  }
  const ocrTime = Date.now() - ocrStartTime;
  console.error(`  [TIMING] OCR: ${(ocrTime / 1000).toFixed(2)}s`);
  globalTimingStats.ocr += ocrTime;

  // Log OCR results for merged pages to help debug
  if (isMerged) {
    console.error(`  [Merged] OCR returned ${extraction.regions.length} regions for ${pageImage.width}x${pageImage.height} image`);
    if (extraction.regions.length === 0) {
      console.error(`  [Merged] WARNING: No text detected on merged image! This may indicate PaddleOCR size limits.`);
    }

    // Log stitch positions
    if (stitchYPositions.length > 0) {
      console.error(`  [Merged] Stitch boundaries at: ${stitchYPositions.map(y => y.toFixed(1) + '%').join(', ')}`);
    }

    // Log ALL detected regions with their positions
    console.error(`  [Merged] === ALL OCR REGIONS ===`);
    extraction.regions.forEach((r, i) => {
      const centerY = r.boundingBox.y + r.boundingBox.height / 2;
      // Check if near any stitch boundary (within 5%)
      const nearStitch = stitchYPositions.some(sy => Math.abs(centerY - sy) < 5);
      const stitchMarker = nearStitch ? ' [NEAR STITCH]' : '';
      console.error(`    ${i}: y=${r.boundingBox.y.toFixed(1)}%-${(r.boundingBox.y + r.boundingBox.height).toFixed(1)}%, ` +
        `h=${r.boundingBox.height.toFixed(2)}%, rot=${(r.rotation || 0).toFixed(1)}°, ` +
        `text="${r.text.substring(0, 25)}..."${stitchMarker}`);
    });
    console.error(`  [Merged] === END OCR REGIONS ===`);
  }

  // Convert traditional to simplified if needed
  if (options.convertTraditional !== false) {
    extraction = convertExtractionToSimplified(extraction);
    extraction = stripUrlTokensFromExtraction(extraction);
  }

  // Apply text filters (same as processMangaPage)
  // Scale height thresholds by aspect ratio (same as single-page filter).
  // For merged images, also divide by mergeRatio since the image is ~Nx taller.
  const mergeRatio = originalPageCount;
  const mergedAspectRatio = pageImage.height / pageImage.width;
  const baselineAspect = 2.6;  // 690x1800 baseline where 1.2% was tuned
  const scaledMaxNonChineseHeight = (12 * (baselineAspect / mergedAspectRatio));

  // Helper: Calculate rotation threshold based on character count, bounding box shape, and height
  // Short text (1-2 chars) with square bounding boxes get unreliable rotation from PaddleOCR
  // But large short text (>5% height) is likely a sound effect - use strict threshold
  // For English, divide letter count by 5 (avg word length) to get equivalent char count
  const isEnglishInput = options.inputLanguage === 'english';

  // Helper to check if a region is near a stitch boundary
  const isNearStitch = (region: TextRegion): boolean => {
    const centerY = region.boundingBox.y + region.boundingBox.height / 2;
    return stitchYPositions.some(sy => Math.abs(centerY - sy) < 5);
  };

  // Track filter results for each region (for merged page diagnostics)
  interface FilterResult {
    region: TextRegion;
    passed: boolean;
    failedFilter?: string;
    failReason?: string;
    nearStitch: boolean;
  }
  const filterResults: FilterResult[] = [];

  const originalCount = extraction.regions.length;


  // Pre-compute text color saturation for all regions. Decode the page ONCE
  // and KEEP it alive for reuse in replaceTextInImage below.
  const saturationMap = new Map<number, { isColorful: boolean; avgChroma: number; textPixelCount: number; hasSolidBackground: boolean; bgStdDev: number }>();
  const _satStart2 = Date.now();
  let sharedPageData: { data: Buffer; width: number; height: number; channels: number } | undefined;
  try {
    const { data: satPageData, info: satPageInfo } = await sharp(pageImage.buffer).raw().toBuffer({ resolveWithObject: true });
    sharedPageData = { data: satPageData, width: satPageInfo.width, height: satPageInfo.height, channels: satPageInfo.channels };
    for (let i = 0; i < extraction.regions.length; i++) {
      try {
        const result = checkTextColorSaturation(satPageData, satPageInfo.width, satPageInfo.height, satPageInfo.channels, extraction.regions[i].boundingBox);
        saturationMap.set(i, result);
      } catch (e) { /* skip */ }
    }
  } catch (e) {
    console.error(`  [Saturation] Page decode failed: ${e}`);
  }
  globalTimingStats.saturation += Date.now() - _satStart2;

  // Helper: count Chinese characters (needed by max height filter below)
  const countChineseCharsMerged = (text: string): number => {
    let count = 0;
    for (const char of text) {
      const code = char.charCodeAt(0);
      if ((code >= 0x4e00 && code <= 0x9fff) || (code >= 0x3400 && code <= 0x4dbf)) {
        count++;
      }
    }
    return count;
  };

  extraction.regions = extraction.regions.filter((region, i) => {
    const text = region.text.trim();
    const nearStitch = isNearStitch(region);

    if (!text) {
      filterResults.push({ region, passed: false, failedFilter: 'empty', failReason: 'Empty text', nearStitch });
      return false;
    }

    // Filter by rotation - threshold varies by character count and bounding box shape
    // PaddleOCR rotation is unreliable for short text with square bounding boxes
    const rotation = Math.abs(region.rotation || 0);
    let rotationThreshold = getRotationThreshold(text, region.boundingBox, isEnglishInput);
    // Large 1-2 char text with punctuation on a solid background is likely dialogue (e.g. "呃！!")
    // Override the strict 3.5° threshold to lenient 17.5°
    if (rotationThreshold === 3.5) {
      const hasPunct = /[！？。，!?,.]/.test(text);
      const solidBg = saturationMap.get(i)?.hasSolidBackground ?? false;
      if (hasPunct && solidBg) {
        rotationThreshold = 17.5;
      }
    }
    if (rotation > rotationThreshold) {
      const charCount = text.replace(/[^\u4e00-\u9fff\u3400-\u4dbf]/g, '').length;
      if (region.confidence >= 0.90 && charCount >= 3) {
        console.error(`  [Rotation] Bypassing filter (${rotation.toFixed(1)}° > ${rotationThreshold}°, conf=${region.confidence.toFixed(2)}, ${charCount} chars): "${text}"`);
      } else {
        console.error(`  Filtering rotated text (${rotation.toFixed(1)}° > ${rotationThreshold}°): "${text}"`);
        filterResults.push({ region, passed: false, failedFilter: 'rotation', failReason: `${rotation.toFixed(1)}° > ${rotationThreshold}°`, nearStitch });
        return false;
      }
    }

    // Min height filter is deferred until after non-height filters run (adaptive baseline by density)
    // Max height filter still runs here
    const heightPct = region.boundingBox.height;

    // Filter text that's too tall (title cards, game names, stylized text, SFX on art)
    if (heightPct > scaledMaxNonChineseHeight) {
      console.error(`  Filtering oversized text (h=${heightPct.toFixed(1)}% > ${scaledMaxNonChineseHeight.toFixed(1)}%): "${text}"`);
      filterResults.push({ region, passed: false, failedFilter: 'max_height', failReason: `${heightPct.toFixed(2)}% > ${scaledMaxNonChineseHeight.toFixed(2)}%`, nearStitch });
      return false;
    }

    // Filter emoji-only text (OCR artifacts from decorative elements)
    const strippedEmoji = text.replace(/[\p{Emoji_Presentation}\p{Extended_Pictographic}]/gu, '').trim();
    if (strippedEmoji.length === 0 && text.length > 0) {
      console.error(`  Filtering emoji: "${text}"`);
      filterResults.push({ region, passed: false, failedFilter: 'emoji', failReason: 'Emoji-only text', nearStitch });
      return false;
    }

    // Filter "111", "1111" etc. — OCR misreads "!!!" exclamation bubbles as repeated 1s
    if (/^1{2,}$/.test(text)) {
      console.error(`  Filtering misread exclamation marks: "${text}"`);
      filterResults.push({ region, passed: false, failedFilter: 'misread_exclamation', failReason: 'Repeated 1s (misread !!!)', nearStitch });
      return false;
    }


    // Log saturation info; chroma filtering is deferred until after block grouping
    // so that multi-line colored UI boxes (e.g. system notifications) can merge first
    const satResult = saturationMap.get(i);
    if (satResult) {
      console.error(`  [Saturation] "${text}" → chroma=${satResult.avgChroma.toFixed(0)}, textPixels=${satResult.textPixelCount}, colorful=${satResult.isColorful}`);
      if (satResult.isColorful) {
        region.isColorful = true;
        region.avgChroma = satResult.avgChroma;
      }
    }

    filterResults.push({ region, passed: true, nearStitch });
    return true;
  });

  if (extraction.regions.length < originalCount) {
    console.error(`  Filtered ${originalCount - extraction.regions.length} regions (rotation/max_height/sound)`);
  }

  // Compute adjacency distance early (needed for comment detection and later filters)
  const beforeNonChineseFilter = extraction.regions.length;
  const baseAdjacencyDistance = parseFloat(process.env.NEIGHBOR_DISTANCE || '15');
  const imageAspectRatio = pageImage.height / pageImage.width;
  const adjacencyDistance = baseAdjacencyDistance / Math.max(1, imageAspectRatio);
  console.error(`  [DEBUG] Image ${pageImage.width}x${pageImage.height} (aspect ${imageAspectRatio.toFixed(1)}:1), adjacency threshold: ${adjacencyDistance.toFixed(1)}%`);

  // Comment section detection via connected component clustering.
  // Find chains of nearby text regions, then check each cluster for comment-section patterns.
  // This handles comment sections spanning page boundaries and avoids false positives
  // on multi-page merges with normal dialogue.
  const normalBaseline = isEnglishInput ? 1.1 : 0.95;
  const commentBaseline = isEnglishInput ? 1.9 : 2.4;
  const commentRegionIndices = new Set<number>(); // indices into extraction.regions

  // Build adjacency graph for clustering. Scale NEIGHBOR_DISTANCE by merge ratio so
  // clustering works consistently: on a 2-page merge, regions that would be 5% apart
  // on a single page are only 2.5% apart in merged coordinates.
  const regions = extraction.regions;
  const mergeRatioForCluster = isMerged ? (mergedAspectRatio / baselineAspect) : 1;
  const clusterDistance = baseAdjacencyDistance * mergeRatioForCluster * 2; // 2x for loose chaining
  const neighbors: number[][] = regions.map(() => []);
  for (let i = 0; i < regions.length; i++) {
    const ci = regions[i].boundingBox.y + regions[i].boundingBox.height / 2;
    const xi = regions[i].boundingBox.x + regions[i].boundingBox.width / 2;
    for (let j = i + 1; j < regions.length; j++) {
      const cj = regions[j].boundingBox.y + regions[j].boundingBox.height / 2;
      const xj = regions[j].boundingBox.x + regions[j].boundingBox.width / 2;
      const dist = Math.sqrt((ci - cj) ** 2 + (xi - xj) ** 2);
      if (dist < clusterDistance) {
        neighbors[i].push(j);
        neighbors[j].push(i);
      }
    }
  }

  // BFS to find connected components
  const visited = new Set<number>();
  const clusters: number[][] = [];
  for (let i = 0; i < regions.length; i++) {
    if (visited.has(i)) continue;
    const cluster: number[] = [];
    const queue = [i];
    visited.add(i);
    while (queue.length > 0) {
      const node = queue.shift()!;
      cluster.push(node);
      for (const nb of neighbors[node]) {
        if (!visited.has(nb)) {
          visited.add(nb);
          queue.push(nb);
        }
      }
    }
    clusters.push(cluster);
  }

  // Check each cluster for comment section pattern
  console.error(`  [Comment detection] ${clusters.length} clusters found (sizes: ${clusters.map(c => c.length).join(', ')}), clusterDistance=${clusterDistance.toFixed(1)}%`);
  for (const cluster of clusters) {
    const mergeCount = stitchYPositions.length + 1; // 1 for single, 2 for 2-page merge, etc.
    const clusterThreshold = 20 + (mergeCount - 1) * 5;
    if (cluster.length < clusterThreshold) continue;
    const clusterRegions = cluster.map(i => regions[i]);
    const prelimBlocks = groupRegionsIntoBlocks(clusterRegions, adjacencyDistance, 'chinese', mergedAspectRatio);
    const nonEmptyBlocks = prelimBlocks.filter(b => b.regionIndices.length > 0);
    const regionsPerBlock = nonEmptyBlocks.length > 0 ? clusterRegions.length / nonEmptyBlocks.length : 999;
    // Chinese-natural ratio is 1.5-2.0 (each char/line is its own OCR region,
    // bubbles don't merge into multi-region blocks). English/Japanese natural
    // ratio is 4+. For Chinese, also require cluster to be in bottom 30% of
    // the merged page — real comment sections only appear at the end of the
    // chapter, not mid-page.
    // Granularity, not glyph shape — EN/JA/KO share the full-line threshold
    // (see the single-page twin of this check for rationale).
    const commentRatioThreshold = isTranslationSource(options.inputLanguage) ? 2.75 : 1.3;
    const clusterCentroidY = clusterRegions.reduce((s, r) => s + r.boundingBox.y + r.boundingBox.height / 2, 0) / clusterRegions.length;
    const clusterYGuardPassed = isEnglishInput || clusterCentroidY > 70;
    console.error(`  [Comment detection] Cluster (${clusterRegions.length} regions, ${nonEmptyBlocks.length} blocks, ratio=${regionsPerBlock.toFixed(2)}, centroidY=${clusterCentroidY.toFixed(1)}%)`);
    if (regionsPerBlock < commentRatioThreshold && clusterYGuardPassed) {
      for (const idx of cluster) commentRegionIndices.add(idx);
      console.error(`  [Comment detection] Cluster flagged as comment section — ${cluster.length} regions will use raised baseline ${commentBaseline}`);
    }
  }

  // Height filter — use raised baseline only for regions in comment clusters
  const beforeMergedHeightFilter = extraction.regions.length;
  const commentRegionSet = commentRegionIndices; // reference for closure
  // Median cap over NORMAL regions only — comment-flagged clusters keep their
  // raised threshold and are excluded from the pool so a flagged junk wall
  // can't drag the median down for the rest of the page.
  const normalThresholdBase = normalBaseline * (baselineAspect / mergedAspectRatio);
  const normalPool = extraction.regions.filter((_, idx) => !commentRegionSet.has(idx));
  const heightCap = computeMedianHeightCap(normalPool, mergedAspectRatio, normalThresholdBase);
  if (heightCap.capped) {
    console.error(`  [HeightCap] median cap active: threshold ${normalThresholdBase.toFixed(2)}% -> ${heightCap.thresholdPct.toFixed(2)}% of height (median line = ${heightCap.medianWPct.toFixed(2)}% of width, ${normalPool.length} regions pooled)`);
  }
  extraction.regions = extraction.regions.filter((region, idx) => {
    const baseline = commentRegionSet.has(idx) ? commentBaseline : normalBaseline;
    const adaptiveHeightThreshold = commentRegionSet.has(idx)
      ? commentBaseline * (baselineAspect / mergedAspectRatio)
      : heightCap.thresholdPct;
    const heightPct = getEffectiveHeight(region, mergedAspectRatio);
    const nearStitch = isNearStitch(region);
    if (heightPct < adaptiveHeightThreshold) {
      const rawH = region.boundingBox.height;
      const rotNote = (rawH !== heightPct) ? ` (raw=${rawH.toFixed(1)}%, rot-corrected)` : '';
      console.error(`  Filtering short text (h=${heightPct.toFixed(1)}% < ${adaptiveHeightThreshold.toFixed(1)}%, baseline=${baseline})${rotNote}: "${region.text.trim()}"`);
      filterResults.push({ region, passed: false, failedFilter: 'height', failReason: `${heightPct.toFixed(2)}% < ${adaptiveHeightThreshold.toFixed(2)}%`, nearStitch });
      return false;
    }
    return true;
  });
  if (extraction.regions.length < beforeMergedHeightFilter) {
    console.error(`  Height filter (${beforeMergedHeightFilter} regions pre-filter): removed ${beforeMergedHeightFilter - extraction.regions.length} regions`);
  }

  if (isMerged) {
    const normalThreshold = heightCap.thresholdPct;
    const capNote = heightCap.capped ? ', median-capped' : '';
    const commentThreshold = commentBaseline * (baselineAspect / mergedAspectRatio);
    if (commentRegionIndices.size > 0) {
      console.error(`  [Merged] Filter thresholds: normal >= ${normalThreshold.toFixed(2)}% (baseline=${normalBaseline}${capNote}), comment regions >= ${commentThreshold.toFixed(2)}% (baseline=${commentBaseline})`);
    } else {
      console.error(`  [Merged] Filter thresholds scaled: heightPct >= ${normalThreshold.toFixed(2)}% (baseline=${normalBaseline}${capNote})`);
    }
  }

  // English-input mode only: drop "structureless" English blocks. For CN input,
  // English is never content to translate — any all-English block is dropped
  // after merging (see the all-English block drop below), so isolated English
  // just flows through to block grouping instead of being filtered per-region.
  if (options.inputLanguage === 'english') {
    // English-input mode: drop "structureless" BLOCKS — text with no spaces
    // and no punctuation. See the matching single-page branch for the
    // rationale on block-level checking (PaddleOCR over-segments multi-word
    // English text, so per-region checks misfire). Applied here because
    // merged long-strip content takes this code path instead of the
    // single-page one.
    const beforeStructureFilterMerged = extraction.regions.length;
    const prelimBlocksMerged = groupRegionsIntoBlocks(extraction.regions, adjacencyDistance, 'english', mergedAspectRatio);
    // Same check as the single-page branch — including the 2+ spaces bypass
    // and the trailing-dash interruption mark (see that branch for rationale).
    const SENTENCE_PUNCT_MERGED = /[.,!?;:()…。，！？；：（）]/u;
    const TRAILING_DASH_MERGED = /[-–—]$/;
    const structurelessIndicesMerged = new Set<number>();
    for (const block of prelimBlocksMerged) {
      if (block.regionIndices.length === 0) continue;
      const joinedText = block.regionIndices
        .map(i => extraction.regions[i].text.trim())
        .filter(t => t.length > 0)
        .join(' ');
      if (joinedText.length === 0) {
        for (const i of block.regionIndices) structurelessIndicesMerged.add(i);
        continue;
      }
      const spaceCount = (joinedText.match(/\s/g) || []).length;
      if (spaceCount >= 2) continue; // bypass punctuation check
      if (!SENTENCE_PUNCT_MERGED.test(joinedText) && !TRAILING_DASH_MERGED.test(joinedText)) {
        for (const i of block.regionIndices) structurelessIndicesMerged.add(i);
      }
    }
    let filteredStructurelessMerged = 0;
    extraction.regions = extraction.regions.filter((region, i) => {
      if (structurelessIndicesMerged.has(i)) {
        filteredStructurelessMerged++;
        console.error(`  Filtering structureless block (no sentence punct): "${region.text.trim().substring(0, 30)}"`);
        const existing = filterResults.find(r => r.region === region);
        if (existing) {
          existing.passed = false;
          existing.failedFilter = 'structureless_en';
          existing.failReason = `block has no sentence punctuation`;
        }
        return false;
      }
      return true;
    });
    if (filteredStructurelessMerged > 0) {
      console.error(`  Structure filter (EN mode, merged path): removed ${filteredStructurelessMerged} structureless regions of ${beforeStructureFilterMerged}`);
    }
  }

  // Filter punctuation-only
  const beforePunctFilter = extraction.regions.length;
  extraction.regions = extraction.regions.filter((region) => {
    const text = region.text.trim();
    const nonPunctuation = text.replace(/[，。！？、；：""''「」『』【】（）—…·～《》〈〉\s\.,!?;:'"()\-]/g, '');
    if (nonPunctuation.length === 0) {
      console.error(`  Filtering punctuation-only: "${text}"`);
      const existing = filterResults.find(r => r.region === region);
      if (existing) {
        existing.passed = false;
        existing.failedFilter = 'punctuation_only';
        existing.failReason = `Only punctuation: "${text}"`;
      }
      return false;
    }
    return true;
  });

  if (extraction.regions.length < beforePunctFilter) {
    console.error(`  Punctuation filter: removed ${beforePunctFilter - extraction.regions.length} regions`);
  }

  // Filter single-char regions (consistent with single-page path)
  // Skip for English input — single English words are valid
  if (options.inputLanguage !== 'english') {
  const beforeSingleCharFilter = extraction.regions.length;
  extraction.regions = extraction.regions.filter((region) => {
    if (region.text.trim().length <= 1) {
      console.error(`  Filtering single character: "${region.text}"`);
      const existing = filterResults.find(r => r.region === region);
      if (existing) {
        existing.passed = false;
        existing.failedFilter = 'single_char';
        existing.failReason = `Single character: "${region.text}"`;
      }
      return false;
    }
    return true;
  });
  if (extraction.regions.length < beforeSingleCharFilter) {
    console.error(`  Single-char filter: removed ${beforeSingleCharFilter - extraction.regions.length} regions`);
  }
  }

  // Log summary of filter results for merged pages
  if (isMerged) {
    const failedNearStitch = filterResults.filter(r => !r.passed && r.nearStitch);
    const passedNearStitch = filterResults.filter(r => r.passed && r.nearStitch);

    console.error(`  [Merged] === FILTER SUMMARY ===`);
    console.error(`  [Merged] Total regions: ${originalCount} → ${extraction.regions.length} passed`);
    console.error(`  [Merged] Near-stitch regions: ${failedNearStitch.length + passedNearStitch.length} total, ${passedNearStitch.length} passed, ${failedNearStitch.length} filtered`);

    if (failedNearStitch.length > 0) {
      console.error(`  [Merged] !!! FILTERED NEAR-STITCH REGIONS !!!`);
      for (const r of failedNearStitch) {
        console.error(`    - FILTERED by ${r.failedFilter}: "${r.region.text.substring(0, 30)}..." (${r.failReason})`);
      }
    }

    if (passedNearStitch.length > 0) {
      console.error(`  [Merged] Passed near-stitch regions:`);
      for (const r of passedNearStitch) {
        console.error(`    - PASSED: "${r.region.text.substring(0, 30)}..."`);
      }
    }
    console.error(`  [Merged] === END FILTER SUMMARY ===`);
  }

  console.error(`  Found ${extraction.regions.length} text regions`);

  // Check for leading punctuation missed by OCR (e.g. "..." in manwha inner monologue)
  const grayCorrectedRegionIndices = new Set<number>();
  if (process.env.ENABLE_GEMINI_VERIFY === 'true' && extraction.regions.length > 0) {
    try {
      const metadata = await sharp(imageBuffer).metadata();
      const imgWidth = metadata.width || 1;
      const imgHeight = metadata.height || 1;

      const grayChecks: Array<{ regionIndex: number; region: TextRegion }> = [];

      for (let rIdx = 0; rIdx < extraction.regions.length; rIdx++) {
        const region = extraction.regions[rIdx];

        const expandedBox = region.isVertical
          ? { x: region.boundingBox.x, y: Math.max(0, region.boundingBox.y - 20),
              width: region.boundingBox.width, height: region.boundingBox.height + 20 }
          : { x: Math.max(0, region.boundingBox.x - 20), y: region.boundingBox.y,
              width: region.boundingBox.width + 20, height: region.boundingBox.height };

        const pixelX = Math.round((expandedBox.x / 100) * imgWidth);
        const pixelY = Math.round((expandedBox.y / 100) * imgHeight);
        const pixelW = Math.round((expandedBox.width / 100) * imgWidth);
        const pixelH = Math.round((expandedBox.height / 100) * imgHeight);

        const left = Math.max(0, pixelX);
        const top = Math.max(0, pixelY);
        const width = Math.min(pixelW, imgWidth - left);
        const height = Math.min(pixelH, imgHeight - top);

        if (width > 0 && height > 0) {
          const { data, info } = await sharp(imageBuffer)
            .extract({ left, top, width, height })
            .raw()
            .toBuffer({ resolveWithObject: true });

          let totalR = 0, totalG = 0, totalB = 0;
          const step = Math.max(1, Math.floor(data.length / info.channels / 100));
          let samples = 0;
          for (let i = 0; i < data.length; i += info.channels * step) {
            totalR += data[i];
            totalG += data[i + 1];
            totalB += data[i + 2];
            samples++;
          }
          const sampledColor = {
            r: Math.round(totalR / samples),
            g: Math.round(totalG / samples),
            b: Math.round(totalB / samples)
          };

          const grayResult = await detectSuspiciousGray(sampledColor, imageBuffer, imgWidth, imgHeight);
          if (grayResult.isGray) {
            grayChecks.push({ regionIndex: rIdx, region });
          }
        }
      }

      console.error(`  [GeminiVerify] Checked ${extraction.regions.length} regions, ${grayChecks.length} suspicious`);

      if (grayChecks.length > 0) {
        console.error(`  [GeminiVerify] Found ${grayChecks.length} suspicious gray regions, checking for leading punctuation...`);
        const verifyStartTime = Date.now();

        for (const { regionIndex, region } of grayChecks) {
          const result = await checkLeadingPunctuation(
            imageBuffer, imgWidth, imgHeight, region, regionIndex
          );

          if (result.wasModified) {
            extraction.regions[regionIndex] = {
              ...extraction.regions[regionIndex],
              text: result.foundPunctuation + extraction.regions[regionIndex].text,
              boundingBox: result.expandedBoundingBox
            };
            grayCorrectedRegionIndices.add(regionIndex);
            console.error(`  [GeminiVerify] Prepended "${result.foundPunctuation}" → "${extraction.regions[regionIndex].text}"`);
          }
        }

        const verifyTime = Date.now() - verifyStartTime;
        console.error(`  [TIMING] Punctuation check: ${(verifyTime / 1000).toFixed(2)}s`);
      }
    } catch (error: any) {
      console.error(`  [GeminiVerify] Error: ${error.message}`);
    }
  }

  // If analyze only, just return the extraction
  if (options.analyzeOnly) {
    return {
      pageNumber: pageImage.pageNumber,
      originalText: extraction.fullText,
      simplifiedText: extraction.fullText,
      regions: extraction.regions,
      simplifiedRegions: [],
      imageWidth: pageImage.width,
      imageHeight: pageImage.height,
    };
  }

  // Simplify text regions using block-based approach for context
  console.error('  Simplifying text...');
  const llmStartTime = Date.now();
  const simplifiedRegions: ReplacementRegion[] = [];

  // [CLUSTER_FILTER — gated] Pre-merge garbage-band removal. Group the raw regions by LOOSE
  // vertical adjacency (gap/height < 0.7 + horizontal overlap); a band whose combined text is
  // >= MINLEN chars is a UI list / comment wall / status window. Dialogue never chains this
  // far — separate bubbles live in separate columns, so the horizontal-overlap requirement
  // stops them linking (measured region-level ceiling ~43 chars across merged Tower-of-God
  // pages). Hand each band to the LLM, keep the regions it marks story, drop the rest to
  // original art. Running PRE-merge is essential: it removes the UI before Pass 1/2 can fuse a
  // UI cell into a bubble line, so surviving bubble regions merge cleanly. Cheap geometry
  // gates a rare LLM call; if it fails/off, the band just renders (no bug). Off unless
  // CLUSTER_FILTER=true.
  // CLUSTER_FILTER is a Chinese-tuned experiment (off in prod) — scope it to
  // Chinese only rather than letting it fire on JA/KO with untested thresholds.
  if (process.env.CLUSTER_FILTER === 'true' && (options.inputLanguage || 'chinese') === 'chinese' && extraction.regions.length > 2) {
    const Rg = extraction.regions;
    const LOOSE = 0.7, HOVL = 0.5, MINLEN = 50, M = Rg.length;
    const adj = (i: number, j: number) => {
      const a = Rg[i].boundingBox, b = Rg[j].boundingBox;
      const ov = Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x));
      const mw = Math.min(a.width, b.width);
      if (mw <= 0 || ov / mw <= HOVL) return false;                    // need horizontal overlap
      const up = a.y <= b.y ? a : b, lo2 = a.y <= b.y ? b : a;
      const mh = Math.min(a.height, b.height);
      return mh > 0 && (lo2.y - (up.y + up.height)) / mh < LOOSE;      // loose vertical adjacency
    };
    const parent = Array.from({ length: M }, (_, i) => i);
    const find = (x: number): number => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
    for (let i = 0; i < M; i++) for (let j = i + 1; j < M; j++) if (adj(i, j)) parent[find(i)] = find(j);
    const comps: Record<number, number[]> = {};
    Rg.forEach((_, i) => { const r = find(i); (comps[r] = comps[r] || []).push(i); });
    const cleanLen = (s: string) => (s || '').replace(/\s/g, '').length;
    const drop = new Array(M).fill(false);
    for (const idxs of Object.values(comps)) {
      const totalLen = idxs.reduce((s, i) => s + cleanLen(Rg[i].text), 0);
      if (totalLen < MINLEN) continue;                                 // not a garbage band
      const ord = idxs.slice().sort((a, b) => Rg[a].boundingBox.y - Rg[b].boundingBox.y);
      const keep = await identifyStoryBlocks(ord.map(i => Rg[i].text || ''));
      ord.forEach((ri, k) => { if (!keep.has(k)) drop[ri] = true; });
      console.error(`  [ClusterFilter] band ${idxs.length} regions / ${totalLen} chars: kept ${keep.size}, dropped ${idxs.length - keep.size}`);
      if (process.env.CLUSTER_DEBUG === 'true') ord.forEach((ri, k) => console.error(`      [${keep.has(k) ? 'KEEP' : 'DROP'}] ${(Rg[ri].text || '').slice(0, 40)}`));
    }
    const before = M;
    extraction.regions = Rg.filter((_, i) => !drop[i]);
    if (extraction.regions.length < before) console.error(`  [ClusterFilter] total dropped ${before - extraction.regions.length}/${before} regions`);
  }

  // No regions are pre-skipped — preserveAsIs regions (adjacent English) now participate
  // in block grouping so mixed Chinese+English lines merge and render correctly.
  // The simplifier handles English text within blocks (preserves it verbatim).
  const skipRegionIndices = new Set<number>();

  // Filter regions for block grouping (includes short text so Pass 2 can merge them)
  const regionsToSimplify = extraction.regions.filter((_, i) => !skipRegionIndices.has(i));

  // Build mapping from regionsToSimplify index → original extraction.regions index
  // This must be done BEFORE the short text filter modifies skipRegionIndices,
  // because block.regionIndices are indices into regionsToSimplify.
  const simplifyToOriginalIdx: number[] = [];
  extraction.regions.forEach((_, i) => {
    if (!skipRegionIndices.has(i)) {
      simplifyToOriginalIdx.push(i);
    }
  });

  // Group regions into blocks for context-aware simplification.
  // outDroppedByOverlap collects regionsToSimplify indices that Pass 4
  // (rotated-overlap filter) drops; we add the corresponding original region
  // indices to chromaExcludedIndices below so neither inpaint nor render
  // touches them — area is left as original artwork.
  const droppedByOverlap = new Set<number>();
  const blocks = groupRegionsIntoBlocks(regionsToSimplify, adjacencyDistance, options.inputLanguage || 'chinese', imageAspectRatio, droppedByOverlap);

  // Now apply short text filter: skip short regions that are still standalone after grouping
  const mergedRegionIndices = new Set<number>();
  blocks.forEach(block => {
    if (block.regionIndices.length > 1) {
      block.regionIndices.forEach(ri => mergedRegionIndices.add(ri));
    }
  });
  regionsToSimplify.forEach((region, simplifyIdx) => {
    if (options.inputLanguage !== 'english' && region.text.length <= 2 && !mergedRegionIndices.has(simplifyIdx)) {
      const origIdx = simplifyToOriginalIdx[simplifyIdx];
      skipRegionIndices.add(origIdx);
    }
  });

  // Deferred chroma filter: now that blocks are formed, filter colorful regions
  // that are standalone with few Chinese chars (art overlays / SFX).
  // Multi-line colored UI boxes (system notifications) survive by merging into blocks first.
  const chromaExcludedIndices = new Set<number>();
  // Seed exclusion list with any regions dropped by Pass 4 (rotated overlap).
  // These should never be inpainted or rendered — leave original artwork.
  for (const idx of droppedByOverlap) {
    const origIdx = simplifyToOriginalIdx[idx];
    if (origIdx !== undefined) chromaExcludedIndices.add(origIdx);
  }
  const chromaExcludedBlocks = new Set<number>();
  blocks.forEach((block, blockIdx) => {
    let blockChineseChars = 0;
    for (const text of block.texts) {
      for (const char of text) {
        const code = char.charCodeAt(0);
        if ((code >= 0x4e00 && code <= 0x9fff) || (code >= 0x3400 && code <= 0x4dbf)) {
          blockChineseChars++;
        }
      }
    }

    const colorfulIndices = block.regionIndices.filter(ri => regionsToSimplify[ri]?.isColorful);
    if (colorfulIndices.length === 0) return;

    if (colorfulIndices.length < block.regionIndices.length) {
      console.error(`  [Chroma] Keeping mixed colorful/non-colorful block #${blockIdx}: "${block.texts.join(' | ')}"`);
      return;
    }

    // Chroma drop DISABLED — see single-page version for rationale.
    let blockMaxChroma = 0;
    for (const ri of colorfulIndices) {
      const chroma = regionsToSimplify[ri]?.avgChroma || 0;
      if (chroma > blockMaxChroma) blockMaxChroma = chroma;
    }
    console.error(`  [Chroma] Keeping colorful block #${blockIdx} (filter disabled, chroma=${blockMaxChroma.toFixed(0)}): "${block.texts.join(' | ')}"`);
  });

  // English mode: filter blocks that are only numbers (OCR hallucinations or page numbers)
  if (options.inputLanguage === 'english') {
    blocks.forEach((block, blockIdx) => {
      if (chromaExcludedBlocks.has(blockIdx)) return;
      const combined = block.texts.join(' ').trim();
      if (/^[\d\s.,-]+$/.test(combined)) {
        console.error(`  [English] Filtering numbers-only block: "${combined}"`);
        chromaExcludedBlocks.add(blockIdx);
        block.regionIndices.forEach(ri => {
          chromaExcludedIndices.add(simplifyToOriginalIdx[ri]);
        });
      }
    });
  }

  // English mode: fully exclude standalone single-word blocks with no punctuation.
  // Only . , ! ? and a TRAILING dash (interruption mark — see single-page
  // branch for rationale) count as punctuation — mid-word hyphens,
  // apostrophes, etc. don't protect garbled text.
  if (options.inputLanguage === 'english') {
    blocks.forEach((block, blockIdx) => {
      if (chromaExcludedBlocks.has(blockIdx)) return;
      if (block.regionIndices.length !== 1) return;
      const text = block.texts[0].trim();
      const stripped = text.replace(/[^a-zA-Z]/g, '');
      if (stripped.length === 0) return;
      const hasSpace = /\s/.test(text);
      const hasPunctuation = /[.!?,]/.test(text) || /[-–—]$/.test(text);
      if (!hasSpace && !hasPunctuation) {
        console.error(`  [English] Filtering standalone single-word without punctuation: "${text}"`);
        chromaExcludedBlocks.add(blockIdx);
        block.regionIndices.forEach(ri => {
          chromaExcludedIndices.add(simplifyToOriginalIdx[ri]);
        });
      }
    });
  }

  // Alignment-based UI filter (see single-page version above for rationale).
  // Skip entirely for Chinese input — Chinese dialogue bubbles legitimately
  // have precise left-alignment with varying line widths, which the filter
  // mistakes for UI blocks.
  if (!isEnglishInput) {
    console.error(`  [Alignment] Skipping alignment filter for Chinese input (merged)`);
  } else {
  const alignmentThresholdMerged = parseFloat(process.env.UI_ALIGNMENT_THRESHOLD || '0.05');
  const alignmentSameSizeRatioMerged = parseFloat(process.env.UI_ALIGNMENT_SAMESIZE_RATIO || '0.90');
  blocks.forEach((block, blockIdx) => {
    if (chromaExcludedBlocks.has(blockIdx)) return;
    if (block.regionIndices.length < 2) return;
    const blockRegions = block.regionIndices.map(ri => regionsToSimplify[ri]).filter(Boolean);
    if (blockRegions.length < 2) return;
    // Same-width override (see single-page version).
    const widths = blockRegions.map(r => r.boundingBox.width);
    const maxW = Math.max(...widths);
    const minW = Math.min(...widths);
    if (maxW > 0 && minW / maxW >= alignmentSameSizeRatioMerged) return;
    const lefts = blockRegions.map(r => r.boundingBox.x);
    const centers = blockRegions.map(r => r.boundingBox.x + r.boundingBox.width / 2);
    const meanLeft = lefts.reduce((s, x) => s + x, 0) / lefts.length;
    const meanCenter = centers.reduce((s, x) => s + x, 0) / centers.length;
    const leftVar = lefts.reduce((s, x) => s + (x - meanLeft) ** 2, 0) / lefts.length;
    const centerVar = centers.reduce((s, x) => s + (x - meanCenter) ** 2, 0) / centers.length;
    if (centerVar === 0) return;
    const ratio = leftVar / centerVar;
    if (ratio < alignmentThresholdMerged) {
      console.error(`  [Alignment] Filtering left-aligned UI block (ratio=${ratio.toFixed(3)}): "${block.texts.join(' | ').slice(0, 80)}"`);
      chromaExcludedBlocks.add(blockIdx);
      block.regionIndices.forEach(ri => {
        chromaExcludedIndices.add(simplifyToOriginalIdx[ri]);
      });
    }
  });
  }

  // Filter out chroma-excluded blocks + non-dialogue long blocks (CN-only)
  const longBlocksToDrop = identifyLongNonDialogueBlocks(blocks, options.inputLanguage || 'chinese');
  // A long non-dialogue block is dropped ENTIRELY, not merely skipped for
  // simplification. Fold it into the chroma-exclusion sets so both the render
  // loop (skips chromaExcludedBlocks) and the inpaint loop (skips
  // chromaExcludedIndices) leave it untouched. Without this, the block falls
  // back to smartJoin(original) and gets re-rendered with its original,
  // unsimplified text — the bug that let OCR'd comment sections survive.
  longBlocksToDrop.forEach((blockIdx) => {
    chromaExcludedBlocks.add(blockIdx);
    blocks[blockIdx].regionIndices.forEach((ri) => {
      const origIdx = simplifyToOriginalIdx[ri];
      if (origIdx !== undefined) chromaExcludedIndices.add(origIdx);
    });
  });

  // CN/JA/KO input: any block with zero source-content chars is incidental — a game
  // title, UI label, SFX, sign, or watermark. It's never content to translate
  // in CN→CN, so drop it (same fold as longBlocksToDrop → skipped by both the
  // inpaint and render loops, leaving the original pixels untouched). English
  // that belongs to a Chinese line already merged INTO that line's block during
  // grouping, so it survives via the has-Chinese test here.
  if (options.inputLanguage !== 'english') {
    blocks.forEach((block, blockIdx) => {
      if (chromaExcludedBlocks.has(blockIdx)) return;
      // Misdetection-guard tally. Pages may pass through here more than once
      // (inpaint/LLM/render passes each regroup) — the drop RATIO is
      // pass-invariant, which is all the guard reads.
      if (options._langStats) options._langStats.totalBlocks++;
      // Language-aware content test — see the single-page twin for rationale.
      const hasSourceContent = block.texts.some(t => countSourceContentChars(t, options.inputLanguage || 'chinese') > 0);
      if (!hasSourceContent) {
        if (options._langStats) options._langStats.allEnglishDropped++;
        console.error(`  [NonChinese] Dropping no-source-content block #${blockIdx}: "${block.texts.join(' | ').slice(0, 60)}"`);
        chromaExcludedBlocks.add(blockIdx);
        block.regionIndices.forEach((ri) => {
          const origIdx = simplifyToOriginalIdx[ri];
          if (origIdx !== undefined) chromaExcludedIndices.add(origIdx);
        });
      }
    });
  }
  // [OVERLAP_DROP] Change 2 — leave heavily-overlapping homogeneous clusters as
  // original art (folded into chromaExcludedBlocks, so neither inpainted nor
  // re-typeset). Runs post-merge on the corrected oriented boxes. Off unless
  // OVERLAP_DROP=true.
  if (process.env.OVERLAP_DROP === 'true') {
    const overlapDrop = overlappingClustersToDrop(blocks, regionsToSimplify, pageImage.width, pageImage.height,
      (i) => chromaExcludedBlocks.has(i) || longBlocksToDrop.has(i));
    overlapDrop.forEach((blockIdx) => {
      chromaExcludedBlocks.add(blockIdx);
      blocks[blockIdx].regionIndices.forEach((ri) => {
        const origIdx = simplifyToOriginalIdx[ri];
        if (origIdx !== undefined) chromaExcludedIndices.add(origIdx);
      });
    });
  }
  const blocksToSimplify = blocks.filter((_, idx) => !chromaExcludedBlocks.has(idx) && !longBlocksToDrop.has(idx));
  const simplifyBlockToOrigBlock: number[] = [];
  blocks.forEach((_, idx) => {
    if (!chromaExcludedBlocks.has(idx) && !longBlocksToDrop.has(idx)) simplifyBlockToOrigBlock.push(idx);
  });

  // Compute bbox per simplify-block (union of its OCR region bboxes).
  // Used by the cache layer for bbox-matching fresh blocks ↔ cached blocks.
  const blocksToSimplifyBboxes = blocksToSimplify.map(block => {
    const bboxes = block.regionIndices.map(ri => regionsToSimplify[ri].boundingBox);
    const minX = Math.min(...bboxes.map(b => b.x));
    const minY = Math.min(...bboxes.map(b => b.y));
    const maxX = Math.max(...bboxes.map(b => b.x + b.width));
    const maxY = Math.max(...bboxes.map(b => b.y + b.height));
    return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
  });

  // In batched mode (skipSimplification), return early with block data — no LLM, no inpainting
  if (options.skipSimplification && !options.precomputedBlockResults && !options.inpaintOnly) {
    return {
      pageNumber: pageImage.pageNumber,
      originalText: extraction.fullText,
      simplifiedText: '',
      regions: extraction.regions,
      simplifiedRegions: [],
      blocksToSimplify,
      blocksToSimplifyBboxes,
      simplifyBlockToOrigBlock,
    };
  }

  // Inpaint-only mode: skip LLM and metadata, just produce the cleaned image buffer.
  // Used by the batched pipeline to overlap inpainting with LLM simplification.
  if (options.inpaintOnly) {
    const inpaintRegions: ReplacementRegion[] = [];
    for (let i = 0; i < extraction.regions.length; i++) {
      if (chromaExcludedIndices.has(i)) continue;
      inpaintRegions.push({ original: extraction.regions[i], simplifiedText: '' });
    }

    const { cleanedImageBuffer, inpaintMs, decodePageMs } = await inpaintImageRegions(
      pageImage.buffer, inpaintRegions, { inputLanguage: options.inputLanguage }
    );
    globalTimingStats.inpaintServer += inpaintMs;
    globalTimingStats.decodePage += decodePageMs || 0;
    return {
      pageNumber: pageImage.pageNumber,
      originalText: extraction.fullText,
      simplifiedText: '',
      regions: extraction.regions,
      simplifiedRegions: [],
      blocksToSimplify,
      blocksToSimplifyBboxes,
      simplifyBlockToOrigBlock,
      cleanedImageBuffer,
      inpaintMs,
    };
  }

  // Simplify blocks (with chapter context if available from Approach 4)
  const useMultiLevel = (options.targetLevels && options.targetLevels.length > 1) || options.includeEnglish;
  let blockResults = new Map<number, string>();  // keyed by original block index
  let multiLevelBlockResults: Map<number, Map<string, string>> | undefined;
  if (options.precomputedBlockResults) {
    // Use pre-computed results from batched simplification
    blockResults = options.precomputedBlockResults;
  } else if (blocksToSimplify.length > 0 && !options.skipSimplification) {
    try {
      if (useMultiLevel && options.targetLevels) {
        const rawResults = await simplifyBlocksMultiLevel(
          blocksToSimplify, options.targetLevels, options.includeEnglish || false, options.chapterContext, pageNumber, options.inputLanguage
        );
        multiLevelBlockResults = new Map();
        const primaryLevel = options.targetLevel;
        rawResults.forEach((levelMap, simplifyIdx) => {
          const origIdx = simplifyBlockToOrigBlock[simplifyIdx];
          multiLevelBlockResults!.set(origIdx, levelMap);
          blockResults.set(origIdx, levelMap.get(primaryLevel) || blocks[origIdx].texts.join(''));
        });
      } else {
        const rawResults = await simplifyBlocks(blocksToSimplify, options.targetLevel, options.chapterContext, pageNumber, options.inputLanguage);
        rawResults.forEach((text, simplifyIdx) => {
          const origIdx = simplifyBlockToOrigBlock[simplifyIdx];
          blockResults.set(origIdx, text);
        });
      }
    } catch (e) {
      console.error('  Block simplification failed, using originals');
    }
  }
  const llmTime = Date.now() - llmStartTime;
  console.error(`  [TIMING] LLM: ${(llmTime / 1000).toFixed(2)}s (${regionsToSimplify.length} texts in ${blocksToSimplify.length} blocks)`);
  globalTimingStats.llm += llmTime;

  // Log simplifier output for each block
  blockResults.forEach((text, blockIdx) => {
    const original = blocks[blockIdx]?.texts.join(' | ');
    console.error(`  [Simplifier] Block ${blockIdx}: "${original}" → "${text}"`);
  });

  // Build maps for block-based rendering: original region index → blockId and blockText
  const blockIdMap = new Map<number, number>();
  const blockTextMap = new Map<number, string>();
  const originalBlockTextMap = new Map<number, string>();

  blocks.forEach((block, blockIdx) => {
    if (chromaExcludedBlocks.has(blockIdx)) return;
    const blockText = blockResults.get(blockIdx) || smartJoin(block.texts);
    blockTextMap.set(blockIdx, blockText);
    originalBlockTextMap.set(blockIdx, smartJoin(block.texts));
    block.regionIndices.forEach((regIdx) => {
      const originalIdx = simplifyToOriginalIdx[regIdx];
      blockIdMap.set(originalIdx, blockIdx);
    });
  });

  // DEBUG: dump the block-index → text mapping at the moment we join LLM output
  // with the block array. This is where any indexing mismatch between phase 4a
  // (LLM input builder) and phase 4c (this call) would surface.
  if (process.env.DEBUG_EMPTY_SENTINEL) {
    console.error(`  [DEBUG_BLOCK_MAP] Page ${pageImage.pageNumber}: blocks.length=${blocks.length}, blockResults.size=${blockResults.size}, precomputed=${options.precomputedBlockResults ? options.precomputedBlockResults.size : 0}`);
    blocks.forEach((block, blockIdx) => {
      const orig = smartJoin(block.texts).slice(0, 40);
      const llmOut = blockResults.get(blockIdx);
      const usedFallback = !blockResults.has(blockIdx) || blockResults.get(blockIdx) === '' || blockResults.get(blockIdx) === undefined;
      console.error(`    B${blockIdx} orig="${orig}" | llm=${llmOut === undefined ? 'MISSING' : JSON.stringify((llmOut || '').slice(0, 40))} | fallback=${usedFallback}`);
    });
  }

  // Reconstruct regions with block data
  // For merged images, detect cross-page continuation using stitch positions, not just top 3%
  for (let i = 0; i < extraction.regions.length; i++) {
    const region = extraction.regions[i];

    // Chroma-excluded regions are fully skipped — no inpainting, no rendering
    if (chromaExcludedIndices.has(i)) continue;

    const centerY = region.boundingBox.y + region.boundingBox.height / 2;

    // Cross-page continuation: near stitch boundary OR at very top of image
    const isNearStitchBoundary = stitchYPositions.some(sy => Math.abs(centerY - sy) < 5);
    const isTopEdge = region.boundingBox.y < 3;
    const isCrossPage = isNearStitchBoundary || isTopEdge;

    // Force white fill for gray-corrected regions
    const forceFill = grayCorrectedRegionIndices.has(i) ? { forceFillColor: { r: 255, g: 255, b: 255 } } : {};

    if (skipRegionIndices.has(i)) {
      // Short text - keep original, no block
      simplifiedRegions.push({
        original: region,
        simplifiedText: region.text,
        isCrossPageContinuation: isCrossPage,
        ...forceFill,
      });
    } else {
      // Assign block data for block-based rendering
      const blockIdx = blockIdMap.get(i);
      const blockText = blockIdx !== undefined ? blockTextMap.get(blockIdx) : undefined;
      const originalBlockText = blockIdx !== undefined ? originalBlockTextMap.get(blockIdx) : undefined;
      simplifiedRegions.push({
        original: region,
        simplifiedText: region.text,
        isCrossPageContinuation: isCrossPage,
        blockId: blockIdx,
        blockText: blockText,
        originalBlockText: originalBlockText,
        ...forceFill,
      });
    }
  }

  // Log simplified regions for merged pages
  if (isMerged && simplifiedRegions.length > 0) {
    console.error(`  [Merged] === SIMPLIFIED REGIONS ===`);
    for (const sr of simplifiedRegions) {
      const centerY = sr.original.boundingBox.y + sr.original.boundingBox.height / 2;
      const nearStitch = stitchYPositions.some(sy => Math.abs(centerY - sy) < 5);
      const stitchMarker = nearStitch ? ' [NEAR STITCH]' : '';
      const crossPageMarker = sr.isCrossPageContinuation ? ' [CROSS-PAGE]' : '';
      const displayText = sr.blockText || sr.simplifiedText;
      console.error(`    y=${centerY.toFixed(1)}%: "${sr.original.text.substring(0, 20)}..." → "${displayText.substring(0, 20)}..."${stitchMarker}${crossPageMarker}`);
    }
    console.error(`  [Merged] === END SIMPLIFIED REGIONS ===`);
  }

  // Replace text in image (inpainting + metadata only, no text rendering)
  // If precomputedCleanedImage is supplied, the inpaint step is skipped — pages were
  // inpainted in parallel with the LLM phase upstream.
  console.error('  Replacing text in image...');
  let cleanedBuffer: Buffer = pageImage.buffer; // fallback: original image
  let renderedBlocks: RenderedBlockInfo[] = [];
  try {
    const replaceResult = await replaceTextInImage(
      pageImage.buffer, simplifiedRegions, { inputLanguage: options.inputLanguage }, options.precomputedCleanedImage, sharedPageData
    );
    cleanedBuffer = replaceResult.cleanedImageBuffer;
    renderedBlocks = replaceResult.renderedBlocks;
    if (!options.precomputedCleanedImage) {
      globalTimingStats.inpaintServer += replaceResult.timing.inpaintServer;
    }
    globalTimingStats.textRendering += replaceResult.timing.bgDetection + replaceResult.timing.varianceCheck + replaceResult.timing.textRendering;
    globalTimingStats.colorSampling += replaceResult.timing.colorSampling;
    globalTimingStats.metaBuild += replaceResult.timing.metaBuild;
    globalTimingStats.decodePage += replaceResult.timing.decodePage;
    if (replaceResult.timing.inpaintServer > 0) {
      console.error(`  [TIMING] Inpaint server: ${(replaceResult.timing.inpaintServer / 1000).toFixed(2)}s`);
    }
  } catch (err) {
    console.error(`  Warning: Text replacement failed, using original image: ${err}`);
  }

  // Save output if outputDir specified
  let originalImagePath: string | undefined;
  let cleanedImagePath: string | undefined;
  if (options.outputDir) {
    const baseName = String(pageNumber).padStart(3, '0');

    const originalDir = path.join(options.outputDir, 'original');
    const cleanedDir = path.join(options.outputDir, 'cleaned');
    if (!fs.existsSync(originalDir)) fs.mkdirSync(originalDir, { recursive: true });
    if (!fs.existsSync(cleanedDir)) fs.mkdirSync(cleanedDir, { recursive: true });

    const imageExt = pickImageExtension(pageImage.width, pageImage.height);
    originalImagePath = path.join(originalDir, `${baseName}${imageExt}`);
    cleanedImagePath = path.join(cleanedDir, `${baseName}${imageExt}`);
    // Fire-and-forget: launch both saves, don't await. Disk I/O happens in
    // parallel with the next page's CPU work. Both are tracked and checked
    // before the reader is generated.
    const _saveStart1 = Date.now();
    trackImageSave(saveImage(pageImage.buffer, originalImagePath)
      .then(() => { globalTimingStats.saveImage += Date.now() - _saveStart1; }), originalImagePath);
    const _saveStart2 = Date.now();
    trackImageSave(saveImage(cleanedBuffer, cleanedImagePath)
      .then(() => { globalTimingStats.saveImage += Date.now() - _saveStart2; }), cleanedImagePath);

    console.error(`  Saved: ${cleanedImagePath}`);
  }

  const simplifiedFullText = simplifiedRegions.map((r) => r.simplifiedText).join('\n');

  const pageTotalTime = Date.now() - pageStartTime;
  globalTimingStats.total += pageTotalTime;
  processedPageCount++;
  console.error(`  [TIMING] Page total: ${(pageTotalTime / 1000).toFixed(2)}s`);

  return {
    pageNumber: pageImage.pageNumber,
    originalText: extraction.fullText,
    simplifiedText: simplifiedFullText,
    regions: extraction.regions,
    simplifiedRegions,
    originalImagePath,
    cleanedImagePath,
    imageWidth: pageImage.width,
    imageHeight: pageImage.height,
    renderedBlocks,
    multiLevelBlockResults,
    blocksToSimplify,
    simplifyBlockToOrigBlock,
  };
}

/**
 * Process multiple manga pages
 */
export async function processMangaPages(
  imagePaths: string[],
  options: MangaProcessingOptions
): Promise<MangaProcessingResult> {
  const startTime = Date.now();
  const pages: PageProcessingResult[] = [];
  // Text-only cache data — populated in batched mode, written by jobService.
  let pipelineCacheBlocks: Array<{ pageNumber: number; blocks: Array<{ bbox: { x: number; y: number; width: number; height: number }; simplifiedText: string }> }> | undefined;

  // Reset timing + token stats for this job
  resetTimingStats();
  resetLLMUsage();
  resetImageSaveTracking();
  resetStoryBlockCache();

  // Ensure output directory exists
  if (options.outputDir && !fs.existsSync(options.outputDir)) {
    fs.mkdirSync(options.outputDir, { recursive: true });
  }

  const ocrServerCount = getOCRServerCount(options.inputLanguage);
  const inpaintServerCount = (process.env.INPAINTING_SERVER_URLS || '').split(',').filter(u => u.trim()).length || 1;
  // For translation input (EN/JA/KO→Chinese), LLM is the bottleneck (much slower
  // per call), so reduce concurrency to avoid Qwen timeouts from request queuing
  const defaultConcurrency = isTranslationSource(options.inputLanguage)
    ? Math.min(2, Math.max(ocrServerCount, inpaintServerCount) + 2)
    : Math.max(ocrServerCount, inpaintServerCount) + 2;
  const concurrency = parseInt(process.env.PIPELINE_CONCURRENCY || String(defaultConcurrency), 10);
  // OCR scan in Phase 3 is wall-time-sensitive (blocks chapter analysis) and not
  // user-latency-sensitive (no UI waits on individual OCR results), so we push
  // queue depth aggressively. Empirical tests: 1 PaddleOCR server saturates at
  // ~6 in-flight requests; 2 servers at ~10. Multiplier of 6 lands near sweet
  // spot for both. Override with OCR_CONCURRENCY if needed.
  const ocrConcurrency = parseInt(process.env.OCR_CONCURRENCY || String(Math.max(ocrServerCount * 6, concurrency)), 10);
  console.error(`  Using ${ocrServerCount} OCR server(s), ${inpaintServerCount} inpainting server(s), concurrency=${concurrency}, ocrConcurrency=${ocrConcurrency}`);

  // PHASE 1: Load all images, filtering out small promotional thumbnails
  // Webtoon pages download promotional banners/cover images for other webtoons
  // that appear below the actual chapter content. These are typically small
  // (92x97, 218x218) while real panels are ~690px wide.
  const loadStartTime = Date.now();
  console.error(`  Loading ${imagePaths.length} images...`);
  let imageBuffers: Buffer[] = [];
  const MIN_PAGE_WIDTH = 400; // Real webtoon panels are ~690px wide
  let skippedCount = 0;
  for (const imagePath of imagePaths) {
    const pageImage = await loadImage(imagePath);
    const meta = await sharp(pageImage.buffer).metadata();
    if (meta.width && meta.width < MIN_PAGE_WIDTH) {
      skippedCount++;
      continue; // Skip promotional thumbnails
    }
    imageBuffers.push(pageImage.buffer);
    // Track page dimensions for the cost dashboard. We record the original
    // source dimensions here, BEFORE splitTallImages runs, so the dashboard
    // reflects what the source site actually served (not our internal chunks).
    if (meta.width && meta.height) {
      pageWidthSum += meta.width;
      pageHeightSum += meta.height;
      dimSampleCount++;
    }
  }
  if (skippedCount > 0) {
    console.error(`  Skipped ${skippedCount} small images (width < ${MIN_PAGE_WIDTH}px)`);
  }
  globalTimingStats.loading += Date.now() - loadStartTime;

  // Pre-merge split: any source image taller than INPUT_SPLIT_THRESHOLD_PX
  // gets divided into ~INPUT_SPLIT_TARGET_PX-tall chunks. The existing merge
  // logic below (`determineMergeGroups`) then re-stitches chunks where text
  // spans the cut, so each final processing unit lands in PaddleOCR's sweet
  // spot (≤4000 px) instead of crashing on multi-thousand-pixel monsters.
  imageBuffers = await splitTallImages(imageBuffers);

  // PHASE 2: Detect which pages need to be merged
  // NOTE: Re-enabled for diagnostic testing
  const ENABLE_PAGE_MERGING = true;

  interface ProcessingUnit {
    buffer: Buffer;
    originalIndices: number[]; // Which original page indices this represents
    stitchYPositions: number[]; // Y positions (as %) where pages were stitched
    pageHeights: number[]; // Heights of each original page in pixels
  }

  const processingUnits: ProcessingUnit[] = [];

  const stitchStartTime = Date.now();
  if (ENABLE_PAGE_MERGING) {
    console.error(`  Detecting cross-page text...`);
    const { groups: mergeGroups, boundaryTexts } = await determineMergeGroups(imageBuffers, options.onProgress);

    // Auto-detect language if set to 'auto' or unset.
    // Sample pages spread EVENLY across the chapter rather than the first few:
    // chapter openings are often textless title/art panels (Hand Jumper Ep1's
    // first three slices have zero OCR-detectable text), which left the old
    // first-3-pages sample empty and fell through to the chinese default — the
    // all-English block drop then silently deleted every block of the English
    // chapter. Sample count scales with chapter length; early-stop once enough
    // text is collected. Full-page OCR (not boundary text) because boundary
    // text is biased toward sound effects — often English in Chinese manhua.
    // Always samples via the CN multilingual pool: it reads both supported
    // languages, while the EN pool can't read Chinese and so can't arbitrate.
    if (!options.inputLanguage || options.inputLanguage === 'auto') {
      const pageCount = imageBuffers.length;
      const sampleTarget = Math.min(12, Math.max(3, Math.ceil(pageCount / 12)));
      const MIN_SAMPLE_SEGMENTS = 5;
      const step = pageCount / sampleTarget;
      const sampleIndices = Array.from(new Set(
        Array.from({ length: sampleTarget }, (_, k) => Math.min(pageCount - 1, Math.round(k * step)))
      ));
      const sampleTexts: string[] = [];
      let sampleOcrErrors = 0;
      let pagesSampled = 0;
      for (const idx of sampleIndices) {
        try {
          const extraction = await extractTextFromImage(imageBuffers[idx], idx + 1, options.visionProvider || 'paddleocr');
          sampleTexts.push(...extraction.regions.map(r => r.text));
          pagesSampled++;
        } catch (e) {
          sampleOcrErrors++;
          console.error(`  [Language] Sample OCR failed for page ${idx + 1}: ${e}`);
        }
        // Early-stop needs BOTH enough text and ≥2 successfully-sampled pages:
        // a single SFX-heavy action page ("BAM", "OW!") can supply 5 Latin-only
        // segments on its own and would misclassify a Chinese chapter as
        // English if allowed to decide alone.
        if (pagesSampled >= 2 && sampleTexts.length >= MIN_SAMPLE_SEGMENTS) break;
      }
      let detected = detectLanguageFromText(sampleTexts);
      console.error(`  [Language] Auto-detected from ${pagesSampled} sampled pages (${sampleIndices.length} planned, chapter has ${pageCount}): ${detected} (${sampleTexts.length} segments, ${sampleOcrErrors} OCR errors)`);

      // Korean probe. The CN pool cannot emit hangul, so on the paddleocr
      // provider a Korean chapter NEVER samples as 'korean': hangul-only pages
      // sample as 'empty', and pages with incidental latin (logos, SFX, ads)
      // sample as 'english' — the hangul is simply invisible to the sampler.
      // When either outcome occurs and a Korean pool is configured, re-sample
      // a few pages through the Korean head before trusting the result. Safe
      // for genuine English chapters: the Korean rec model reads English too,
      // so the probe returns the same latin text and detection stays english —
      // only real hangul flips it. (gemini/claude read hangul directly and
      // never reach this probe path with 'empty'/'english' + hangul content.)
      // Third disguise — thin 'chinese': a Korean chapter can leak a few
      // high-confidence han misreads (observed: 早10% @0.96), enough to tip a
      // nearly-empty sample into a 'chinese' verdict but never enough to build
      // a rich one. Genuine Chinese samples carry full dialogue lines (20+ han
      // chars even when sparse); garbage leakage stays in single digits. A
      // rich 'chinese' verdict therefore skips the probe — no Korean-server
      // coupling for normal Chinese traffic.
      const THIN_CHINESE_HAN_CHARS = 15;
      let sampleHanChars = 0;
      for (const t of sampleTexts) for (const ch of t) if (isHanChar(ch)) sampleHanChars++;
      const koProbeEligible =
        (detected === 'empty' && sampleOcrErrors === 0) ||
        detected === 'english' ||
        (detected === 'chinese' && sampleHanChars < THIN_CHINESE_HAN_CHARS);
      if (koProbeEligible && (options.visionProvider || 'paddleocr') === 'paddleocr' && hasKoreanOCRPool()) {
        console.error(`  [Language] Korean probe firing (verdict=${detected}${detected === 'chinese' ? `, thin: ${sampleHanChars} han chars` : ''})`);
        // Per-page timeout: the probe runs on EVERY auto-detected-english
        // chapter, and extractTextFromImage retries with long backoff — a dead
        // Korean pool must not stall English chapters for minutes. 15s cap per
        // page bounds the worst case at ~45s; a timed-out probe just means
        // detection keeps its original verdict (pre-probe behavior).
        // Note: extractTextFromImage applies the standard 0.55 confidence gate,
        // so the flimsiest hangul misreads never reach the proportion count —
        // surviving garbage stays ~5% of letters vs the 20% flip threshold.
        const KO_PROBE_TIMEOUT_MS = 15000;
        const koTexts: string[] = [];
        for (const idx of sampleIndices.slice(0, 3)) {
          try {
            const probeCall = extractTextFromImage(imageBuffers[idx], idx + 1, 'paddleocr', 'korean');
            probeCall.catch(() => {}); // may be abandoned by the race — never surface as unhandled
            const koExtraction = await Promise.race([
              probeCall,
              new Promise<never>((_, reject) =>
                setTimeout(() => reject(new Error('Korean-probe timeout')), KO_PROBE_TIMEOUT_MS)),
            ]);
            koTexts.push(...koExtraction.regions.map(r => r.text));
          } catch (e) {
            console.error(`  [Language] Korean-probe OCR failed for page ${idx + 1}: ${e}`);
          }
        }
        if (detectLanguageFromText(koTexts) === 'korean') {
          detected = 'korean';
          console.error(`  [Language] Korean probe: hangul dominant on re-sample (${koTexts.length} segments) — detected korean`);
        }
      }

      if (detected === 'unsupported') {
        // Real text, but dominated by letters outside every supported script
        // (Thai, Cyrillic, ...). Fail before any processing happens.
        const err: any = new Error('This chapter does not appear to be in a supported language (Chinese, English, Japanese, or Korean).');
        err.code = 'UNSUPPORTED_LANGUAGE';
        throw err;
      }
      if (detected === 'empty') {
        if (sampleOcrErrors > 0 && sampleOcrErrors === sampleIndices.length) {
          // Every sample OCR call failed — infrastructure, not content.
          const err: any = new Error('Text extraction failed on every sampled page — the OCR server may be unavailable.');
          err.code = 'OCR_UNAVAILABLE';
          throw err;
        }
        // Zero readable text on pages spread through the whole chapter (and
        // the Korean probe above, if configured, found no hangul either). For
        // URL-sourced jobs fail fast instead of defaulting to chinese and
        // silently emitting an untouched chapter. Requires zero OCR errors so
        // a flaky server never masquerades as an unsupported-content rejection.
        if (options.failOnNoText && sampleOcrErrors === 0) {
          const err: any = new Error('No readable text found in this chapter — it may be textless, or in a language the automatic detection cannot read. Try selecting the input language manually.');
          err.code = 'NO_TEXT_FOUND';
          throw err;
        }
      }

      const resolvedLang: Exclude<InputLanguage, 'auto'> =
        detected === 'english' || detected === 'japanese' || detected === 'korean'
          ? detected
          : 'chinese';
      if (resolvedLang === 'chinese' && !options._langRerun) {
        // Arm the misdetection guard (checked at the end of processMangaPages):
        // page processing tallies how many blocks the all-English drop removes;
        // ~all-dropped means this detection was wrong. Only Chinese arms it —
        // for JA/KO the drop uses language-aware content counting, so a high
        // drop ratio there means mixed/edge content, not misdetection.
        options = { ...options, _langStats: { totalBlocks: 0, allEnglishDropped: 0 } };
      }
      options = { ...options, inputLanguage: resolvedLang };
      if (isTranslationSource(resolvedLang)) {
        // No opencc for translation sources: EN has nothing to convert, and
        // JA shinjitai kanji must not be run through a zh-Hant→Hans table.
        options = { ...options, convertTraditional: false };
      }
    }

    // Stitch groups sequentially. Sharp/libvips already parallelizes each
    // stitch internally over the libuv thread pool — running multiple stitches
    // concurrently just oversubscribes that pool and makes wall time worse.
    // Effective parallelism is controlled by UV_THREADPOOL_SIZE (set in
    // package.json scripts to 24 to match Computer 2's core count).
    const stitchPhaseStart = Date.now();
    const stitchTimings: Array<{ groupIdx: number; pages: number; ms: number }> = [];
    for (let groupIdx = 0; groupIdx < mergeGroups.length; groupIdx++) {
      const group = mergeGroups[groupIdx];
      const stitchOpStart = Date.now();
      if (group.length === 1) {
        const meta = await sharp(imageBuffers[group[0]]).metadata();
        processingUnits.push({
          buffer: imageBuffers[group[0]],
          originalIndices: group,
          stitchYPositions: [],
          pageHeights: [meta.height || 0],
        });
      } else {
        const groupBuffers = group.map(i => imageBuffers[i]);
        const stitchResult = await stitchPages(groupBuffers);
        processingUnits.push({
          buffer: stitchResult.buffer,
          originalIndices: group,
          stitchYPositions: stitchResult.stitchYPositions,
          pageHeights: stitchResult.pageHeights,
        });
      }
      stitchTimings.push({ groupIdx, pages: group.length, ms: Date.now() - stitchOpStart });
    }

    const stitchWallMs = Date.now() - stitchPhaseStart;
    const uvPool = process.env.UV_THREADPOOL_SIZE || '4 (default)';
    const slowest = stitchTimings.filter(t => t.pages > 1).sort((a, b) => b.ms - a.ms).slice(0, 3);
    console.error(`  [Stitch] ${mergeGroups.length} groups stitched sequentially — wall=${(stitchWallMs / 1000).toFixed(2)}s, UV_THREADPOOL_SIZE=${uvPool}`);
    if (slowest.length > 0) {
      const detail = slowest.map(t => `g${t.groupIdx}(${t.pages}p):${t.ms}ms`).join(', ');
      console.error(`  [Stitch] Slowest ops: ${detail}`);
    }
  } else {
    // No merging - process each page individually
    console.error(`  Page merging disabled, processing pages individually...`);
    for (let i = 0; i < imageBuffers.length; i++) {
      const meta = await sharp(imageBuffers[i]).metadata();
      processingUnits.push({
        buffer: imageBuffers[i],
        originalIndices: [i],
        stitchYPositions: [],
        pageHeights: [meta.height || 0],
      });
    }
  }

  globalTimingStats.stitching += Date.now() - stitchStartTime;
  console.error(`  Processing ${processingUnits.length} image(s) (${imagePaths.length} original pages)`);

  // In-memory OCR cache: stores full ExtractionResult from chapter analysis scan
  // so page processing can reuse it instead of re-running OCR
  const ocrCache = new Map<number, ExtractionResult>();

  // Check for pre-loaded chapter context file (skips chapter analysis entirely)
  const contextFile = process.env.CHAPTER_CONTEXT_FILE;
  if (contextFile && fs.existsSync(contextFile)) {
    try {
      const raw = JSON.parse(fs.readFileSync(contextFile, 'utf-8'));
      const { buildChapterDefinitions } = require('./simplifier');
      const definitions = buildChapterDefinitions(raw.characters || [], raw.keep || [], raw.glosses || {});
      const chapterContext = { ...raw, definitions };
      // KEEP-token bias override (same as fresh-analysis path)
      try {
        chapterContext.keepTokenIds = getKeepNegBiasIds(chapterContext.keep || [], options.targetLevel, options.inputLanguage || 'chinese');
        if (chapterContext.keepTokenIds.length > 0) {
          console.error(`  [Chapter] KEEP override: ${chapterContext.keepTokenIds.length} negative-biased token IDs will be zeroed at simplification time`);
        }
      } catch (e) {
        console.error(`  [Chapter] KEEP override skipped (${e})`);
      }
      options = { ...options, chapterContext };
      console.error(`  [Chapter] Loaded pre-computed context from ${contextFile}: ${raw.keep?.length || 0} KEEP terms, ${raw.characters?.length || 0} characters`);
    } catch (e) {
      console.error(`  [Chapter] Failed to load context file ${contextFile}: ${e}`);
    }
  }

  // PHASE 3: OCR scan (always) + chapter analysis (conditional)
  // OCR scan populates ocrCache so Phase 4a/4b/4c can reuse extractions
  // instead of re-running OCR per page. Chapter analysis runs in parallel
  // with OCR, but is skipped on cache hit (skipChapterAnalysis=true) — the
  // cached simplifications already encode any context we'd derive.
  let hasLLMProvider = false;
  try { detectProvider(); hasLLMProvider = true; } catch {}
  const wantChapterAnalysis = !options.chapterContext && !options.skipChapterAnalysis && hasLLMProvider;
  if (processingUnits.length > 1 && !options.analyzeOnly) {
    if (wantChapterAnalysis) {
      console.error(`  [Chapter] Running interleaved OCR scan + chapter analysis...`);
    } else {
      console.error(`  [Chapter] Running OCR scan only (chapter analysis skipped — cache hit or pre-loaded context)`);
    }
    if (options.onProgress) {
      options.onProgress(0, processingUnits.length, wantChapterAnalysis ? 'Reading and analyzing text...' : 'Reading text...');
    }
    const scanStartTime = Date.now();
    const scanLimit = pLimit(ocrConcurrency);

    // Ordered OCR results array — filled as pages complete (may be out of order)
    // Store both text (for chapter analysis) and full extraction (for reuse in page processing)
    const ocrResults: Array<{ pageNum: number; text: string } | undefined> = new Array(processingUnits.length);
    let scannedCount = 0;
    let nonEmptyScanned = 0;

    // Event notification for OCR completion
    const ocrWaiters: Set<() => void> = new Set();
    function notifyOcrComplete() {
      for (const resolve of ocrWaiters) resolve();
      ocrWaiters.clear();
    }
    function waitForOcrProgress(): Promise<void> {
      return new Promise(resolve => ocrWaiters.add(resolve));
    }

    // OCR scan (producer) — runs all pages in parallel with ocrConcurrency limit
    const ocrPromises = processingUnits.map((unit, index) =>
      scanLimit(async () => {
        try {
          let extraction = await extractTextFromImage(
            unit.buffer,
            index + 1,
            options.visionProvider || 'paddleocr',
            options.inputLanguage
          );
          // Convert traditional to simplified for consistent text
          if (options.convertTraditional !== false) {
            extraction = convertExtractionToSimplified(extraction);
            extraction = stripUrlTokensFromExtraction(extraction);
          }
          const isEnglish = options.inputLanguage === 'english';
          const meta = await sharp(unit.buffer).metadata();
          const imageAspectRatio = (meta.height || 1) / (meta.width || 1);
          const filtered = filterRegionsForAnalysis(extraction.regions, isEnglish, imageAspectRatio);
          const text = filtered.map(r => r.text).join('\n');
          ocrResults[index] = { pageNum: index + 1, text };
          ocrCache.set(index, extraction); // Cache full unmodified extraction for page processing
          scannedCount++;
          if (text.length > 0) nonEmptyScanned++;
        } catch (e) {
          console.error(`  [Chapter] OCR scan failed for unit ${index + 1}: ${e}`);
          ocrResults[index] = { pageNum: index + 1, text: '' };
          scannedCount++;
        }
        notifyOcrComplete();
      })
    );

    // Chunk provider (consumer) — iterates pages in order, buffers non-empty results,
    // and yields chunks when enough non-empty pages are accumulated
    let nextConsumeIndex = 0;
    // Initial chunk target estimate; scale by provider (cloud models have larger context windows)
    const totalUnitsForAnalysis = processingUnits.length;
    let currentProvider: string;
    try { currentProvider = detectProvider(); } catch { currentProvider = 'qwen'; }
    const isLocalProvider = currentProvider === 'qwen';
    const targetTurns = isLocalProvider ? 5 : 2;
    // Floor: a short chapter analyzes in a SINGLE content turn rather than being
    // chopped up — splitting a short chapter buys nothing and gives the model
    // less context per turn. Cloud models have huge context windows so the floor
    // is generous (≤30-page chapters = 1 content turn); the local model's small
    // window keeps a low floor.
    const chunkFloor = isLocalProvider ? 10 : 30;
    let chunkTarget = Math.max(chunkFloor, Math.ceil(totalUnitsForAnalysis / targetTurns));
    let chunkTargetRefined = false;

    async function getNextChunk(): Promise<{ pages: { pageNum: number; text: string }[]; isLast: boolean } | null> {
      const chunk: { pageNum: number; text: string }[] = [];

      while (nextConsumeIndex < totalUnitsForAnalysis) {
        // Wait for this page's OCR to complete
        while (ocrResults[nextConsumeIndex] === undefined) {
          await waitForOcrProgress();
        }

        const result = ocrResults[nextConsumeIndex]!;
        nextConsumeIndex++;

        if (result.text.length > 0) {
          chunk.push(result);
        }

        // Refine chunk target after ~40% of pages scanned (better text ratio estimate)
        if (!chunkTargetRefined && scannedCount >= totalUnitsForAnalysis * 0.4 && nonEmptyScanned > 0) {
          const textRatio = nonEmptyScanned / scannedCount;
          const estimatedNonEmpty = Math.ceil(totalUnitsForAnalysis * textRatio);
          chunkTarget = Math.max(chunkFloor, Math.ceil(estimatedNonEmpty / targetTurns));
          chunkTargetRefined = true;
          console.error(`  [Chapter] Chunk target refined: ${chunkTarget} non-empty pages per chunk (textRatio=${textRatio.toFixed(2)})`);
        }

        // Send chunk when we have enough non-empty pages (and there are more pages to come)
        if (chunk.length >= chunkTarget && nextConsumeIndex < totalUnitsForAnalysis) {
          return { pages: chunk, isLast: false };
        }
      }

      // All pages consumed — return remaining as final chunk
      return chunk.length > 0 ? { pages: chunk, isLast: true } : null;
    }

    // Run OCR and chapter analysis concurrently
    const analysisStartTime = Date.now();
    const ocrDonePromise = Promise.all(ocrPromises).then(() => {
      const ocrScanTime = Date.now() - scanStartTime;
      globalTimingStats.ocrScan += ocrScanTime;
      const totalNonEmpty = ocrResults.filter(r => r && r.text.length > 0).length;
      console.error(`  [Chapter] OCR scan complete: ${totalNonEmpty}/${totalUnitsForAnalysis} pages with text (${(ocrScanTime / 1000).toFixed(1)}s)`);
    });

    if (wantChapterAnalysis) {
      const analysisPromise = analyzeChapterStreaming(getNextChunk, options.targetLevel, chunkTarget, options.inputLanguage);

      // Wait for both OCR scan and chapter analysis to finish
      const [, chapterContext] = await Promise.all([ocrDonePromise, analysisPromise]);
      const totalPhase3Time = Date.now() - scanStartTime;
      // chapterAnalysis = extra time beyond OCR (the non-overlapping portion)
      globalTimingStats.chapterAnalysis += Math.max(0, totalPhase3Time - globalTimingStats.ocrScan);
      console.error(`  [Chapter] Phase 3 complete (${(totalPhase3Time / 1000).toFixed(1)}s wall time): OCR + analysis interleaved`);

      if (chapterContext.keep.length > 0 || chapterContext.characters.length > 0) {
        console.error(`  [Chapter] Analysis result: ${chapterContext.keep.length} KEEP terms, ${chapterContext.characters.length} characters`);
        // KEEP-token bias override: compute which bias-map token IDs to zero
        // so the simplifier's "PROTECTED TERMS" rule isn't fought by the
        // unified bias map's negative entries on HSK 4+ tokens (e.g. 猎人).
        try {
          chapterContext.keepTokenIds = getKeepNegBiasIds(chapterContext.keep, options.targetLevel, options.inputLanguage || 'chinese');
          if (chapterContext.keepTokenIds.length > 0) {
            console.error(`  [Chapter] KEEP override: ${chapterContext.keepTokenIds.length} negative-biased token IDs will be zeroed at simplification time`);
          }
        } catch (e) {
          console.error(`  [Chapter] KEEP override skipped (${e})`);
        }
        options = { ...options, chapterContext };
      }

      // A/B test support: dump chapter context to a file so subsequent runs
      // can load it via CHAPTER_CONTEXT_FILE and skip re-analysis. Strips the
      // `definitions` Map (not JSON-serializable; loader rebuilds it from keep + glosses).
      const dumpFile = process.env.CHAPTER_CONTEXT_DUMP_FILE;
      if (dumpFile) {
        try {
          const { definitions, ...serializable } = chapterContext;
          fs.writeFileSync(dumpFile, JSON.stringify(serializable, null, 2), 'utf-8');
          console.error(`  [Chapter] Dumped context to ${dumpFile}`);
        } catch (e) {
          console.error(`  [Chapter] Failed to dump context to ${dumpFile}: ${e}`);
        }
      }
    } else {
      // OCR-only path: just wait for the scan to finish.
      await ocrDonePromise;
      const totalPhase3Time = Date.now() - scanStartTime;
      console.error(`  [Chapter] Phase 3 complete (${(totalPhase3Time / 1000).toFixed(1)}s wall time): OCR scan only`);
    }
  }

  // Determine if we should use batched simplification (cloud providers benefit from fewer calls)
  const useBatchedSimplification = !options.analyzeOnly && processingUnits.length > 1 && hasLLMProvider;

  const limit = pLimit(concurrency);
  const totalUnits = processingUnits.length;
  let completedCount = 0;

  if (useBatchedSimplification) {
    // === BATCHED MODE: Phase 4a → 4b → 4c ===

    // Phase 4a: OCR + filter + group blocks (parallel, no LLM, no inpainting)
    console.error(`  [Batched] Phase 4a: OCR + block grouping (${totalUnits} pages)...`);
    const phase4aOptions = { ...options, skipSimplification: true };
    const phase4aPromises = processingUnits.map((unit, index) =>
      limit(async () => {
        try {
          const result = await processMangaPageFromBuffer(
            unit.buffer, index + 1, unit.originalIndices, phase4aOptions,
            unit.originalIndices.length, unit.stitchYPositions,
            ocrCache.get(index)
          );
          completedCount++;
          if (options.onProgress) {
            options.onProgress(completedCount, totalUnits * 2, `Scanning pages ${completedCount}/${totalUnits}...`);
          }
          return { result, index };
        } catch (err) {
          console.error(`  Error scanning unit ${index + 1}: ${err}`);
          completedCount++;
          return {
            result: { pageNumber: index + 1, originalText: '', simplifiedText: '', regions: [], simplifiedRegions: [], blocksToSimplify: [], simplifyBlockToOrigBlock: [] } as PageProcessingResult,
            index,
          };
        }
      })
    );
    const phase4aResults = await Promise.all(phase4aPromises);
    phase4aResults.sort((a, b) => a.index - b.index);

    // Phase 4b: Batched simplification — pages split into block-count-balanced
    // batches, 1 LLM call per batch, all batches run in parallel
    console.error(`  [Batched] Phase 4b: Batched simplification...`);
    const batchStartTime = Date.now();

    // Collect blocks from all pages
    const allPageBlocks = new Map<number, TextBlock[]>(); // pageNum → blocks
    const pageBlockMapping = new Map<number, number[]>(); // pageNum → simplifyBlockToOrigBlock
    const pageBlockBboxes = new Map<number, Array<{ x: number; y: number; width: number; height: number }>>(); // pageNum → bbox per simplify-block
    for (const { result, index } of phase4aResults) {
      const pageNum = index + 1;
      if (result.blocksToSimplify && result.blocksToSimplify.length > 0) {
        allPageBlocks.set(pageNum, result.blocksToSimplify);
        pageBlockMapping.set(pageNum, result.simplifyBlockToOrigBlock || []);
        pageBlockBboxes.set(pageNum, result.blocksToSimplifyBboxes || []);
      }
    }

    // Split pages into batches balanced by BLOCK COUNT (not section), so every
    // batch is roughly equal work and the parallel LLM calls finish together —
    // wall time drops to the slowest *sub-batch* instead of the slowest section.
    // Pages are kept whole and batches stay contiguous page ranges. Plot-section
    // summaries are then attached by page-range overlap (a batch inside one
    // section gets that summary; one straddling a boundary gets both) — so batch
    // size is fully decoupled from chapter-analysis section boundaries.
    const totalBlocksForBatching = [...allPageBlocks.values()].reduce((s, b) => s + b.length, 0);
    const baseTarget = Math.max(1, parseInt(process.env.SIMPLIFY_BATCH_TARGET_BLOCKS || '20', 10));
    const maxBatches = Math.max(1, parseInt(process.env.SIMPLIFY_MAX_BATCHES || '16', 10));
    // Raise the target if needed so the batch count never exceeds maxBatches
    // (caps parallel LLM requests to avoid provider rate-limiting).
    const targetBlocksPerBatch = Math.max(baseTarget, Math.ceil(totalBlocksForBatching / maxBatches));

    const batches: Array<{ pages: Map<number, TextBlock[]>; summary: string }> = [];
    if (totalBlocksForBatching < 2 * targetBlocksPerBatch) {
      // Short chapter — not enough material for two full-size batches. Splitting
      // here buys no meaningful parallelism and costs cross-block context, so
      // simplify the whole chapter in a single LLM call.
      console.error(`  [Batched] Short chapter (${totalBlocksForBatching} blocks) — single batch`);
      batches.push({ pages: new Map(allPageBlocks), summary: '' });
    } else {
      let current = new Map<number, TextBlock[]>();
      let currentBlocks = 0;
      for (const pageNum of [...allPageBlocks.keys()].sort((a, b) => a - b)) {
        const blocks = allPageBlocks.get(pageNum)!;
        current.set(pageNum, blocks);
        currentBlocks += blocks.length;
        if (currentBlocks >= targetBlocksPerBatch) {
          batches.push({ pages: current, summary: '' });
          current = new Map();
          currentBlocks = 0;
        }
      }
      // Fold a tiny leftover tail into the previous batch rather than spawn a
      // near-empty LLM call (its ~550-token fixed overhead would be wasted).
      if (current.size > 0) {
        if (currentBlocks < targetBlocksPerBatch / 2 && batches.length > 0) {
          for (const [p, b] of current) batches[batches.length - 1].pages.set(p, b);
        } else {
          batches.push({ pages: current, summary: '' });
        }
      }
    }

    // Attach plot-section summaries by page-range overlap. The summary is soft
    // narrative context for the simplifier — a batch overlapping two sections
    // simply gets both summaries joined (~40 tokens each, negligible).
    if (options.chapterContext?.plot && options.chapterContext.plot.length > 0) {
      const sections = options.chapterContext.plot
        .map(s => {
          const m = s.pages.match(/(\d+)-(\d+)/);
          return m ? { start: parseInt(m[1]), end: parseInt(m[2]), summary: s.summary } : null;
        })
        .filter((s): s is { start: number; end: number; summary: string } => s !== null);
      for (const batch of batches) {
        const pageNums = [...batch.pages.keys()];
        const minPage = Math.min(...pageNums);
        const maxPage = Math.max(...pageNums);
        const overlapping = sections
          .filter(s => s.start <= maxPage && s.end >= minPage && s.summary)
          .map(s => s.summary);
        batch.summary = [...new Set(overlapping)].join(' ');
      }
    }

    console.error(`  [Batched] ${batches.length} batches (~${targetBlocksPerBatch} blocks/batch, ${totalBlocksForBatching} total), ${allPageBlocks.size} pages with blocks`);

    // Phase 4b runs LLM simplification AND per-page inpainting concurrently.
    // The two phases are independent: inpaint depends only on OCR bounding boxes
    // (which are already known after phase 4a), while simplification depends on
    // block text. Running them in parallel removes the LLM↔inpaint sequential bar.
    const phase4bWallStart = Date.now();
    const llmStartedAt = Date.now();
    let llmFinishedAt = 0;

    // Advance the UI stepper to "Simplifying & cleaning" the moment Phase 4b
    // begins — LLM simplification and inpaint cleaning both run in this phase.
    // Without this the message stays at 4a's "Scanning…" and the stepper sits
    // on the previous step for the whole (longest) phase.
    if (options.onProgress) {
      options.onProgress(totalUnits, totalUnits * 2, `Simplifying and cleaning pages 0/${totalUnits}...`);
    }

    // ---- LLM task: parallel batches (block-count balanced) ----
    // CACHE SHORT-CIRCUIT: when cachedSimplifications is set, skip the LLM
    // entirely and build batchedResults by bbox-matching cached blocks to
    // fresh blocks. Saves ~30s + ~$0.04 per chapter on cache hit.
    const llmTask = (async () => {
      if (options.cachedSimplifications) {
        const { matchCachedBlocks } = require('./services/cacheService');
        const batchedResults = new Map<number, Map<number, string>>();
        const cachePagesByNum = new Map<number, Array<{ bbox: { x: number; y: number; width: number; height: number }; simplifiedText: string }>>();
        for (const p of options.cachedSimplifications.pages) {
          cachePagesByNum.set(p.pageNumber, p.blocks);
        }
        let totalMatched = 0;
        let totalFresh = 0;
        for (const [pageNum, freshBboxes] of pageBlockBboxes) {
          const cachedBlocks = cachePagesByNum.get(pageNum) || [];
          const matched: Map<number, string> = matchCachedBlocks(freshBboxes, cachedBlocks);
          batchedResults.set(pageNum, matched);
          totalMatched += matched.size;
          totalFresh += freshBboxes.length;
        }
        llmFinishedAt = Date.now();
        console.error(`  [Batched|LLM] CACHE HIT — skipped LLM, bbox-matched ${totalMatched}/${totalFresh} blocks (${((totalMatched / Math.max(totalFresh, 1)) * 100).toFixed(0)}%)`);
        return batchedResults;
      }

      console.error(`  [Batched|LLM] Started at +${((llmStartedAt - phase4bWallStart) / 1000).toFixed(2)}s`);
      const batchedResults = new Map<number, Map<number, string>>(); // pageNum → blockIdx → text
      const batchPromises = batches.map((batch, i) => {
        const pageRange = [...batch.pages.keys()].sort((a, b) => a - b);
        const startedAt = Date.now();
        console.error(`  [Batched|LLM] Batch ${i + 1}/${batches.length}: pages ${pageRange[0]}-${pageRange[pageRange.length - 1]} (${batch.pages.size} pages) started`);
        return simplifyPageBatch(
          batch.pages, options.targetLevel, options.chapterContext, batch.summary, options.inputLanguage
        ).then(results => {
          const elapsed = ((Date.now() - startedAt) / 1000).toFixed(1);
          console.error(`  [Batched|LLM] Batch ${i + 1}/${batches.length}: pages ${pageRange[0]}-${pageRange[pageRange.length - 1]} done in ${elapsed}s`);
          return results;
        });
      });

      let allResults: Array<Map<number, Map<number, string>>>;
      try {
        allResults = await Promise.all(batchPromises);
      } catch (e: any) {
        const msg = e.message || String(e);
        const isRateLimit = msg.includes('too_many_requests') || msg.includes('queue_exceeded') || msg.includes('429');
        const userMsg = isRateLimit
          ? 'The AI translation service is currently overloaded. Please try again in a few minutes.'
          : `AI translation service error: ${msg}`;
        const err: any = new Error(userMsg);
        // Preserve a typed error code so the job surfaces a specific, friendly
        // message (LLM_TIMEOUT → "taking too long, try again in a minute")
        // instead of the generic INTERNAL. callLLMMultiTurn already tags
        // timeouts with 'LLM_TIMEOUT'; a provider rate-limit/overload gets the
        // same code since the user advice — retry shortly — is identical.
        err.code = e.code || (isRateLimit || /timed out/i.test(msg) ? 'LLM_TIMEOUT' : undefined);
        throw err;
      }
      for (const results of allResults) {
        for (const [pageNum, blockMap] of results) {
          batchedResults.set(pageNum, blockMap);
        }
      }
      llmFinishedAt = Date.now();
      const llmElapsed = ((llmFinishedAt - llmStartedAt) / 1000).toFixed(1);
      console.error(`  [Batched|LLM] Done at +${((llmFinishedAt - phase4bWallStart) / 1000).toFixed(2)}s (took ${llmElapsed}s) — ${batchedResults.size} pages`);
      return batchedResults;
    })();

    // ---- Inpaint task: per-page in parallel, runs alongside LLM ----
    const inpaintStartedAt = Date.now();
    let inpaintFinishedAt = 0;
    const inpaintTask = (async () => {
      console.error(`  [Batched|Inpaint] Started at +${((inpaintStartedAt - phase4bWallStart) / 1000).toFixed(2)}s — inpainting ${totalUnits} pages in parallel with LLM`);
      const cleanedBuffers = new Map<number, Buffer>(); // pageNum → cleaned buffer
      const inpaintOptions = { ...options, inpaintOnly: true };
      let inpaintCompleted = 0;
      const promises = processingUnits.map((unit, index) =>
        limit(async () => {
          const pageNum = index + 1;
          try {
            const result = await processMangaPageFromBuffer(
              unit.buffer, pageNum, unit.originalIndices, inpaintOptions,
              unit.originalIndices.length, unit.stitchYPositions,
              ocrCache.get(index)
            );
            if (result.cleanedImageBuffer) {
              cleanedBuffers.set(pageNum, result.cleanedImageBuffer);
            }
            inpaintCompleted++;
            if (options.onProgress) {
              options.onProgress(totalUnits, totalUnits * 2, `Simplifying and cleaning pages ${inpaintCompleted}/${totalUnits}...`);
            }
            if (inpaintCompleted % 20 === 0 || inpaintCompleted === totalUnits) {
              console.error(`  [Batched|Inpaint] Progress ${inpaintCompleted}/${totalUnits}`);
            }
          } catch (err) {
            console.error(`  [Batched|Inpaint] Page ${pageNum} failed: ${err}`);
          }
        })
      );
      await Promise.all(promises);
      inpaintFinishedAt = Date.now();
      const inpaintElapsed = ((inpaintFinishedAt - inpaintStartedAt) / 1000).toFixed(1);
      console.error(`  [Batched|Inpaint] Done at +${((inpaintFinishedAt - phase4bWallStart) / 1000).toFixed(2)}s (took ${inpaintElapsed}s) — ${cleanedBuffers.size}/${totalUnits} cleaned`);
      return cleanedBuffers;
    })();

    const [batchedResults, cleanedBuffers] = await Promise.all([llmTask, inpaintTask]);

    const batchTime = Date.now() - batchStartTime;
    globalTimingStats.llm += llmFinishedAt > 0 ? (llmFinishedAt - llmStartedAt) : batchTime;

    // Verification log: did the two phases actually overlap?
    const llmDurMs = llmFinishedAt - llmStartedAt;
    const inpaintDurMs = inpaintFinishedAt - inpaintStartedAt;
    const wallDurMs = Date.now() - phase4bWallStart;
    const sequentialEstimate = llmDurMs + inpaintDurMs;
    const overlapMs = Math.max(0, sequentialEstimate - wallDurMs);
    const overlapPct = sequentialEstimate > 0 ? ((overlapMs / sequentialEstimate) * 100).toFixed(0) : '0';
    console.error(`  [Batched] Phase 4b wall=${(wallDurMs / 1000).toFixed(1)}s | LLM=${(llmDurMs / 1000).toFixed(1)}s + Inpaint=${(inpaintDurMs / 1000).toFixed(1)}s (sequential would be ${(sequentialEstimate / 1000).toFixed(1)}s) | overlap=${(overlapMs / 1000).toFixed(1)}s (${overlapPct}%)`);

    // Phase 4c: Apply results + render (uses precomputed cleaned images, skips inpaint)
    console.error(`  [Batched] Phase 4c: Rendering text + saving (${totalUnits} pages)...`);
    completedCount = 0;
    const phase4cPromises = processingUnits.map((unit, index) =>
      limit(async () => {
        const pageNum = index + 1;
        try {
          // Convert batched results (blockIdx in simplifyBlocks space) to original block index space
          const precomputed = new Map<number, string>();
          const batchResult = batchedResults.get(pageNum);
          const mapping = pageBlockMapping.get(pageNum) || [];
          if (batchResult) {
            for (const [simplifyIdx, text] of batchResult) {
              const origIdx = mapping[simplifyIdx] ?? simplifyIdx;
              precomputed.set(origIdx, text);
            }
          }

          const phase4cOptions = {
            ...options,
            precomputedBlockResults: precomputed,
            precomputedCleanedImage: cleanedBuffers.get(pageNum),
          };
          const result = await processMangaPageFromBuffer(
            unit.buffer, pageNum, unit.originalIndices, phase4cOptions,
            unit.originalIndices.length, unit.stitchYPositions,
            ocrCache.get(index)
          );
          completedCount++;
          if (options.onProgress) {
            options.onProgress(totalUnits + completedCount, totalUnits * 2, `Rendering pages ${completedCount}/${totalUnits}...`);
          } else {
            console.log(`  Progress: ${completedCount}/${totalUnits} images completed`);
          }
          return { result, index };
        } catch (err) {
          console.error(`  Error processing unit ${pageNum}: ${err}`);
          completedCount++;
          return {
            result: { pageNumber: pageNum, originalText: '', simplifiedText: '', regions: [], simplifiedRegions: [] } as PageProcessingResult,
            index,
          };
        }
      })
    );
    const phase4cResults = await Promise.all(phase4cPromises);
    phase4cResults.sort((a, b) => a.index - b.index);
    for (const { result } of phase4cResults) {
      pages.push(result);
    }

    // Build cacheBlocks (text-only) for cache write by jobService.
    // Each entry: { pageNumber, blocks: [{bbox, simplifiedText}] }.
    // Skip blocks whose LLM result is missing OR identical to the original
    // (no point caching a no-op simplification).
    pipelineCacheBlocks = [];
    for (const [pageNum, freshBboxes] of pageBlockBboxes) {
      const batchResult = batchedResults.get(pageNum);
      if (!batchResult || batchResult.size === 0) continue;
      const origBlocks = allPageBlocks.get(pageNum) || [];
      const blocks: Array<{ bbox: { x: number; y: number; width: number; height: number }; simplifiedText: string }> = [];
      freshBboxes.forEach((bbox, simplifyIdx) => {
        const text = batchResult.get(simplifyIdx);
        if (text === undefined || text === '') return;
        const orig = origBlocks[simplifyIdx];
        const origText = orig ? smartJoin(orig.texts) : '';
        if (text === origText) return; // no-op; not worth caching
        blocks.push({ bbox, simplifiedText: text });
      });
      if (blocks.length > 0) {
        pipelineCacheBlocks.push({ pageNumber: pageNum, blocks });
      }
    }
  } else {
    // === LEGACY MODE: Per-page simplification (single pages, analyze-only, or no LLM) ===
    const pagePromises = processingUnits.map((unit, index) =>
      limit(async () => {
        try {
          const result = await processMangaPageFromBuffer(
            unit.buffer, index + 1, unit.originalIndices, options,
            unit.originalIndices.length, unit.stitchYPositions,
            ocrCache.get(index)
          );
          completedCount++;
          if (options.onProgress) {
            options.onProgress(completedCount, totalUnits, `Simplifying and cleaning pages ${completedCount}/${totalUnits}...`);
          } else {
            console.log(`  Progress: ${completedCount}/${totalUnits} images completed`);
          }
          return { result, index };
        } catch (err) {
          console.error(`  Error processing unit ${index + 1}: ${err}`);
          completedCount++;
          if (options.onProgress) {
            options.onProgress(completedCount, totalUnits, `Simplifying and cleaning pages ${completedCount}/${totalUnits}...`);
          } else {
            console.log(`  Progress: ${completedCount}/${totalUnits} images completed`);
          }
          return {
            result: { pageNumber: index + 1, originalText: '', simplifiedText: '', regions: [], simplifiedRegions: [] } as PageProcessingResult,
            index,
          };
        }
      })
    );
    const results = await Promise.all(pagePromises);
    results.sort((a, b) => a.index - b.index);
    for (const { result } of results) {
      pages.push(result);
    }
  }

  const totalRegions = pages.reduce((sum, p) => sum + p.regions.length, 0);
  const processingTime = (Date.now() - startTime) / 1000;

  let combinedOutputPath: string | undefined;

  // Generate interactive HTML viewer (unless separateFiles is true or analyzeOnly)
  if (options.outputDir && !options.analyzeOnly && !options.separateFiles) {
    // Every page image must be on disk before the reader references it.
    await assertAllImageSavesSucceeded();
    const hasProcessedPages = pages.some((p) => p.cleanedImagePath || p.originalImagePath);

    if (hasProcessedPages) {
      console.error('  Generating interactive HTML viewer...');
      try {
        // Generate combined output filename from first input file
        const firstInputName = path.basename(imagePaths[0], path.extname(imagePaths[0]));
        // Remove page number suffix if present (e.g., "test2-1" -> "test2")
        const baseName = firstInputName.replace(/-\d+$/, '');
        // Prefer the resolved chapter title (browser tab); the filename-derived
        // name is only a last-ditch fallback for direct/CLI runs with no title.
        const title = options.readerTitle || `${baseName} - Simplified`;

        // Build reader data structure with merged bubble regions
        // Use a reference dimension for grouping (percentages scale proportionally)
        const refDimension = 1000;

        // Build available levels list for multi-level reader
        const availableLevels: string[] = ['original'];
        if (options.targetLevels && options.targetLevels.length > 1) {
          availableLevels.push(...options.targetLevels);
        } else {
          availableLevels.push(options.targetLevel);
        }
        if (options.includeEnglish) {
          availableLevels.push('english');
        }

        const readerData: ReaderData = {
          title,
          createdAt: new Date().toISOString(),
          targetLevel: options.targetLevel,
          availableLevels: availableLevels.length > 2 ? availableLevels : undefined,
          // Propagate chapter-analysis character map so the reader's
          // "show names in English" toggle can swap deterministically.
          // Requiring BOTH name and chineseName shipped an empty array for every
          // Chinese-source chapter: the analysis puts the Chinese surface form in
          // `name` and omits `chineseName` entirely, so the filter dropped
          // everything and the reader's toggle was a permanent no-op. Accept an
          // entry with either, and normalise the missing side — for a CN chapter
          // `name` IS the Chinese form, so mirroring it keeps the map keyed on
          // what the reader looks up.
          characters: options.chapterContext?.characters
            ?.filter((c: any) => c && c.role && (c.chineseName || c.name))
            ?.map((c: any) => ({
              name: c.name || c.chineseName,
              chineseName: c.chineseName || c.name,
              role: c.role,
            })),
          // Chapter-specific Chinese terms the names toggle must NEVER
          // dictionary-swap: every character's Chinese surface form (chineseName
          // for EN input, the Chinese `name` for CN input) plus all KEEP terms
          // (invented places, skills, concepts). These are protected because we
          // can't reliably romanize invented names — only genuine dictionary
          // proper nouns (real cities) get swapped. See resolveEnglishName.
          protectedNames: options.chapterContext
            ? Array.from(new Set([
                ...(options.chapterContext.characters || [])
                  .map((c: any) => c.chineseName || c.name)
                  .filter(Boolean),
                ...(options.chapterContext.keep || []),
              ]))
            : undefined,
          pages: pages.map((page, index) => {
            // Get just the filename from the full path
            const originalFileName = page.originalImagePath ? path.basename(page.originalImagePath) : `page_${index + 1}.png`;
            const cleanedFileName = page.cleanedImagePath ? path.basename(page.cleanedImagePath) : originalFileName;

            // Build rendered block data maps for reader overlays
            const renderedBlockBboxes = new Map<number, { x: number; y: number; width: number; height: number }>();
            const renderedBlockData = new Map<number, RenderedBlockInfo>();
            if (page.renderedBlocks) {
              for (const rb of page.renderedBlocks) {
                renderedBlockBboxes.set(rb.blockId, rb.bbox);
                renderedBlockData.set(rb.blockId, rb);
              }
            }

            // Merge nearby regions into bubbles for better click targets
            const mergedBubbles = mergeBubbleRegions(
              page.simplifiedRegions,
              refDimension,
              refDimension,
              renderedBlockBboxes.size > 0 ? renderedBlockBboxes : undefined
            );

            // Debug: DEBUG_REGION_MERGE dumps the input→output ratio per page so we
            // can spot when N input regions produce N output bubbles instead of the
            // expected M<<N. Also dumps identical-bbox clusters (see groupNearbyRegions).
            if (process.env.DEBUG_REGION_MERGE) {
              // Group input regions by identical bbox to spot duplicates
              const bboxKey = (b: { x: number; y: number; width: number; height: number }) =>
                `${b.x.toFixed(3)},${b.y.toFixed(3)},${b.width.toFixed(3)},${b.height.toFixed(3)}`;
              const inputClusters = new Map<string, number>();
              for (const r of page.simplifiedRegions) {
                const key = bboxKey(r.original.boundingBox);
                inputClusters.set(key, (inputClusters.get(key) || 0) + 1);
              }
              const outputClusters = new Map<string, number>();
              for (const b of mergedBubbles) {
                const key = bboxKey(b.bbox);
                outputClusters.set(key, (outputClusters.get(key) || 0) + 1);
              }
              const inputDupes = [...inputClusters.entries()].filter(([_, n]) => n > 1).length;
              const outputDupes = [...outputClusters.entries()].filter(([_, n]) => n > 1).length;
              console.error(`  [DEBUG_REGION_MERGE] Page ${page.pageNumber}: ${page.simplifiedRegions.length} in → ${mergedBubbles.length} out (input dup-bbox clusters: ${inputDupes}, output dup-bbox clusters: ${outputDupes})`);
              if (outputDupes > 0) {
                for (const [key, n] of outputClusters) {
                  if (n <= 1) continue;
                  const b = mergedBubbles.find(mb => bboxKey(mb.bbox) === key);
                  console.error(`    OUTPUT DUP × ${n}: text="${(b?.originalText || '').slice(0, 60)}"`);
                }
              }
            }

            // Build block-level reader data: phrase tokenization + rendered block metadata
            const blockRegions: ReaderPageData['blocks'] = [];
            if (page.renderedBlocks) {
              for (const rb of page.renderedBlocks) {
                if (!rb.text) continue;
                const blockEntry: ReaderBlockData = {
                  blockId: rb.blockId,
                  bbox: rb.bbox,
                  text: rb.text,
                  isVertical: rb.isVertical || false,
                  fontSize: rb.fontSize || 16,
                  textColor: rb.textColor || '#000000',
                  rotation: rb.rotation || 0,
                  minFontSize: rb.minFontSize,
                  phrases: buildPhraseData(rb.text, options.targetLevel, options.chapterContext?.definitions, options.chapterContext?.keep),
                };

                // Build multi-level phrase data if available
                if (page.multiLevelBlockResults) {
                  const levelMap = page.multiLevelBlockResults.get(rb.blockId);
                  if (levelMap) {
                    const textByLevel: { [level: string]: string } = {};
                    const phrasesByLevel: { [level: string]: PhraseData[] } = {};
                    for (const [levelKey, rawLevelText] of levelMap) {
                      // Apply traditional→simplified conversion and compact punctuation to Chinese level texts
                      const levelText = levelKey === 'english' ? compactAdjacentPunctuation(rawLevelText) : compactPunctuation(traditionalToSimplified(rawLevelText));
                      textByLevel[levelKey] = levelText;
                      if (levelKey === 'english') {
                        // English: single phrase, no pinyin/HSK analysis
                        phrasesByLevel[levelKey] = [{
                          text: levelText,
                          pinyin: '',
                          isAboveLevel: false,
                          hskLevel: null,
                        }];
                      } else {
                        // Chinese HSK level: full tokenization + phrase data
                        phrasesByLevel[levelKey] = buildPhraseData(
                          levelText, levelKey as HSKLevel, options.chapterContext?.definitions, options.chapterContext?.keep
                        );
                      }
                    }
                    blockEntry.textByLevel = textByLevel;
                    blockEntry.phrasesByLevel = phrasesByLevel;
                  }
                }

                blockRegions.push(blockEntry);
              }
            }

            return {
              pageNumber: page.pageNumber,
              originalImage: originalFileName,
              cleanedImage: cleanedFileName,
              imageWidth: page.imageWidth,
              imageHeight: page.imageHeight,
              regions: mergedBubbles.map(bubble => ({
                bbox: bubble.bbox,
                originalText: bubble.originalText,
                simplifiedText: bubble.simplifiedText,
                originalPinyin: bubble.originalPinyin,
                simplifiedPinyin: bubble.simplifiedPinyin,
              })),
              blocks: blockRegions,
            };
          }),
        };

        // Merge cross-page bubble continuations
        // Detect bottom-edge regions on page N and top-edge regions on page N+1, merge by X overlap
        for (let pageIdx = 0; pageIdx < readerData.pages.length - 1; pageIdx++) {
          const currentPage = readerData.pages[pageIdx];
          const nextPage = readerData.pages[pageIdx + 1];

          // Find bottom-edge regions on current page (y + height > 97%)
          const bottomEdgeIndices: number[] = [];
          currentPage.regions.forEach((r, i) => {
            if (r.bbox.y + r.bbox.height > 97) {
              bottomEdgeIndices.push(i);
            }
          });

          // Find top-edge regions on next page (y < 3%)
          const topEdgeIndices: number[] = [];
          nextPage.regions.forEach((r, i) => {
            if (r.bbox.y < 3) {
              topEdgeIndices.push(i);
            }
          });

          // Match pairs by X position overlap
          const mergedTopIndices = new Set<number>();
          for (const bottomIdx of bottomEdgeIndices) {
            const bottomRegion = currentPage.regions[bottomIdx];
            const bottomLeft = bottomRegion.bbox.x;
            const bottomRight = bottomRegion.bbox.x + bottomRegion.bbox.width;

            for (const topIdx of topEdgeIndices) {
              if (mergedTopIndices.has(topIdx)) continue;

              const topRegion = nextPage.regions[topIdx];
              const topLeft = topRegion.bbox.x;
              const topRight = topRegion.bbox.x + topRegion.bbox.width;

              // Check X overlap (at least 50% overlap)
              const overlapLeft = Math.max(bottomLeft, topLeft);
              const overlapRight = Math.min(bottomRight, topRight);
              const overlapWidth = Math.max(0, overlapRight - overlapLeft);
              const minWidth = Math.min(bottomRegion.bbox.width, topRegion.bbox.width);

              if (overlapWidth > minWidth * 0.5) {
                // Merge: combine text, expand bottom region's bbox, mark top for removal
                console.error(`  Merging cross-page bubble: page ${pageIdx + 1} bottom → page ${pageIdx + 2} top`);

                // Append top region's text to bottom region
                currentPage.regions[bottomIdx] = {
                  bbox: bottomRegion.bbox, // Keep original bbox (on current page)
                  originalText: bottomRegion.originalText + '\n' + topRegion.originalText,
                  simplifiedText: bottomRegion.simplifiedText + '\n' + topRegion.simplifiedText,
                  originalPinyin: bottomRegion.originalPinyin + ' ' + topRegion.originalPinyin,
                  simplifiedPinyin: bottomRegion.simplifiedPinyin + ' ' + topRegion.simplifiedPinyin,
                };

                mergedTopIndices.add(topIdx);
                break; // One bottom region can only match one top region
              }
            }
          }

          // Remove merged top-edge regions from next page
          if (mergedTopIndices.size > 0) {
            nextPage.regions = nextPage.regions.filter((_, i) => !mergedTopIndices.has(i));
          }
        }

        // Pronunciation audio, opt-in via CHAPTER_AUDIO=1. Kicked off HERE,
        // concurrent with the gloss call below: block phrases are final by
        // this point (the cross-page merge above only touches `regions`, not
        // `blocks`), and the gloss LLM call is the natural window to hide
        // synthesis in — audio's wall-time cost is max(0, audio - gloss),
        // which is ~0 in practice. Failure never fails a chapter.
        let audioPromise: Promise<{ bubbles: number; total: number; wall: number } | null> =
          Promise.resolve(null);
        if (process.env.CHAPTER_AUDIO === '1') {
          // Narrowing from the enclosing `options.outputDir &&` guard does not
          // reach into the async closure; capture the narrowed value here.
          const audioOutputDir = options.outputDir;
          audioPromise = (async () => {
            const t0 = Date.now();
            try {
              const { generateChapterAudio } = await import('./audio-generator');
              const manifest = await generateChapterAudio(readerData, {
                outputDir: audioOutputDir,
              });
              const bubbles = Object.keys(manifest.bubbles).length;
              const total = readerData.pages.reduce(
                (n: number, p: any) => n + (p.blocks || []).length, 0);
              return { bubbles, total, wall: Date.now() - t0 };
            } catch (err) {
              console.error(`  Audio generation skipped: ${err instanceof Error ? err.message : String(err)}`);
              return null;
            }
          })();
        }

        // Per-bubble English glosses. Runs on the final block text, so it
        // sees exactly what the reader will show. Failure is non-fatal: an
        // unglossed bubble simply omits the section in the popup.
        const glossStart = Date.now();
        if (options.includeGlosses !== false) {
          if (options.onProgress) {
            options.onProgress(totalUnits, totalUnits, 'Translating bubbles...');
          }
          try {
            const targets: GlossTarget[] = [];
            for (const page of readerData.pages) {
              for (const block of (page.blocks || [])) {
                if (block.text) targets.push({ key: `${page.pageNumber}|${block.blockId}`, text: block.text });
              }
            }
            const glosses = await generateBubbleGlosses(targets, title, readerData.characters || []);
            let applied = 0;
            for (const page of readerData.pages) {
              for (const block of (page.blocks || [])) {
                const g = glosses.get(`${page.pageNumber}|${block.blockId}`);
                if (g) { block.gloss = g; applied++; }
              }
            }
            console.error(`  Bubble glosses: ${applied}/${targets.length}`);
          } catch (err) {
            console.error('  Bubble glosses skipped:', err instanceof Error ? err.message : err);
          }
        }

        // Join the audio task started before the glosses. The three numbers
        // logged here are the ones that decide whether CHAPTER_AUDIO can
        // default on: audio wall, gloss wall, and what audio actually added.
        {
          const glossWall = Date.now() - glossStart;
          const audio = await audioPromise;
          if (audio) {
            const added = Math.max(0, audio.wall - glossWall);
            console.error(
              `  [TIMING] Audio: ${(audio.wall / 1000).toFixed(1)}s for ` +
              `${audio.bubbles}/${audio.total} bubbles, gloss ${(glossWall / 1000).toFixed(1)}s ` +
              `-> audio added ${(added / 1000).toFixed(1)}s to the chapter`,
            );
          }
        }

        if (options.onProgress) {
          options.onProgress(totalUnits, totalUnits, 'Building reader...');
        }

        // Save data.json for potential future use (API, other viewers)
        const dataJsonPath = path.join(options.outputDir, 'data.json');
        fs.writeFileSync(dataJsonPath, JSON.stringify(readerData, null, 2), 'utf-8');
        console.error(`  Data file: ${dataJsonPath}`);

        // Write simplification/translation log (human-readable quality review)
        {
          const isTranslationLog = isTranslationSource(options.inputLanguage);
          const srcLabel = sourceLanguageName(options.inputLanguage);
          const provider = process.env.SIMPLIFY_PROVIDER || 'auto';
          const model = process.env.SIMPLIFY_MODEL || '(default)';
          const logLines: string[] = [
            `=== ${isTranslationLog ? `${srcLabel} → Chinese Translation` : 'Chinese Simplification'} Log ===`,
            `Level: ${options.targetLevel}`,
            `Provider: ${provider}`,
            `Model: ${model}`,
            `Date: ${new Date().toISOString()}`,
            '',
          ];

          // Include chapter analysis context
          if (options.chapterContext) {
            const ctx = options.chapterContext;
            logLines.push('=== Chapter Analysis ===');
            if (ctx.setting) logLines.push(`Setting: ${ctx.setting}`);
            if (ctx.characters && ctx.characters.length > 0) {
              logLines.push(`Characters: ${ctx.characters.map((c: any) => `${c.name} (${c.role})`).join(', ')}`);
            }
            if (ctx.keep && ctx.keep.length > 0) {
              logLines.push(`KEEP terms: ${ctx.keep.join('、')}`);
            }
            if (ctx.plot && ctx.plot.length > 0) {
              logLines.push('Plot sections:');
              for (const section of ctx.plot) {
                logLines.push(`  Pages ${section.pages}: ${section.summary}`);
              }
            }
            logLines.push('');
          }

          // Per-page simplification results
          for (const page of readerData.pages) {
            if (!page.blocks || page.blocks.length === 0) continue;
            logLines.push(`--- Page ${page.pageNumber} ---`);
            for (const block of page.blocks) {
              const region = page.regions.find((r: any) =>
                Math.abs(r.bbox.x - block.bbox.x) < 2 && Math.abs(r.bbox.y - block.bbox.y) < 2
              );
              const original = region?.originalText || '(original not found)';
              if (isTranslationLog) {
                logLines.push(`  ${srcLabel === 'Japanese' ? 'JA' : srcLabel === 'Korean' ? 'KO' : 'EN'}: ${original}`);
                logLines.push(`  CN: ${block.text}`);
              } else {
                logLines.push(`  原文: ${original}`);
                logLines.push(`  简化: ${block.text}`);
              }
              logLines.push('');
            }
          }

          const logPath = path.join(options.outputDir, 'translation-log.txt');
          fs.writeFileSync(logPath, logLines.join('\n'), 'utf-8');
          console.error(`  Translation log: ${logPath}`);
        }

        // Generate interactive reader HTML
        const html = generateInteractiveReaderHTML(readerData);
        combinedOutputPath = path.join(options.outputDir, 'reader.html');

        fs.writeFileSync(combinedOutputPath, html, 'utf-8');
        console.error(`  Interactive reader: ${combinedOutputPath}`);

        // NOTE: Do NOT delete individual page files - the interactive reader references them
      } catch (e) {
        console.error(`  Warning: Failed to generate HTML viewer: ${e}`);
      }
    }
  }

  // Print timing summary
  const stats = getTimingStats();
  const wallTime = processingTime;
  const fmt = (ms: number) => (ms / 1000).toFixed(2) + 's';
  const pct = (ms: number) => ((ms / (wallTime * 1000)) * 100).toFixed(1) + '%';

  console.error('\n' + '='.repeat(60));
  console.error('TIMING SUMMARY');
  console.error('='.repeat(60));
  console.error(`Pages processed: ${stats.pageCount}`);
  console.error(`Total wall time: ${wallTime.toFixed(2)}s`);
  console.error(`Concurrency:     ${concurrency}`);
  console.error('');
  console.error('Sequential phases (wall time):');
  console.error(`  Image loading:      ${fmt(stats.loading)}  (${pct(stats.loading)})`);
  console.error(`  Stitching:          ${fmt(stats.stitching)}  (${pct(stats.stitching)})`);
  if (stats.ocrScan > 0) {
    console.error(`  OCR scan:           ${fmt(stats.ocrScan)}  (${pct(stats.ocrScan)})`);
  }
  if (stats.chapterAnalysis > 0) {
    console.error(`  Chapter analysis:   ${fmt(stats.chapterAnalysis)}  (${pct(stats.chapterAnalysis)})`);
  }
  const parallelWallTime = wallTime * 1000 - stats.loading - stats.stitching - stats.ocrScan - stats.chapterAnalysis;
  console.error(`  Parallel processing: ${fmt(parallelWallTime)}  (${pct(parallelWallTime)})`);
  console.error('');
  console.error(`Parallel phase breakdown (cumulative across ${stats.pageCount} pages):`);
  const parallelTotalTracked = stats.ocr + stats.llm + stats.inpaintServer + stats.textRendering
    + stats.saveImage + stats.blockGrouping + stats.filterPasses + stats.colorSampling + stats.metaBuild
    + stats.saturation + stats.decodePage;
  const pctParallel = (ms: number) => parallelTotalTracked > 0 ? ((ms / parallelTotalTracked) * 100).toFixed(1) + '%' : '0%';
  console.error(`  OCR (cached):        ${fmt(stats.ocr)}  (${pctParallel(stats.ocr)})`);
  console.error(`  LLM:                 ${fmt(stats.llm)}  (${pctParallel(stats.llm)})`);
  console.error(`  Inpaint server:      ${fmt(stats.inpaintServer)}  (${pctParallel(stats.inpaintServer)})`);
  console.error(`  Metadata (legacy):   ${fmt(stats.textRendering)}  (${pctParallel(stats.textRendering)})`);
  console.error(`  Save image (JPG):    ${fmt(stats.saveImage)}  (${pctParallel(stats.saveImage)})`);
  console.error(`  Block grouping:      ${fmt(stats.blockGrouping)}  (${pctParallel(stats.blockGrouping)})`);
  console.error(`  Filter passes:       ${fmt(stats.filterPasses)}  (${pctParallel(stats.filterPasses)})`);
  console.error(`  Color sampling:      ${fmt(stats.colorSampling)}  (${pctParallel(stats.colorSampling)})`);
  console.error(`  Meta build:          ${fmt(stats.metaBuild)}  (${pctParallel(stats.metaBuild)})`);
  console.error(`  Saturation:          ${fmt(stats.saturation)}  (${pctParallel(stats.saturation)})`);
  console.error(`  Decode page:         ${fmt(stats.decodePage)}  (${pctParallel(stats.decodePage)})`);
  // Untracked = parallel wall × concurrency - sum of tracked cumulative
  const parallelWallMs = parallelWallTime;
  const expectedCumulative = parallelWallMs * concurrency;
  const untrackedCumulative = Math.max(0, expectedCumulative - parallelTotalTracked);
  console.error(`  (untracked):         ${fmt(untrackedCumulative)}  (estimated as parallel_wall×concurrency − tracked sum)`);
  console.error('');
  console.error('Average per page:');
  console.error(`  LLM:                 ${fmt(stats.llm / stats.pageCount)}`);
  console.error(`  Inpaint server:      ${fmt(stats.inpaintServer / stats.pageCount)}`);
  console.error(`  Metadata:            ${fmt(stats.textRendering / stats.pageCount)}`);
  console.error('='.repeat(60) + '\n');

  // Language-misdetection guard. Armed only when auto-detection resolved to
  // chinese (_langStats set): if essentially every block was dropped by the
  // all-English filter, the chapter is English that was misdetected — e.g.
  // every sampled page happened to be textless art. Rerun once with English
  // forced: slower than being right the first time, but never ships a
  // silently-empty chapter. The first pass's chapterContext / cached data are
  // products of the wrong-language pass, so they're stripped for the rerun.
  const langStats = options._langStats;
  if (langStats && !options._langRerun && langStats.totalBlocks >= 5
      && langStats.allEnglishDropped / langStats.totalBlocks >= 0.9) {
    console.error(`  [Language] Misdetection guard tripped: ${langStats.allEnglishDropped}/${langStats.totalBlocks} block passes dropped as all-English under auto-detected Chinese — re-running chapter as English`);
    if (options.onProgress) {
      options.onProgress(0, 1, 'Language re-detected as English — reprocessing...');
    }
    const { _langStats, chapterContext, cachedSimplifications, skipChapterAnalysis, ...rest } = options;
    return processMangaPages(imagePaths, {
      ...rest,
      inputLanguage: 'english',
      convertTraditional: false,
      _langRerun: true,
    });
  }

  return {
    inputPath: imagePaths[0],
    pages,
    totalRegions,
    processingTime,
    combinedOutputPath,
    cacheBlocks: pipelineCacheBlocks,
    chapterContext: options.chapterContext,
  };
}

/**
 * Analyze manga text without simplification
 */
export async function analyzeMangaPage(
  imagePath: string,
  targetLevel: HSKLevel,
  visionProvider: VisionProvider = 'gemini'
): Promise<{
  extraction: ExtractionResult;
  stats: ReturnType<typeof getVocabularyStats>;
  hskAnalysis: string;
}> {
  const pageImage = await loadImage(imagePath);
  let extraction = await extractTextFromImage(pageImage.buffer, pageImage.pageNumber, visionProvider);

  // Convert to simplified for analysis
  extraction = convertExtractionToSimplified(extraction);
  extraction = stripUrlTokensFromExtraction(extraction);

  // Get vocabulary stats
  const stats = getVocabularyStats(extraction.fullText, targetLevel);

  // Determine HSK level
  let hskAnalysis: string;
  if (stats.outsideHSK > stats.total * 0.3) {
    hskAnalysis = 'HSK 5+ (很多超纲词汇)';
  } else if (stats.hsk4 > stats.hsk3 && stats.hsk4 > stats.hsk2) {
    hskAnalysis = 'HSK 4-5';
  } else if (stats.hsk3 > stats.hsk2) {
    hskAnalysis = 'HSK 3-4';
  } else if (stats.hsk2 > stats.hsk1) {
    hskAnalysis = 'HSK 2-3';
  } else {
    hskAnalysis = 'HSK 1-2';
  }

  return {
    extraction,
    stats,
    hskAnalysis,
  };
}

/**
 * Format analysis results for display
 */
export function formatAnalysisResult(
  analysis: Awaited<ReturnType<typeof analyzeMangaPage>>
): string {
  const lines: string[] = [];

  lines.push('═'.repeat(60));
  lines.push('漫画文本分析');
  lines.push('═'.repeat(60));
  lines.push('');

  lines.push(`估计HSK等级: ${analysis.hskAnalysis}`);
  lines.push('');

  lines.push('词汇统计:');
  lines.push(`  总词数: ${analysis.stats.total}`);
  lines.push(`  HSK1: ${analysis.stats.hsk1}`);
  lines.push(`  HSK2: ${analysis.stats.hsk2}`);
  lines.push(`  HSK3: ${analysis.stats.hsk3}`);
  lines.push(`  HSK4: ${analysis.stats.hsk4}`);
  lines.push(`  超纲: ${analysis.stats.outsideHSK}`);
  lines.push(`  覆盖率: ${analysis.stats.coverage.toFixed(1)}%`);
  lines.push('');

  lines.push('提取的文本:');
  for (const region of analysis.extraction.regions) {
    lines.push(`  - "${region.text}"`);
  }

  return lines.join('\n');
}

/**
 * Format processing results for display
 */
export function formatProcessingResult(result: MangaProcessingResult): string {
  const lines: string[] = [];

  lines.push('═'.repeat(60));
  lines.push('漫画简化完成');
  lines.push('═'.repeat(60));
  lines.push('');

  for (const page of result.pages) {
    lines.push(`第${page.pageNumber}页:`);

    for (let i = 0; i < page.simplifiedRegions.length; i++) {
      const sr = page.simplifiedRegions[i];
      if (sr.original.text !== sr.simplifiedText) {
        lines.push(`  "${sr.original.text}"`);
        lines.push(`  → "${sr.simplifiedText}"`);
        lines.push('');
      }
    }
    lines.push('');
  }

  lines.push(`处理时间: ${result.processingTime.toFixed(2)}秒`);
  lines.push(`文本区域: ${result.totalRegions}个`);

  if (result.combinedOutputPath) {
    lines.push('');
    lines.push(`输出文件: ${result.combinedOutputPath}`);
  }

  return lines.join('\n');
}
