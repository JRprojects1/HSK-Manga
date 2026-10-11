import sharp from 'sharp';
import { TextRegion, traditionalToSimplified } from './text-extractor';
import { createMask, inpaintImage } from './inpainting';
import { InputLanguage } from './language';

export interface ReplacementRegion {
  original: TextRegion;
  simplifiedText: string;
  skipRender?: boolean;  // Set by post-inpaint filter when background is not uniform (signs/artwork)
  skipInpaint?: boolean;  // Set when background is uniform - use solid fill instead of inpainting
  fillColor?: { r: number; g: number; b: number };  // Background color for solid fill
  forceFillColor?: { r: number; g: number; b: number };  // Override fill color (e.g., for gray-corrected regions)
  isCrossPageContinuation?: boolean;  // Set for top-edge regions that continue from previous page
  blockId?: number;      // Block index for block-based rendering (regions in same block share this)
  blockText?: string;    // Full simplified text for the entire block
  originalBlockText?: string;  // Concatenated original text for the entire block (for reader)
}

export interface RenderedBlockInfo {
  blockId: number;
  bbox: { x: number; y: number; width: number; height: number }; // percentages (0-100)
  text?: string;         // rendered block text (post compactPunctuation + trad→simplified)
  isVertical?: boolean;
  fontSize?: number;
  textColor?: string;    // '#000000' or '#FFFFFF' based on background brightness
  rotation?: number;     // degrees, 0 = axis-aligned
  minFontSize?: number;  // floor derived from the original lettering height (EN dialogue only)
  originalLineHeight?: number; // median per-line OCR box height in px — kept for tuning/debug
}

export interface ReplaceTextResult {
  imageBuffer: Buffer;
  cleanedImageBuffer: Buffer;  // inpainted image with text removed, no new text rendered
  renderedBlocks: RenderedBlockInfo[];
  timing: {
    bgDetection: number;
    inpaintServer: number;
    varianceCheck: number;
    textRendering: number;
    colorSampling: number; // wall time of standalone+block sampleRegionColor parallel calls
    metaBuild: number;     // wall time of post-sampling metadata construction (font, pinyin, block boxes)
    decodePage: number;    // wall time of decodePage() calls inside runInpaintStep + replaceTextInImage
  };
}

export interface TextStyle {
  fontFamily: string;
  fontSize: number;
  fontColor: string;
  backgroundColor: string;
  padding: number;
  useInpainting: boolean;
  dynamicTextColor: boolean;
  // Source language drives the solid-fill padding scale: PaddleOCR's bboxes for
  // English text are systematically looser than Chinese (Latin letters have
  // natural padding from descenders/whitespace), so the inpaint mask needs less
  // expansion. Undefined falls back to the Chinese-tuned default.
  // JA/KO deliberately take the ZH padding scale below: their glyphs are
  // CJK-shaped (no Latin descender whitespace), so only EN gets the EN scale.
  inputLanguage?: InputLanguage;
}

const DEFAULT_STYLE: TextStyle = {
  fontFamily: 'Microsoft YaHei',
  fontSize: 24,
  fontColor: '#000000',
  backgroundColor: '#FFFFFF',
  padding: 4, // Margin around text box to cover original text
  useInpainting: process.env.ENABLE_INPAINTING !== 'false',
  dynamicTextColor: true,
};

// Solid-fill / inpaint mask padding scale (% of min image dimension).
// Both languages default to 0.005 — historically Chinese ran at 0.015 because
// PaddleOCR's tighter Chinese bboxes needed more expansion to fully cover the
// glyphs, but that produced visibly oversized white plates on vertical-manhua
// content (KuaiKan, Webtoons-style) where the original bbox already had
// surrounding whitespace. Env vars let you push either back up per-deploy if
// you see uncovered glyph slop in a specific content type.
const SOLID_FILL_PADDING_SCALE_ZH = parseFloat(process.env.SOLID_FILL_PADDING_SCALE_ZH || '0.005');
const SOLID_FILL_PADDING_SCALE_EN = parseFloat(process.env.SOLID_FILL_PADDING_SCALE_EN || '0.005');

// ─────────────────────────────────────────────────────────────────────────
// Decode-once helpers. Every per-region read function used to call
// sharp(imageBuffer).extract(...).toBuffer() — which re-decodes the source
// each call. For ~15 regions × 4-6 reads each per page, that's 60-100
// redundant full-image decodes per page. Benchmark showed 7× speedup by
// decoding ONCE per page and slicing pixel data in JS.
//
// Pattern: caller does `await decodePage(buffer)` once, then passes the
// returned RawPageData to every helper. Helpers become synchronous.
// ─────────────────────────────────────────────────────────────────────────
export interface RawPageData {
  data: Buffer;
  width: number;
  height: number;
  channels: number;
}
async function decodePage(imageBuffer: Buffer): Promise<RawPageData> {
  const { data, info } = await sharp(imageBuffer).raw().toBuffer({ resolveWithObject: true });
  return { data, width: info.width, height: info.height, channels: info.channels };
}
// Pre-allocated TypedArray RGB collector — pushes into typed arrays are 2-3x
// faster than into regular arrays. Caller passes in approximate capacity.
function collectStripRGB(
  page: RawPageData,
  strip: { left: number; top: number; width: number; height: number }
): { reds: Uint8Array; greens: Uint8Array; blues: Uint8Array; count: number } {
  const stride = page.width * page.channels;
  const total = strip.width * strip.height;
  const reds = new Uint8Array(total);
  const greens = new Uint8Array(total);
  const blues = new Uint8Array(total);
  let idx = 0;
  for (let y = 0; y < strip.height; y++) {
    const rowStart = (strip.top + y) * stride + strip.left * page.channels;
    for (let x = 0; x < strip.width; x++) {
      const i = rowStart + x * page.channels;
      reds[idx] = page.data[i];
      greens[idx] = page.data[i + 1];
      blues[idx] = page.data[i + 2];
      idx++;
    }
  }
  return { reds, greens, blues, count: idx };
}
function stddev(arr: Uint8Array, count: number): number {
  if (count === 0) return Infinity;
  let sum = 0;
  for (let i = 0; i < count; i++) sum += arr[i];
  const mean = sum / count;
  let sqSum = 0;
  for (let i = 0; i < count; i++) {
    const d = arr[i] - mean;
    sqSum += d * d;
  }
  return Math.sqrt(sqSum / count);
}
function medianU8(arr: Uint8Array, count: number): number {
  if (count === 0) return 0;
  // For small arrays, sort directly; for larger ones this is still O(n log n)
  // but typed-array sort is fast.
  const sorted = arr.subarray(0, count).slice().sort();
  return sorted[Math.floor(count / 2)];
}

/**
 * Sample background color from OUTSIDE the bounding box for narrow regions.
 * For characters like 一 (yi), the bounding box contains mostly character pixels,
 * so we expand the sampling area and sample only from the outer ring.
 */
function sampleOuterRing(
  data: Buffer,
  totalWidth: number,
  totalHeight: number,
  channels: number,
  innerOffsetX: number,
  innerOffsetY: number,
  innerWidth: number,
  innerHeight: number
): { r: number; g: number; b: number } {
  const samples: Array<{ r: number; g: number; b: number; brightness: number }> = [];

  const addSample = (x: number, y: number) => {
    if (x >= 0 && x < totalWidth && y >= 0 && y < totalHeight) {
      // Skip if inside the original (inner) region - we only want the expanded outer area
      if (x >= innerOffsetX && x < innerOffsetX + innerWidth &&
          y >= innerOffsetY && y < innerOffsetY + innerHeight) {
        return; // Skip - this is inside the character region
      }
      const idx = (y * totalWidth + x) * channels;
      const r = data[idx];
      const g = data[idx + 1];
      const b = data[idx + 2];
      const brightness = 0.299 * r + 0.587 * g + 0.114 * b;
      samples.push({ r, g, b, brightness });
    }
  };

  // Sample the entire outer ring area (every 2 pixels for efficiency)
  for (let y = 0; y < totalHeight; y += 2) {
    for (let x = 0; x < totalWidth; x += 2) {
      addSample(x, y);
    }
  }

  if (samples.length === 0) {
    return { r: 0, g: 0, b: 0 }; // Default to black for dark backgrounds
  }

  // Use median filtering - take the darker half (background, not any light artifacts)
  samples.sort((a, b) => a.brightness - b.brightness);
  const darkHalf = samples.slice(0, Math.ceil(samples.length / 2));

  const avgR = darkHalf.reduce((s, p) => s + p.r, 0) / darkHalf.length;
  const avgG = darkHalf.reduce((s, p) => s + p.g, 0) / darkHalf.length;
  const avgB = darkHalf.reduce((s, p) => s + p.b, 0) / darkHalf.length;

  return { r: Math.round(avgR), g: Math.round(avgG), b: Math.round(avgB) };
}

/**
 * Result of gray background detection.
 * Used to identify regions where undetected white text (like 一) contaminated the background sample.
 */
export interface GrayDetectionResult {
  isGray: boolean;
  brightness: number;
  cornerBrightness: number;
  sampledColor: { r: number; g: number; b: number };
}

/**
 * Sample brightness from image corners to determine overall background type.
 * Corners are far from text content and give true background color.
 */
export async function sampleCornerBrightness(
  imageBuffer: Buffer,
  imageWidth: number,
  imageHeight: number
): Promise<number> {
  const cornerSize = 15;
  const corners = [
    { left: 0, top: 0 },
    { left: Math.max(0, imageWidth - cornerSize), top: 0 },
    { left: 0, top: Math.max(0, imageHeight - cornerSize) },
    { left: Math.max(0, imageWidth - cornerSize), top: Math.max(0, imageHeight - cornerSize) }
  ];

  let totalBrightness = 0;
  let sampleCount = 0;

  for (const corner of corners) {
    try {
      const extractWidth = Math.min(cornerSize, imageWidth - corner.left);
      const extractHeight = Math.min(cornerSize, imageHeight - corner.top);
      if (extractWidth <= 0 || extractHeight <= 0) continue;

      const { data, info } = await sharp(imageBuffer)
        .extract({ left: corner.left, top: corner.top, width: extractWidth, height: extractHeight })
        .raw()
        .toBuffer({ resolveWithObject: true });

      // Sample center pixel of corner region
      const centerX = Math.floor(extractWidth / 2);
      const centerY = Math.floor(extractHeight / 2);
      const centerIdx = (centerY * extractWidth + centerX) * info.channels;

      if (centerIdx + 2 < data.length) {
        const r = data[centerIdx];
        const g = data[centerIdx + 1];
        const b = data[centerIdx + 2];
        totalBrightness += 0.299 * r + 0.587 * g + 0.114 * b;
        sampleCount++;
      }
    } catch (e) {
      // Skip corners that fail
    }
  }

  return sampleCount > 0 ? totalBrightness / sampleCount : 128;
}

/**
 * Detect if a sampled background color is "suspicious gray" on a dark image.
 * This indicates contamination from undetected white text (like 一).
 *
 * Gray zone: brightness 40-180
 * - Below 40: dark enough to be true dark background
 * - Above 180: light enough to be intentional light background
 *
 * If gray detected on dark image (corners < 50), returns isGray: true
 */
export async function detectSuspiciousGray(
  sampledColor: { r: number; g: number; b: number },
  imageBuffer: Buffer,
  imageWidth: number,
  imageHeight: number
): Promise<GrayDetectionResult> {
  const brightness = 0.299 * sampledColor.r + 0.587 * sampledColor.g + 0.114 * sampledColor.b;

  // Gray zone: 40-180 (too dark to be intentional light, too light to be true black)
  if (brightness > 40 && brightness < 180) {
    const cornerBrightness = await sampleCornerBrightness(imageBuffer, imageWidth, imageHeight);
    if (cornerBrightness < 50) {
      // Dark image with gray sample = likely undetected white text
      console.log(`[TextReplacer] Suspicious gray detected: brightness=${brightness.toFixed(0)}, corners=${cornerBrightness.toFixed(0)}`);
      return { isGray: true, brightness, cornerBrightness, sampledColor };
    }
    return { isGray: false, brightness, cornerBrightness, sampledColor };
  }

  return { isGray: false, brightness, cornerBrightness: 0, sampledColor };
}

/**
 * Sample the average color from a region using Sharp extract (crops only the needed area).
 * Used to determine background color for choosing black/white overlay text.
 *
 * For narrow regions (like the character 一), expands the crop and samples from
 * OUTSIDE the bounding box to avoid sampling the character pixels.
 */
function sampleRegionColor(
  page: RawPageData,
  boundingBox: { x: number; y: number; width: number; height: number },
): { r: number; g: number; b: number } {
  const imageWidth = page.width;
  const imageHeight = page.height;
  // Convert percentage to pixels
  const pixelX = Math.round((boundingBox.x / 100) * imageWidth);
  const pixelY = Math.round((boundingBox.y / 100) * imageHeight);
  const pixelW = Math.round((boundingBox.width / 100) * imageWidth);
  const pixelH = Math.round((boundingBox.height / 100) * imageHeight);

  const left = Math.max(0, pixelX);
  const top = Math.max(0, pixelY);
  const regionWidth = Math.min(pixelW, imageWidth - left);
  const regionHeight = Math.min(pixelH, imageHeight - top);

  if (regionWidth <= 0 || regionHeight <= 0) {
    return { r: 255, g: 255, b: 255 };
  }

  // Detect narrow regions that need EXPANDED sampling
  const isNarrowRegion = (regionWidth < 20 || regionHeight < 20 ||
                         regionWidth / regionHeight > 4 ||
                         regionHeight / regionWidth > 4);

  const channels = page.channels;
  const stride = imageWidth * channels;

  if (isNarrowRegion) {
    const expandBy = 10;
    const expandedLeft = Math.max(0, left - expandBy);
    const expandedTop = Math.max(0, top - expandBy);
    const expandedRight = Math.min(imageWidth, left + regionWidth + expandBy);
    const expandedBottom = Math.min(imageHeight, top + regionHeight + expandBy);
    const expandedWidth = expandedRight - expandedLeft;
    const expandedHeight = expandedBottom - expandedTop;

    if (expandedWidth <= 0 || expandedHeight <= 0) {
      return { r: 255, g: 255, b: 255 };
    }

    // Copy the expanded region into a contiguous buffer so sampleOuterRing's
    // (totalWidth, totalHeight)-indexed math still works unchanged.
    const expandedData = Buffer.alloc(expandedWidth * expandedHeight * channels);
    for (let y = 0; y < expandedHeight; y++) {
      const srcOffset = (expandedTop + y) * stride + expandedLeft * channels;
      const dstOffset = y * expandedWidth * channels;
      page.data.copy(expandedData, dstOffset, srcOffset, srcOffset + expandedWidth * channels);
    }

    const innerOffsetX = left - expandedLeft;
    const innerOffsetY = top - expandedTop;

    return sampleOuterRing(expandedData, expandedWidth, expandedHeight, channels,
                           innerOffsetX, innerOffsetY, regionWidth, regionHeight);
  }

  // Standard: sample center pixels directly from page data
  let totalR = 0, totalG = 0, totalB = 0;
  let sampleCount = 0;

  const startX = Math.floor(regionWidth * 0.25);
  const endX = Math.floor(regionWidth * 0.75);
  const startY = Math.floor(regionHeight * 0.25);
  const endY = Math.floor(regionHeight * 0.75);
  const step = Math.max(1, Math.floor(Math.min(endX - startX, endY - startY) / 10));

  for (let sy = startY; sy < endY; sy += step) {
    for (let sx = startX; sx < endX; sx += step) {
      const idx = (top + sy) * stride + (left + sx) * channels;
      totalR += page.data[idx];
      totalG += page.data[idx + 1];
      totalB += page.data[idx + 2];
      sampleCount++;
    }
  }

  if (sampleCount === 0) {
    return { r: 255, g: 255, b: 255 };
  }

  return {
    r: Math.round(totalR / sampleCount),
    g: Math.round(totalG / sampleCount),
    b: Math.round(totalB / sampleCount),
  };
}

/**
 * Convert fullwidth CJK punctuation to halfwidth for compact rendering.
 * Microsoft YaHei renders fullwidth punctuation at full character width,
 * creating large gaps. Halfwidth equivalents render much more compactly.
 */
/**
 * Convert fullwidth punctuation to ASCII only when adjacent to another punctuation char.
 * Preserves single fullwidth punctuation (which renders well at CJK size).
 */
export function compactAdjacentPunctuation(text: string): string {
  const fwToAscii: Record<string, string> = {
    '！': '!', '？': '?', '。': '.', '，': ',', '；': ';', '：': ':',
    '（': '(', '）': ')', '【': '[', '】': ']', '「': '"', '」': '"',
    '『': '"', '』': '"', '《': '<', '》': '>',
  };
  const fwChars = new Set(Object.keys(fwToAscii));

  const chars = [...text];
  const result: string[] = [];
  for (let i = 0; i < chars.length; i++) {
    if (fwChars.has(chars[i])) {
      const prevIsFW = i > 0 && fwChars.has(chars[i - 1]);
      const nextIsFW = i < chars.length - 1 && fwChars.has(chars[i + 1]);
      if (prevIsFW || nextIsFW) {
        result.push(fwToAscii[chars[i]]);
        continue;
      }
    }
    result.push(chars[i]);
  }
  return result.join('');
}

export function compactPunctuation(text: string): string {
  return text
    // Punctuation with halfwidth CJK equivalents (preserve shape, reduce width)
    .replace(/。/g, '｡')    // U+FF61 halfwidth ideographic full stop (circle)
    .replace(/、/g, '､')    // U+FF64 halfwidth ideographic comma
    .replace(/「/g, '｢')    // U+FF62 halfwidth left corner bracket
    .replace(/」/g, '｣')    // U+FF63 halfwidth right corner bracket
    // Punctuation converted to ASCII halfwidth (no CJK halfwidth form exists)
    .replace(/，/g, ',')
    .replace(/！/g, '!')
    .replace(/？/g, '?')
    .replace(/；/g, ';')
    .replace(/：/g, ':')
    .replace(/（/g, '(')
    .replace(/）/g, ')')
    .replace(/【/g, '[')
    .replace(/】/g, ']')
    .replace(/『/g, '"')
    .replace(/』/g, '"')
    // Curly quotes are FULL-WIDTH in CJK fonts (a full em — same width as a
    // hanzi; ASCII " is 0.44em). They were the one full-width pair this list
    // missed: “宇成” burned 4 ems for 2 chars of content, shrinking fonts and
    // forcing wraps mid-name. The reader resolves opener/closer by parity
    // since ASCII quotes are directionless (see renderBlockPhrases).
    .replace(/[“”]/g, '"')
    .replace(/[‘’]/g, "'")
    .replace(/《/g, '<')
    .replace(/》/g, '>')
    .replace(/——/g, '—')
    .replace(/…+/g, '…')           // Collapse multiple ellipsis chars to one
    .replace(/\.{2,}/g, '…');      // Convert 2+ ASCII periods to ellipsis
}

/**
 * Group nearby text regions that likely belong to the same speech bubble
 * Returns array of group indices (regions with same index are in same group)
 */
export function groupNearbyRegions(
  replacements: ReplacementRegion[],
  imageWidth: number,
  imageHeight: number
): number[] {
  const groups: number[] = new Array(replacements.length).fill(-1);
  let nextGroup = 0;

  // Distance threshold: regions within 5% of image dimensions are considered same bubble
  const distanceThreshold = Math.min(imageWidth, imageHeight) * 0.05;

  for (let i = 0; i < replacements.length; i++) {
    if (groups[i] === -1) {
      groups[i] = nextGroup++;
    }

    const boxA = replacements[i].original.boundingBox;
    const centerAX = (boxA.x + boxA.width / 2) / 100 * imageWidth;
    const centerAY = (boxA.y + boxA.height / 2) / 100 * imageHeight;

    for (let j = i + 1; j < replacements.length; j++) {
      if (groups[j] !== -1) continue;

      const boxB = replacements[j].original.boundingBox;
      const centerBX = (boxB.x + boxB.width / 2) / 100 * imageWidth;
      const centerBY = (boxB.y + boxB.height / 2) / 100 * imageHeight;

      // Calculate distance between region centers
      const distance = Math.sqrt(
        Math.pow(centerAX - centerBX, 2) + Math.pow(centerAY - centerBY, 2)
      );

      // If close enough, assign to same group
      if (distance < distanceThreshold) {
        groups[j] = groups[i];
      }
    }
  }

  // Debug: when DEBUG_REGION_MERGE is set, log identical-bbox clusters that
  // ended up in different groups (that's the "7 duplicate regions" bug from
  // job d53784a5 — 7 regions with byte-identical bbox produced 7 output entries).
  if (process.env.DEBUG_REGION_MERGE) {
    const bboxKey = (b: { x: number; y: number; width: number; height: number }) =>
      `${b.x.toFixed(3)},${b.y.toFixed(3)},${b.width.toFixed(3)},${b.height.toFixed(3)}`;
    const clusters = new Map<string, number[]>();
    replacements.forEach((r, idx) => {
      const key = bboxKey(r.original.boundingBox);
      if (!clusters.has(key)) clusters.set(key, []);
      clusters.get(key)!.push(idx);
    });
    let splitCount = 0;
    for (const [bbox, indices] of clusters) {
      if (indices.length < 2) continue;
      const uniqueGroups = new Set(indices.map(i => groups[i]));
      if (uniqueGroups.size > 1) {
        splitCount++;
        console.error(`  [DEBUG_REGION_MERGE] SPLIT: ${indices.length} identical-bbox regions in ${uniqueGroups.size} groups. bbox=${bbox}`);
        console.error(`    text: "${(replacements[indices[0]].original.text || '').slice(0, 60)}"`);
        console.error(`    group assignments: ${indices.map(i => `R${i}→g${groups[i]}`).join(', ')}`);
      } else {
        console.error(`  [DEBUG_REGION_MERGE] MERGED: ${indices.length} identical-bbox regions all in group ${[...uniqueGroups][0]}. text="${(replacements[indices[0]].original.text || '').slice(0, 40)}"`);
      }
    }
    console.error(`  [DEBUG_REGION_MERGE] Summary: ${replacements.length} regions → ${nextGroup} groups; identical-bbox splits: ${splitCount}`);
  }

  return groups;
}

/**
 * Check if a margin strip area has uniform color.
 * Used to decide if it's safe to expand inpainting into that area.
 * Synchronous — caller passes pre-decoded RawPageData (see decodePage).
 * RGB-aware: requires all three channels to have low variance, so that
 * uniform-green or uniform-mid-tone backgrounds are correctly detected
 * (the old brightness-only check missed these).
 */
function isMarginUniform(
  page: RawPageData,
  margin: { x: number; y: number; width: number; height: number },
): boolean {
  const px = Math.round((margin.x / 100) * page.width);
  const py = Math.round((margin.y / 100) * page.height);
  const pw = Math.max(1, Math.round((margin.width / 100) * page.width));
  const ph = Math.max(1, Math.round((margin.height / 100) * page.height));
  const left = Math.max(0, Math.min(px, page.width - 1));
  const top = Math.max(0, Math.min(py, page.height - 1));
  const w = Math.min(pw, page.width - left);
  const h = Math.min(ph, page.height - top);
  if (w <= 0 || h <= 0) return false;
  const { reds, greens, blues, count } = collectStripRGB(page, { left, top, width: w, height: h });
  if (count === 0) return false;
  return stddev(reds, count) < CHANNEL_STD_THRESHOLD
    && stddev(greens, count) < CHANNEL_STD_THRESHOLD
    && stddev(blues, count) < CHANNEL_STD_THRESHOLD;
}

// \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
// Skip-inpaint optimization: when a text region's surrounding margin is highly
// uniform (clean speech bubble interior), we can replace the slow LaMa call
// with a cheap solid-color rectangle and produce visually identical output.
// On chapters with mostly-clean bubbles this skips ~40-60% of LaMa work.
//
// Tunable: set MIN_MARGIN_UNIFORM_PCT (default 0.85) \u2014 fraction of margin
// pixels that must be very-bright (>220) or very-dark (<35) for skip-fill
// to engage. Lower = more aggressive skip (faster, riskier);
// higher = more conservative.
//
// Critical: solid-fill MUST run before LaMa, never after. LaMa uses pixels
// outside its mask as inpaint context. If skip regions still contain text
// when LaMa runs, the text glyphs leak into LaMa's reconstruction at the
// mask boundary. Filling them first gives LaMa a clean canvas.
// \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
const MIN_MARGIN_UNIFORM_PCT = parseFloat(process.env.MIN_MARGIN_UNIFORM_PCT || '0.85');
const MARGIN_STRIP_PCT = 1.5; // sample 1.5%-of-image-height strip just outside bbox
// Path-A classifier parameters: thin L/R ring + percentile uniformity.
// Why L/R only: T/B strips overlap adjacent text lines in multi-line blocks,
// failing uniformity even when the bubble itself is uniform. L/R strips
// (above/below text-row clutter) stay inside the bubble.
// Why ring + gap: leaving 2px gap inside the bbox edge skips OCR anti-aliasing
// and tight-bbox boundary noise. A 5px ring width gives enough samples.
// Why percentile vs std: ≤5% outlier pixels (bubble borders, JPEG noise)
// dominate std but barely move percentile — matches "99% uniform = solid-fill"
// human intuition.
const RING_INNER_GAP_PX = parseInt(process.env.RING_INNER_GAP_PX || '2', 10);
const RING_WIDTH_PX = parseInt(process.env.RING_WIDTH_PX || '5', 10);
const RGB_DEVIATION = parseInt(process.env.MARGIN_RGB_DEVIATION || '15', 10);
const UNIFORM_PCT_THRESHOLD = parseFloat(process.env.MARGIN_UNIFORM_PCT || '0.95');
// Per-channel std threshold (0-255 scale). 20 = pixels stay within ~±20 of
// each other on each color channel. Catches uniform mid-tone colors that the
// old brightness-only check missed.
const CHANNEL_STD_THRESHOLD = parseFloat(process.env.MARGIN_CHANNEL_STD || '20');

// Max allowed per-channel difference between L/R median and either T or B
// strip median. Catches vertical gradient backgrounds that horizontal L/R
// sampling alone misses (e.g. narration text laid over a sky → ground fade —
// L/R cleanly read the local horizontal slice but the actual fill color
// changes top-to-bottom across the bbox, producing a flat rectangle that
// doesn't blend). Speech-bubble interiors empirically show T/B diff of 0–1,
// problem-case gradients show 11+, so 10 sits comfortably in the gap.
const TB_AGREE_TOLERANCE = parseInt(process.env.MARGIN_TB_AGREE_TOLERANCE || '10', 10);

interface SkipDecision {
  isSkip: boolean;
  fillColor?: { r: number; g: number; b: number };
  uniformPct: number;
}

function classifyForSkipFill(
  page: RawPageData,
  bboxPct: { x: number; y: number; width: number; height: number },
  // T/B sampling is only safe when no other region's bbox intrudes into the
  // strip area — otherwise we'd sample text pixels from an adjacent line and
  // falsely fail uniformity. Callers pre-compute these geometrically against
  // the rest of the page's regions.
  adjacentAbove: boolean = true,
  adjacentBelow: boolean = true,
): SkipDecision {
  const imgW = page.width;
  const imgH = page.height;
  const bx = Math.round((bboxPct.x / 100) * imgW);
  const by = Math.round((bboxPct.y / 100) * imgH);
  const bw = Math.round((bboxPct.width / 100) * imgW);
  const bh = Math.round((bboxPct.height / 100) * imgH);

  // Sample LEFT and RIGHT ring strips ONLY. Top/bottom strips skipped because
  // in multi-line blocks they overlap adjacent lines' bboxes and fail
  // uniformity even when the local bubble interior IS uniform. L/R strips
  // typically stay inside the bubble (text is centered, sides have padding).
  // Ring shape: 2px gap inside (skip OCR anti-aliasing) + 5px sample width.
  const gap = RING_INNER_GAP_PX;
  const w = RING_WIDTH_PX;
  const strips = [
    // left ring
    {
      left: Math.max(0, bx - gap - w),
      top: by,
      width: Math.min(w, Math.max(0, bx - gap)),
      height: bh,
    },
    // right ring
    {
      left: Math.min(imgW - 1, bx + bw + gap),
      top: by,
      width: Math.min(w, imgW - (bx + bw + gap)),
      height: bh,
    },
  ].filter(s => s.width >= 2 && s.height >= 2 && s.left + s.width <= imgW && s.top + s.height <= imgH);

  if (strips.length === 0) return { isSkip: false, uniformPct: 0 };

  // Concatenate all strip RGB samples into typed-array buffers.
  let totalCount = 0;
  for (const s of strips) totalCount += s.width * s.height;
  const allReds = new Uint8Array(totalCount);
  const allGreens = new Uint8Array(totalCount);
  const allBlues = new Uint8Array(totalCount);
  let writeIdx = 0;
  for (const s of strips) {
    const { reds, greens, blues, count } = collectStripRGB(page, s);
    allReds.set(reds.subarray(0, count), writeIdx);
    allGreens.set(greens.subarray(0, count), writeIdx);
    allBlues.set(blues.subarray(0, count), writeIdx);
    writeIdx += count;
  }
  if (writeIdx === 0) return { isSkip: false, uniformPct: 0 };

  // Percentile-based uniformity: per-channel "what fraction of pixels are
  // within ±RGB_DEVIATION of the median?". Robust to small outlier groups
  // (bubble borders, JPEG noise) that would spike std but represent <5% of
  // pixels. Matches the human intuition of "99% uniform → solid-fill is fine".
  const medR = medianU8(allReds, writeIdx);
  const medG = medianU8(allGreens, writeIdx);
  const medB = medianU8(allBlues, writeIdx);
  let inR = 0, inG = 0, inB = 0;
  for (let i = 0; i < writeIdx; i++) {
    if (Math.abs(allReds[i] - medR) <= RGB_DEVIATION) inR++;
    if (Math.abs(allGreens[i] - medG) <= RGB_DEVIATION) inG++;
    if (Math.abs(allBlues[i] - medB) <= RGB_DEVIATION) inB++;
  }
  const pctR = inR / writeIdx;
  const pctG = inG / writeIdx;
  const pctB = inB / writeIdx;
  const minPct = Math.min(pctR, pctG, pctB);

  if (minPct < UNIFORM_PCT_THRESHOLD) {
    return { isSkip: false, uniformPct: minPct };
  }

  // T/B median-agreement check — catches vertical gradients that L/R alone
  // misses. Only runs for the side(s) where no neighboring region's bbox would
  // contaminate the sample (otherwise we'd be measuring adjacent-line text
  // pixels and force-failing legitimate solid-fillable bubbles).
  // Strip layout mirrors L/R: GAP px outside the bbox edge, then WIDTH px deep.
  // T/B span the bbox width (vs L/R spanning bbox height) so we sample the
  // background immediately above / below the actual text glyphs.
  function medianForStrip(strip: { left: number; top: number; width: number; height: number }): { r: number; g: number; b: number } | null {
    if (strip.width < 2 || strip.height < 2) return null;
    if (strip.left < 0 || strip.top < 0) return null;
    if (strip.left + strip.width > imgW || strip.top + strip.height > imgH) return null;
    const { reds, greens, blues, count } = collectStripRGB(page, strip);
    if (count === 0) return null;
    return { r: medianU8(reds, count), g: medianU8(greens, count), b: medianU8(blues, count) };
  }
  if (!adjacentAbove) {
    const tMed = medianForStrip({
      left: bx,
      top: by - RING_INNER_GAP_PX - RING_WIDTH_PX,
      width: bw,
      height: RING_WIDTH_PX,
    });
    if (tMed) {
      const diff = Math.max(Math.abs(tMed.r - medR), Math.abs(tMed.g - medG), Math.abs(tMed.b - medB));
      if (diff > TB_AGREE_TOLERANCE) {
        return { isSkip: false, uniformPct: minPct };
      }
    }
  }
  if (!adjacentBelow) {
    const bMed = medianForStrip({
      left: bx,
      top: by + bh + RING_INNER_GAP_PX,
      width: bw,
      height: RING_WIDTH_PX,
    });
    if (bMed) {
      const diff = Math.max(Math.abs(bMed.r - medR), Math.abs(bMed.g - medG), Math.abs(bMed.b - medB));
      if (diff > TB_AGREE_TOLERANCE) {
        return { isSkip: false, uniformPct: minPct };
      }
    }
  }

  return {
    isSkip: true,
    uniformPct: minPct,
    fillColor: { r: medR, g: medG, b: medB },
  };
}

/**
 * Inpaint regions of an image (Step 1 of text replacement) \u2014 removes original text
 * without rendering anything new. Used to decouple the slow inpaint server work
 * from LLM simplification so they can run in parallel.
 */
export async function inpaintImageRegions(
  imageBuffer: Buffer,
  replacements: ReplacementRegion[],
  style: Partial<TextStyle> = {}
): Promise<{ cleanedImageBuffer: Buffer; inpaintMs: number; decodePageMs?: number }> {
  const mergedStyle = { ...DEFAULT_STYLE, ...style };
  const metadata = await sharp(imageBuffer).metadata();
  const width = metadata.width || 800;
  const height = metadata.height || 600;
  const paddingScale = mergedStyle.inputLanguage === 'english'
    ? SOLID_FILL_PADDING_SCALE_EN
    : SOLID_FILL_PADDING_SCALE_ZH;
  const scaledPadding = Math.max(
    mergedStyle.padding,
    Math.round(Math.min(width, height) * paddingScale)
  );
  // SHRINK the solid-fill rect by 2px on each side (negative padding) for both
  // languages. Keeps the rect strictly inside the OCR bbox so it never touches
  // the bbox edge — eliminates the bright-edge halo from any glyph that
  // touches the bbox border. (Previously only English shrank; Chinese expanded
  // by scaledPadding, which over-filled on vertical-manhua content.)
  const solidFillPadX = -2;

  if (!mergedStyle.useInpainting || replacements.length === 0) {
    return { cleanedImageBuffer: imageBuffer, inpaintMs: 0 };
  }

  return runInpaintStep(imageBuffer, replacements, width, height, scaledPadding, solidFillPadX);
}

/**
 * Internal: inpaint step shared by inpaintImageRegions() and replaceTextInImage().
 *
 * Two-phase pipeline:
 *   1. Classify each region by margin uniformity. Uniform-bubble regions get
 *      solid-fill (cheap composite, no LaMa server call). Textured regions
 *      go to LaMa as before.
 *   2. Solid-fill the skip regions FIRST, then call LaMa on the now-cleaner
 *      image with a mask covering only the LaMa regions. Order is critical:
 *      LaMa uses the surrounding pixels as inpaint context, so nearby
 *      un-skipped text would leak into LaMa's output.
 *
 * Both groups still use the existing left/right bbox expansion to catch
 * orphan punctuation outside the OCR bbox.
 */
async function runInpaintStep(
  imageBuffer: Buffer,
  replacements: ReplacementRegion[],
  width: number,
  height: number,
  scaledPadding: number,
  solidFillPadX: number = scaledPadding,
  precomputedPageData?: RawPageData
): Promise<{ cleanedImageBuffer: Buffer; inpaintMs: number; decodePageMs: number }> {
  // Decode the source page ONCE. classifyForSkipFill and isMarginUniform are
  // synchronous and slice from this shared raw buffer instead of each calling
  // sharp(imageBuffer).extract() (which would re-decode the whole image per
  // call \u2014 benchmark showed 7x speedup from this single change).
  // If caller already decoded the page (e.g. for saturation check), reuse it
  // to skip the second decode entirely.
  let page: RawPageData;
  let decodePageMs = 0;
  if (precomputedPageData) {
    page = precomputedPageData;
  } else {
    const _decodeStart = Date.now();
    page = await decodePage(imageBuffer);
    decodePageMs = Date.now() - _decodeStart;
  }

  // Step 0a: compute T/B sampling adjacency per region. A region is "adjacent
  // above" when ANY other region's bbox geometrically intersects its T sample
  // strip area (5px tall, GAP px above the bbox, full bbox width). Same shape
  // for "adjacent below". When adjacent, the classifier skips T/B sampling on
  // that side because the strip would contain neighbor-line text pixels and
  // fail uniformity for the wrong reason. Pure bbox-intersection (no block
  // grouping required) — different blocks stacked close enough to intrude
  // are still correctly treated as adjacent.
  const adjacency = replacements.map((r, i) => {
    const me = r.original.boundingBox;
    const myBx = Math.round((me.x / 100) * page.width);
    const myBy = Math.round((me.y / 100) * page.height);
    const myBw = Math.round((me.width / 100) * page.width);
    const myBh = Math.round((me.height / 100) * page.height);
    const tStrip = {
      l: myBx, r: myBx + myBw,
      t: myBy - RING_INNER_GAP_PX - RING_WIDTH_PX, b: myBy - RING_INNER_GAP_PX,
    };
    const bStrip = {
      l: myBx, r: myBx + myBw,
      t: myBy + myBh + RING_INNER_GAP_PX, b: myBy + myBh + RING_INNER_GAP_PX + RING_WIDTH_PX,
    };
    let adjacentAbove = false;
    let adjacentBelow = false;
    for (let j = 0; j < replacements.length; j++) {
      if (j === i) continue;
      const o = replacements[j].original.boundingBox;
      const ox = Math.round((o.x / 100) * page.width);
      const oy = Math.round((o.y / 100) * page.height);
      const ow = Math.round((o.width / 100) * page.width);
      const oh = Math.round((o.height / 100) * page.height);
      // 2D bbox intersection: ranges overlap on both axes.
      if (ox < tStrip.r && ox + ow > tStrip.l && oy < tStrip.b && oy + oh > tStrip.t) adjacentAbove = true;
      if (ox < bStrip.r && ox + ow > bStrip.l && oy < bStrip.b && oy + oh > bStrip.t) adjacentBelow = true;
      if (adjacentAbove && adjacentBelow) break;
    }
    return { adjacentAbove, adjacentBelow };
  });

  // Step 0b: classify each region as skip-fill vs LaMa.
  const decisions = replacements.map((r, i) =>
    classifyForSkipFill(page, r.original.boundingBox, adjacency[i].adjacentAbove, adjacency[i].adjacentBelow)
  );

  // Step 0c: block-level failure propagation. If any region in a logical block
  // fails the classifier (gradient, art, suspicious uniformity), every region
  // in that block routes to LaMa. Catches inner-line cases that idea 1 alone
  // can't sample (middle lines of multi-line narration have neighbors on both
  // sides → no T/B sampling possible) but where a sibling line is already
  // suspect. Regions without a blockId are isolated and unaffected.
  const failingBlockIds = new Set<number>();
  for (let i = 0; i < decisions.length; i++) {
    const bid = replacements[i].blockId;
    if (!decisions[i].isSkip && bid !== undefined) failingBlockIds.add(bid);
  }
  if (failingBlockIds.size > 0) {
    for (let i = 0; i < decisions.length; i++) {
      const bid = replacements[i].blockId;
      if (bid !== undefined && failingBlockIds.has(bid) && decisions[i].isSkip) {
        decisions[i] = { isSkip: false, uniformPct: decisions[i].uniformPct };
      }
    }
  }

  // Step 1: expand bboxes left/right to catch orphan punctuation. Same logic
  // as before, applied to ALL regions regardless of skip/LaMa classification.
  const expandedAll = replacements.map((r) => {
    const box = r.original.boundingBox;
    const charCount = r.original.text.replace(/[^\u4e00-\u9fff\u3400-\u4dbf]/g, '').length || 1;
    const expandPct = Math.min(box.height / charCount, 3);

    let finalX = box.x;
    let finalW = box.width;

    try {
      const leftMargin = {
        x: Math.max(0, box.x - expandPct),
        y: box.y,
        width: box.x - Math.max(0, box.x - expandPct),
        height: box.height,
      };
      if (leftMargin.width > 0.1 && isMarginUniform(page, leftMargin)) {
        finalX = leftMargin.x;
        finalW += (box.x - leftMargin.x);
      }

      const rightEdge = box.x + box.width;
      const rightMargin = {
        x: rightEdge,
        y: box.y,
        width: Math.min(expandPct, 100 - rightEdge),
        height: box.height,
      };
      if (rightMargin.width > 0.1 && isMarginUniform(page, rightMargin)) {
        finalW += rightMargin.width;
      }
    } catch (e) {
      // On error, fall back to original box
    }

    return { x: finalX, y: box.y, width: finalW, height: box.height };
  });

  // Step 2: solid-fill the skip regions onto the working image FIRST.
  // (Composite uses pixel coords. X-padding is independent of Y-padding —
  // X uses solidFillPadX (-2 by default for both languages, to stay strictly
  // inside the bbox and avoid bright-edge halos from glyphs touching the
  // border); Y uses scaledPadding to catch ascender/descender slop.)
  let workingImage = imageBuffer;
  let skipCount = 0;
  const skipComposites: sharp.OverlayOptions[] = [];
  for (let i = 0; i < replacements.length; i++) {
    const d = decisions[i];
    if (!d.isSkip || !d.fillColor) continue;
    skipCount++;
    const exp = expandedAll[i];
    const x = Math.max(0, Math.round((exp.x / 100) * width) - solidFillPadX);
    const y = Math.max(0, Math.round((exp.y / 100) * height) - scaledPadding);
    const wRaw = Math.round((exp.width / 100) * width) + solidFillPadX * 2;
    const hRaw = Math.round((exp.height / 100) * height) + scaledPadding * 2;
    const w = Math.min(wRaw, width - x);
    const h = Math.min(hRaw, height - y);
    if (w <= 0 || h <= 0) continue;
    skipComposites.push({
      input: { create: { width: w, height: h, channels: 3, background: d.fillColor } },
      left: x, top: y,
    });
  }
  if (skipComposites.length > 0) {
    // Feed raw pixel data (from the decode-once step above) into sharp
    // instead of `sharp(workingImage)` which would re-decode the entire PNG.
    // Same final output, one decode skipped.
    workingImage = await sharp(page.data, {
      raw: { width: page.width, height: page.height, channels: page.channels as 1 | 2 | 3 | 4 },
    }).composite(skipComposites).png().toBuffer();
  }

  // Step 3: LaMa pass (only on regions that failed the uniform check).
  // Final colorspace+encode only runs when LaMa actually ran — LaMa server
  // may return non-sRGB output that needs normalization. Solid-fill-only pages
  // skip the redundant re-encode entirely (workingImage from step 2 is already
  // a valid PNG buffer in the source colorspace).
  let inpaintMs = 0;
  const lamaRegions = expandedAll.filter((_, i) => !decisions[i].isSkip);
  if (lamaRegions.length > 0) {
    const maskBuffer = await createMask(width, height, lamaRegions, scaledPadding);
    const inpaintStart = Date.now();
    workingImage = await inpaintImage(workingImage, maskBuffer, lamaRegions);
    inpaintMs = Date.now() - inpaintStart;
    workingImage = await sharp(workingImage).toColourspace('srgb').png().toBuffer();
  }

  const total = replacements.length;
  const skipPct = total > 0 ? Math.round((skipCount / total) * 100) : 0;
  console.log(`[TextReplacer] Inpaint: ${skipCount} solid-fill / ${lamaRegions.length} LaMa (${skipPct}% skip), LaMa ${inpaintMs}ms`);
  return { cleanedImageBuffer: workingImage, inpaintMs, decodePageMs };
}

/**
 * Replace text in an image by painting over original text and adding new text.
 *
 * If `precomputedCleanedImage` is provided, the inpaint step is skipped and the
 * supplied buffer is used directly. This enables the pipeline to inpaint pages
 * in parallel with LLM simplification.
 */
export async function replaceTextInImage(
  imageBuffer: Buffer,
  replacements: ReplacementRegion[],
  style: Partial<TextStyle> = {},
  precomputedCleanedImage?: Buffer,
  precomputedPageData?: RawPageData
): Promise<ReplaceTextResult> {
  const mergedStyle = { ...DEFAULT_STYLE, ...style };

  // Get image dimensions
  const metadata = await sharp(imageBuffer).metadata();
  const width = metadata.width || 800;
  const height = metadata.height || 600;

  let workingImage = imageBuffer;
  const timingBreakdown = { bgDetection: 0, inpaintServer: 0, varianceCheck: 0, textRendering: 0, colorSampling: 0, metaBuild: 0, decodePage: 0 };

  // Calculate scaled padding for inpainting mask. Scale depends on input
  // language — see SOLID_FILL_PADDING_SCALE_* constants for why.
  const baseMargin = mergedStyle.padding;
  const paddingScale = mergedStyle.inputLanguage === 'english'
    ? SOLID_FILL_PADDING_SCALE_EN
    : SOLID_FILL_PADDING_SCALE_ZH;
  const scaledPadding = Math.max(baseMargin, Math.round(Math.min(width, height) * paddingScale));
  // SHRINK the solid-fill rect by 2px on each side (negative padding) for both
  // languages. See runInpaintStep header comment for the rationale.
  const solidFillPadX = -2;

  // Step 1: Inpaint all regions to remove original text
  if (precomputedCleanedImage) {
    workingImage = precomputedCleanedImage;
    console.log(`[TextReplacer] Using precomputed cleaned image (skipped inpaint step)`);
  } else if (mergedStyle.useInpainting && replacements.length > 0) {
    const result = await runInpaintStep(imageBuffer, replacements, width, height, scaledPadding, solidFillPadX, precomputedPageData);
    workingImage = result.cleanedImageBuffer;
    timingBreakdown.inpaintServer += result.inpaintMs;
    timingBreakdown.decodePage += result.decodePageMs;
  }

  const cleanedImageBuffer = workingImage;

  // Step 2: Compute metadata for interactive reader overlays
  const metadataStart = Date.now();

  // Group replacements by blockId
  const blockGroupMap = new Map<number, ReplacementRegion[]>();
  const standaloneReplacements: ReplacementRegion[] = [];
  for (const r of replacements) {
    if (r.blockId !== undefined && r.blockText) {
      if (!blockGroupMap.has(r.blockId)) blockGroupMap.set(r.blockId, []);
      blockGroupMap.get(r.blockId)!.push(r);
    } else {
      standaloneReplacements.push(r);
    }
  }

  const renderedBlocks: RenderedBlockInfo[] = [];

  // Decode the working image ONCE so sampleRegionColor can slice the raw
  // pixel buffer instead of each region triggering its own sharp.extract
  // (each of which would re-decode the entire page).
  const colorStart = Date.now();
  const _decodeStart = Date.now();
  const workingPage = await decodePage(workingImage);
  timingBreakdown.decodePage += Date.now() - _decodeStart;
  const standaloneColors = standaloneReplacements.map(r => {
    const box = r.original.boundingBox;
    const pixelW = Math.round((box.width / 100) * width);
    const pixelH = Math.round((box.height / 100) * height);
    if (pixelW < 5 || pixelH < 5) return null;
    return sampleRegionColor(workingPage, box);
  });

  const blockEntries = [...blockGroupMap.entries()];
  // Oriented render box per block — un-rotates the polygons in pixel space so
  // rotated text isn't sized/placed from the inflated axis-aligned box. See
  // computeRenderBoxPct.
  const blockBoxes = blockEntries.map(([, regions]) => computeRenderBoxPct(regions, width, height));
  const blockColors = blockBoxes.map(box => {
    const pixelW = Math.round((box.width / 100) * width);
    const pixelH = Math.round((box.height / 100) * height);
    if (pixelW < 5 || pixelH < 5) return null;
    return sampleRegionColor(workingPage, box);
  });

  const colorTime = Date.now() - colorStart;
  timingBreakdown.colorSampling += colorTime;
  const metaBuildStart = Date.now();
  // Use the solid-fill bbox dimensions (original OCR bbox + inpaint padding)
  // for rendering instead of the tight OCR bbox. The inpaint step clears a
  // larger area than the OCR bbox to avoid edge halos, so rendering text into
  // the cleared area gives simplified text more room to fit cleanly.
  // Use only HALF of the inpaint padding for rendering — the full padding
  // was tuned for halo prevention (text edges) not for visual fit, and
  // using all of it overflows the visible bubble outline.
  const RENDER_PADDING_FRACTION = 0.5;
  const xPadPct = (solidFillPadX * RENDER_PADDING_FRACTION / width) * 100;
  const yPadPct = (scaledPadding * RENDER_PADDING_FRACTION / height) * 100;

  // Standalone region metadata
  for (let i = 0; i < standaloneReplacements.length; i++) {
    const bgColor = standaloneColors[i];
    if (!bgColor) continue;

    const r = standaloneReplacements[i];
    // Oriented render box — for a rotated standalone line this is the true
    // (un-inflated) extent, so the font isn't sized from the tall AABB. Falls
    // back to the plain bbox for non-rotated text. See computeRenderBoxPct.
    const box = computeRenderBoxPct([r], width, height);
    const expandedBox = {
      x: box.x - xPadPct,
      y: box.y - yPadPct,
      width: box.width + 2 * xPadPct,
      height: box.height + 2 * yPadPct,
    };

    const brightness = 0.299 * bgColor.r + 0.587 * bgColor.g + 0.114 * bgColor.b;
    const textColor = brightness > 128 ? '#000000' : '#FFFFFF';

    const finalText = compactPunctuation(traditionalToSimplified(r.simplifiedText));
    const pixelW = Math.round((expandedBox.width / 100) * width);
    const pixelH = Math.round((expandedBox.height / 100) * height);
    const fontSize = calculateFontSize(finalText, pixelW, pixelH, r.original.isVertical);
    // A standalone region is one OCR line, so its oriented (un-inflated) box
    // height IS the original lettering height.
    const lineHeightPx = (box.height / 100) * height;
    const minFontSize = computeMinFontSize(r.original.text || '', lineHeightPx, width, r.original.isVertical);

    renderedBlocks.push({
      blockId: -1 - renderedBlocks.length,
      bbox: expandedBox,
      text: finalText,
      isVertical: r.original.isVertical,
      fontSize,
      textColor,
      rotation: r.original.rotation || 0,
      minFontSize,
      originalLineHeight: Math.round(lineHeightPx),
    });
  }

  // Block group metadata
  for (let i = 0; i < blockEntries.length; i++) {
    const bgColor = blockColors[i];
    if (!bgColor) continue;

    const [blockId, regions] = blockEntries[i];
    const mergedBoxPct = blockBoxes[i];
    const expandedBlockBox = {
      x: mergedBoxPct.x - xPadPct,
      y: mergedBoxPct.y - yPadPct,
      width: mergedBoxPct.width + 2 * xPadPct,
      height: mergedBoxPct.height + 2 * yPadPct,
    };
    const blockText = compactPunctuation(traditionalToSimplified(regions[0].blockText!));
    const isVertical = regions[0].original.isVertical;
    const pixelW = Math.round((expandedBlockBox.width / 100) * width);
    const pixelH = Math.round((expandedBlockBox.height / 100) * height);

    const brightness = 0.299 * bgColor.r + 0.587 * bgColor.g + 0.114 * bgColor.b;
    const textColor = brightness > 128 ? '#000000' : '#FFFFFF';
    const fontSize = calculateFontSize(blockText, pixelW, pixelH, isVertical);

    const avgRotation = regions.reduce((sum, r) => sum + (r.original.rotation || 0), 0) / regions.length;

    // Each region in a block is one OCR line of the original text; the median
    // oriented height is the original lettering size. This is the last point
    // in the pipeline where per-line boxes still exist — downstream only sees
    // the merged block bbox, which spans ALL lines.
    const lineHeights = regions
      .map(r => (computeRenderBoxPct([r], width, height).height / 100) * height)
      .sort((a, b) => a - b);
    const medianLineH = lineHeights[Math.floor(lineHeights.length / 2)];
    const blockOriginalText = regions[0].originalBlockText || regions.map(r => r.original.text).join(' ');
    const minFontSize = computeMinFontSize(blockOriginalText, medianLineH, width, isVertical);

    renderedBlocks.push({
      blockId,
      bbox: expandedBlockBox,
      text: blockText,
      isVertical,
      fontSize,
      textColor,
      rotation: avgRotation,
      minFontSize,
      originalLineHeight: Math.round(medianLineH),
    });
  }

  timingBreakdown.metaBuild += Date.now() - metaBuildStart;
  console.log(`[TextReplacer] Metadata: color sampling=${colorTime}ms, total=${Date.now() - metadataStart}ms`);
  timingBreakdown.textRendering += Date.now() - metadataStart;
  return { imageBuffer: cleanedImageBuffer, cleanedImageBuffer, renderedBlocks, timing: timingBreakdown };
}

/**
 * Estimate the effective width of text in "full-width character units".
 * CJK characters = 1.0, ASCII/half-width = 0.5
 */
function effectiveCharWidth(text: string): number {
  let width = 0;
  for (const char of text) {
    const code = char.charCodeAt(0);
    // CJK Unified Ideographs, CJK Extension A, fullwidth forms
    if ((code >= 0x4e00 && code <= 0x9fff) || (code >= 0x3400 && code <= 0x4dbf) ||
        (code >= 0xff01 && code <= 0xff60) || (code >= 0xffe0 && code <= 0xffe6)) {
      width += 1.0;
    } else {
      width += 0.5;
    }
  }
  return width;
}

// Minimum-size rule (2026-08 font-size investigation): Chinese dialogue should
// not render smaller than the original lettering. Chinese says the same thing
// in far fewer glyphs, so box-driven sizing produces lines that sit at or below
// the artist's lettering height inside boxes shaped for long English ribbons.
// Target: CN ink = MIN_SIZE_INK_RATIO × original ink height; CN ink occupies
// ~CJK_INK_RATIO of the em (measured on Microsoft YaHei), so
// minFontSize = MIN_SIZE_INK_RATIO / CJK_INK_RATIO × lineHeight. The reader's
// fit pass enforces the floor, overflowing the OCR box into surrounding bubble
// whitespace when necessary. Validated on EN sources only (195 bubbles across
// two Hand Jumper chapters: 68% unaffected, 30% harmless overflow, 3% caught
// by the display exemption / fit fallback) — non-Latin sources are exempt.
// 1.25 ink-to-ink ≈ nominal-size parity with the letterer: comic caps fill
// ~72-80% of their em vs ~95% for hanzi, so 1.25 × (0.72..0.80)/0.95 lands at
// 0.95-1.05× the English's own nominal font size — "at least the size the
// letterer typed", with a hair of safety margin below it.
const MIN_SIZE_INK_RATIO = 1.25;
const CJK_INK_RATIO = 0.95;
// Original lettering taller than this fraction of image width is display/title
// text, not dialogue — multiplying it would demand absurd sizes.
const DISPLAY_LETTERING_MAX_FRAC = 0.05;

// Latin-letter share of non-whitespace chars — gates the minimum-size rule.
function isMostlyLatin(text: string): boolean {
  let latin = 0, total = 0;
  for (const ch of text) {
    if (/\s/.test(ch)) continue;
    total++;
    if (/[A-Za-z]/.test(ch)) latin++;
  }
  return total > 0 && latin / total > 0.5;
}

function computeMinFontSize(
  originalText: string,
  lineHeightPx: number,
  imageWidth: number,
  isVertical: boolean | undefined,
): number | undefined {
  if (isVertical || lineHeightPx <= 0) return undefined;
  if (!isMostlyLatin(originalText)) return undefined;
  if (lineHeightPx > DISPLAY_LETTERING_MAX_FRAC * imageWidth) return undefined;
  return Math.min(72, Math.round((MIN_SIZE_INK_RATIO * lineHeightPx) / CJK_INK_RATIO));
}

/**
 * Estimate font size to fit text in a box using math.
 * Accounts for half-width characters (ASCII, punctuation) being narrower than CJK.
 */
function calculateFontSize(
  text: string,
  maxWidth: number,
  maxHeight: number,
  isVertical: boolean,
): number {
  const charCount = text.length;
  if (charCount === 0) return 10;

  const effWidth = effectiveCharWidth(text);
  let bestSize = 10;

  if (isVertical) {
    const charSpacing = 1.05;
    const colSpacing = 1.05;
    for (let cols = 1; cols <= 10; cols++) {
      const charsPerCol = Math.ceil(charCount / cols);
      const size = Math.min(maxHeight / (charsPerCol * charSpacing), maxWidth / (cols * colSpacing));
      if (size > bestSize) bestSize = size;
    }
  } else {
    const lineSpacing = 1.1;
    for (let lines = 1; lines <= 10; lines++) {
      const effPerLine = effWidth / lines;
      const sizeByWidth = maxWidth / effPerLine;
      const sizeByHeight = maxHeight / (lines * lineSpacing);
      const size = Math.min(sizeByWidth, sizeByHeight);
      if (size > bestSize) bestSize = size;
    }
  }

  // No round-down normalization: snapping to "standard" sizes only ever
  // shrank text, and a size sitting at a wrap threshold that shrinks by even
  // 2% collapses a planned 2-line layout into 1 small line. The reader's fit
  // pass measures the real rendering, so this estimate no longer needs a
  // safety bias in either direction.
  return Math.round(Math.min(Math.max(bestSize, 10), 72));
}

// Compute the render box (percentage coords) for one or more regions. For
// rotated text the axis-aligned bounding box is far larger than the text — a
// wide line tilted even 16° has a tall AABB (AABB_h ≈ line_width·sinθ) — which
// over-sizes the font and makes stacked tilted lines' boxes overlap. We recover
// the true oriented box by un-rotating the OCR polygons.
//
// CRUCIAL: do the un-rotation in PIXEL space, not raw %. Polygon x is % of image
// width and y is % of image height, so on a tall webtoon page (e.g. 690×3731)
// those axes have wildly different pixel scales; rotating in % space distorts
// the geometry and actually *inflates* the box further. Convert to px, un-rotate
// around the centroid, measure the tight box there, then convert back to %.
export function computeRenderBoxPct(
  regions: Array<{ original: { rotation?: number; polygon?: number[][]; boundingBox: { x: number; y: number; width: number; height: number } } }>,
  width: number,
  height: number,
): { x: number; y: number; width: number; height: number } {
  const avgRot = regions.reduce((s, r) => s + (r.original.rotation || 0), 0) / regions.length;
  const hasPolygons = regions.every(r => r.original.polygon && r.original.polygon.length >= 3);

  if (Math.abs(avgRot) > 2 && hasPolygons) {
    const theta = -avgRot * Math.PI / 180; // un-rotate angle
    const cosT = Math.cos(theta), sinT = Math.sin(theta);
    const pts: Array<{ x: number; y: number }> = [];
    for (const r of regions) {
      for (const p of r.original.polygon!) {
        pts.push({ x: (p[0] / 100) * width, y: (p[1] / 100) * height }); // → pixels
      }
    }
    const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
    const cy = pts.reduce((s, p) => s + p.y, 0) / pts.length;
    const u = pts.map(p => ({
      x: cx + (p.x - cx) * cosT - (p.y - cy) * sinT,
      y: cy + (p.x - cx) * sinT + (p.y - cy) * cosT,
    }));
    const wPx = Math.max(...u.map(p => p.x)) - Math.min(...u.map(p => p.x));
    const hPx = Math.max(...u.map(p => p.y)) - Math.min(...u.map(p => p.y));
    const wPct = (wPx / width) * 100, hPct = (hPx / height) * 100;
    const cxPct = (cx / width) * 100, cyPct = (cy / height) * 100;
    return { x: cxPct - wPct / 2, y: cyPct - hPct / 2, width: wPct, height: hPct };
  }

  // Axis-aligned fallback (little/no rotation, or no polygon data)
  const minX = Math.min(...regions.map(r => r.original.boundingBox.x));
  const minY = Math.min(...regions.map(r => r.original.boundingBox.y));
  const maxX = Math.max(...regions.map(r => r.original.boundingBox.x + r.original.boundingBox.width));
  const maxY = Math.max(...regions.map(r => r.original.boundingBox.y + r.original.boundingBox.height));
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
}

/**
 * Create a simple text overlay without replacing original
 * (fallback option if inpainting doesn't work well)
 */
export async function createTextOverlay(
  imageBuffer: Buffer,
  replacements: ReplacementRegion[],
  style: Partial<TextStyle> = {}
): Promise<Buffer> {
  const mergedStyle = { ...DEFAULT_STYLE, ...style, backgroundColor: 'rgba(255, 255, 255, 0.9)' };
  const result = await replaceTextInImage(imageBuffer, replacements, mergedStyle);
  return result.imageBuffer;
}
