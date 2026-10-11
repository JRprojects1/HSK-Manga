import { GoogleGenerativeAI } from '@google/generative-ai';
import Anthropic from '@anthropic-ai/sdk';
import * as fs from 'fs';
import * as http from 'http';
import * as OpenCC from 'opencc-js';
import sharp from 'sharp';
import { imageToBase64 } from './image-processor';
import { fixEnglishSpacing } from './english-splitter';
import { InputLanguage } from './language';

export type VisionProvider = 'gemini' | 'claude' | 'paddleocr' | 'manga-ocr';

// OCR server configuration - supports multiple servers for load balancing
// Set PADDLEOCR_URLS to comma-separated list: "http://host:5000,http://host:5001,http://host:5002"
const PADDLEOCR_URLS = (process.env.PADDLEOCR_URLS || process.env.PADDLEOCR_URL || 'http://localhost:5000')
  .split(',')
  .map(url => url.trim())
  .filter(url => url.length > 0);
// English-input chapters route here. These servers must run the English
// recognition model (en_PP-OCRv5_mobile_rec), which scores 85% on English vs
// the multilingual model's 65%. Falls back to PADDLEOCR_URLS if unset.
const PADDLEOCR_URLS_EN = (process.env.PADDLEOCR_URLS_EN || '')
  .split(',')
  .map(url => url.trim())
  .filter(url => url.length > 0);
// Korean-input chapters route here. These servers must run
// korean_PP-OCRv5_mobile_rec — the default CN pool cannot output hangul at
// all, so without this pool a Korean chapter OCRs to nothing. Japanese needs
// no pool of its own: the CN pool's PP-OCRv5/v6 models read Japanese natively.
const PADDLEOCR_URLS_KO = (process.env.PADDLEOCR_URLS_KO || '')
  .split(',')
  .map(url => url.trim())
  .filter(url => url.length > 0);
const MANGAOCR_URL = process.env.MANGAOCR_URL || 'http://localhost:5000';

// Round-robin counters for load balancing across OCR servers (separate per pool)
let ocrServerIndex = 0;
let ocrServerIndexEn = 0;
let ocrServerIndexKo = 0;
let warnedMissingKoPool = false;

/**
 * Get the next OCR server URL using round-robin. English-input chapters use the
 * English-model pool (PADDLEOCR_URLS_EN) and Korean-input chapters the Korean
 * pool (PADDLEOCR_URLS_KO) when configured; everything else (Chinese, Japanese,
 * auto) uses the default multilingual pool.
 */
function getNextOCRServer(inputLanguage: InputLanguage = 'chinese'): string {
  if (inputLanguage === 'english' && PADDLEOCR_URLS_EN.length > 0) {
    const url = PADDLEOCR_URLS_EN[ocrServerIndexEn % PADDLEOCR_URLS_EN.length];
    ocrServerIndexEn++;
    return url;
  }
  if (inputLanguage === 'korean') {
    if (PADDLEOCR_URLS_KO.length > 0) {
      const url = PADDLEOCR_URLS_KO[ocrServerIndexKo % PADDLEOCR_URLS_KO.length];
      ocrServerIndexKo++;
      return url;
    }
    if (!warnedMissingKoPool) {
      warnedMissingKoPool = true;
      console.error('[OCR] WARNING: Korean input requested but PADDLEOCR_URLS_KO is not set — the default CN pool cannot read hangul, so pages will OCR to empty text.');
    }
  }
  const url = PADDLEOCR_URLS[ocrServerIndex % PADDLEOCR_URLS.length];
  ocrServerIndex++;
  return url;
}

/** True when a dedicated Korean OCR pool is configured (PADDLEOCR_URLS_KO). */
export function hasKoreanOCRPool(): boolean {
  return PADDLEOCR_URLS_KO.length > 0;
}

/**
 * Get the number of available OCR servers for the given input language.
 */
export function getOCRServerCount(inputLanguage: InputLanguage = 'chinese'): number {
  if (inputLanguage === 'english' && PADDLEOCR_URLS_EN.length > 0) {
    return PADDLEOCR_URLS_EN.length;
  }
  if (inputLanguage === 'korean' && PADDLEOCR_URLS_KO.length > 0) {
    return PADDLEOCR_URLS_KO.length;
  }
  return PADDLEOCR_URLS.length;
}


const MAX_RETRIES = 5;
const BASE_DELAY_MS = 5000; // Start with 5 seconds

/**
 * Sleep for specified milliseconds
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Check if error is a rate limit error and extract retry delay
 */
function getRateLimitDelay(error: any): number | null {
  const message = error?.message || String(error);
  if (message.includes('429') || message.includes('Too Many Requests') || message.includes('quota')) {
    // Try to extract retry delay from error message
    const retryMatch = message.match(/retry in (\d+(?:\.\d+)?)/i);
    if (retryMatch) {
      return Math.ceil(parseFloat(retryMatch[1]) * 1000) + 1000; // Add 1 second buffer
    }
    return null; // Use exponential backoff
  }
  return null;
}

export interface TextRegion {
  text: string;
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  confidence: number;
  isVertical: boolean;
  rotation?: number; // Angle in degrees (-180 to 180)
  polygon?: number[][]; // Original 4-point polygon [[x,y], ...]
  preserveAsIs?: boolean; // If true, skip simplification and render original text (e.g., English adjacent to Chinese)
  isColorful?: boolean; // If true, text has high chroma (colored text like red dialogue)
  avgChroma?: number; // Average chroma value of text pixels
}

export interface ExtractionResult {
  regions: TextRegion[];
  fullText: string;
  pageNumber: number;
}

// Initialize Traditional to Simplified converter
const converter = OpenCC.Converter({ from: 'tw', to: 'cn' });

/**
 * Convert Traditional Chinese to Simplified Chinese
 */
export function traditionalToSimplified(text: string): string {
  // OpenCC doesn't convert Taiwan-specific gendered pronouns (妳→你, 祢→你)
  // These aren't traditional/simplified variants but Taiwan-only characters
  return converter(text).replace(/妳/g, '你').replace(/祢/g, '你');
}

// Reverse direction, for the reader's Traditional display toggle. Target is
// 'tw' (Taiwan standard) rather than 't': 'tw' gives 裡面 where 't' gives the
// older 裏面. Deliberately NOT 'twp' — that preset does *lexical* substitution
// (自行车→腳踏車, 服务器→伺服器, 发送→傳送), which would replace the word the
// baked pinyin and CC-CEDICT definition were resolved for.
//
// Unlike the T→S direction above, this is genuinely ambiguous (发→發/髮,
// 干→乾/幹, 里→裡/裏), so callers should pass as much surrounding context as
// they can — OpenCC disambiguates from its phrase dictionary, not per char.
// The 妳/祢 normalization is intentionally not reversed: it is a one-way
// cleanup, and we have no record of which 你 was originally 妳.
const toTraditional = OpenCC.Converter({ from: 'cn', to: 'tw' });

/**
 * Convert Simplified Chinese to Traditional (Taiwan standard).
 * Character-level and length-preserving — see simplifiedToTraditionalInContext
 * for the token-level caller that relies on that property.
 */
export function simplifiedToTraditional(text: string): string {
  return toTraditional(text);
}

/**
 * Convert a single token to Traditional using its neighbours as disambiguation
 * context, then slice the token's own span back out.
 *
 * Converting a bare token loses cross-boundary context: 头发 tokenized as
 * 头 + 发 would yield 頭 + 發 ("send") instead of 頭 + 髮 ("hair"). Feeding
 * prev+token+next to OpenCC lets its phrase dictionary see 头发 and pick 髮,
 * and because the 'tw' mapping is length-preserving we can recover the token's
 * slice by character offset.
 *
 * The length check is a guard, not decoration: if a future OpenCC dictionary
 * ever emits a different-length result the offsets would silently misalign, so
 * we fall back to the context-free conversion instead of returning garbage.
 */
export function simplifiedToTraditionalInContext(
  token: string,
  prev: string,
  next: string,
): string {
  if (!token) return token;
  const context = prev + token + next;
  const converted = toTraditional(context);
  if (converted.length !== context.length) return toTraditional(token);
  return converted.substr(prev.length, token.length);
}

const EXTRACTION_PROMPT = `You are analyzing a Chinese manga/manhua page. Extract ALL text from speech bubbles, text boxes, and any other text in the image.

For each text region, provide:
1. The exact text (keep original Traditional or Simplified Chinese)
2. Approximate bounding box as percentage of image dimensions (x%, y%, width%, height%)
3. Whether the text is vertical (true) or horizontal (false)

Return a JSON object in this exact format:
{
  "regions": [
    {
      "text": "extracted text here",
      "boundingBox": {"x": 10, "y": 20, "width": 30, "height": 15},
      "isVertical": false
    }
  ]
}

Important:
- Extract ALL visible Chinese text in speech bubbles and text boxes
- Coordinates are percentages (0-100) of image width/height
- x,y is the top-left corner of the text region
- EXCLUDE sound effects and onomatopoeia (象声词如：嗙、咔、啪、轰、砰、嘎、嘶、嗖、咻、呜、哗、咚、嘣、噗、嘭、呲)
- EXCLUDE stylized text that is part of the artwork (colored text, angled/rotated text, decorative fonts)
- Only extract dialogue and narrative text in standard speech bubbles or text boxes
- Do not translate, keep original Chinese characters`;

/**
 * Parse extraction response JSON
 */
function parseExtractionResponse(responseText: string, pageNumber: number): ExtractionResult {
  const jsonMatch = responseText.match(/\{[\s\S]*\}/);
  if (!jsonMatch) {
    console.error('Failed to parse extraction response:', responseText);
    return { regions: [], fullText: '', pageNumber };
  }

  try {
    const parsed = JSON.parse(jsonMatch[0]);
    const regions: TextRegion[] = (parsed.regions || []).map((r: any) => ({
      text: r.text || '',
      boundingBox: {
        x: r.boundingBox?.x || 0,
        y: r.boundingBox?.y || 0,
        width: r.boundingBox?.width || 0,
        height: r.boundingBox?.height || 0,
      },
      confidence: r.confidence || 0.9,
      isVertical: r.isVertical || false,
    }));

    const fullText = regions.map((r) => r.text).join('\n');
    return { regions, fullText, pageNumber };
  } catch (e) {
    console.error('JSON parse error:', e);
    return { regions: [], fullText: '', pageNumber };
  }
}

/**
 * Extract text using Claude Vision API
 */
async function extractWithClaude(
  imageBuffer: Buffer,
  pageNumber: number
): Promise<ExtractionResult> {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    throw new Error('ANTHROPIC_API_KEY not found');
  }

  const client = new Anthropic({ apiKey });
  const base64Image = imageToBase64(imageBuffer);

  const response = await client.messages.create({
    model: 'claude-sonnet-4-20250514',
    max_tokens: 4096,
    messages: [
      {
        role: 'user',
        content: [
          {
            type: 'image',
            source: {
              type: 'base64',
              media_type: 'image/png',
              data: base64Image,
            },
          },
          {
            type: 'text',
            text: EXTRACTION_PROMPT,
          },
        ],
      },
    ],
  });

  const textContent = response.content.find((c) => c.type === 'text');
  const responseText = textContent?.type === 'text' ? textContent.text : '';
  return parseExtractionResponse(responseText, pageNumber);
}

/**
 * Extract text using Gemini Vision API
 */
async function extractWithGemini(
  imageBuffer: Buffer,
  pageNumber: number
): Promise<ExtractionResult> {
  const apiKey = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY not found');
  }

  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });
  const base64Image = imageToBase64(imageBuffer);

  // Retry loop with exponential backoff
  let lastError: any;
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      const result = await model.generateContent([
        {
          inlineData: {
            mimeType: 'image/png',
            data: base64Image,
          },
        },
        EXTRACTION_PROMPT,
      ]);

      const responseText = result.response.text();
      return parseExtractionResponse(responseText, pageNumber);
    } catch (error: any) {
      lastError = error;

      // Check if it's a rate limit error
      const suggestedDelay = getRateLimitDelay(error);
      if (suggestedDelay !== null || (error?.message && error.message.includes('429'))) {
        const delayMs = suggestedDelay || BASE_DELAY_MS * Math.pow(2, attempt);
        console.error(`  Rate limited. Waiting ${(delayMs / 1000).toFixed(1)}s before retry ${attempt + 1}/${MAX_RETRIES}...`);
        await sleep(delayMs);
        continue;
      }

      // Not a rate limit error, throw immediately
      throw error;
    }
  }

  // All retries exhausted
  throw lastError || new Error('Max retries exceeded for Gemini API');
}

// Left padding - disabled as it didn't help detect 一
// The post-OCR repair approach works better
const LEFT_PADDING_PX = 0;

// Upscale factor - disabled, upscaling caused bracket misreads
const UPSCALE_FACTOR = 1;

/**
 * Common patterns where 一 (yi) is often missed by PaddleOCR at line start.
 * Maps: detected text → corrected text with 一 prepended.
 * Only applies if the detected text starts with these patterns.
 */
/**
 * Extract text using PaddleOCR server (local)
 */
async function extractWithPaddleOCR(
  imageBuffer: Buffer,
  pageNumber: number,
  inputLanguage: InputLanguage = 'chinese'
): Promise<ExtractionResult> {
  const serverUrl = getNextOCRServer(inputLanguage);
  const url = new URL(serverUrl);

  // Preprocess image to ensure consistent format for PaddleOCR
  // This fixes "axes don't match array" errors from RGBA, grayscale, or unusual color spaces
  let processedBuffer: Buffer;
  let originalWidth = 0;
  let originalHeight = 0;
  let paddingApplied = 0;

  try {
    const metadata = await sharp(imageBuffer).metadata();
    originalWidth = metadata.width || 0;
    originalHeight = metadata.height || 0;

    // Skip images that are too small to contain meaningful text
    if (originalWidth < 20 || originalHeight < 20) {
      return {
        fullText: '',
        regions: [],
        pageNumber,
      };
    }

    // Normalize format: flatten alpha onto white, convert to sRGB, ensure PNG
    // Also upscale to improve detection of thin strokes like 一
    // At low resolution, 一 can be 1-2 pixels tall and gets filtered as noise
    const scaledWidth = originalWidth * UPSCALE_FACTOR;
    const scaledHeight = originalHeight * UPSCALE_FACTOR;

    const normalizedBuffer = await sharp(imageBuffer)
      .flatten({ background: { r: 255, g: 255, b: 255 } })
      .toColourspace('srgb')
      .resize(scaledWidth, scaledHeight, { kernel: 'lanczos3' })  // High-quality upscale
      .sharpen()  // Sharpen to enhance thin strokes
      .png()
      .toBuffer();

    console.log(`[OCR] Upscaled ${originalWidth}x${originalHeight} → ${scaledWidth}x${scaledHeight} (${UPSCALE_FACTOR}x)`);

    // Add left padding to help detect line-start characters like 一
    // PaddleOCR's DBNet suppresses horizontal strokes at left edge (thinks they're borders)
    // Adding padding moves text away from edge, bypassing this suppression
    // Determine padding color based on image brightness (sample top-left corner)
    const scaledPadding = LEFT_PADDING_PX * UPSCALE_FACTOR;
    const cornerSample = await sharp(normalizedBuffer)
      .extract({ left: 0, top: 0, width: Math.min(40, scaledWidth), height: Math.min(40, scaledHeight) })
      .raw()
      .toBuffer();
    const avgBrightness = cornerSample.reduce((sum, val, i) => i % 3 === 0 ? sum + val : sum, 0) / (cornerSample.length / 3);
    const paddingColor = avgBrightness < 128 ? { r: 0, g: 0, b: 0 } : { r: 255, g: 255, b: 255 };

    processedBuffer = await sharp({
      create: {
        width: scaledWidth + scaledPadding,
        height: scaledHeight,
        channels: 3,
        background: paddingColor,
      }
    })
      .composite([{ input: normalizedBuffer, left: scaledPadding, top: 0 }])
      .png()
      .toBuffer();

    paddingApplied = scaledPadding;
    console.log(`[OCR] Added ${scaledPadding}px left padding (color: ${avgBrightness < 128 ? 'black' : 'white'})`);

  } catch (preprocessError) {
    console.error(`  Image preprocessing failed: ${preprocessError}`);
    // Fall back to original buffer if preprocessing fails
    processedBuffer = imageBuffer;
    paddingApplied = 0;
  }

  const base64Image = processedBuffer.toString('base64');

  // `lang` selects the recognition head on the unified OCR server
  // (scripts/ocr-server/ocr_server.py). Legacy single-model servers ignore it,
  // so it is always safe to send. JA rides the cn head (reads Japanese natively).
  const ocrLang = inputLanguage === 'korean' ? 'ko' : inputLanguage === 'english' ? 'en' : 'cn';
  const requestData = JSON.stringify({
    image: base64Image,
    lang: ocrLang,
  });

  return new Promise((resolve, reject) => {
    const options = {
      hostname: url.hostname,
      port: url.port || 5000,
      path: '/ocr',
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(requestData),
      },
    };

    const req = http.request(options, (res) => {
      let body = '';
      res.on('data', (chunk) => (body += chunk));
      res.on('end', () => {
        try {
          const result = JSON.parse(body);
          if (result.error) {
            reject(new Error(result.error));
            return;
          }

          // Get image dimensions from server response - MUST be valid
          const imgWidth = result.image_width;
          const imgHeight = result.image_height;
          if (!imgWidth || !imgHeight || imgWidth < 1 || imgHeight < 1) {
            reject(new Error('Invalid image dimensions from OCR server'));
            return;
          }

          // Log ALL raw OCR results BEFORE filtering (for debugging 一 detection issue)
          console.log(`[OCR] Raw results from PaddleOCR: ${(result.results || []).length} detections`);
          (result.results || []).forEach((d: any, i: number) => {
            const xs = d.box?.map((p: any) => p[0]) || [];
            const ys = d.box?.map((p: any) => p[1]) || [];
            const w = xs.length ? Math.max(...xs) - Math.min(...xs) : 0;
            const h = ys.length ? Math.max(...ys) - Math.min(...ys) : 0;
            const rdx = d.box?.[1]?.[0] - d.box?.[0]?.[0] || 0;
            const rdy = d.box?.[1]?.[1] - d.box?.[0]?.[1] || 0;
            const rot = Math.atan2(rdy, rdx) * (180 / Math.PI);
            console.log(`[OCR]   ${i}: text="${d.text}" conf=${(d.confidence || 0).toFixed(2)} size=${Math.round(w)}x${Math.round(h)} area=${Math.round(w*h)} rot=${rot.toFixed(1)}°`);
          });

          // Convert PaddleOCR format to our format (with percentage coordinates)
          const regions: TextRegion[] = (result.results || [])
            .filter((d: any) => {
              // Validate box structure
              if (!d.box || !Array.isArray(d.box) || d.box.length !== 4) {
                console.log(`[OCR] FILTERED (invalid box): "${d.text}"`);
                return false;
              }
              if (!d.box.every((p: any) => Array.isArray(p) && p.length === 2)) {
                console.log(`[OCR] FILTERED (invalid box points): "${d.text}"`);
                return false;
              }

              // Validate text exists
              if (!d.text || d.text.trim().length === 0) {
                console.log(`[OCR] FILTERED (empty text)`);
                return false;
              }

              // Calculate size for logging
              const xs = d.box.map((p: any) => p[0]);
              const ys = d.box.map((p: any) => p[1]);
              const w = Math.max(...xs) - Math.min(...xs);
              const h = Math.max(...ys) - Math.min(...ys);

              // Minimum confidence threshold
              if ((d.confidence || 0) < 0.55) {
                console.log(`[OCR] FILTERED (low confidence): "${d.text}" conf=${(d.confidence || 0).toFixed(2)} size=${Math.round(w)}x${Math.round(h)}`);
                return false;
              }

              // Minimum size filter - use area-based filtering to allow narrow characters like 一 (yi)
              // A 30x4 pixel region (120 sq px) is valid, but 5x5 (25 sq px) is noise
              if (w * h < 100) {
                console.log(`[OCR] FILTERED (too small): "${d.text}" size=${Math.round(w)}x${Math.round(h)} area=${Math.round(w*h)}`);
                return false;
              }

              return true;
            })
            .map((d: any) => {
              // PaddleOCR returns box as [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
              // This is: top-left, top-right, bottom-right, bottom-left
              const box = d.box;

              // Calculate rotation angle from the top edge
              const dx = box[1][0] - box[0][0];
              const dy = box[1][1] - box[0][1];
              const rotation = Math.atan2(dy, dx) * (180 / Math.PI);

              // Get axis-aligned bounding box
              const xs = box.map((p: any) => p[0]);
              const ys = box.map((p: any) => p[1]);
              const x1 = Math.min(...xs);
              const y1 = Math.min(...ys);
              const x2 = Math.max(...xs);
              const y2 = Math.max(...ys);

              const width = x2 - x1;
              const height = y2 - y1;

              // Detect if text is vertical - more conservative threshold
              // Only consider vertical if height > width * 1.5 AND short text
              const isVertical = height > width * 1.5 && d.text.length <= 10;

              // Adjust coordinates to account for:
              // 1. Left padding (subtract paddingApplied from x)
              // 2. Upscaling (divide by UPSCALE_FACTOR to get back to original space)
              const adjustedX1 = Math.max(0, (x1 - paddingApplied) / UPSCALE_FACTOR);
              const adjustedX2 = Math.max(0, (x2 - paddingApplied) / UPSCALE_FACTOR);
              const adjustedY1 = y1 / UPSCALE_FACTOR;
              const adjustedY2 = y2 / UPSCALE_FACTOR;
              const adjustedWidth = adjustedX2 - adjustedX1;
              const adjustedHeight = adjustedY2 - adjustedY1;

              // Use original dimensions for percentage calculations
              const refWidth = originalWidth > 0 ? originalWidth : imgWidth / UPSCALE_FACTOR;
              const refHeight = originalHeight > 0 ? originalHeight : imgHeight / UPSCALE_FACTOR;

              // Chinese-trained OCR models drop spaces inside mixed Chinese+English
              // line detections. Apply dictionary-based word splitting to recover
              // spaces in long all-letter runs — see src/english-splitter.ts.
              const fixedText = fixEnglishSpacing(d.text);
              if (fixedText !== d.text) {
                console.log(`[OCR] English spacing fix: "${d.text}" → "${fixedText}"`);
              }

              // Convert pixel coordinates to percentages
              return {
                text: fixedText,
                boundingBox: {
                  x: (adjustedX1 / refWidth) * 100,
                  y: (adjustedY1 / refHeight) * 100,
                  width: (adjustedWidth / refWidth) * 100,
                  height: (adjustedHeight / refHeight) * 100,
                },
                confidence: d.confidence || 0.9,
                isVertical,
                rotation: Math.abs(rotation) > 5 ? rotation : 0, // Only store if significant (5° threshold — below this is OCR noise)
                polygon: box.map((p: any) => [
                  (Math.max(0, (p[0] - paddingApplied) / UPSCALE_FACTOR) / refWidth) * 100,
                  ((p[1] / UPSCALE_FACTOR) / refHeight) * 100
                ]),
              };
            });

          // Log final accepted regions
          console.log(`[OCR] Accepted ${regions.length} regions:`);
          regions.forEach((r, i) => console.log(`[OCR]   ${i}: "${r.text}"`));

          const fullText = regions.map((r) => r.text).join('\n');
          const extractionResult = { regions, fullText, pageNumber };

          resolve(extractionResult);
        } catch (e) {
          reject(new Error(`Failed to parse PaddleOCR response: ${body}`));
        }
      });
    });

    req.on('error', (e) => {
      const err: any = new Error(`PaddleOCR request failed: ${e.message}`);
      err.code = 'OCR_TIMEOUT';
      reject(err);
    });
    req.setTimeout(120000, () => {
      req.destroy();
      const err: any = new Error('PaddleOCR request timed out');
      err.code = 'OCR_TIMEOUT';
      reject(err);
    });
    req.write(requestData);
    req.end();
  });
}

/**
 * Extract text using MangaOCR server (local)
 * MangaOCR is specialized for manga/manhua text recognition
 * Note: MangaOCR only does recognition, not detection - we use PaddleOCR for detection
 */
async function extractWithMangaOCR(
  imageBuffer: Buffer,
  pageNumber: number
): Promise<ExtractionResult> {
  const url = new URL(MANGAOCR_URL);
  const base64Image = imageBuffer.toString('base64');

  const requestData = JSON.stringify({
    image: base64Image,
  });

  return new Promise((resolve, reject) => {
    const options = {
      hostname: url.hostname,
      port: url.port || 5000,
      path: '/manga-ocr',
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(requestData),
      },
    };

    const req = http.request(options, (res) => {
      let body = '';
      res.on('data', (chunk) => (body += chunk));
      res.on('end', () => {
        try {
          const result = JSON.parse(body);
          if (result.error) {
            reject(new Error(result.error));
            return;
          }

          // Get image dimensions from server response - MUST be valid
          const imgWidth = result.image_width;
          const imgHeight = result.image_height;
          if (!imgWidth || !imgHeight || imgWidth < 1 || imgHeight < 1) {
            reject(new Error('Invalid image dimensions from MangaOCR server'));
            return;
          }

          // Convert MangaOCR format to our format (with percentage coordinates)
          const regions: TextRegion[] = (result.detections || [])
            .filter((d: any) => {
              // Validate box structure
              if (!d.box || !Array.isArray(d.box) || d.box.length !== 4) {
                return false;
              }
              if (!d.box.every((p: any) => Array.isArray(p) && p.length === 2)) {
                return false;
              }

              // Validate text exists
              if (!d.text || d.text.trim().length === 0) {
                return false;
              }

              // Minimum confidence threshold
              if ((d.confidence || 0) < 0.55) {
                return false;
              }

              // Calculate size for minimum size filter
              const xs = d.box.map((p: any) => p[0]);
              const ys = d.box.map((p: any) => p[1]);
              const w = Math.max(...xs) - Math.min(...xs);
              const h = Math.max(...ys) - Math.min(...ys);

              // Minimum size filter - use area-based filtering to allow narrow characters like 一 (yi)
              // A 30x4 pixel region (120 sq px) is valid, but 5x5 (25 sq px) is noise
              if (w * h < 100) {
                return false;
              }

              return true;
            })
            .map((d: any) => {
              const box = d.box;

              // Calculate rotation angle from the top edge
              const dx = box[1][0] - box[0][0];
              const dy = box[1][1] - box[0][1];
              const rotation = Math.atan2(dy, dx) * (180 / Math.PI);

              // Get axis-aligned bounding box
              const xs = box.map((p: any) => p[0]);
              const ys = box.map((p: any) => p[1]);
              const x1 = Math.min(...xs);
              const y1 = Math.min(...ys);
              const x2 = Math.max(...xs);
              const y2 = Math.max(...ys);

              const width = x2 - x1;
              const height = y2 - y1;

              // Detect if text is vertical
              const isVertical = height > width * 1.5 && d.text.length <= 10;

              // Chinese-trained OCR models drop spaces inside mixed Chinese+English
              // line detections. Apply dictionary-based word splitting to recover
              // spaces in long all-letter runs — see src/english-splitter.ts.
              const fixedText = fixEnglishSpacing(d.text);
              if (fixedText !== d.text) {
                console.log(`[OCR] English spacing fix: "${d.text}" → "${fixedText}"`);
              }

              // Convert pixel coordinates to percentages
              return {
                text: fixedText,
                boundingBox: {
                  x: (x1 / imgWidth) * 100,
                  y: (y1 / imgHeight) * 100,
                  width: (width / imgWidth) * 100,
                  height: (height / imgHeight) * 100,
                },
                confidence: d.confidence || 0.9,
                isVertical,
                rotation: Math.abs(rotation) > 2 ? rotation : 0,
                polygon: box.map((p: any) => [(p[0] / imgWidth) * 100, (p[1] / imgHeight) * 100]),
              };
            });

          const fullText = regions.map((r) => r.text).join('\n');
          resolve({ regions, fullText, pageNumber });
        } catch (e) {
          reject(new Error(`Failed to parse MangaOCR response: ${body}`));
        }
      });
    });

    req.on('error', (e) => reject(new Error(`MangaOCR request failed: ${e.message}`)));
    req.setTimeout(120000, () => {
      req.destroy();
      reject(new Error('MangaOCR request timed out'));
    });
    req.write(requestData);
    req.end();
  });
}

/**
 * Extract text and bounding boxes from an image using Vision API
 * Supports Gemini, Claude, PaddleOCR, and MangaOCR as providers
 */
export async function extractTextFromImage(
  imageBuffer: Buffer,
  pageNumber: number = 1,
  provider: VisionProvider = 'gemini',
  inputLanguage: InputLanguage = 'chinese'
): Promise<ExtractionResult> {
  if (provider === 'claude') {
    return extractWithClaude(imageBuffer, pageNumber);
  }
  if (provider === 'paddleocr') {
    return extractWithPaddleOCR(imageBuffer, pageNumber, inputLanguage);
  }
  if (provider === 'manga-ocr') {
    return extractWithMangaOCR(imageBuffer, pageNumber);
  }
  return extractWithGemini(imageBuffer, pageNumber);
}

/**
 * Extract text from multiple pages
 */
export async function extractTextFromPages(
  imageBuffers: Buffer[],
  provider: VisionProvider = 'gemini',
  inputLanguage: InputLanguage = 'chinese'
): Promise<ExtractionResult[]> {
  const results: ExtractionResult[] = [];

  for (let i = 0; i < imageBuffers.length; i++) {
    console.error(`Extracting text from page ${i + 1}/${imageBuffers.length}...`);
    const result = await extractTextFromImage(imageBuffers[i], i + 1, provider, inputLanguage);
    results.push(result);
  }

  return results;
}

/**
 * Convert all extracted text from Traditional to Simplified Chinese
 */
export function convertExtractionToSimplified(
  extraction: ExtractionResult
): ExtractionResult {
  return {
    ...extraction,
    regions: extraction.regions.map((r) => ({
      ...r,
      text: traditionalToSimplified(r.text),
    })),
    fullText: traditionalToSimplified(extraction.fullText),
  };
}
