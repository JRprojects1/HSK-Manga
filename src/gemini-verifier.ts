import { GoogleGenerativeAI } from '@google/generative-ai';
import sharp from 'sharp';
import { TextRegion } from './text-extractor';
import { GrayDetectionResult } from './text-replacer';

/**
 * Result of Gemini verification for a gray-detected region.
 */
export interface GeminiVerificationResult {
  originalText: string;
  correctedText: string;
  originalBoundingBox: { x: number; y: number; width: number; height: number };
  expandedBoundingBox: { x: number; y: number; width: number; height: number };
  wasModified: boolean;
}

/**
 * Crop a region from an image buffer.
 * Coordinates are in percentages (0-100).
 */
async function cropRegion(
  imageBuffer: Buffer,
  boundingBox: { x: number; y: number; width: number; height: number },
  imageWidth: number,
  imageHeight: number
): Promise<Buffer> {
  const left = Math.max(0, Math.round((boundingBox.x / 100) * imageWidth));
  const top = Math.max(0, Math.round((boundingBox.y / 100) * imageHeight));
  const width = Math.min(
    Math.round((boundingBox.width / 100) * imageWidth),
    imageWidth - left
  );
  const height = Math.min(
    Math.round((boundingBox.height / 100) * imageHeight),
    imageHeight - top
  );

  if (width <= 0 || height <= 0) {
    throw new Error('Invalid crop dimensions');
  }

  return sharp(imageBuffer)
    .extract({ left, top, width, height })
    .png()
    .toBuffer();
}

/**
 * Expand a bounding box to the left to capture potential missed characters.
 * Returns new bounding box in percentage coordinates.
 */
function expandBoundingBoxLeft(
  boundingBox: { x: number; y: number; width: number; height: number },
  expandPercent: number = 20
): { x: number; y: number; width: number; height: number } {
  const newX = Math.max(0, boundingBox.x - expandPercent);
  const addedWidth = boundingBox.x - newX;
  return {
    x: newX,
    y: boundingBox.y,
    width: boundingBox.width + addedWidth,
    height: boundingBox.height
  };
}

/**
 * Expand a bounding box upward to capture potential missed characters above.
 * Used for vertical text where leading punctuation appears above the detected region.
 */
function expandBoundingBoxUp(
  boundingBox: { x: number; y: number; width: number; height: number },
  expandPercent: number = 20
): { x: number; y: number; width: number; height: number } {
  const newY = Math.max(0, boundingBox.y - expandPercent);
  const addedHeight = boundingBox.y - newY;
  return {
    x: boundingBox.x,
    y: newY,
    width: boundingBox.width,
    height: boundingBox.height + addedHeight
  };
}

/**
 * Result of leading punctuation check.
 */
export interface LeadingPunctuationResult {
  regionIndex: number;
  foundPunctuation: string;
  expandedBoundingBox: { x: number; y: number; width: number; height: number };
  wasModified: boolean;
}

/**
 * Check a single gray-detected region for leading punctuation that OCR missed.
 * Crops just the area before the detected text (left for horizontal, up for vertical)
 * and asks Gemini a simple yes/no question about punctuation.
 */
export async function checkLeadingPunctuation(
  imageBuffer: Buffer,
  imageWidth: number,
  imageHeight: number,
  region: TextRegion,
  regionIndex: number
): Promise<LeadingPunctuationResult> {
  const noChange: LeadingPunctuationResult = {
    regionIndex,
    foundPunctuation: '',
    expandedBoundingBox: region.boundingBox,
    wasModified: false
  };

  try {
    // Expand in reading direction: left for horizontal, up for vertical
    // Expand by ~1.5x the character dimension to capture a few characters of punctuation
    let cropBox: { x: number; y: number; width: number; height: number };

    if (region.isVertical) {
      const expandAmount = Math.min(region.boundingBox.width * 1.5, region.boundingBox.y);
      cropBox = {
        x: region.boundingBox.x,
        y: Math.max(0, region.boundingBox.y - expandAmount),
        width: region.boundingBox.width,
        height: expandAmount + region.boundingBox.height * 0.3
      };
    } else {
      const expandAmount = Math.min(region.boundingBox.height * 1.5, region.boundingBox.x);
      cropBox = {
        x: Math.max(0, region.boundingBox.x - expandAmount),
        y: region.boundingBox.y,
        width: expandAmount + region.boundingBox.width * 0.3,
        height: region.boundingBox.height
      };
    }

    if (cropBox.width < 1 || cropBox.height < 1) {
      return noChange;
    }

    const croppedImage = await cropRegion(imageBuffer, cropBox, imageWidth, imageHeight);

    const punctuationPrompt = `Look at this image crop from a manga speech bubble.
Is there any punctuation visible (such as ... or 。or ？ or ！ or 、 or ……)?
If YES: respond with ONLY the exact punctuation characters you see, nothing else.
If NO: respond with exactly "NONE".`;

    console.log(`[PunctuationCheck] Checking region "${region.text.substring(0, 20)}..." for leading punctuation`);
    const geminiResponse = await callGeminiFlash(croppedImage, punctuationPrompt);

    if (!geminiResponse || geminiResponse.trim().toUpperCase() === 'NONE' || geminiResponse.trim() === '') {
      console.log(`[PunctuationCheck] No leading punctuation found`);
      return noChange;
    }

    // Validate response is actually punctuation
    const cleaned = geminiResponse.trim();
    if (!/^[.。？！、…，\s]+$/.test(cleaned)) {
      console.log(`[PunctuationCheck] Gemini returned non-punctuation: "${cleaned}" — ignoring`);
      return noChange;
    }

    // Expand the bounding box to cover the punctuation area
    let expandedBox: { x: number; y: number; width: number; height: number };
    if (region.isVertical) {
      expandedBox = expandBoundingBoxUp(region.boundingBox, cropBox.y === 0 ? region.boundingBox.y : region.boundingBox.y - cropBox.y);
    } else {
      expandedBox = expandBoundingBoxLeft(region.boundingBox, cropBox.x === 0 ? region.boundingBox.x : region.boundingBox.x - cropBox.x);
    }

    console.log(`[PunctuationCheck] Found leading "${cleaned}" before "${region.text.substring(0, 20)}..."`);

    return {
      regionIndex,
      foundPunctuation: cleaned,
      expandedBoundingBox: expandedBox,
      wasModified: true
    };
  } catch (error: any) {
    console.error(`[PunctuationCheck] Error checking region "${region.text.substring(0, 20)}...":`, error.message);
    return noChange;
  }
}

// Track which API key index to use (for rotation on quota errors)
let currentApiKeyIndex = 0;

/**
 * Get list of available Gemini API keys.
 * Supports comma-separated keys for rotation.
 */
function getApiKeys(): string[] {
  const keyStr = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY || '';
  return keyStr.split(',').map(k => k.trim()).filter(k => k.length > 0);
}

/**
 * Call Gemini Flash to read text from an image region.
 * Uses the fast and cheap gemini-2.5-flash model.
 * Supports multiple API keys with automatic rotation on quota errors.
 */
async function callGeminiFlash(imageBuffer: Buffer, customPrompt?: string): Promise<string> {
  const apiKeys = getApiKeys();
  if (apiKeys.length === 0) {
    throw new Error('GEMINI_API_KEY not found - cannot verify gray regions');
  }

  const prompt = customPrompt || `What Chinese text is shown in this image?
Return ONLY the exact text you see, nothing else.
If there are multiple lines, separate them with newlines.
Include all characters including 一 (yi) if present.
Include all punctuation exactly as shown (。？，！etc).`;

  // Try each API key until one works
  for (let attempts = 0; attempts < apiKeys.length; attempts++) {
    const keyIndex = (currentApiKeyIndex + attempts) % apiKeys.length;
    const apiKey = apiKeys[keyIndex];

    try {
      const genAI = new GoogleGenerativeAI(apiKey);
      const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });

      const result = await model.generateContent([
        {
          inlineData: {
            mimeType: 'image/png',
            data: imageBuffer.toString('base64'),
          },
        },
        prompt,
      ]);

      return result.response.text().trim();
    } catch (error: any) {
      const isQuotaError = error.message?.includes('quota') ||
                          error.message?.includes('429') ||
                          error.message?.includes('RESOURCE_EXHAUSTED');

      if (isQuotaError && attempts < apiKeys.length - 1) {
        console.error(`[GeminiVerifier] API key ${keyIndex + 1} quota exceeded, trying next key...`);
        currentApiKeyIndex = (keyIndex + 1) % apiKeys.length;
        continue;
      }

      console.error('[GeminiVerifier] API error:', error.message);
      return '';
    }
  }

  return '';
}

/**
 * Verify gray-detected regions with Gemini Flash.
 * For each region where gray background was detected (indicating possible missed text),
 * uses Gemini to read the actual text and returns corrected regions.
 *
 * @param imageBuffer - The full original image
 * @param imageWidth - Image width in pixels
 * @param imageHeight - Image height in pixels
 * @param grayRegions - Regions with gray detection results and their original TextRegion
 * @returns Array of verification results for regions that were modified
 */
export async function verifyGrayRegionsWithGemini(
  imageBuffer: Buffer,
  imageWidth: number,
  imageHeight: number,
  grayRegions: Array<{ grayResult: GrayDetectionResult; region: TextRegion }>
): Promise<GeminiVerificationResult[]> {
  const results: GeminiVerificationResult[] = [];

  for (const { grayResult, region } of grayRegions) {
    try {
      // Expand crop area to the left to include potential missed characters
      const expandedBox = expandBoundingBoxLeft(region.boundingBox, 20);

      // Crop the expanded region from the original image
      const croppedImage = await cropRegion(imageBuffer, expandedBox, imageWidth, imageHeight);

      // Ask Gemini to read the text
      console.log(`[GeminiVerifier] Verifying region with text: "${region.text}"`);
      const geminiText = await callGeminiFlash(croppedImage);

      if (geminiText && geminiText !== region.text) {
        console.log(`[GeminiVerifier] Corrected: "${region.text}" → "${geminiText}"`);
        results.push({
          originalText: region.text,
          correctedText: geminiText,
          originalBoundingBox: region.boundingBox,
          expandedBoundingBox: expandedBox,
          wasModified: true
        });
      } else {
        console.log(`[GeminiVerifier] No change needed for: "${region.text}"`);
        results.push({
          originalText: region.text,
          correctedText: region.text,
          originalBoundingBox: region.boundingBox,
          expandedBoundingBox: region.boundingBox,
          wasModified: false
        });
      }
    } catch (error: any) {
      console.error(`[GeminiVerifier] Error verifying region "${region.text}":`, error.message);
      // On error, keep original
      results.push({
        originalText: region.text,
        correctedText: region.text,
        originalBoundingBox: region.boundingBox,
        expandedBoundingBox: region.boundingBox,
        wasModified: false
      });
    }
  }

  return results;
}
