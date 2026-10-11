import sharp from 'sharp';

export interface InpaintingConfig {
  serverUrls: string[];
  enabled: boolean;
}

// Parse multiple server URLs from environment variable
function parseServerUrls(): string[] {
  const urlsEnv = process.env.INPAINTING_SERVER_URLS || process.env.INPAINTING_SERVER_URL;
  if (!urlsEnv) {
    return ['http://localhost:8080'];
  }
  return urlsEnv.split(',').map(url => url.trim()).filter(url => url.length > 0);
}

const DEFAULT_CONFIG: InpaintingConfig = {
  serverUrls: parseServerUrls(),
  enabled: process.env.ENABLE_INPAINTING === 'true',
};

// Round-robin counter for server selection
let serverIndex = 0;

/**
 * Get the next server URL using round-robin selection
 */
function getNextServerUrl(): string {
  const urls = DEFAULT_CONFIG.serverUrls;
  const url = urls[serverIndex % urls.length];
  serverIndex++;
  return url;
}

// Log config at startup for debugging
console.log(`[Inpainting] Config: serverUrls=${DEFAULT_CONFIG.serverUrls.join(', ')}, enabled=${DEFAULT_CONFIG.enabled}`);
console.log(`[Inpainting] Env vars: INPAINTING_SERVER_URLS=${process.env.INPAINTING_SERVER_URLS}, ENABLE_INPAINTING=${process.env.ENABLE_INPAINTING}`);

/**
 * Create a binary mask from bounding boxes (white = area to inpaint)
 */
export async function createMask(
  imageWidth: number,
  imageHeight: number,
  regions: Array<{ x: number; y: number; width: number; height: number }>,
  padding: number = 4
): Promise<Buffer> {
  // Create black image (0 = keep)
  const mask = sharp({
    create: {
      width: imageWidth,
      height: imageHeight,
      channels: 3,
      background: { r: 0, g: 0, b: 0 },
    },
  });

  // Create white rectangles for each region (255 = inpaint)
  const composites: sharp.OverlayOptions[] = [];

  for (const region of regions) {
    // Convert percentage to pixels if needed (values > 100 are already pixels)
    const x = region.x > 100 ? region.x : Math.round((region.x / 100) * imageWidth);
    const y = region.y > 100 ? region.y : Math.round((region.y / 100) * imageHeight);
    const w = region.width > 100 ? region.width : Math.round((region.width / 100) * imageWidth);
    const h = region.height > 100 ? region.height : Math.round((region.height / 100) * imageHeight);

    // Add padding and clamp to image bounds
    const left = Math.max(0, x - padding);
    const top = Math.max(0, y - padding);
    const width = Math.min(w + padding * 2, imageWidth - left);
    const height = Math.min(h + padding * 2, imageHeight - top);

    if (width > 0 && height > 0) {
      // Create white rectangle
      const rect = await sharp({
        create: {
          width,
          height,
          channels: 3,
          background: { r: 255, g: 255, b: 255 },
        },
      })
        .png()
        .toBuffer();

      composites.push({ input: rect, left, top });
    }
  }

  if (composites.length === 0) {
    return mask.png().toBuffer();
  }

  // Create the mask with white rectangles
  let maskBuffer = await mask.composite(composites).png().toBuffer();

  // Apply feathering (blur) to soften edges for smoother LaMa inpainting
  const MASK_FEATHER_RADIUS = parseFloat(process.env.MASK_FEATHER_RADIUS || '5');
  if (MASK_FEATHER_RADIUS > 0) {
    const sigma = Math.max(0.3, MASK_FEATHER_RADIUS);
    maskBuffer = await sharp(maskBuffer)
      .blur(sigma)
      .png()
      .toBuffer();
    console.log(`[Inpainting] Applied mask feathering with radius ${sigma}`);
  }

  return maskBuffer;
}

// Resize scale for faster inpainting (0.5 = half size, 1.0 = no resize)
const INPAINT_SCALE = parseFloat(process.env.INPAINT_SCALE || '0.5');

/**
 * Resize image and mask for faster inpainting, then scale result back up
 */
async function resizeForInpainting(
  imageBuffer: Buffer,
  maskBuffer: Buffer,
  scale: number
): Promise<{ resizedImage: Buffer; resizedMask: Buffer; originalWidth: number; originalHeight: number }> {
  const metadata = await sharp(imageBuffer).metadata();
  const originalWidth = metadata.width || 800;
  const originalHeight = metadata.height || 600;

  if (scale >= 1.0) {
    return { resizedImage: imageBuffer, resizedMask: maskBuffer, originalWidth, originalHeight };
  }

  const newWidth = Math.round(originalWidth * scale);
  const newHeight = Math.round(originalHeight * scale);

  const resizedImage = await sharp(imageBuffer)
    .resize(newWidth, newHeight)
    .png()
    .toBuffer();

  const resizedMask = await sharp(maskBuffer)
    .resize(newWidth, newHeight)
    .png()
    .toBuffer();

  console.log(`[Inpainting] Resized from ${originalWidth}x${originalHeight} to ${newWidth}x${newHeight} (scale: ${scale})`);

  return { resizedImage, resizedMask, originalWidth, originalHeight };
}

/**
 * Scale inpainted result back to original size
 */
async function scaleBackResult(
  resultBuffer: Buffer,
  originalWidth: number,
  originalHeight: number,
  scale: number
): Promise<Buffer> {
  if (scale >= 1.0) {
    return resultBuffer;
  }

  return sharp(resultBuffer)
    .resize(originalWidth, originalHeight)
    .png()
    .toBuffer();
}

// Context margin (px) kept around the masked area when cropping a page down
// for inpainting — gives LaMa surrounding pixels to reconstruct from.
const INPAINT_CONTEXT_MARGIN = parseInt(process.env.INPAINT_CONTEXT_MARGIN || '64', 10);

interface PixelBox { left: number; top: number; right: number; bottom: number; }

function boxesOverlap(a: PixelBox, b: PixelBox): boolean {
  return a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom;
}

function unionBox(a: PixelBox, b: PixelBox): PixelBox {
  return {
    left: Math.min(a.left, b.left),
    top: Math.min(a.top, b.top),
    right: Math.max(a.right, b.right),
    bottom: Math.max(a.bottom, b.bottom),
  };
}

/**
 * Group region rectangles into crop boxes. Each region is expanded by the
 * context margin, then boxes that overlap are merged. Adjacent regions end up
 * sharing one crop (so LaMa never inpaints the same pixels twice); far-apart
 * regions get separate crops (so the empty gaps between them are never sent).
 */
function clusterRegions(
  regions: Array<{ x: number; y: number; width: number; height: number }>,
  imgW: number,
  imgH: number
): PixelBox[] {
  // Region coords may be percentages (<=100) or already pixels (>100) — the
  // same convention createMask uses.
  const boxes: PixelBox[] = regions.map(r => {
    const x = r.x > 100 ? r.x : (r.x / 100) * imgW;
    const y = r.y > 100 ? r.y : (r.y / 100) * imgH;
    const w = r.width > 100 ? r.width : (r.width / 100) * imgW;
    const h = r.height > 100 ? r.height : (r.height / 100) * imgH;
    return {
      left: Math.max(0, Math.floor(x - INPAINT_CONTEXT_MARGIN)),
      top: Math.max(0, Math.floor(y - INPAINT_CONTEXT_MARGIN)),
      right: Math.min(imgW, Math.ceil(x + w + INPAINT_CONTEXT_MARGIN)),
      bottom: Math.min(imgH, Math.ceil(y + h + INPAINT_CONTEXT_MARGIN)),
    };
  });

  // Iteratively merge overlapping boxes until stable.
  let merged = true;
  while (merged) {
    merged = false;
    for (let i = 0; i < boxes.length && !merged; i++) {
      for (let j = i + 1; j < boxes.length; j++) {
        if (boxesOverlap(boxes[i], boxes[j])) {
          boxes[i] = unionBox(boxes[i], boxes[j]);
          boxes.splice(j, 1);
          merged = true;
          break;
        }
      }
    }
  }
  return boxes;
}

/**
 * Send one image+mask pair to a LaMa server (round-robin) with retries and the
 * optional INPAINT_SCALE downscale. Internal — `inpaintImage` wraps this with a
 * bounding-box crop. Returns a buffer the same dimensions as `imageBuffer`.
 */
async function inpaintBuffer(imageBuffer: Buffer, maskBuffer: Buffer): Promise<Buffer> {
  const MAX_RETRIES = 3;
  const TIMEOUT_MS = 300000; // 5 minutes - IOPaint queues requests internally

  // Select server using round-robin
  const serverUrl = getNextServerUrl();

  // Resize for faster processing
  const { resizedImage, resizedMask, originalWidth, originalHeight } = await resizeForInpainting(
    imageBuffer,
    maskBuffer,
    INPAINT_SCALE
  );

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      console.log(`[Inpainting] Attempt ${attempt}/${MAX_RETRIES} - Server: ${serverUrl}/api/v1/inpaint`);
      console.log(`[Inpainting] Image size: ${resizedImage.length} bytes, Mask size: ${resizedMask.length} bytes`);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);

      // IOPaint API has a bug with multipart form data - use JSON with base64
      const requestBody = {
        image: resizedImage.toString('base64'),
        mask: resizedMask.toString('base64'),
      };

      const response = await fetch(`${serverUrl}/api/v1/inpaint`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (response.ok) {
        const resultBuffer = Buffer.from(await response.arrayBuffer());
        console.log(`[Inpainting] Success on attempt ${attempt}! Server: ${serverUrl}, Result size: ${resultBuffer.length} bytes`);

        // Validate the inpainted image is valid before using it
        try {
          const metadata = await sharp(resultBuffer).metadata();
          if (!metadata.width || !metadata.height) {
            console.error('[Inpainting] Server returned invalid image (no dimensions)');
            continue;  // Try next attempt
          }
        } catch (validationError) {
          console.error('[Inpainting] Server returned corrupted image data:', validationError);
          continue;  // Try next attempt
        }

        // Composite approach: scale inpainted result back up, then paste only the
        // inpainted regions onto the original full-res image. This keeps untouched
        // areas pixel-perfect while only the small text regions have reduced quality.
        if (INPAINT_SCALE < 1.0) {
          const scaledResult = await scaleBackResult(resultBuffer, originalWidth, originalHeight, INPAINT_SCALE);
          // Resize mask to original dimensions, output as raw single-channel
          // so the next pipeline can consume it without re-decoding a PNG.
          const fullMask = await sharp(maskBuffer)
            .resize(originalWidth, originalHeight)
            .extractChannel(0)
            .raw()
            .toBuffer();
          // Create RGBA version of inpainted result with mask as alpha,
          // output as raw RGBA so the final composite skips another decode.
          const inpaintedWithAlpha = await sharp(scaledResult)
            .ensureAlpha()
            .removeAlpha()
            .joinChannel(fullMask, { raw: { width: originalWidth, height: originalHeight, channels: 1 } })
            .raw()
            .toBuffer();
          // Composite: original image with inpainted regions pasted on top via mask alpha
          const finalResult = await sharp(imageBuffer)
            .ensureAlpha()
            .composite([{
              input: inpaintedWithAlpha,
              raw: { width: originalWidth, height: originalHeight, channels: 4 },
              blend: 'over',
            }])
            .removeAlpha()
            .png()
            .toBuffer();
          console.log(`[Inpainting] Composited inpainted regions onto original (scale: ${INPAINT_SCALE})`);
          return finalResult;
        }
        return resultBuffer;
      }

      const errorText = await response.text();
      console.error(`[Inpainting] Server error: ${response.status} ${response.statusText}`);
      console.error(`[Inpainting] Error details: ${errorText}`);
    } catch (error: any) {
      console.error(`[Inpainting] Attempt ${attempt} failed on ${serverUrl}: ${error.message}`);

      if (attempt < MAX_RETRIES) {
        const delay = 2000 * attempt; // 2s, 4s, 6s
        console.log(`[Inpainting] Retrying in ${delay}ms...`);
        await new Promise(r => setTimeout(r, delay));
      }
    }
  }

  console.error('[Inpainting] All attempts failed, using original image');
  return imageBuffer;
}

/**
 * Inpaint an image on the IOPaint/LaMa server.
 *
 * Instead of sending the whole page, the masked regions are clustered into one
 * or more crop boxes and each crop is inpainted separately, in parallel. LaMa's
 * cost scales with input pixels, so this skips all the empty space between
 * scattered text regions. Quality is unaffected: each crop is inpainted at full
 * resolution and composited back onto the pixel-identical original.
 *
 * Falls back to a whole-page pass when no regions are supplied.
 */
export async function inpaintImage(
  imageBuffer: Buffer,
  maskBuffer: Buffer,
  regions: Array<{ x: number; y: number; width: number; height: number }> = [],
  config: Partial<InpaintingConfig> = {}
): Promise<Buffer> {
  const mergedConfig = { ...DEFAULT_CONFIG, ...config };

  if (!mergedConfig.enabled) {
    return imageBuffer;
  }

  const meta = await sharp(imageBuffer).metadata();
  const imgW = meta.width || 0;
  const imgH = meta.height || 0;

  // No regions (or no dimensions) — fall back to a whole-page pass.
  if (regions.length === 0 || imgW === 0 || imgH === 0) {
    return inpaintBuffer(imageBuffer, maskBuffer);
  }

  const cropBoxes = clusterRegions(regions, imgW, imgH);

  // Inpaint every crop in parallel; each crop carries the slice of the
  // full-page mask that falls inside it. Use sharp.clone() so the source
  // image and mask are decoded ONCE and shared across all crops, instead
  // of `sharp(imageBuffer).extract()` triggering a fresh decode per crop.
  const srcImage = sharp(imageBuffer);
  const srcMask = sharp(maskBuffer);
  const composites = await Promise.all(cropBoxes.map(async (b) => {
    const w = b.right - b.left;
    const h = b.bottom - b.top;
    if (w <= 0 || h <= 0) return null;
    const cropImage = await srcImage.clone()
      .extract({ left: b.left, top: b.top, width: w, height: h })
      .png()
      .toBuffer();
    const cropMask = await srcMask.clone()
      .extract({ left: b.left, top: b.top, width: w, height: h })
      .png()
      .toBuffer();
    const inpainted = await inpaintBuffer(cropImage, cropMask);
    return { input: inpainted, left: b.left, top: b.top };
  }));

  const valid = composites.filter(
    (c): c is { input: Buffer; left: number; top: number } => c !== null
  );
  if (valid.length === 0) {
    return imageBuffer;
  }

  const cropPx = cropBoxes.reduce((s, b) => s + (b.right - b.left) * (b.bottom - b.top), 0);
  console.log(`[Inpainting] ${cropBoxes.length} crop(s) covering ${((cropPx / (imgW * imgH)) * 100).toFixed(0)}% of the ${imgW}x${imgH} page sent to LaMa`);

  // Paste every inpainted crop back onto the full-resolution original; all
  // pixels outside the crops stay identical. Reuse srcImage.clone() so the
  // source PNG decode that already happened (for the per-crop extracts above)
  // is shared with this final composite step instead of repeated.
  return srcImage.clone()
    .composite(valid)
    .png()
    .toBuffer();
}

/**
 * Calculate average brightness of a region in an image
 * Returns a value from 0 (black) to 255 (white)
 */
export async function calculateRegionBrightness(
  imageBuffer: Buffer,
  x: number,
  y: number,
  width: number,
  height: number,
  imageWidth: number,
  imageHeight: number
): Promise<number> {
  // Convert percentage to pixels if needed
  const pixelX = x > 100 ? Math.round(x) : Math.round((x / 100) * imageWidth);
  const pixelY = y > 100 ? Math.round(y) : Math.round((y / 100) * imageHeight);
  const pixelW = width > 100 ? Math.round(width) : Math.round((width / 100) * imageWidth);
  const pixelH = height > 100 ? Math.round(height) : Math.round((height / 100) * imageHeight);

  // Clamp to image bounds
  const left = Math.max(0, pixelX);
  const top = Math.max(0, pixelY);
  const regionWidth = Math.min(pixelW, imageWidth - left);
  const regionHeight = Math.min(pixelH, imageHeight - top);

  if (regionWidth <= 0 || regionHeight <= 0) {
    return 255; // Default to light background
  }

  try {
    // Extract the region and get raw pixel data
    const { data, info } = await sharp(imageBuffer)
      .extract({ left, top, width: regionWidth, height: regionHeight })
      .raw()
      .toBuffer({ resolveWithObject: true });

    // Calculate average brightness using luminance formula
    let totalBrightness = 0;
    const pixelCount = info.width * info.height;
    const channels = info.channels;

    for (let i = 0; i < data.length; i += channels) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      // Luminance formula: 0.299*R + 0.587*G + 0.114*B
      totalBrightness += 0.299 * r + 0.587 * g + 0.114 * b;
    }

    return totalBrightness / pixelCount;
  } catch (error) {
    console.error('Error calculating brightness:', error);
    return 255; // Default to light background on error
  }
}

/**
 * Determine text color based on background brightness
 */
export function getTextColorForBrightness(brightness: number): string {
  // Use white text on dark backgrounds, black text on light backgrounds
  return brightness < 128 ? '#FFFFFF' : '#000000';
}

/**
 * Calculate the standard deviation of brightness INSIDE a region
 * Used for post-inpaint bubble detection - after text is removed by inpainting,
 * speech bubbles have uniform backgrounds (low variance), while artwork/signs
 * have textured backgrounds that MI-GAN reconstructs (high variance).
 * Returns standard deviation (0 = perfectly uniform, higher = more varied)
 */
export async function calculateBackgroundVariance(
  imageBuffer: Buffer,
  x: number,
  y: number,
  width: number,
  height: number,
  imageWidth: number,
  imageHeight: number
): Promise<number> {
  // Convert percentage to pixels if needed
  const pixelX = x > 100 ? Math.round(x) : Math.round((x / 100) * imageWidth);
  const pixelY = y > 100 ? Math.round(y) : Math.round((y / 100) * imageHeight);
  const pixelW = width > 100 ? Math.round(width) : Math.round((width / 100) * imageWidth);
  const pixelH = height > 100 ? Math.round(height) : Math.round((height / 100) * imageHeight);

  // Clamp to image bounds
  const left = Math.max(0, pixelX);
  const top = Math.max(0, pixelY);
  const regionWidth = Math.min(pixelW, imageWidth - left);
  const regionHeight = Math.min(pixelH, imageHeight - top);

  if (regionWidth <= 0 || regionHeight <= 0) {
    return 0; // Default to uniform (bubble) if invalid region
  }

  try {
    // Extract the region and get raw pixel data
    const { data, info } = await sharp(imageBuffer)
      .extract({ left, top, width: regionWidth, height: regionHeight })
      .raw()
      .toBuffer({ resolveWithObject: true });

    const pixelCount = info.width * info.height;
    const channels = info.channels;

    // First pass: calculate mean brightness
    let totalBrightness = 0;
    for (let i = 0; i < data.length; i += channels) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      totalBrightness += 0.299 * r + 0.587 * g + 0.114 * b;
    }
    const meanBrightness = totalBrightness / pixelCount;

    // Second pass: calculate variance
    let sumSquaredDiff = 0;
    for (let i = 0; i < data.length; i += channels) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const brightness = 0.299 * r + 0.587 * g + 0.114 * b;
      const diff = brightness - meanBrightness;
      sumSquaredDiff += diff * diff;
    }
    const variance = sumSquaredDiff / pixelCount;

    // Return standard deviation
    return Math.sqrt(variance);
  } catch (error) {
    console.error('Error calculating background variance:', error);
    return 0; // Default to uniform (bubble) on error
  }
}
