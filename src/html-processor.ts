import * as fs from 'fs';
import * as path from 'path';
import * as https from 'https';
import * as http from 'http';
import sharp from 'sharp';
import { HSKLevel } from './types';
import { VisionProvider } from './text-extractor';
import { processMangaPage } from './manga-pipeline';

// Minimum image dimensions to process (skip small icons/buttons)
const MIN_IMAGE_WIDTH = 200;
const MIN_IMAGE_HEIGHT = 200;

export interface ImageSource {
  type: 'base64' | 'file' | 'url';
  src: string;           // Original src attribute
  data?: string;         // Base64 data (for embedded images)
  mimeType?: string;     // Image MIME type
  element: string;       // The full <img> tag for replacement
  index: number;         // Position in HTML (for ordering)
}

export interface HTMLProcessOptions {
  level: HSKLevel;
  provider: VisionProvider;
  outputPath?: string;
  convertTraditional?: boolean;
}

/**
 * Parse HTML and extract all image sources
 */
export function extractImagesFromHTML(htmlContent: string): ImageSource[] {
  const images: ImageSource[] = [];

  // Match <img> tags with various attributes
  // Handles: src, data-src, data-lazy-src
  const imgRegex = /<img\s+[^>]*(?:src|data-src|data-lazy-src)\s*=\s*["']([^"']+)["'][^>]*>/gi;

  let match;
  let index = 0;

  while ((match = imgRegex.exec(htmlContent)) !== null) {
    const fullTag = match[0];
    const src = match[1];

    // Determine source type
    if (src.startsWith('data:image/')) {
      // Base64 embedded image
      const mimeMatch = src.match(/^data:(image\/[^;]+);base64,(.+)$/);
      if (mimeMatch) {
        images.push({
          type: 'base64',
          src: src,
          mimeType: mimeMatch[1],
          data: mimeMatch[2],
          element: fullTag,
          index: index++,
        });
      }
    } else if (src.startsWith('http://') || src.startsWith('https://')) {
      // Remote URL
      images.push({
        type: 'url',
        src: src,
        element: fullTag,
        index: index++,
      });
    } else {
      // Local file reference
      images.push({
        type: 'file',
        src: src,
        element: fullTag,
        index: index++,
      });
    }
  }

  return images;
}

/**
 * Load image data from various source types
 */
export async function loadImageData(source: ImageSource, basePath: string): Promise<Buffer> {
  switch (source.type) {
    case 'base64':
      if (!source.data) {
        throw new Error('Base64 image has no data');
      }
      return Buffer.from(source.data, 'base64');

    case 'file':
      // Resolve path relative to HTML file location
      const filePath = path.resolve(basePath, source.src);
      if (!fs.existsSync(filePath)) {
        throw new Error(`Image file not found: ${filePath}`);
      }
      return fs.readFileSync(filePath);

    case 'url':
      return downloadImage(source.src);

    default:
      throw new Error(`Unknown image source type: ${source.type}`);
  }
}

/**
 * Download image from URL
 */
function downloadImage(url: string): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const protocol = url.startsWith('https://') ? https : http;

    protocol.get(url, (response) => {
      // Handle redirects
      if (response.statusCode === 301 || response.statusCode === 302) {
        const redirectUrl = response.headers.location;
        if (redirectUrl) {
          downloadImage(redirectUrl).then(resolve).catch(reject);
          return;
        }
      }

      if (response.statusCode !== 200) {
        reject(new Error(`Failed to download image: HTTP ${response.statusCode}`));
        return;
      }

      const chunks: Buffer[] = [];
      response.on('data', (chunk) => chunks.push(chunk));
      response.on('end', () => resolve(Buffer.concat(chunks)));
      response.on('error', reject);
    }).on('error', reject);
  });
}

/**
 * Convert Buffer to base64 data URL
 */
function bufferToDataUrl(buffer: Buffer, mimeType: string = 'image/png'): string {
  const base64 = buffer.toString('base64');
  return `data:${mimeType};base64,${base64}`;
}

/**
 * Detect MIME type from buffer
 */
function detectMimeType(buffer: Buffer): string {
  // Check magic bytes
  if (buffer[0] === 0x89 && buffer[1] === 0x50 && buffer[2] === 0x4E && buffer[3] === 0x47) {
    return 'image/png';
  }
  if (buffer[0] === 0xFF && buffer[1] === 0xD8 && buffer[2] === 0xFF) {
    return 'image/jpeg';
  }
  if (buffer[0] === 0x47 && buffer[1] === 0x49 && buffer[2] === 0x46) {
    return 'image/gif';
  }
  if (buffer[0] === 0x52 && buffer[1] === 0x49 && buffer[2] === 0x46 && buffer[3] === 0x46) {
    return 'image/webp';
  }
  // Default to PNG
  return 'image/png';
}

/**
 * Replace image in HTML with processed version
 */
function replaceImageInHTML(
  html: string,
  originalElement: string,
  newImageBuffer: Buffer
): string {
  const mimeType = detectMimeType(newImageBuffer);
  const dataUrl = bufferToDataUrl(newImageBuffer, mimeType);

  // Create new img tag with base64 data
  // Preserve any class, style, id attributes from original
  const classMatch = originalElement.match(/class\s*=\s*["']([^"']+)["']/i);
  const styleMatch = originalElement.match(/style\s*=\s*["']([^"']+)["']/i);
  const idMatch = originalElement.match(/id\s*=\s*["']([^"']+)["']/i);
  const altMatch = originalElement.match(/alt\s*=\s*["']([^"']+)["']/i);

  let newElement = `<img src="${dataUrl}"`;
  if (classMatch) newElement += ` class="${classMatch[1]}"`;
  if (styleMatch) newElement += ` style="${styleMatch[1]}"`;
  if (idMatch) newElement += ` id="${idMatch[1]}"`;
  if (altMatch) newElement += ` alt="${altMatch[1]}"`;
  newElement += ' />';

  return html.replace(originalElement, newElement);
}

/**
 * Process HTML file end-to-end
 */
export async function processHTMLFile(
  htmlPath: string,
  options: HTMLProcessOptions
): Promise<string> {
  // Read HTML content
  const htmlContent = fs.readFileSync(htmlPath, 'utf-8');
  const basePath = path.dirname(htmlPath);

  // Extract images
  const images = extractImagesFromHTML(htmlContent);
  console.error(`Found ${images.length} images in HTML`);

  if (images.length === 0) {
    console.error('No images found to process');
    return htmlContent;
  }

  let processedHTML = htmlContent;
  let processedCount = 0;
  let skippedCount = 0;

  // Create temp directory for processing
  const tempDir = path.join(basePath, '.temp-processing');
  if (!fs.existsSync(tempDir)) {
    fs.mkdirSync(tempDir, { recursive: true });
  }

  // Process each image
  for (let i = 0; i < images.length; i++) {
    const image = images[i];
    console.error(`\nProcessing image ${i + 1}/${images.length}...`);

    try {
      // Load image data
      const imageBuffer = await loadImageData(image, basePath);

      // Check image dimensions - skip small images (icons, buttons, etc.)
      const metadata = await sharp(imageBuffer).metadata();
      const width = metadata.width || 0;
      const height = metadata.height || 0;

      if (width < MIN_IMAGE_WIDTH || height < MIN_IMAGE_HEIGHT) {
        console.error(`  ⏭ Skipping small image (${width}x${height})`);
        skippedCount++;
        continue;
      }

      // Convert to RGB format to avoid PaddleOCR issues with RGBA/grayscale
      const rgbBuffer = await sharp(imageBuffer)
        .flatten({ background: { r: 255, g: 255, b: 255 } }) // Remove alpha by flattening onto white
        .toColourspace('srgb')
        .png()
        .toBuffer();

      // Save to temp file for processing
      const tempInputPath = path.join(tempDir, `input_${i}.png`);
      const tempOutputPath = path.join(tempDir, `output_${i}.png`);
      fs.writeFileSync(tempInputPath, rgbBuffer);

      // Process through manga pipeline
      const result = await processMangaPage(tempInputPath, {
        targetLevel: options.level,
        outputDir: tempDir,
        convertTraditional: options.convertTraditional ?? true,
        visionProvider: options.provider,
      });

      // Read cleaned (inpainted) image
      const cleanedPath = result.cleanedImagePath;
      if (cleanedPath && fs.existsSync(cleanedPath)) {
        const processedBuffer = fs.readFileSync(cleanedPath);

        // Replace in HTML
        processedHTML = replaceImageInHTML(processedHTML, image.element, processedBuffer);
        processedCount++;

        console.error(`  ✓ Processed: ${result.regions.length} regions, ${result.simplifiedRegions.length} replaced`);

        // Clean up output file
        fs.unlinkSync(cleanedPath);
      } else {
        // Use original image if no output was created
        console.error(`  ⚠ No processed output, keeping original`);
        skippedCount++;
      }

      // Clean up temp files
      if (fs.existsSync(tempInputPath)) fs.unlinkSync(tempInputPath);
      if (fs.existsSync(tempOutputPath)) fs.unlinkSync(tempOutputPath);

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error(`  ✗ Failed to process: ${errorMessage}`);
      skippedCount++;
    }
  }

  // Clean up temp directory
  try {
    fs.rmdirSync(tempDir);
  } catch {
    // Ignore if not empty
  }

  console.error(`\n✓ Processed ${processedCount} images`);
  if (skippedCount > 0) {
    console.error(`⚠ Skipped ${skippedCount} images due to errors`);
  }

  return processedHTML;
}

/**
 * Save processed HTML to file
 */
export function saveProcessedHTML(
  htmlContent: string,
  outputPath: string
): void {
  fs.writeFileSync(outputPath, htmlContent, 'utf-8');
  console.error(`\n✓ Output saved to: ${outputPath}`);
}
