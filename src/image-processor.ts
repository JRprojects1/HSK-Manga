import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import sharp from 'sharp';
import * as pdfPoppler from 'pdf-poppler';

export interface PageImage {
  pageNumber: number;
  imagePath: string;
  width: number;
  height: number;
  buffer: Buffer;
}

/**
 * Extract pages from a PDF as images using pdf-poppler
 */
export async function extractPagesFromPDF(pdfPath: string, outputDir?: string): Promise<PageImage[]> {
  const ext = path.extname(pdfPath).toLowerCase();

  if (ext !== '.pdf') {
    // For image files, process directly
    return [await loadImage(pdfPath, 1)];
  }

  // Create temp directory for PDF conversion if not provided
  const tempDir = outputDir || fs.mkdtempSync(path.join(os.tmpdir(), 'manga-pdf-'));

  // Ensure output directory exists
  if (!fs.existsSync(tempDir)) {
    fs.mkdirSync(tempDir, { recursive: true });
  }

  const baseName = path.basename(pdfPath, '.pdf');

  // Convert PDF to PNG images
  const opts: pdfPoppler.Options = {
    format: 'png',
    out_dir: tempDir,
    out_prefix: baseName,
    scale: 2048, // Higher resolution for better text recognition
  };

  console.error(`Converting PDF to images...`);
  await pdfPoppler.convert(pdfPath, opts);

  // Find all generated images
  const files = fs.readdirSync(tempDir);
  const imageFiles = files
    .filter((f) => f.startsWith(baseName) && f.endsWith('.png'))
    .sort((a, b) => {
      // Sort by page number (e.g., "test-1.png", "test-2.png")
      const numA = parseInt(a.match(/-(\d+)\.png$/)?.[1] || '0');
      const numB = parseInt(b.match(/-(\d+)\.png$/)?.[1] || '0');
      return numA - numB;
    });

  console.error(`Extracted ${imageFiles.length} pages from PDF`);

  // Load each image
  const pages: PageImage[] = [];
  for (let i = 0; i < imageFiles.length; i++) {
    const imagePath = path.join(tempDir, imageFiles[i]);
    const page = await loadImage(imagePath, i + 1);
    pages.push(page);
  }

  return pages;
}

/**
 * Check if file is a PDF
 */
export function isPDF(filePath: string): boolean {
  return path.extname(filePath).toLowerCase() === '.pdf';
}

/**
 * Load a single image file
 */
export async function loadImage(imagePath: string, pageNumber: number = 1): Promise<PageImage> {
  const buffer = fs.readFileSync(imagePath);
  const metadata = await sharp(buffer).metadata();

  return {
    pageNumber,
    imagePath,
    width: metadata.width || 0,
    height: metadata.height || 0,
    buffer,
  };
}

/**
 * Load multiple image files (e.g., from a directory or list)
 */
export async function loadImages(imagePaths: string[]): Promise<PageImage[]> {
  const pages: PageImage[] = [];

  for (let i = 0; i < imagePaths.length; i++) {
    const page = await loadImage(imagePaths[i], i + 1);
    pages.push(page);
  }

  return pages;
}

/**
 * Encoder dimension limits. libjpeg refuses any side above 65,500 px and WebP
 * any side above 16,383 px; both throw inside sharp. PNG has no practical cap.
 * A merged webtoon unit can exceed the JPEG limit (observed: 800x68515), and
 * a page whose save throws is a page the reader cannot show.
 */
export const JPEG_MAX_DIMENSION = 65500;
export const WEBP_MAX_DIMENSION = 16383;

export function exceedsEncoderLimit(width: number, height: number, ext: string): boolean {
  const e = ext.toLowerCase();
  if (e === '.jpg' || e === '.jpeg') return width > JPEG_MAX_DIMENSION || height > JPEG_MAX_DIMENSION;
  if (e === '.webp') return width > WEBP_MAX_DIMENSION || height > WEBP_MAX_DIMENSION;
  return false;
}

/** Pick the output extension for a page: the preferred one unless it cannot encode these dimensions. */
export function pickImageExtension(width: number, height: number, preferred: string = '.jpg'): string {
  return exceedsEncoderLimit(width, height, preferred) ? '.png' : preferred;
}

/**
 * Save a processed image. Falls back to PNG encoding when the requested format
 * cannot hold the image's dimensions (the file keeps the caller's path; browsers
 * sniff image bytes, so a PNG payload under a .jpg name still renders).
 */
export async function saveImage(buffer: Buffer, outputPath: string): Promise<void> {
  let ext = path.extname(outputPath).toLowerCase();

  let sharpInstance = sharp(buffer);
  const meta = await sharpInstance.metadata();
  if (meta.width && meta.height && exceedsEncoderLimit(meta.width, meta.height, ext)) {
    console.error(`  [SaveImage] ${meta.width}x${meta.height} exceeds the ${ext} encoder limit; writing PNG to ${path.basename(outputPath)}`);
    ext = '.png';
  }

  switch (ext) {
    case '.png':
      sharpInstance = sharpInstance.png();
      break;
    case '.jpg':
    case '.jpeg':
      sharpInstance = sharpInstance.jpeg({ quality: 95 });
      break;
    case '.webp':
      sharpInstance = sharpInstance.webp({ quality: 95 });
      break;
    default:
      sharpInstance = sharpInstance.png();
  }

  await sharpInstance.toFile(outputPath);
}

/**
 * Convert image to base64 for API calls
 */
export function imageToBase64(buffer: Buffer): string {
  return buffer.toString('base64');
}

/**
 * Get supported image extensions
 */
export function getSupportedExtensions(): string[] {
  return ['.png', '.jpg', '.jpeg', '.webp', '.gif'];
}

/**
 * Check if file is a supported image
 */
export function isSupportedImage(filePath: string): boolean {
  const ext = path.extname(filePath).toLowerCase();
  return getSupportedExtensions().includes(ext);
}

/**
 * Generate an HTML viewer for scrollable manga reading
 * @param imagePaths Array of image file paths to include (in order)
 * @param title Title for the HTML page
 * @returns HTML string
 */
export function generateScrollableHTML(
  imagePaths: string[],
  title: string = 'Simplified Manga'
): string {
  // Convert images to base64 for embedding in HTML
  const imageDataUrls: string[] = [];

  for (const imagePath of imagePaths) {
    const buffer = fs.readFileSync(imagePath);
    const ext = path.extname(imagePath).toLowerCase();
    const mimeType = ext === '.jpg' || ext === '.jpeg' ? 'image/jpeg' : 'image/png';
    const base64 = buffer.toString('base64');
    imageDataUrls.push(`data:${mimeType};base64,${base64}`);
  }

  const html = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${title}</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    body {
      background-color: #1a1a1a;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 0;
      line-height: 0;
    }
    .container {
      max-width: 100%;
      width: 800px;
    }
    .page {
      width: 100%;
      margin: 0;
      padding: 0;
      display: block;
      line-height: 0;
    }
    .page img {
      width: 100%;
      height: auto;
      display: block;
    }
    .nav {
      position: fixed;
      bottom: 20px;
      right: 20px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .nav button {
      width: 50px;
      height: 50px;
      border-radius: 50%;
      border: none;
      background: rgba(255,255,255,0.9);
      cursor: pointer;
      font-size: 20px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    .nav button:hover {
      background: #fff;
    }
    .progress {
      position: fixed;
      top: 0;
      left: 0;
      height: 3px;
      background: #4CAF50;
      transition: width 0.1s;
    }
  </style>
</head>
<body>
  <div class="progress" id="progress"></div>
  <div class="container">
${imageDataUrls.map((dataUrl, i) => `    <div class="page"><img src="${dataUrl}" alt="Page ${i + 1}" loading="lazy"></div>`).join('\n')}
  </div>
  <div class="nav">
    <button onclick="scrollToTop()" title="Top">↑</button>
    <button onclick="scrollToBottom()" title="Bottom">↓</button>
  </div>
  <script>
    function scrollToTop() {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
    function scrollToBottom() {
      window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
    }
    window.addEventListener('scroll', () => {
      const scrolled = window.scrollY;
      const maxScroll = document.body.scrollHeight - window.innerHeight;
      const progress = maxScroll > 0 ? (scrolled / maxScroll) * 100 : 0;
      document.getElementById('progress').style.width = progress + '%';
    });
  </script>
</body>
</html>`;

  return html;
}

/**
 * Data structure for interactive reader metadata
 */
// `t` is the Traditional (Taiwan-standard) form, present only when it differs
// from `text`. `text` remains canonical Simplified — it is the flashcard key
// and HSK lookup key. See PhraseData in manga-pipeline.ts.
type PhraseInfo = { text: string; pinyin: string; isAboveLevel: boolean; hskLevel: number | null; definition?: string; properNoun?: boolean; t?: string };

export interface ReaderBlockData {
  blockId: number;
  bbox: { x: number; y: number; width: number; height: number };
  text: string;
  isVertical: boolean;
  fontSize: number;
  textColor: string;
  rotation?: number;     // degrees, 0 = axis-aligned
  /**
   * Floor for the reader's fit pass, derived from the original lettering
   * height (EN dialogue only — see computeMinFontSize in text-replacer).
   * Absent on vertical blocks, non-Latin sources, display lettering, and
   * jobs cached before PIPELINE_VERSION 17; the fit pass then sizes to the
   * box alone.
   */
  minFontSize?: number;
  phrases: PhraseInfo[];
  /**
   * Plain-English rendering of the whole bubble, shown in the word popup under
   * "THIS BUBBLE". Optional: a failed gloss batch leaves it absent and the
   * reader omits the section.
   */
  gloss?: string;
  // Multi-level support: text and phrases per level (e.g. { hsk2: {...}, hsk3: {...}, english: {...} })
  textByLevel?: { [level: string]: string };
  phrasesByLevel?: { [level: string]: PhraseInfo[] };
}

export interface ReaderPageData {
  pageNumber: number;
  originalImage: string;
  cleanedImage: string;
  // Native pixel dimensions of the page image, propagated so the reader can
  // set <img width/height> attributes. This lets the browser reserve the
  // correct vertical space before download completes — scrollHeight is
  // accurate at DOM ready and scroll-to-top/bottom lands correctly.
  imageWidth?: number;
  imageHeight?: number;
  regions: {
    bbox: { x: number; y: number; width: number; height: number };
    originalText: string;
    simplifiedText: string;
    originalPinyin?: string;
    simplifiedPinyin?: string;
  }[];
  blocks: ReaderBlockData[];
}

export interface ReaderData {
  title: string;
  createdAt: string;
  targetLevel?: string;
  availableLevels?: string[]; // e.g. ['original', 'hsk2', 'hsk3', 'english']
  pages: ReaderPageData[];
  // Chinese↔English character map from the chapter analysis. Drives the
  // reader's "show names in English" toggle so swap targets are deterministic
  // (catches single-char surnames like 朴 that the no-CEDICT heuristic misses).
  characters?: { name: string; chineseName: string; role?: string }[];
  // Chapter-specific Chinese terms (character names + KEEP terms) the names
  // toggle must never dictionary-swap. See resolveEnglishName in the reader.
  protectedNames?: string[];
}

/**
 * Generate an interactive HTML reader with hover pinyin and HSK highlighting
 * Uses cleaned images with HTML text overlays positioned over speech bubbles
 * @param data Reader metadata including page info, block data with phrases
 * @returns HTML string
 */
export interface ReaderHtmlOptions {
  /**
   * Bake the audio manifest into the page instead of fetching it at runtime.
   *
   * Needed for readers opened straight off disk. Every file:// document is an
   * opaque origin, so fetching a sibling JSON is a cross-origin request nobody
   * can authorise and CORS refuses it — the reader then concludes the chapter
   * has no audio and hides every control, silently. The mp3s themselves load
   * fine, because <audio> and <img> predate CORS; it is only the index that
   * cannot be read. Served over http this changes nothing.
   */
  inlineAudioManifest?: unknown;
}

export function generateInteractiveReaderHTML(data: ReaderData, options: ReaderHtmlOptions = {}): string {
  const inlineManifest = options.inlineAudioManifest
    ? JSON.stringify(options.inlineAudioManifest).replace(/<\//g, '<\\/')
    : null;
  // The names toggle is only rendered when something can actually be swapped:
  // the analysis has to have produced an English name distinct from the Chinese
  // one. That is true for English-input chapters and false for Chinese-input
  // ones, which mirror the Chinese form into both fields.
  //
  // When false the row is omitted entirely rather than shown with a caveat. A
  // toggle that provably cannot change anything is worse than no toggle, and the
  // CC-CEDICT proper-noun fallback does not rescue it — measured across two
  // chapters, that path swaps nothing; every swappable word comes from
  // characters.
  const hasEnglishNames = (data.characters || []).some(c => c && c.chineseName && c.name && c.chineseName !== c.name);
  const html = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${data.title}</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    :root {
      --r-bg: #15120e;
      --r-panel: #221f19;
      --r-panel-2: #2d2920;
      --r-border: #373126;
      --r-text: #f2ecdf;
      --r-text-2: #b8af9b;
      --r-muted: #8a806f;
      --r-accent: #e6855e;
      --r-accent-sf: #3a2b21;
    }
    body {
      background-color: var(--r-bg);
      min-height: 100vh;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .container {
      max-width: 800px;
      margin: 0 auto;
    }
    .page {
      position: relative;
      width: 100%;
      line-height: 0;
      /* overflow: visible (vs hidden) lets long simplified bubble text
         continue past the page boundary onto the next page's area instead
         of being clipped mid-line. Trade-off: overflowing text renders
         over the next page's art with no white plate behind it. Worth it
         for comprehension — full text is more important than no-bleed. */
      overflow: visible;
    }
    .page img {
      width: 100%;
      height: auto;
      display: block;
    }

    /* Bubble text overlay — positioned over each speech bubble */
    .bubble-text {
      position: absolute;
      padding: 2px;
      font-family: "Noto Sans SC", "Microsoft YaHei", "PingFang SC", "Hiragino Sans GB", sans-serif;
      /* Traditional mode swaps to the Taiwan-region counterparts below. */
      line-height: 1.1;
      overflow: visible;
      pointer-events: auto;
      text-align: center;
      display: flex;
      align-items: center;
      justify-content: center;
      /* Bug 3 fix — legibility on two-color / non-uniform backgrounds. The
         pipeline renders each block in ONE sampled color; where the original
         art's background changes brightness under the text (bright→dark
         caption, text straddling a panel edge), the wrong-contrast half of the
         text would disappear. A contrasting outline drawn BEHIND the glyph
         (paint-order: stroke fill keeps the fill crisp, halo only on the outer
         edge) makes it readable on any background without knowing where the
         boundary is. Default = white halo for dark text; .dark-bg flips it to
         black for white text. Crucially this is a NO-OP on normal bubbles: a
         white halo over a white bubble is invisible, so it only "activates"
         exactly where the background contrasts wrongly. em-based so it scales
         with each block's font size. */
      -webkit-text-stroke: 0.05em #fff;
      paint-order: stroke fill;
    }
    /* Traditional script mode.
       Unicode unifies Han characters, so 骨/直/令/過 are each a single codepoint
       shared across mainland China, Taiwan, Hong Kong and Japan — but the
       standard STROKE SHAPES differ by region, and a font picks one convention.
       Ideally Traditional text uses a Taiwan-region font.

       The catch is that only SOME regional fonts are design siblings. Noto Sans
       SC/TC and PingFang SC/TC are variants of ONE typeface — identical design,
       metrics and weights, differing only in those regional glyph forms — so
       swapping between them is visually seamless. Microsoft YaHei and Microsoft
       JhengHei are NOT siblings despite the parallel naming: they are separate
       typefaces, and swapping mid-document is a jarring, visible change of face.

       So: try the true siblings first, then fall back to the SAME stack the
       Simplified text uses rather than to a mismatched regional font. On a
       machine with the Noto or PingFang pair that means authentic Traditional
       glyphs; on stock Windows it means YaHei for both scripts — mainland glyph
       forms, but one consistent typeface. Verified that YaHei covers all 365
       Traditional characters this chapter produces, so the fallback costs
       coverage nothing. JhengHei is deliberately absent from this list. */
    body.script-trad .bubble-text,
    body.script-trad .definition-popover .def-word {
      font-family: "Noto Sans TC", "PingFang TC", "Noto Sans SC", "Microsoft YaHei", "PingFang SC", "Hiragino Sans GB", sans-serif;
    }
    /* White text (light-on-dark original) → black halo instead. */
    .bubble-text.dark-bg {
      -webkit-text-stroke: 0.05em #000;
    }

    .bubble-inner {
      max-width: 100%;
    }

    /* Vertical text layout */
    .bubble-text.vertical {
      writing-mode: vertical-rl;
      text-orientation: upright;
      text-align: start;
    }

    /* Rotated text layout */
    .bubble-text.rotated {
      transform-origin: center center;
    }

    /* Each word/phrase */
    .phrase {
      cursor: default;
      border-radius: 2px;
      transition: background-color 0.15s;
      /* inline-block (vs inline) prevents Firefox from breaking mid-phrase at
         character boundaries — forces wrap to happen between phrase spans,
         matching Chrome's natural balanced-wrap behavior on CJK bubble text. */
      display: inline-block;
      position: relative;
    }
    .phrase:hover {
      text-shadow: 0 0 8px rgba(245, 215, 175, 0.85), 0 0 3px rgba(230, 175, 130, 0.6);
    }

    /* Touch-target expansion. Bubble text scales with the artwork, so on a
       390px phone a word span measures about 16x18px — every word in a chapter
       lands under Apple's 44px minimum, and adjacent words touch with no gap,
       so a near-miss opens the neighbour's definition instead of nothing.
       This pseudo-element enlarges the HIT AREA without touching layout or
       pixels. Only on coarse pointers: on a mouse the spans are already
       precise, and overlapping hit boxes there would make hover ambiguous. */
    @media (pointer: coarse) {
      .phrase:not(.punct)::before {
        content: '';
        position: absolute;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        width: max(100%, 40px);
        height: max(100%, 40px);
        /* Overlapping neighbours are unavoidable when words are contiguous;
           the topmost still wins, but a fingertip now lands on SOME word
           rather than between two. */
        z-index: -1;
      }
    }

    /* Pinyin — floating above word on hover, no layout impact.
       Gated to devices that actually hover. On touch there is no hover to
       trigger it, and worse: tapping a word leaves :hover STUCK on that span,
       so the plate would appear and stay until you tapped elsewhere. Phones
       read pinyin from the bubble panel instead. */
    @media (hover: hover) and (pointer: fine) {
    .phrase::after {
      content: attr(data-pinyin);
      position: absolute;
      bottom: 100%;
      left: 50%;
      transform: translateX(-50%);
      opacity: 0;
      font-size: 0.55em;
      white-space: nowrap;
      pointer-events: none;
      transition: opacity 0.15s;
      color: #2f2a24;
      background: #fffdf7;
      padding: 1px 5px;
      border-radius: 4px;
      z-index: 10;
      /* .bubble-text carries a -webkit-text-stroke halo for legibility over
         two-color art, and text-stroke INHERITS — including into this
         pseudo-element. The halo is meant for glyphs sitting on the artwork;
         on the pinyin's own solid plate it just muddies the color. Reset it. */
      -webkit-text-stroke: 0;
    }
    .phrase:hover::after {
      opacity: 1;
    }
    /* No pinyin pseudo-element for punctuation */
    .phrase.punct::after {
      display: none;
    }
    }  /* end @media (hover: hover) */

    /* Words above target HSK level — always highlighted orange */
    .phrase.above-level {
      text-shadow: 0 0 8px rgba(255, 140, 0, 0.8), 0 0 3px rgba(255, 140, 0, 0.5);
    }
    .phrase.above-level:hover {
      text-shadow: 0 0 8px rgba(245, 215, 175, 0.85), 0 0 3px rgba(230, 175, 130, 0.6);
    }

    /* Dark background overrides — wider glow spread for visibility */
    .dark-bg .phrase:hover {
      text-shadow: 0 0 14px rgba(248, 222, 188, 0.95), 0 0 6px rgba(240, 195, 150, 0.85), 0 0 2px rgba(255, 235, 210, 0.9);
    }
    .dark-bg .phrase.above-level {
      text-shadow: 0 0 14px rgba(255, 180, 50, 0.9), 0 0 6px rgba(255, 200, 80, 0.8), 0 0 2px rgba(255, 220, 120, 0.9);
    }
    .dark-bg .phrase.above-level:hover {
      text-shadow: 0 0 14px rgba(248, 222, 188, 0.95), 0 0 6px rgba(240, 195, 150, 0.85), 0 0 2px rgba(255, 235, 210, 0.9);
    }

    /* Punctuation — no hover effect */
    .phrase.punct {
      pointer-events: none;
    }

    /* Each non-punct phrase + its trailing punctuation run is wrapped in a
       .phrase-group with white-space: nowrap. Stops puncts (commas, periods,
       ellipsis dots) from getting orphaned on their own line when names
       toggle swaps Chinese chars to longer Latin text. Groups themselves
       still wrap freely between non-punct phrases.
       display: inline-block matches the .phrase fix above — Firefox
       suppresses line-break opportunities between adjacent inline elements
       that both have white-space:nowrap, so the bubble text wouldn't wrap
       at all. inline-block makes each group an atomic break point. */
    .phrase-group {
      display: inline-block;
      white-space: nowrap;
    }

    /* English chunks with internal spaces (merged tokens like "DUNGEON AND
       STONE") render as inline so the spaces are real break points instead of
       being trapped inside an unbreakable inline-block. Combined with the
       .has-wrap rule below, this lets long bracketed names wrap at their
       internal spaces instead of overflowing the bubble. */
    .phrase.has-spaces { display: inline; }
    /* has-wrap groups also drop the default inline-block display so they stop
       being atomic from the outer line-layout's perspective. Otherwise the
       group's inline-block shrink-to-fit width can claim a whole bubble line,
       orphaning the next short phrase (e.g. 中) onto its own line even though
       it would fit alongside the following content. With display: inline, the
       group's contents flow as part of the surrounding text and the browser
       breaks lines based on actual rendered width. */
    .phrase-group.has-wrap {
      display: inline;
      white-space: normal;
    }
    /* Every phrase INSIDE a has-wrap group must also be inline, not the
       default inline-block. If e.g. '[' stays inline-block, the browser
       treats it as its own atomic word and may put it at the end of the
       previous line, orphaning it from its bracketed term:
           "...学习["
           "DUNGEON AND"
           "STONE]｡"
       Making every child inline turns the whole group into one continuous
       inline text run; the only break points are the spaces inside the
       English chunk, so '[' stays glued to its term. */
    .phrase-group.has-wrap .phrase { display: inline; }

    /* === Reader toggles (settings popup) === */

    /* Proper-noun swap. Each name span contains both .zh-text and .en-text
       children; toggle by hiding one or the other via body class. */
    .phrase.is-name .en-text { display: none; }
    body.show-en-names .phrase.is-name .zh-text { display: none; }
    body.show-en-names .phrase.is-name .en-text {
      display: inline;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    /* Hide pinyin pseudo-element when a name is shown in English (chinese
       pinyin floating above a Latin name reads as broken). */
    body.show-en-names .phrase.is-name:hover::after { opacity: 0; }
    /* When a name is shown in English, suppress the above-level orange glow on
       it. Scoped :not(:hover) so the hover glow still fires correctly. */
    body.show-en-names .phrase.is-name.above-level:not(:hover) {
      text-shadow: none !important;
    }

    /* Hide above-level orange highlight. :not(:hover) so the hover glow
       (visual focus cue) keeps working. */
    body.no-above-highlight .phrase.above-level:not(:hover),
    body.no-above-highlight .dark-bg .phrase.above-level:not(:hover) {
      text-shadow: none !important;
    }

    /* Settings gear active state */
    .controls button.active {
      background: rgba(217, 104, 63, 0.35);
      color: #fff;
    }

    /* Settings popup — positioned to the LEFT of the controls column,
       vertically aligned with the gear button. z-index above .controls. */
    .settings-popup {
      position: fixed;
      bottom: 136px;
      right: 88px;
      background: var(--r-panel);
      border: 1px solid var(--r-border);
      border-radius: 14px;
      padding: 16px;
      min-width: 260px;
      box-shadow: 0 12px 32px rgba(0,0,0,0.5);
      z-index: 110;
      opacity: 0;
      transform: translateX(8px);
      pointer-events: none;
      transition: opacity 0.15s, transform 0.15s;
    }
    .settings-popup.open {
      opacity: 1;
      transform: translateX(0);
      pointer-events: auto;
    }
    .settings-popup .settings-title {
      font-size: 12px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--r-muted);
      font-weight: 700;
      margin-bottom: 12px;
    }
    .settings-row {
      display: flex; align-items: center; justify-content: space-between;
      padding: 8px 0;
      gap: 14px;
    }
    .settings-row + .settings-row { border-top: 1px solid var(--r-border); }
    .settings-label { font-size: 13.5px; color: var(--r-text); }
    /* iOS-style toggle switch */
    .toggle-switch {
      position: relative;
      width: 38px; height: 22px;
      flex-shrink: 0;
    }
    .toggle-switch input { opacity: 0; width: 0; height: 0; }
    .toggle-slider {
      position: absolute; cursor: pointer;
      top: 0; left: 0; right: 0; bottom: 0;
      background: #555;
      border-radius: 22px;
      transition: background 0.2s;
    }
    .toggle-slider::before {
      content: "";
      position: absolute;
      height: 18px; width: 18px;
      left: 2px; top: 2px;
      background: white;
      border-radius: 50%;
      transition: transform 0.2s;
    }
    .toggle-switch input:checked + .toggle-slider { background: var(--r-accent); }
    .toggle-switch input:checked + .toggle-slider::before { transform: translateX(16px); }

    /* Definition popover — appears on click */
    .definition-popover {
      position: absolute;
      background: var(--r-panel);
      color: var(--r-text);
      border: 1px solid var(--r-border);
      border-radius: 16px;
      padding: 16px;
      max-width: 280px;
      font-size: 14px;
      line-height: 1.5;
      z-index: 200;
      pointer-events: auto;
      box-shadow: 0 10px 30px rgba(0,0,0,0.6);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    .definition-popover .def-word {
      display: block;
      font-size: 24px;
      font-weight: 800;
      line-height: 1.1;
      font-family: "Noto Sans SC", "Microsoft YaHei", sans-serif;
    }
    .definition-popover .def-pinyin {
      display: block;
      color: var(--r-accent);
      font-size: 15px;
      font-weight: 700;
      margin-top: 3px;
    }
    .definition-popover .def-level {
      display: inline-block;
      background: var(--r-accent-sf);
      color: var(--r-accent);
      padding: 3px 10px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 800;
      margin-top: 10px;
    }
    .definition-popover .def-meaning {
      margin-top: 9px;
      color: var(--r-text-2);
    }

    /* Phase 3.2: ⭐ save-to-flashcards button. Appears in the popover header.
       Three states: default (dark outlined star), saved (amber filled star),
       disabled (greyed — when CC-CEDICT has no entry, e.g., character names). */
    .definition-popover .def-header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
    }
    .definition-popover .def-header-text {
      flex: 1;
      min-width: 0;
    }
    .definition-popover .def-save {
      background: var(--r-panel-2);
      border: 1px solid var(--r-border);
      color: var(--r-muted);
      width: 34px;
      height: 34px;
      border-radius: 50%;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
      padding: 0;
      transition: color 0.15s, border-color 0.15s, background-color 0.15s, transform 0.1s;
    }
    .definition-popover .def-save svg {
      width: 16px;
      height: 16px;
      stroke: currentColor;
      stroke-width: 2.25;
      fill: none;
      transition: fill 0.15s;
    }
    .definition-popover .def-save:hover:not(:disabled) {
      color: var(--r-accent);
      border-color: var(--r-accent);
    }
    .definition-popover .def-save.saved {
      color: #fff;
      border-color: var(--r-accent);
      background: var(--r-accent);
    }
    .definition-popover .def-save.saved svg {
      fill: #fff;
    }
    .definition-popover .def-save:disabled {
      cursor: not-allowed;
      opacity: 0.4;
    }
    .definition-popover .def-save:active {
      transform: scale(0.92);
    }
    .definition-popover .def-save-note {
      margin-top: 8px;
      color: var(--r-muted);
      font-size: 11px;
      font-style: italic;
    }
    /* Bubble-level gloss inside the word popup. Muted and set below the word
       meaning so it reads as supporting context, not the primary answer. */
    /* ---- Bubble panel: the whole-sentence translation ----
       Sits directly under the speech bubble it belongs to. Measured before
       building: median clearance below a bubble is 681px, and this panel
       collides with the next bubble on 4 of 191 (2%) — those flip above,
       the same escape hatch positionPopover() already uses. */
    .bubble-panel {
      position: absolute;
      display: none;
      flex-direction: column;
      align-items: flex-start;
      gap: 8px;
      max-width: 380px;
      background: var(--r-panel);
      color: var(--r-text);
      border: 1px solid var(--r-border);
      border-radius: 12px;
      padding: 11px 14px;
      z-index: 190;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      box-shadow: 0 10px 30px rgba(0,0,0,.55);
    }
    .bubble-panel.open { display: flex; }
    .bubble-panel .bp-gloss { font-size: 16px; line-height: 1.45; color: var(--r-text); }
    /* The English gloss is opt-in per bubble: it renders veiled (blurred past
       legibility) until the reader taps to reveal it, so selecting a bubble or
       looking up a word can never spoil the sentence. The pinyin above stays
       visible - it is a pronunciation aid, not a meaning spoiler. Blur rather
       than collapse so the panel keeps its height and never re-anchors on
       reveal. */
    .bubble-panel .bp-gloss-wrap { position: relative; align-self: stretch; }
    /* In tap mode reserve a row under the gloss for the hide control - in
       BOTH states - so revealing and re-hiding never changes the panel height. */
    .bubble-panel .bp-gloss-wrap.can-hide { padding-bottom: 20px; }
    .bubble-panel .bp-hide-gloss {
      display: none; position: absolute; right: 0; bottom: 0; margin: 0; padding: 0 2px;
      background: transparent; border: 0; cursor: pointer;
      font: inherit; font-size: 12px; color: var(--r-text-2); -webkit-tap-highlight-color: transparent;
    }
    .bubble-panel .bp-hide-gloss:hover { color: var(--r-accent); }
    .bubble-panel .bp-gloss-wrap.is-revealed .bp-hide-gloss { display: inline-block; }
    .bubble-panel .bp-gloss-wrap.is-veiled .bp-gloss {
      filter: blur(7px); opacity: .7; user-select: none; pointer-events: none;
    }
    .bubble-panel .bp-reveal {
      display: none; position: absolute; inset: 0; margin: 0; padding: 0;
      align-items: center; justify-content: center;
      background: transparent; border: 0; cursor: pointer;
      font: inherit; -webkit-tap-highlight-color: transparent;
    }
    .bubble-panel .bp-gloss-wrap.is-veiled .bp-reveal { display: flex; }
    .bubble-panel .bp-reveal span {
      font-size: 13px; font-weight: 600; letter-spacing: .01em; color: var(--r-accent);
      background: var(--r-panel); border: 1px solid var(--r-border);
      border-radius: 999px; padding: 5px 12px; white-space: nowrap;
    }
    /* Three-way control in settings: Tap to show / Always / Off. */
    .seg { display: inline-flex; border: 1px solid var(--r-border); border-radius: 999px; overflow: hidden; }
    .seg button {
      background: transparent; border: 0; color: var(--r-text-2); cursor: pointer;
      font: inherit; font-size: 12px; padding: 5px 10px; white-space: nowrap;
    }
    .seg button + button { border-left: 1px solid var(--r-border); }
    .seg button.is-on { background: var(--r-accent); color: #fff; }
    /* Whole-sentence pinyin, interlinear: each word's pinyin sits over its own
       characters. This cannot go inside the speech bubble — ruby needs about
       1.5x the line height and the median bubble only has 1.28x of slack, so
       72% of them would spill pinyin onto the artwork. The panel is outside the
       art and has the room.
       A plain pinyin LINE was tried first and rejected: matching 12 syllables
       to 12 characters by counting is a decoding puzzle, not a reading aid,
       and multi-syllable words like 小时候 -> xiao shi hou break naive 1:1
       counting anyway. */
    .bubble-panel .bp-inter {
      display: flex;
      flex-wrap: wrap;
      align-items: flex-end;
      gap: 4px 9px;
      font-family: "Noto Sans SC", "Microsoft YaHei", "PingFang SC", sans-serif;
    }
    .bubble-panel .bp-inter:empty { display: none; }
    .bp-w { display: flex; flex-direction: column; align-items: center; line-height: 1.15; }
    .bp-py {
      font-size: 13px; font-weight: 700; color: var(--r-accent);
      white-space: nowrap; letter-spacing: .01em;
    }
    /* The hanzi stays at full strength. It is the thing being learned, so
       dimming it to make the panel look less like a copy of the bubble would
       trade the wrong way. What stops it reading as duplication is the stacked
       pairing, not a contrast trick. */
    .bp-hz { font-size: 18px; color: var(--r-text); }
    /* Punctuation keeps the sentence's shape but carries no pinyin. */
    .bp-w.is-punct .bp-py { visibility: hidden; }
    .bp-w.is-punct .bp-hz { color: var(--r-text-2); }

    .bubble-panel .bp-sep {
      align-self: stretch; height: 1px; background: var(--r-border); margin: 1px 0;
    }
    .bubble-panel .bp-inter:empty + .bp-sep { display: none; }
    @media (min-width: 900px) {
      .bp-py { font-size: 14px; }
      .bp-hz { font-size: 20px; }
    }

    /* Dismiss where you meet it. Making someone hunt through settings to remove
       something that is bothering them right now is the wrong shape — settings
       is the way back, not the way out. Hover-revealed on a mouse (free, no
       permanent chrome); on touch there is no hover to hide it, so it is a
       persistent, full-size target instead. */
    .bp-hide {
      position: absolute; top: 4px; right: 6px;
      background: none; border: 0; color: var(--r-muted);
      font-size: 16px; line-height: 1; cursor: pointer; padding: 3px 5px;
      border-radius: 5px; opacity: 0;
      transition: opacity .15s, color .15s;
    }
    .bp-hide:hover { color: var(--r-accent); }
    .bp-hide:focus-visible { opacity: 1; outline: 2px solid var(--r-accent); outline-offset: 2px; }
    @media (hover: hover) and (pointer: fine) {
      .bubble-panel:hover .bp-hide { opacity: 1; }
    }
    @media (pointer: coarse) {
      .bp-hide {
        opacity: 1; width: 44px; height: 44px; top: 0; right: 0; padding: 0;
        font-size: 21px; display: flex; align-items: center; justify-content: center;
      }
      .bubble-panel { padding-right: 46px; }
    }
    body.pref-no-translation .bubble-panel { display: none !important; }

    /* Brief confirmation that also says where the thing went. */
    .reader-toast {
      position: fixed; left: 50%; bottom: 26px; transform: translate(-50%, 14px);
      background: var(--r-panel); color: var(--r-text);
      border: 1px solid var(--r-border); border-radius: 999px;
      padding: 10px 18px; font-size: 13.5px; font-weight: 600;
      box-shadow: 0 10px 30px rgba(0,0,0,.5);
      z-index: 400; opacity: 0; pointer-events: none;
      transition: opacity .18s, transform .18s;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    .reader-toast.show { opacity: 1; transform: translate(-50%, 0); }


    /* On a phone the controls column sits over the artwork: 48 of 191 bubbles
       reach under it horizontally, and 12 have more than 15% of their width
       covered (worst case 54%). On desktop it is 0 of 191, because the column
       lives in the dead margin beside the 800px art. So this is a touch-only
       problem, and gets a touch-only answer: step aside while the reader is
       actually scrolling, come back the moment they stop. pointer-events is
       dropped too, so a word underneath stays tappable while they are hidden. */
    @media (pointer: coarse) {
      .controls { transition: opacity .18s ease; }
      body.scrolling .controls { opacity: .12; pointer-events: none; }
    }
    @media (min-width: 900px) {
      .bubble-panel { max-width: 440px; padding: 13px 16px; }
      .bubble-panel .bp-gloss { font-size: 17px; }
    }
    /* ---- Audio ----
       Speaker buttons only exist when the chapter was generated with audio; a
       chapter without an audio/manifest.json shows none of this. */
    .hm-speak {
      background: var(--r-panel-2);
      border: 1px solid var(--r-border);
      color: var(--r-muted);
      border-radius: 50%;
      cursor: pointer;
      display: inline-flex; align-items: center; justify-content: center;
      flex-shrink: 0; padding: 0;
      transition: color .15s, border-color .15s, background-color .15s, transform .1s;
    }
    .hm-speak svg { stroke: currentColor; stroke-width: 2; fill: none; stroke-linejoin: round; stroke-linecap: round; }
    .hm-speak:hover { color: var(--r-accent); border-color: var(--r-accent); }
    .hm-speak:active { transform: scale(.92); }
    .hm-speak.speaking { color: #fff; border-color: var(--r-accent); background: var(--r-accent); }
    .hm-speak:focus-visible { outline: 2px solid var(--r-accent); outline-offset: 2px; }
    /* Anchored to the bubble's edge, so the control that speaks Chinese sits
       against the Chinese and never against the English translation. 44px is
       the touch minimum; this is the one control we fully own. */
    #bubbleSpeak {
      position: absolute; display: none;
      width: 44px; height: 44px; z-index: 195;
      border: 2px solid var(--r-panel);
      background: var(--r-accent); color: #fff;
    }
    #bubbleSpeak svg { width: 19px; height: 19px; }
    #bubbleSpeak:hover { background: #b8532b; border-color: var(--r-panel); color: #fff; }
    /* Word-level speaker, in the definition popover header. */
    .definition-popover .hm-speak { width: 34px; height: 34px; }
    .definition-popover .hm-speak svg { width: 16px; height: 16px; }
    body.pref-no-audio .hm-speak, body.pref-no-audio #bubbleSpeak { display: none !important; }
    body.no-audio-data .hm-speak, body.no-audio-data #bubbleSpeak { display: none !important; }

    /* ---- Karaoke ----
       Higher specificity than .phrase.above-level and .phrase:hover so it wins
       over their glow rules. text-stroke is reset for the same reason
       .phrase::after resets it: the halo exists for glyphs sitting on artwork,
       and this word now has its own solid plate.
       #b8532b, not the accent: white on #e6855e is 2.66:1, so the word being
       spoken would have been the least legible word on screen. */
    .bubble-text .phrase.is-speaking {
      background: #b8532b; color: #fff; border-radius: 3px;
      -webkit-text-stroke: 0; text-shadow: none;
    }
    .bubble-text .phrase.is-spoken { background: rgba(184,83,43,.20); border-radius: 3px; }
    /* The same highlight in the panel, which is the readable surface — the
       bubble copy can be 16px, rotated, or vertical. */
    .bp-w.is-speaking .bp-hz { background: #b8532b; color: #fff; border-radius: 3px; }
    .bp-w.is-speaking .bp-py { color: #fff; }
    .bp-w.is-spoken .bp-hz { background: rgba(184,83,43,.20); border-radius: 3px; }
    body.pref-no-audio .phrase.is-speaking, body.pref-no-audio .phrase.is-spoken,
    body.pref-no-audio .bp-w.is-speaking .bp-hz, body.pref-no-audio .bp-w.is-spoken .bp-hz {
      background: none; color: inherit; text-shadow: inherit;
    }
    @media (prefers-reduced-motion: reduce) {
      .bubble-text .phrase.is-speaking, .bp-w.is-speaking .bp-hz { transition: none; }
    }

    /* Active bubble outline — tells the reader which bubble the panel is for. */
    .bubble-text.is-active {
      outline: 2px solid var(--r-accent);
      outline-offset: 5px;
      border-radius: 6px;
      background: rgba(230,133,94,.16);
    }
    /* An unselected bubble is one big target while nothing is selected. */
    body.two-stage .bubble-text:not(.is-active) { cursor: pointer; }

    .definition-popover .def-gloss {
      margin-top: 10px;
      padding-top: 9px;
      border-top: 1px solid var(--r-border);
      font-size: 13px;
      line-height: 1.45;
      color: var(--r-text);
      opacity: 0.92;
    }
    .definition-popover .def-gloss-label {
      display: block;
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--r-text-2);
      opacity: 0.75;
      margin-bottom: 3px;
    }

    /* Controls */
    .controls {
      position: fixed;
      bottom: 20px;
      right: 20px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      z-index: 100;
      align-items: flex-end;
    }
    .controls button {
      width: 48px;
      height: 48px;
      border-radius: 50%;
      border: 1px solid var(--r-border);
      background: var(--r-panel);
      color: var(--r-text-2);
      cursor: pointer;
      font-size: 18px;
      font-weight: 700;
      box-shadow: 0 6px 20px rgba(0,0,0,0.5);
      transition: transform 0.1s, background-color 0.2s, color 0.2s;
    }
    .controls button:hover {
      background: var(--r-panel-2);
      color: var(--r-accent);
      transform: scale(1.05);
    }
    .controls button:active {
      transform: scale(0.95);
    }
    .controls button.active {
      background: var(--r-accent);
      color: #fff;
      border-color: var(--r-accent);
    }
    .controls select {
      padding: 9px 14px;
      border-radius: 999px;
      border: 1px solid var(--r-border);
      background: var(--r-panel);
      color: var(--r-text);
      cursor: pointer;
      font-size: 14px;
      font-weight: 700;
      box-shadow: 0 6px 20px rgba(0,0,0,0.5);
      outline: none;
    }

    /* Progress bar */
    .progress {
      position: fixed;
      top: 0;
      left: 0;
      height: 3px;
      background: var(--r-accent);
      transition: width 0.1s;
      z-index: 101;
    }

    /* Mode indicator */
    .mode-indicator {
      position: fixed;
      top: 12px;
      left: 50%;
      transform: translateX(-50%);
      background: var(--r-panel);
      color: var(--r-text);
      border: 1px solid var(--r-border);
      padding: 7px 16px;
      border-radius: 999px;
      font-size: 13px;
      font-weight: 700;
      z-index: 101;
      transition: opacity 0.3s;
    }

    /* Keyboard shortcut hints */
    /* Advertises a KEYBOARD shortcut, so it is meaningless on a phone — and it
       is permanent chrome sitting on the artwork there, unlike on desktop where
       it lands in the margin beside the 800px column. Same class of bug as
       telling touch users to "hover". */
    .shortcut-hint { display: none; }
    @media (hover: hover) and (pointer: fine) {
      .shortcut-hint { display: block; }
    }
    .shortcut-hint {
      position: fixed;
      bottom: 20px;
      left: 20px;
      background: var(--r-panel);
      color: var(--r-muted);
      border: 1px solid var(--r-border);
      padding: 8px 12px;
      border-radius: 10px;
      font-size: 12px;
      font-weight: 600;
      z-index: 100;
    }
    .shortcut-hint kbd {
      background: var(--r-panel-2);
      color: var(--r-text-2);
      padding: 2px 7px;
      border-radius: 5px;
      margin: 0 2px;
      font-weight: 700;
    }

    /* First-visit tooltip-discoverability hint */
    .tooltip-hint {
      position: fixed;
      bottom: 90px;
      left: 50%;
      transform: translateX(-50%) translateY(8px);
      /* Not --r-accent: white on #e6855e measures 2.66:1, and WCAG AA wants
         4.5:1 for text this size. #b8532b is the same hue deepened and
         measures 4.87:1. The accent token is left alone because it is used
         elsewhere as a foreground colour on dark panels, where it is fine. */
      background: #b8532b;
      color: #fff;
      padding: 14px 14px 14px 18px;
      border-radius: 14px;
      font-size: 14px;
      font-weight: 600;
      display: flex;
      align-items: flex-start;
      gap: 10px;
      z-index: 200;
      box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.4s ease, transform 0.4s ease;
      max-width: calc(100% - 40px);
    }
    .tooltip-hint.show {
      opacity: 1;
      pointer-events: auto;
      transform: translateX(-50%) translateY(0);
    }
    .tooltip-hint-body {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .tooltip-hint-row {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .tooltip-hint-icon {
      font-size: 14px;
      flex-shrink: 0;
    }
    .tooltip-hint-dismiss {
      background: rgba(255, 255, 255, 0.18);
      border: none;
      color: white;
      font-size: 18px;
      width: 24px;
      height: 24px;
      border-radius: 50%;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 0;
      line-height: 1;
      transition: background 0.15s;
      margin-left: 4px;
      flex-shrink: 0;
      align-self: flex-start;
    }
    .tooltip-hint-dismiss:hover {
      background: rgba(255, 255, 255, 0.32);
    }
  </style>
</head>
<body>
  <div class="progress" id="progress"></div>
  <div class="mode-indicator" id="modeIndicator">Simplified</div>

  <!-- First-visit tooltip hint. Rows are written by buildTooltipHint() from
       the device's actual pointer type — the previous fixed copy told phone
       users to "hover", a gesture their device does not have. -->
  <div class="tooltip-hint" id="tooltipHint" role="status" aria-live="polite">
    <div class="tooltip-hint-body"></div>
    <button class="tooltip-hint-dismiss" onclick="dismissTooltipHint()" aria-label="Dismiss hint">×</button>
  </div>

  <!-- Whole-bubble translation. One instance, repositioned under whichever
       bubble is active. -->
  <div class="bubble-panel" id="bubblePanel">
    <button class="bp-hide" onclick="hideTranslations(event)"
            title="Hide bubble translations" aria-label="Hide bubble translations">&times;</button>
    <div class="bp-inter" id="bubblePanelInter"></div>
    <div class="bp-sep"></div>
    <div class="bp-gloss-wrap" id="bubblePanelGlossWrap">
      <div class="bp-gloss" id="bubblePanelGloss"></div>
      <button type="button" class="bp-reveal" id="bubblePanelReveal" onclick="revealGloss(event)"
              aria-label="Show translation"><span>Tap to show translation</span></button>
      <button type="button" class="bp-hide-gloss" id="bubblePanelHideGloss" onclick="hideGloss(event)">Hide translation</button>
    </div>
  </div>

  <button class="hm-speak" id="bubbleSpeak" onclick="playActiveBubble(event)"
          title="Play this bubble in Chinese" aria-label="Play this bubble in Chinese"></button>

  <div class="reader-toast" id="readerToast" role="status" aria-live="polite"></div>


  <div class="container" id="container"></div>

  <div class="controls">
    ${data.availableLevels ? `<select id="levelSelect" onchange="switchLevel(this.value)">
      ${data.availableLevels.map(l => {
        const label = l === 'original' ? 'Original' : l === 'english' ? 'English' : l.toUpperCase();
        const selected = l === (data.availableLevels![1] || data.targetLevel) ? ' selected' : '';
        return `<option value="${l}"${selected}>${label}</option>`;
      }).join('\n      ')}
    </select>` : `<button onclick="toggleMode()" id="toggleBtn" title="Toggle between simplified and original (O)">⇄</button>`}
    <button onclick="toggleSettings()" id="settingsBtn" title="Reader settings">⚙</button>
    <button onclick="scrollToTop()" title="Scroll to Top">↑</button>
    <button onclick="scrollToBottom()" title="Scroll to Bottom">↓</button>
  </div>

  <!-- Settings popup. Two toggle switches — proper-noun swap and above-level
       highlight hide. Outside .controls so its position: fixed isn't clipped. -->
  <div class="settings-popup" id="settingsPopup" role="dialog" aria-label="Reader settings">
    <div class="settings-title">Reader settings</div>
    ${hasEnglishNames ? `<div class="settings-row">
      <span class="settings-label">Show names &amp; places in English</span>
      <label class="toggle-switch">
        <input type="checkbox" id="namesCheckbox" onchange="toggleNames()">
        <span class="toggle-slider"></span>
      </label>
    </div>` : ''}
    <div class="settings-row">
      <span class="settings-label">Hide above-level highlight</span>
      <label class="toggle-switch">
        <input type="checkbox" id="highlightCheckbox" onchange="toggleHighlight()">
        <span class="toggle-slider"></span>
      </label>
    </div>
    <div class="settings-row">
      <span class="settings-label">Traditional characters</span>
      <label class="toggle-switch">
        <input type="checkbox" id="scriptCheckbox" onchange="toggleScript()">
        <span class="toggle-slider"></span>
      </label>
    </div>
    <div class="settings-row">
      <span class="settings-label">Sentence translation</span>
      <div class="seg" id="prefTranslationSeg" role="radiogroup" aria-label="Sentence translation">
        <button type="button" role="radio" data-mode="tap" onclick="setTranslationMode('tap')">Tap to show</button>
        <button type="button" role="radio" data-mode="always" onclick="setTranslationMode('always')">Always</button>
        <button type="button" role="radio" data-mode="off" onclick="setTranslationMode('off')">Off</button>
      </div>
    </div>
    <div class="settings-row" id="audioSettingRow" style="display:none">
      <span class="settings-label">Pronunciation audio</span>
      <label class="toggle-switch">
        <input type="checkbox" id="prefAudio" checked onchange="applyReaderPrefs()">
        <span class="toggle-slider"></span>
      </label>
    </div>
  </div>

  <div class="shortcut-hint">
    <kbd>O</kbd> Toggle view
  </div>

  <script>
    const readerData = ${JSON.stringify(data, null, 2)};

    const availableLevels = readerData.availableLevels || null;
    let currentLevel = availableLevels ? availableLevels[1] || availableLevels[0] : 'interactive';

    // Script display mode: 'simp' | 'trad'. Display-only — phrase.text stays
    // canonical Simplified for flashcard keys, HSK data and name lookups; the
    // Traditional form rides along as phrase.t (baked at generation time).
    // Persisted, unlike the other two reader toggles: which script a learner
    // reads is a stable property of that learner, not a per-chapter tweak.
    const SCRIPT_KEY = 'hm_script';
    let currentScript = 'simp';
    try {
      if (localStorage.getItem(SCRIPT_KEY) === 'trad') currentScript = 'trad';
    } catch (e) {}

    // ---- Usage beacon (anonymous, first-party) ----
    // Small fire-and-forget events to /api/beacon so real reading activity is
    // measurable server-side (reader opens, word lookups, toggles, flashcard
    // saves — previously all of this died in localStorage). No account or IP
    // linkage: hm_bid is a random per-browser id, used only to count distinct
    // and returning readers.
    function hmBid() {
      try {
        var v = localStorage.getItem('hm_bid');
        if (!v) {
          v = Math.random().toString(36).slice(2) + Date.now().toString(36);
          localStorage.setItem('hm_bid', v);
        }
        return v;
      } catch (e) { return 'anon'; }
    }
    // What kind of device is reading. Without this the desktop/mobile split is
    // unmeasurable, and every layout decision is a guess. Pointer type matters
    // more than width: a touchscreen laptop is wide AND hoverless.
    function hmDevice() {
      try {
        return {
          vw: window.innerWidth,
          vh: window.innerHeight,
          ptr: window.matchMedia('(pointer: coarse)').matches ? 'coarse' : 'fine',
          hover: window.matchMedia('(hover: hover)').matches ? 1 : 0,
          dpr: Math.round((window.devicePixelRatio || 1) * 100) / 100,
        };
      } catch (e) { return {}; }
    }

    function hmBeacon(event, props) {
      try {
        var payload = JSON.stringify({ event: event, bid: hmBid(), props: props || {} });
        if (navigator.sendBeacon) {
          navigator.sendBeacon('/api/beacon', new Blob([payload], { type: 'application/json' }));
        } else {
          fetch('/api/beacon', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: payload, keepalive: true });
        }
      } catch (e) { /* never let telemetry break the reader */ }
    }

    function init() {
      const container = document.getElementById('container');

      readerData.pages.forEach((page, pageIndex) => {
        const pageDiv = document.createElement('div');
        pageDiv.className = 'page';
        pageDiv.dataset.pageIndex = pageIndex;

        // Image — native lazy loading (loading="lazy"). Browser defers offscreen
        // downloads without JS. Also sets width/height so the browser can reserve
        // page space upfront — scrollHeight becomes accurate at DOM ready, killing
        // layout shift and fixing scroll-to-top/bottom accuracy.
        const img = document.createElement('img');
        img.src = 'cleaned/' + page.cleanedImage;
        img.loading = 'lazy';
        img.alt = 'Page ' + page.pageNumber;
        img.dataset.cleaned = 'cleaned/' + page.cleanedImage;
        img.dataset.original = 'original/' + page.originalImage;
        if (page.imageWidth && page.imageHeight) {
          img.width = page.imageWidth;
          img.height = page.imageHeight;
        }
        pageDiv.appendChild(img);

        // Create text overlays for each block
        const fitQueue = [];
        if (page.blocks) {
          page.blocks.forEach((block, blockIdx) => {
            const bubbleDiv = document.createElement('div');
            const isDark = block.textColor === '#FFFFFF' || block.textColor === '#ffffff';
            bubbleDiv.className = 'bubble-text' + (block.isVertical ? ' vertical' : '') + (isDark ? ' dark-bg' : '');
            bubbleDiv.style.left = block.bbox.x + '%';
            bubbleDiv.style.top = block.bbox.y + '%';
            bubbleDiv.style.width = block.bbox.width + '%';
            bubbleDiv.style.height = block.bbox.height + '%';
            bubbleDiv.dataset.fontSize = String(block.fontSize);
            bubbleDiv.dataset.blockIdx = String(blockIdx);
            bubbleDiv.dataset.pageIdx = String(pageIndex);
            bubbleDiv.style.color = block.textColor || '#000';

            if (block.rotation && Math.abs(block.rotation) > 0.5) {
              bubbleDiv.classList.add('rotated');
              bubbleDiv.style.transform = 'rotate(' + block.rotation + 'deg)';
            }

            // Bubble-level gloss: what the whole speech bubble means. Surfaced
            // in the word popup so a reader who knows the words but can't
            // assemble the sentence gets both at once.
            if (block.gloss) bubbleDiv.dataset.gloss = block.gloss;
            renderBlockPhrases(bubbleDiv, block.phrases);
            pageDiv.appendChild(bubbleDiv);
            fitQueue.push({ el: bubbleDiv, block: block });
          });
        }

        container.appendChild(pageDiv);
        // Fit AFTER the page is in the document — offsetWidth needs layout.
        fitQueue.forEach(q => fitBubbleText(q.el, q.block, page.imageWidth, page.imageHeight, currentLevel));
      });

      updateModeIndicator();
    }

    // Chinese → English character override map, built once from the chapter
    // analysis's characters array (which has explicit { chineseName, name }
    // pairs). Catches the single-character surname case (e.g. 朴 → Park) that
    // the no-CEDICT heuristic would miss because CEDICT has 朴 as "Korean
    // surname; also simple/plain".
    // c.chineseName !== c.name matters: Chinese-input chapters have no English
    // name, so the pipeline mirrors the Chinese form into both fields to keep
    // the characters array populated. Mapping 比约恩 to 比约恩 is not a swap: it
    // mark the span .is-name, strip its level highlight, and make the toggle
    // appear to do nothing but remove colour.
    const characterMap = (readerData.characters || [])
      .filter(c => c && c.chineseName && c.name && c.chineseName !== c.name)
      .reduce((m, c) => { m[c.chineseName] = c.name; return m; }, {});

    // Chapter-specific Chinese terms (character names + KEEP terms) that must
    // NEVER be dictionary-swapped — we can't reliably romanize invented names,
    // and story names like 克拉克/Clark are characters, not real proper nouns.
    const protectedNames = new Set(readerData.protectedNames || []);

    // Resolve the English form for a phrase, or null if it shouldn't be swapped.
    // See resolveEnglishName below for the rule order. extractProperNoun pulls
    // the clean proper-noun run off the front of a CC-CEDICT definition.
    // Strips multi-clause descriptions so "Sung Jin Woo, the protagonist"
    // → "Sung Jin Woo".
    // Connector words allowed inside a multi-word proper noun (e.g. "United
    // States of America", "Joan of Arc", "Bank of the West"). Only kept when
    // sandwiched between capitalized tokens.
    const PROPER_NOUN_CONNECTORS = new Set(['of', 'the', 'and', 'de', 'la', 'du', 'al', 'van', 'von']);

    // Extract the clean display form of a proper noun from a CEDICT definition
    // or chapter-analysis gloss. CEDICT is messy ("South Korea (Republic of
    // Korea)/(historical) Han...", "Beijing municipality, capital of..."),
    // but proper nouns always START with a run of Title-Case words. Everything
    // after the first lowercase non-connector word is descriptive context.
    function extractProperNoun(definition) {
      // NOTE: this function runs INSIDE the reader HTML template literal,
      // so backslashes must be double-escaped to survive template-literal
      // parsing. \\s → \s, \\( → \( in the emitted HTML.
      const cleaned = definition.replace(/^\\([^)]*\\)\\s*/, '');
      const firstClause = cleaned.split(/[,;/(]/)[0].trim();
      if (!firstClause) return null;
      const words = firstClause.split(/\\s+/);
      const result = [];
      const pending = [];
      for (const w of words) {
        if (/^[A-Z]/.test(w)) {
          result.push(...pending, w);
          pending.length = 0;
        } else if (PROPER_NOUN_CONNECTORS.has(w.toLowerCase())) {
          pending.push(w);
        } else {
          break;
        }
      }
      if (result.length === 0) return null;
      const display = result.join(' ');
      return display.length > 1 ? display : null;
    }

    function resolveEnglishName(phrase) {
      // 1. Reliable name from the analysis map. Only populated when the analysis
      //    gave both a Chinese form and an English name — true for English-input
      //    chapters (the English name is the source of truth), empty for
      //    Chinese-input chapters (no reliable romanization exists). Swap it.
      if (characterMap[phrase.text]) return characterMap[phrase.text];
      // 2. Chapter-specific terms (character names + KEEP: invented places,
      //    skills, concepts) → never swap. Catches names like 克拉克 that happen
      //    to be dictionary proper nouns but are the story's own characters.
      if (protectedNames.has(phrase.text)) return null;
      // 3. Latin-mixed terms (E级, B超, K歌) are jargon/codes, not proper nouns.
      if (/[A-Za-z]/.test(phrase.text)) return null;
      // 4. HSK 1–3 is basic vocab the learner already reads (我, 中国, 一月) — a
      //    real proper noun the learner would struggle with is essentially never
      //    in this band.
      if (phrase.hskLevel && phrase.hskLevel <= 3) return null;
      // 5. Single-char tokens are the CC-CEDICT polysemy traps (神→"God",
      //    壮→"Zhuang"); real swappable proper nouns (cities/countries) are 2+.
      if ([...phrase.text].length < 2) return null;
      // 6. Must be a genuine CC-CEDICT proper noun (capitalized pinyin, baked in
      //    at build time). This is the reliable path that keeps 北京→Beijing,
      //    首尔→Seoul while dropping common-noun definitions that merely start
      //    with a capital.
      if (!phrase.properNoun) return null;
      // 7. Language names (中文, 英文, 俄语, 印地语…) are CC-CEDICT proper nouns
      //    but common vocab, not names/places. The whole X文/X语 "…language…"
      //    family is the one systematic false positive left, so skip it.
      if (/[文语]$/.test(phrase.text) && /language/i.test(phrase.definition || '')) return null;
      if (phrase.definition) return extractProperNoun(phrase.definition);
      return null;
    }

    function renderBlockPhrases(bubbleDiv, phrases) {
      // Clear existing content
      const existing = bubbleDiv.querySelector('.bubble-inner');
      if (existing) existing.remove();

      const innerWrap = document.createElement('div');
      innerWrap.className = 'bubble-inner';

      // Group each non-punct phrase with its surrounding punctuation inside a
      // .phrase-group (white-space: nowrap) so punctuation can't get orphaned
      // on its own line. Trailing/closing puncts attach BACKWARD to the
      // preceding non-punct (default); LEADING/opening puncts attach FORWARD
      // to the following non-punct. Otherwise an opener like '[' before a
      // bracketed name (e.g. '[DUNGEON AND STONE]') gets locked into the
      // previous word's group, forcing them to wrap together unnecessarily.
      // Curly quotes included: the simplifier emits directional quote pairs,
      // so “ and ‘ are unambiguous openers. Without them, “ attaches BACKWARD
      // and dangles at line end (就是“ / 熙英” instead of 就是 / “熙英”).
      const OPENERS = new Set(['[', '(', '（', '「', '『', '【', '〈', '《', '｢', '“', '‘']);
      let currentGroup = null;
      let pendingOpenerGroup = null; // opener-led group waiting for a non-punct to absorb it
      // ASCII quotes (compactPunctuation converts curly “”‘’ to " and ') are
      // directionless, so opener/closer is decided by parity: quotes pair up
      // within a bubble, so the 1st/3rd/5th… occurrence opens, the rest close.
      let dquoteSeen = 0, squoteSeen = 0;

      phrases.forEach((phrase, phIdx) => {
        const span = document.createElement('span');
        const enName = resolveEnglishName(phrase);
        const hasInternalSpace = phrase.text && phrase.text.indexOf(' ') !== -1;
        // Display form only. resolveEnglishName above and every dataset key
        // below deliberately keep using phrase.text: the name map, HSK data and
        // flashcard storage are all keyed on Simplified.
        const display = (currentScript === 'trad' && phrase.t) ? phrase.t : phrase.text;
        if (!phrase.pinyin) {
          span.className = 'phrase punct' + (hasInternalSpace ? ' has-spaces' : '');
          span.textContent = display;
        } else {
          span.className = 'phrase' + (phrase.isAboveLevel ? ' above-level' : '') + (enName ? ' is-name' : '');
          if (enName) {
            // Both versions in the DOM; CSS swaps visibility via body class.
            // Use child spans (textContent escapes user data safely).
            const zh = document.createElement('span');
            zh.className = 'zh-text';
            zh.textContent = display;
            const en = document.createElement('span');
            en.className = 'en-text';
            en.textContent = enName;
            span.appendChild(zh);
            span.appendChild(en);
          } else {
            span.textContent = display;
          }
          span.dataset.pinyin = phrase.pinyin;
          // Canonical Simplified + Traditional forms, independent of what is
          // currently displayed. The popover reads these so a word saved in
          // Traditional mode lands under the same flashcard key as the same
          // word saved in Simplified mode.
          span.dataset.simplified = phrase.text;
          if (phrase.t) span.dataset.trad = phrase.t;
          if (phrase.definition) span.dataset.definition = phrase.definition;
          // Phase 3.2: cedictDefinition is set ONLY when CC-CEDICT had an entry.
          // Story-specific names (e.g., character names) lack a dictionary entry,
          // so this attr is absent and the popup's ⭐ button is disabled.
          if (phrase.cedictDefinition) span.dataset.cedict = phrase.cedictDefinition;
          if (phrase.hskLevel) span.dataset.hskLevel = String(phrase.hskLevel);
          // Index into the phrase array — the popover uses it to find this
          // word's time span in the audio manifest (ph is index-parallel).
          span.dataset.phIdx = String(phIdx);
        }

        const isPunct = !phrase.pinyin;
        let isOpener = isPunct && OPENERS.has(phrase.text);
        if (isPunct && phrase.text === '"') { isOpener = dquoteSeen % 2 === 0; dquoteSeen++; }
        if (isPunct && phrase.text === "'") { isOpener = squoteSeen % 2 === 0; squoteSeen++; }

        if (isOpener) {
          // Start a new group for the opener (don't attach to the preceding
          // non-punct). The next non-punct will join this group.
          if (!pendingOpenerGroup) {
            pendingOpenerGroup = document.createElement('span');
            pendingOpenerGroup.className = 'phrase-group';
            innerWrap.appendChild(pendingOpenerGroup);
          }
          pendingOpenerGroup.appendChild(span);
          currentGroup = pendingOpenerGroup;
        } else if (!isPunct) {
          // Non-punct: join the pending opener group if one is waiting,
          // otherwise start a fresh group.
          if (pendingOpenerGroup) {
            pendingOpenerGroup.appendChild(span);
            currentGroup = pendingOpenerGroup;
            pendingOpenerGroup = null;
          } else {
            currentGroup = document.createElement('span');
            currentGroup.className = 'phrase-group';
            innerWrap.appendChild(currentGroup);
            currentGroup.appendChild(span);
          }
        } else {
          // Trailing/closer punct: attach to current group, or start one if
          // we're at the very beginning of the block (nothing to attach to).
          if (!currentGroup) {
            currentGroup = document.createElement('span');
            currentGroup.className = 'phrase-group';
            innerWrap.appendChild(currentGroup);
          }
          currentGroup.appendChild(span);
        }

        // Mark the containing group as wrap-eligible so its white-space rule
        // becomes 'normal'. Without this, the merged English phrase's inline
        // display alone isn't enough — the parent group's 'nowrap' would still
        // prevent the browser from breaking at the internal spaces.
        if (hasInternalSpace && currentGroup) {
          currentGroup.classList.add('has-wrap');
        }
      });
      bubbleDiv.appendChild(innerWrap);
    }

    // === Fit pass (v19) ===
    // Sizes each bubble by MEASURING its real rendered text instead of trusting
    // the server's character-count estimate: binary-search the largest font
    // size whose NATURAL browser wrap fits the box, then enforce the block's
    // minimum size (derived server-side from the original lettering height),
    // overflowing the OCR box into surrounding bubble whitespace within a
    // bound when the floor demands it.
    //
    // Deliberately NO layout search, NO forced line breaks, NO preference
    // rules: lines exist only where width forces them, so a line is short only
    // as a natural remainder. The v18 predecessor searched (line count, size)
    // with fullness/balance/height-bound rules and inserted <br>s — every one
    // of those rules existed to patch a failure of another, and a width-pinned
    // atomic group still stacked short groups into 1-2 character towers
    // (我们 / 就 / 来演 / 好警察坏警察。). Shrink-from-above lands on the same
    // sizes without the machinery: three-way comparison on Ep120 agreed with
    // v18 within 1px on 96% of bubbles and beat it on the rest.
    // All math runs in image-space px; scaleFonts converts to display px.
    const FIT_MIN_SIZE = 10;
    const FIT_MAX_SIZE = 72;
    const FIT_LINE_SPACING = 1.1;      // must match .bubble-text line-height
    const FIT_MIN_OVERFLOW_CAP = 1.35; // minimum-size rule may exceed box height up to this

    // Greedy wrap of the atomic groups at a given size — mirrors exactly what
    // the browser will do. Returns line count and the widest line's px width.
    function fitWrap(widths, size, maxW) {
      let lines = 1, cur = 0, widest = 0;
      for (let i = 0; i < widths.length; i++) {
        const w = widths[i] * size;
        if (cur > 0 && cur + w > maxW) {
          lines++;
          if (cur > widest) widest = cur;
          cur = w;
        } else {
          cur += w;
        }
      }
      if (cur > widest) widest = cur;
      return { lines: lines, widest: widest };
    }

    // Shared 2D context for word-level measurement. measureText is sub-pixel
    // and lives in layout space, so parent rotation transforms can't skew it.
    let fitCanvasCtx = null;
    function fitMeasureCtx(el) {
      if (!fitCanvasCtx) fitCanvasCtx = document.createElement('canvas').getContext('2d');
      fitCanvasCtx.font = '100px ' + getComputedStyle(el).fontFamily;
      return fitCanvasCtx;
    }

    function fitBubbleText(el, block, imageWidth, imageHeight, level) {
      if (!block || block.isVertical) return;
      const inner = el.querySelector('.bubble-inner');
      if (!inner) return;
      const groups = Array.from(inner.children).filter(c => c.classList && c.classList.contains('phrase-group'));
      if (!groups.length) return;

      // Measure each wrap-atomic unit once at a 100px reference size with
      // wrapping disabled (max-content). Widths scale linearly with font size,
      // so the whole search after this is pure math with no further reflows.
      //
      // CJK phrase-groups are atomic — one unit each. But a group with
      // internal spaces (merged English text, class has-wrap) can wrap INSIDE
      // itself at the spaces; measured as one unit it becomes a "sentence
      // brick", and the English layer — whose whole text is a single phrase —
      // would shrink until the entire sentence fit on ONE line. Split those
      // into per-word units instead. Every word but the group's last carries
      // its trailing space: when a line break lands on that space the browser
      // drops it, so this overcounts a line by ≤1 space width — erring toward
      // fitting, never overflowing.
      const prevFont = el.style.fontSize;
      const prevWidth = inner.style.width;
      el.style.fontSize = '100px';
      inner.style.width = 'max-content';
      const unit = []; // px per 1px of font size, one entry per wrap opportunity
      let ctx = null;
      for (let gi = 0; gi < groups.length; gi++) {
        const g = groups[gi];
        if (g.classList.contains('has-wrap')) {
          if (!ctx) ctx = fitMeasureCtx(el);
          const words = (g.textContent || '').split(' ').filter(w => w.length > 0);
          const spaceW = ctx.measureText(' ').width / 100;
          for (let wi = 0; wi < words.length; wi++) {
            const w = ctx.measureText(words[wi]).width / 100;
            unit.push(wi < words.length - 1 ? w + spaceW : w);
          }
        } else {
          unit.push(g.offsetWidth / 100);
        }
      }
      el.style.fontSize = prevFont;
      inner.style.width = prevWidth;
      if (!unit.length || unit.every(w => w === 0)) return; // not laid out (hidden) — keep server size

      // Box in image-space px (.bubble-text has 2px CSS padding per side).
      const boxW = (block.bbox.width / 100) * imageWidth - 4;
      const boxH = (block.bbox.height / 100) * imageHeight - 4;
      if (boxW < FIT_MIN_SIZE || boxH < FIT_MIN_SIZE) return;

      // Largest size whose natural wrap fits, by binary search. Wrapped height
      // is monotonic in size, so shrink-from-above lands on the biggest size
      // with the fewest lines automatically. The 1px width margin absorbs
      // offsetWidth's integer rounding (±0.5px per group at the reference
      // size) so a rounding wobble can't push a line past the box edge and
      // wrap where the math said it wouldn't.
      let lo = FIT_MIN_SIZE, hi = FIT_MAX_SIZE, best = FIT_MIN_SIZE;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        const w = fitWrap(unit, mid, boxW);
        if (w.widest <= boxW - 1 && w.lines * FIT_LINE_SPACING * mid <= boxH) {
          best = mid;
          lo = mid + 1;
        } else {
          hi = mid - 1;
        }
      }
      let finalSize = best;

      // Minimum-size rule: don't render dialogue smaller than the original
      // lettering. Skipped for the English layer — its long lines are what
      // shaped the box in the first place. When the floor wins, natural wrap
      // at box width decides the lines and the (bounded) vertical overflow
      // spills symmetrically into bubble whitespace via flex centering.
      const minSize = (level === 'english') ? 0 : (block.minFontSize || 0);
      if (minSize > finalSize) {
        const wMin = fitWrap(unit, minSize, boxW);
        if (wMin.lines * FIT_LINE_SPACING * minSize <= boxH * FIT_MIN_OVERFLOW_CAP) {
          finalSize = minSize;
        }
      }

      el.dataset.fontSize = String(finalSize);
    }

    // === Settings toggles ===

    function toggleNames() {
      // The row is absent when the chapter has no English names to swap to.
      const box = document.getElementById('namesCheckbox');
      if (!box) return;
      document.body.classList.toggle('show-en-names', box.checked);
    }

    function toggleHighlight() {
      const on = document.getElementById('highlightCheckbox').checked;
      document.body.classList.toggle('no-above-highlight', on);
    }

    // Re-renders every overlay rather than rewriting the existing spans in
    // place. Deliberate: switchLevel() already tears out .bubble-inner and
    // rebuilds from the phrase data, so any in-place DOM conversion would be
    // silently discarded the moment the reader changed HSK level. Driving the
    // display off currentScript inside renderBlockPhrases makes every rebuild —
    // this one, switchLevel's, the initial paint — correct by construction.
    function toggleScript() {
      const on = document.getElementById('scriptCheckbox').checked;
      currentScript = on ? 'trad' : 'simp';
      try { localStorage.setItem(SCRIPT_KEY, currentScript); } catch (e) {}
      applyScript();
      hmBeacon('toggle_script', { to: currentScript });
    }

    // Swaps the font stack and repaints the visible text. Also called on load so
    // a persisted Traditional preference applies before the reader is seen.
    function applyScript() {
      document.body.classList.toggle('script-trad', currentScript === 'trad');
      dismissPopover();
      document.querySelectorAll('.bubble-text').forEach(el => {
        const pageIdx = parseInt(el.dataset.pageIdx);
        const blockIdx = parseInt(el.dataset.blockIdx);
        const block = readerData.pages[pageIdx]?.blocks?.[blockIdx];
        if (!block) return;
        let phrases = block.phrases;
        if (block.phrasesByLevel && block.phrasesByLevel[currentLevel]) {
          phrases = block.phrasesByLevel[currentLevel];
        }
        renderBlockPhrases(el, phrases);
        // The rebuild dropped any forced line breaks — re-fit for the new
        // script's rendering (Traditional glyph advances match Simplified,
        // but the layout must be recomputed to restore the breaks).
        const pg = readerData.pages[pageIdx];
        if (pg) fitBubbleText(el, block, pg.imageWidth, pg.imageHeight, currentLevel);
      });
      scaleFonts();
      updateModeIndicator();
    }

    function toggleSettings() {
      const popup = document.getElementById('settingsPopup');
      const btn = document.getElementById('settingsBtn');
      const open = popup.classList.toggle('open');
      btn.classList.toggle('active', open);
    }


    function switchLevel(level) {
      dismissPopover();
      currentLevel = level;
      hmBeacon('toggle_mode', { to: level });

      const images = document.querySelectorAll('.page img');
      const overlays = document.querySelectorAll('.bubble-text');

      if (level === 'original') {
        // Show original images, hide overlays
        images.forEach(img => { img.src = img.dataset.original; });
        overlays.forEach(el => { el.style.display = 'none'; });
      } else {
        // Show cleaned images, show overlays
        images.forEach(img => { img.src = img.dataset.cleaned; });

        // Rebuild overlays with the selected level's phrases
        overlays.forEach(el => {
          el.style.display = '';
          const pageIdx = parseInt(el.dataset.pageIdx);
          const blockIdx = parseInt(el.dataset.blockIdx);
          const block = readerData.pages[pageIdx]?.blocks?.[blockIdx];
          if (!block) return;

          // Use phrasesByLevel if available, otherwise fall back to default phrases
          let phrases = block.phrases;
          if (block.phrasesByLevel && block.phrasesByLevel[level]) {
            phrases = block.phrasesByLevel[level];
          }
          if (block.gloss) el.dataset.gloss = block.gloss;
          renderBlockPhrases(el, phrases);

          // Re-fit for this level's actual text. This replaces the old
          // character-count English ratio hack — the fit pass measures the
          // real rendered words, so English gets a correct wrapped layout
          // instead of an estimated shrink.
          el.dataset.fontSize = String(block.fontSize);
          const pg = readerData.pages[pageIdx];
          fitBubbleText(el, block, pg.imageWidth, pg.imageHeight, level);
        });

        scaleFonts();
      }

      updateModeIndicator();
    }

    function toggleMode() {
      if (availableLevels) {
        // Cycle through available levels
        const idx = availableLevels.indexOf(currentLevel);
        const nextIdx = (idx + 1) % availableLevels.length;
        const nextLevel = availableLevels[nextIdx];
        const select = document.getElementById('levelSelect');
        if (select) select.value = nextLevel;
        switchLevel(nextLevel);
      } else {
        // Simple toggle: interactive <-> original. Every image has a src from
        // initial render (native lazy loading defers the actual download, but
        // the src attribute is set) so no guard needed.
        currentLevel = currentLevel === 'interactive' ? 'original' : 'interactive';
        const images = document.querySelectorAll('.page img');
        const overlays = document.querySelectorAll('.bubble-text');
        images.forEach(img => {
          img.src = currentLevel === 'interactive' ? img.dataset.cleaned : img.dataset.original;
        });
        overlays.forEach(el => {
          el.style.display = currentLevel === 'interactive' ? '' : 'none';
        });
        // The overlays just vanished (or came back) and the artwork swapped —
        // a panel anchored to a bubble is now pointing at the wrong place, or
        // at a bubble that is no longer visible.
        if (currentLevel === 'interactive') reanchorBubblePanel(); else deselectBubble();
        hmBeacon('toggle_mode', { to: currentLevel });
        updateModeIndicator();
        const btn = document.getElementById('toggleBtn');
        if (btn) btn.classList.toggle('active', currentLevel !== 'interactive');
      }
    }

    function updateModeIndicator() {
      const indicator = document.getElementById('modeIndicator');
      let label = currentLevel;
      if (currentLevel === 'interactive') label = 'Simplified';
      else if (currentLevel === 'original') label = 'Original';
      else if (currentLevel === 'english') label = 'English';
      else if (currentLevel.startsWith('hsk')) label = currentLevel.toUpperCase();
      // "Simplified" here means simplified VOCABULARY, not Simplified script —
      // a distinction nobody makes when Traditional characters are on screen
      // right next to the word. Suffix the script so the pill states both facts
      // instead of appearing to contradict what the reader can see. Skipped for
      // Original (no overlay) and English (nothing Chinese to label).
      if (currentScript === 'trad' && currentLevel !== 'original' && currentLevel !== 'english') {
        label += ' · 繁體';
      }
      indicator.textContent = label;
      indicator.style.opacity = '1';
      setTimeout(() => { indicator.style.opacity = '0.5'; }, 1500);
    }

    function scrollToTop() {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    function scrollToBottom() {
      window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
    }

    // Fade the controls out of the way while scrolling (see the CSS note).
    // Only meaningful on touch, but the class is harmless elsewhere.
    let hmScrollIdle = 0;
    window.addEventListener('scroll', () => {
      document.body.classList.add('scrolling');
      clearTimeout(hmScrollIdle);
      hmScrollIdle = setTimeout(() => document.body.classList.remove('scrolling'), 650);
    }, { passive: true });

    // Progress bar
    window.addEventListener('scroll', () => {
      const scrolled = window.scrollY;
      const maxScroll = document.body.scrollHeight - window.innerHeight;
      const progress = maxScroll > 0 ? (scrolled / maxScroll) * 100 : 0;
      document.getElementById('progress').style.width = progress + '%';
    });

    // Scale font sizes based on displayed image size vs source image size
    function scaleFonts() {
      document.querySelectorAll('.page').forEach(pageDiv => {
        const img = pageDiv.querySelector('img');
        if (!img || !img.naturalWidth) return;
        const scale = img.clientWidth / img.naturalWidth;
        pageDiv.querySelectorAll('.bubble-text').forEach(el => {
          const base = parseFloat(el.dataset.fontSize);
          // No 0.92 safety shrink: sizes come from the fit pass, which
          // measured the real text. Any blind shrink here can push a size
          // back under a wrap threshold and collapse the planned layout.
          if (base) el.style.fontSize = (base * scale) + 'px';
        });
      });
      // Bubbles just changed size, so anything anchored to one has moved.
      if (typeof reanchorBubblePanel === 'function') reanchorBubblePanel();
    }

    // Definition popover
    let activePopover = null;

    function dismissPopover() {
      if (activePopover) { activePopover.remove(); activePopover = null; }
    }

    function positionPopover(popover, span) {
      const rect = span.getBoundingClientRect();
      const popW = popover.offsetWidth;
      const popH = popover.offsetHeight;

      // Default: centered above the word
      let left = rect.left + rect.width / 2 - popW / 2 + window.scrollX;
      let top = rect.top - popH - 8 + window.scrollY;

      // Flip below if too close to top
      if (top < window.scrollY + 10) {
        top = rect.bottom + 8 + window.scrollY;
      }
      // Keep within horizontal viewport
      if (left < 4) left = 4;
      if (left + popW > window.innerWidth - 4) left = window.innerWidth - popW - 4;

      popover.style.left = left + 'px';
      popover.style.top = top + 'px';
    }

    // Phase 3.2: inline SVG star icon — renders consistently across systems.
    // Unicode ☆/★ varies (sometimes rendered as emoji on Windows) and would
    // break our color/border styling.
    // ================= Bubble selection =================
    // Two stages. The first tap anywhere in an unselected bubble SELECTS it and
    // shows its translation; a second tap on a word gives that word's meaning.
    //
    // This exists because a bubble is entirely filled with .phrase spans, so
    // "tap the bubble" and "tap a word" were the same gesture and there was no
    // way to ask about the sentence without first picking a word. It also
    // means the common path never requires hitting a 16px target — the bubble
    // is a far larger one, and only word lookup needs precision.
    //
    // Cost is one extra tap PER BUBBLE, not per word: three lookups inside one
    // bubble is four taps, not six.
    var hmSelected = null;
    document.body.classList.add('two-stage');


    // ================= Audio =================
    // Chapters generated without audio have no manifest; the controls stay
    // hidden and everything else works exactly as before.
    var SPEAKER_SVG = '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M11 5L6 9H2v6h4l5 4V5z"/><path d="M15.5 8.5a5 5 0 010 7"/><path d="M18.5 5.5a9 9 0 010 13"/></svg>';
    var hmAudio = { manifest: null, base: 'audio/', el: null, btn: null,
                    file: null, raf: 0, segEnd: null, tick: null };

    document.body.classList.add('no-audio-data');
    ${inlineManifest ? `Promise.resolve(${inlineManifest})` : `fetch('audio/manifest.json')
      .then(r => (r.ok ? r.json() : null))`}
      .then(m => {
        // Format 2: bubbles map to {file, s, e, ph} time spans inside batch
        // mp3 files. Older per-clip manifests predate this reader — ignore
        // them the same way as no audio at all.
        if (!m || m.format !== 2 || !m.bubbles) return;
        hmAudio.manifest = m;
        document.body.classList.remove('no-audio-data');
        const b = document.getElementById('bubbleSpeak');
        if (b) b.innerHTML = SPEAKER_SVG;
        // The setting only appears for chapters that actually have audio —
        // a dead toggle is worse than no toggle.
        const row = document.getElementById('audioSettingRow');
        if (row) row.style.display = '';
        if (hmSelected) positionBubbleSpeak(hmSelected);
      })
      .catch(() => { /* no audio for this chapter */ });

    // Audio spans are parallel to the block's BASE phrase array — the one the
    // chapter was generated from. A per-level phrase override showing a
    // different segmentation means the spans no longer describe what's on
    // screen. Content comparison, not reference: an override whose text
    // matches the base keeps its audio.
    function audioPhrasesMatch(el) {
      const page = readerData.pages[+el.dataset.pageIdx];
      const block = page && page.blocks && page.blocks[+el.dataset.blockIdx];
      if (!block || !block.phrases) return false;
      const shown = bubblePhrasesOf(el);
      if (shown === block.phrases) return true;
      if (shown.length !== block.phrases.length) return false;
      for (var i = 0; i < shown.length; i++) {
        if (shown[i].text !== block.phrases[i].text) return false;
      }
      return true;
    }

    function bubbleAudioOf(el) {
      const m = hmAudio.manifest;
      if (!m || !el || !audioPhrasesMatch(el)) return null;
      return m.bubbles['p' + el.dataset.pageIdx + 'b' + el.dataset.blockIdx] || null;
    }

    // Bubble batches sit next to the chapter ('audio/...'); citation clips
    // come from the site-wide shared library (root path '/word-audio/...').
    function audioSrcOf(entry) {
      return entry.file.charAt(0) === '/' ? entry.file : hmAudio.base + entry.file;
    }

    // ONE Audio element, reused. iOS only grants playback off a user gesture,
    // and a freshly constructed Audio() does not inherit that grant — so
    // creating a new one per clip works on desktop and silently fails on a
    // phone. Swapping src on a single unlocked element does not.
    //
    // Everything plays as a SEGMENT of a batch file: seek to the span's start,
    // stop at its end via the rAF loop below. A bubble and a word are the same
    // mechanism at different spans.
    var SEG_LEAD = 0.02;        // batch sentences carry ~0.1s lead silence
    var SEG_TAIL_WORD = 0.05;   // words sit flush against neighbours — stop tight
    var SEG_TAIL_BUBBLE = 0.25; // let the last syllable decay; inter-bubble gaps are ~0.5s

    function playSeg(entry, s, e, opts) {
      if (!entry || s == null || e == null || !hmPrefs.audio) return;
      opts = opts || {};
      if (!hmAudio.el) {
        hmAudio.el = new Audio();
        hmAudio.el.addEventListener('ended', clearSpeakingState);
        hmAudio.el.addEventListener('error', clearSpeakingState);
      }
      const a = hmAudio.el;
      try { a.pause(); } catch (err) {}
      clearSpeakingState();
      if (hmAudio.file !== entry.file) {
        a.src = audioSrcOf(entry);
        hmAudio.file = entry.file;
      }
      hmAudio.segEnd = e + (opts.tail != null ? opts.tail : SEG_TAIL_WORD);
      hmAudio.tick = opts.tick || null;
      if (opts.btn) { opts.btn.classList.add('speaking'); hmAudio.btn = opts.btn; }
      const begin = Math.max(0, s - SEG_LEAD);
      const go = function () {
        try { a.currentTime = begin; } catch (err) {}
        const p = a.play();
        if (p && p.catch) p.catch(clearSpeakingState);
        if (!hmAudio.raf) hmAudio.raf = requestAnimationFrame(segTick);
      };
      // Seeking needs metadata; the first play of a fresh file must wait for it.
      if (a.readyState >= 1) go();
      else a.addEventListener('loadedmetadata', go, { once: true });
    }

    // One loop drives both stop-at-segment-end and the karaoke highlight. It
    // never bails just because the element is momentarily paused — play() is
    // async and the first frames often run before it resolves. Skipping a
    // frame is correct; stopping is not. clearSpeakingState() is the exit.
    function segTick() {
      const a = hmAudio.el;
      if (!a || hmAudio.segEnd == null) { hmAudio.raf = 0; return; }
      if (a.currentTime >= hmAudio.segEnd && !a.paused) {
        try { a.pause(); } catch (err) {}
        clearSpeakingState();
        return;
      }
      if (hmAudio.tick) hmAudio.tick(a.currentTime);
      hmAudio.raf = requestAnimationFrame(segTick);
    }

    function clearSpeakingState() {
      waStop();
      if (hmAudio.raf) { cancelAnimationFrame(hmAudio.raf); hmAudio.raf = 0; }
      hmAudio.segEnd = null;
      hmAudio.tick = null;
      if (hmAudio.btn) hmAudio.btn.classList.remove('speaking');
      hmAudio.btn = null;
      // Cleared by query rather than from a list captured when playback
      // started. That list was written by karaokeTickFor while the options
      // object was being built — i.e. BEFORE playSeg ran, and playSeg opens
      // by calling this function, which emptied it. The tick closure held its
      // own references and kept painting, so by the end there was nothing
      // left to clean and every word stayed highlighted. Reading the DOM is
      // immune to that ordering, and to any future caller that highlights
      // without registering.
      document.querySelectorAll('.is-speaking, .is-spoken').forEach(function (n) {
        n.classList.remove('is-speaking', 'is-spoken');
      });
    }

    // ---- Word slices ----
    // A word is 100-400ms. Element seeking in an mp3 is bitrate-estimated and
    // lands tens of ms off — inaudible when a 5s bubble starts a hair early,
    // fatal when the slice IS the word. (Measured: the manifest spans wrap the
    // speech correctly — RMS in inter-word gaps is ~9x below speech level — so
    // the error is entirely in playback cutting.) Words therefore play through
    // WebAudio: the batch file decodes once, cuts land on exact samples, and
    // an 8ms gain ramp on each edge means the cut itself never clicks. Falls
    // back to the element path where WebAudio is unavailable.
    var hmWA = { ctx: null, buffers: {}, order: [], src: null };

    function waStop() {
      if (hmWA.src) { try { hmWA.src.stop(); } catch (err) {} hmWA.src = null; }
    }

    function playWordSlice(entry, s, e, opts) {
      opts = opts || {};
      var AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) { playSeg(entry, s, e, opts); return; }
      if (!hmWA.ctx) hmWA.ctx = new AC();
      // iOS creates the context suspended until a user gesture resumes it —
      // and this call is always inside one.
      if (hmWA.ctx.state === 'suspended') { try { hmWA.ctx.resume(); } catch (err) {} }
      var cached = hmWA.buffers[entry.file];
      if (cached) { waPlayBuf(cached, s, e, opts); return; }
      fetch(audioSrcOf(entry))
        .then(function (r) { return r.ok ? r.arrayBuffer() : Promise.reject(new Error('http')); })
        .then(function (ab) { return hmWA.ctx.decodeAudioData(ab); })
        .then(function (buf) {
          hmWA.buffers[entry.file] = buf;
          hmWA.order.push(entry.file);
          // Decoded PCM is ~2MB per ~20s batch; keep only a few around.
          if (hmWA.order.length > 3) delete hmWA.buffers[hmWA.order.shift()];
          waPlayBuf(buf, s, e, opts);
        })
        .catch(function () { playSeg(entry, s, e, opts); });
    }

    function waPlayBuf(buf, s, e, opts) {
      if (hmAudio.el) { try { hmAudio.el.pause(); } catch (err) {} }
      clearSpeakingState();
      var ctx = hmWA.ctx;
      var begin = Math.max(0, s - 0.015);
      var dur = Math.min(buf.duration - begin, (e - begin) + (opts.tail != null ? opts.tail : 0.05));
      if (dur <= 0) return;
      var src = ctx.createBufferSource();
      src.buffer = buf;
      var gain = ctx.createGain();
      src.connect(gain);
      gain.connect(ctx.destination);
      var FADE = 0.008;
      var t0 = ctx.currentTime + 0.01;
      gain.gain.setValueAtTime(0, t0);
      gain.gain.linearRampToValueAtTime(1, t0 + FADE);
      gain.gain.setValueAtTime(1, Math.max(t0 + FADE, t0 + dur - 0.012));
      gain.gain.linearRampToValueAtTime(0, t0 + dur);
      if (opts.btn) { opts.btn.classList.add('speaking'); hmAudio.btn = opts.btn; }
      src.onended = function () {
        if (hmWA.src === src) { hmWA.src = null; clearSpeakingState(); }
      };
      hmWA.src = src;
      src.start(t0, begin, dur);
    }

    // ---- Karaoke ----
    // Exact per-phrase time spans, measured from the TTS engine's word-boundary
    // events at generation time and baked into the manifest — the highlight is
    // wherever the voice actually is, no silence-padding heuristics. Both
    // element lists render from the same phrase array the spans are parallel
    // to, so index alignment is structural. Unvoiced phrases (punctuation,
    // stripped decorations) have null spans and are never highlighted.
    function karaokeTickFor(el, entry) {
      const spans = [].slice.call(el.querySelectorAll('.phrase'));
      const units = [].slice.call(document.querySelectorAll('#bubblePanelInter .bp-w'));
      const ph = entry.ph || [];
      return function (t) {
        for (var i = 0; i < ph.length; i++) {
          const w = ph[i];
          if (!w) continue;
          const speaking = t >= w[0] && t < w[1];
          const spoken = t >= w[1];
          if (spans[i]) { spans[i].classList.toggle('is-speaking', speaking); spans[i].classList.toggle('is-spoken', spoken); }
          if (units[i]) { units[i].classList.toggle('is-speaking', speaking); units[i].classList.toggle('is-spoken', spoken); }
        }
      };
    }

    function playActiveBubble(e) {
      if (e) e.stopPropagation();
      if (!hmSelected) return;
      const entry = bubbleAudioOf(hmSelected);
      if (!entry) return;
      playSeg(entry, entry.s, entry.e, {
        btn: document.getElementById('bubbleSpeak'),
        tick: karaokeTickFor(hmSelected, entry),
        tail: SEG_TAIL_BUBBLE,
      });
      hmBeacon('audio_bubble', { p: +hmSelected.dataset.pageIdx, b: +hmSelected.dataset.blockIdx });
    }

    // Sits just outside the bubble's right edge, vertically centred, clamped so
    // a bubble against the screen edge does not push it off.
    function positionBubbleSpeak(el) {
      const btn = document.getElementById('bubbleSpeak');
      if (!btn || !hmAudio.manifest) return;
      if (!bubbleAudioOf(el)) { btn.style.display = 'none'; return; }
      const r = el.getBoundingClientRect();
      const size = 44;
      const left = Math.min(r.right + 6, window.innerWidth - size - 8) + window.scrollX;
      btn.style.display = 'inline-flex';
      btn.style.left = left + 'px';
      btn.style.top = (r.top + window.scrollY + r.height / 2 - size / 2) + 'px';
    }

    function hideBubbleSpeak() {
      const btn = document.getElementById('bubbleSpeak');
      if (btn) btn.style.display = 'none';
    }

    // ---- Reader preferences ----
    // Deliberately subtractive: these only ever HIDE things, so no combination
    // can produce a broken reader — the bottom of the lattice is the plain
    // reader, which already works. Persisted per browser.
    var READER_PREFS_KEY = 'hm_reader_prefs';
    // translation: 'tap' (gloss veiled until tapped - the default), 'always'
    // (gloss shown as soon as a bubble is selected), or 'off' (no panel).
    var hmPrefs = { translation: 'tap', audio: true };
    var TRANSLATION_MODES = ['tap', 'always', 'off'];

    // Older builds stored a boolean. true was the always-show behaviour and
    // becomes the tap-to-reveal default; false stays off.
    function normalizeTranslationPref(v) {
      if (v === false) return 'off';
      return TRANSLATION_MODES.indexOf(v) >= 0 ? v : 'tap';
    }

    function loadReaderPrefs() {
      try {
        var raw = localStorage.getItem(READER_PREFS_KEY);
        if (raw) Object.assign(hmPrefs, JSON.parse(raw));
      } catch (e) {}
      hmPrefs.translation = normalizeTranslationPref(hmPrefs.translation);
      syncTranslationSeg();
      var a = document.getElementById('prefAudio');
      if (a) a.checked = hmPrefs.audio !== false;
      var rb = document.querySelector('#bubblePanelReveal span');
      if (rb) {
        var coarse = window.matchMedia && window.matchMedia('(pointer: coarse)').matches;
        rb.textContent = (coarse ? 'Tap' : 'Click') + ' to show translation';
      }
    }

    function syncTranslationSeg() {
      var seg = document.getElementById('prefTranslationSeg');
      if (!seg) return;
      seg.querySelectorAll('button').forEach(function (b) {
        var on = b.dataset.mode === hmPrefs.translation;
        b.classList.toggle('is-on', on);
        b.setAttribute('aria-checked', on ? 'true' : 'false');
      });
    }

    function setTranslationMode(mode) {
      hmPrefs.translation = normalizeTranslationPref(mode);
      syncTranslationSeg();
      applyReaderPrefs();
      hmBeacon('pref_translation', { mode: hmPrefs.translation });
    }

    function applyReaderPrefs() {
      var a = document.getElementById('prefAudio');
      hmPrefs.audio = a ? a.checked : true;
      document.body.classList.toggle('pref-no-audio', !hmPrefs.audio);
      if (!hmPrefs.audio && hmAudio.el) { try { hmAudio.el.pause(); } catch (e) {} }
      var on = hmPrefs.translation !== 'off';
      document.body.classList.toggle('pref-no-translation', !on);
      // Two-stage exists to give bubble-level info somewhere to appear. With
      // translations off there is nothing for the first tap to reveal, so it
      // would be pure friction — fall back to one tap.
      document.body.classList.toggle('two-stage', on);
      if (!on) deselectBubble();
      else if (hmSelected) reanchorBubblePanel();  // re-veil / unveil the open panel live
      try { localStorage.setItem(READER_PREFS_KEY, JSON.stringify(hmPrefs)); } catch (e) {}
    }

    function hideTranslations(e) {
      if (e) e.stopPropagation();
      setTranslationMode('off');
      readerToast('Translations hidden · turn back on in ⚙');
    }

    var hmToastTimer = 0;
    function readerToast(msg) {
      var el = document.getElementById('readerToast');
      if (!el) return;
      el.textContent = msg;
      el.classList.add('show');
      clearTimeout(hmToastTimer);
      hmToastTimer = setTimeout(function () { el.classList.remove('show'); }, 2600);
    }

    function bubbleGlossOf(el) { return (el && el.dataset.gloss) || ''; }

    // The same phrase list the overlay was rendered from, so the panel always
    // matches what is on screen — including the current HSK level.
    function bubblePhrasesOf(el) {
      if (!el) return [];
      const page = readerData.pages[+el.dataset.pageIdx];
      const block = page && page.blocks && page.blocks[+el.dataset.blockIdx];
      if (!block) return [];
      let phrases = block.phrases;
      if (block.phrasesByLevel && block.phrasesByLevel[currentLevel]) {
        phrases = block.phrasesByLevel[currentLevel];
      }
      return phrases || [];
    }

    // Interlinear: one column per word, pinyin above its own characters.
    function renderBubbleInterlinear(el) {
      const host = document.getElementById('bubblePanelInter');
      if (!host) return;
      const phrases = bubblePhrasesOf(el);
      if (!phrases.some(p => p.pinyin)) { host.innerHTML = ''; return; }
      host.innerHTML = phrases.map(p => {
        const py = p.pinyin || '';
        // Same display rule the bubble overlay uses: show the Traditional form
        // when the script toggle is on. Without this the panel sat in
        // Simplified while the bubble directly above it was Traditional.
        const hz = (currentScript === 'trad' && p.t) ? p.t : p.text;
        return '<span class="bp-w' + (py ? '' : ' is-punct') + '">' +
                 '<span class="bp-py">' + escapeHtml(py || '.') + '</span>' +
                 '<span class="bp-hz">' + escapeHtml(hz) + '</span>' +
               '</span>';
      }).join('');
    }

    // Bubbles whose translation the reader has revealed this session, keyed
    // like the panel ('pageIdx:blockIdx'). Sticky so re-selecting a bubble you
    // already translated does not veil it again.
    var hmRevealed = {};

    function applyGlossVeil(key) {
      var wrap = document.getElementById('bubblePanelGlossWrap');
      var gloss = document.getElementById('bubblePanelGloss');
      if (!wrap || !gloss) return;
      var tapMode = hmPrefs.translation === 'tap';
      var veiled = tapMode && !hmRevealed[key];
      wrap.classList.toggle('can-hide', tapMode);
      wrap.classList.toggle('is-veiled', veiled);
      wrap.classList.toggle('is-revealed', tapMode && !veiled);
      gloss.setAttribute('aria-hidden', veiled ? 'true' : 'false');
    }

    // Re-veil a translation the reader opened. Only offered in tap mode - in
    // 'always' the reader has asked for translations up front.
    function hideGloss(e) {
      if (e) e.stopPropagation();
      var p = document.getElementById('bubblePanel');
      var key = p && p.dataset.key;
      if (!key) return;
      delete hmRevealed[key];
      applyGlossVeil(key);
      var parts = key.split(':');
      hmBeacon('gloss_hide', { p: +parts[0], b: +parts[1] });
    }

    function revealGloss(e) {
      if (e) e.stopPropagation();
      var p = document.getElementById('bubblePanel');
      var key = p && p.dataset.key;
      if (!key) return;
      hmRevealed[key] = true;
      applyGlossVeil(key);
      var parts = key.split(':');
      hmBeacon('gloss_reveal', { p: +parts[0], b: +parts[1] });
    }

    function selectBubble(el) {
      if (!el) return;
      hmSelected = el;
      document.querySelectorAll('.bubble-text.is-active').forEach(b => b.classList.remove('is-active'));
      el.classList.add('is-active');
      showBubblePanel(el);
      positionBubbleSpeak(el);
    }

    function deselectBubble() {
      hmSelected = null;
      document.querySelectorAll('.bubble-text.is-active').forEach(b => b.classList.remove('is-active'));
      const p = document.getElementById('bubblePanel');
      if (p) { p.classList.remove('open'); delete p.dataset.key; }
      hideBubbleSpeak();
      if (hmAudio.el) { try { hmAudio.el.pause(); } catch (e) {} }
      clearSpeakingState();
    }

    function showBubblePanel(el) {
      const p = document.getElementById('bubblePanel');
      const gloss = bubbleGlossOf(el);
      if (!p || !gloss) { if (p) p.classList.remove('open'); return; }
      document.getElementById('bubblePanelGloss').textContent = gloss;
      renderBubbleInterlinear(el);
      p.classList.add('open');
      p.dataset.key = el.dataset.pageIdx + ':' + el.dataset.blockIdx;
      applyGlossVeil(p.dataset.key);
      positionBubblePanel(el);
    }

    // ALWAYS below the bubble. An earlier version flipped it above when the
    // next bubble was close (about 2% of cases), but above is exactly where
    // the word popover lives — positionPopover() puts it over the word and
    // only drops below near the top of the screen. So a flipped panel and a
    // word definition landed on top of each other every time, which is a
    // guaranteed collision traded against a rare, cosmetic one: below, the
    // panel may cover part of the next bubble, and that is the better failure.
    function positionBubblePanel(el) {
      const p = document.getElementById('bubblePanel');
      if (!p || !p.classList.contains('open') || !p.offsetWidth) return;
      const b = el.getBoundingClientRect();
      const pw = p.offsetWidth;
      let left = b.left + b.width / 2 - pw / 2 + window.scrollX;
      left = Math.max(6, Math.min(left, window.innerWidth - pw - 6));
      p.style.left = left + 'px';
      p.style.top = (b.bottom + window.scrollY + 8) + 'px';
    }

    // Re-anchor after anything that moves the artwork. toggleMode() swaps
    // original/cleaned images, which changes layout — without this the panel
    // stays where the bubble used to be.
    function reanchorBubblePanel() {
      if (!hmSelected || !document.body.contains(hmSelected)) return;
      // switchLevel() rewrites dataset.gloss per level, so refresh the text as
      // well as the position — otherwise the panel shows the old level's
      // translation next to the new level's Chinese.
      const gloss = bubbleGlossOf(hmSelected);
      const p = document.getElementById('bubblePanel');
      if (!gloss) { if (p) p.classList.remove('open'); return; }
      document.getElementById('bubblePanelGloss').textContent = gloss;
      renderBubbleInterlinear(hmSelected);
      if (p && !p.classList.contains('open')) p.classList.add('open');
      if (p) applyGlossVeil(p.dataset.key);
      positionBubblePanel(hmSelected);
      positionBubbleSpeak(hmSelected);
    }
    window.addEventListener('resize', reanchorBubblePanel);

    var STAR_SVG = '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M12 2.5l3.09 6.26 6.91 1-5 4.87 1.18 6.87L12 18.27l-6.18 3.23L7 14.63l-5-4.87 6.91-1L12 2.5z" stroke-linejoin="round" stroke-linecap="round"/></svg>';
    var FLASHCARDS_KEY = 'flashcards';

    function loadFlashcards() {
      try {
        const raw = localStorage.getItem(FLASHCARDS_KEY);
        if (!raw) return {};
        const parsed = JSON.parse(raw);
        return parsed && typeof parsed === 'object' ? parsed : {};
      } catch (e) { return {}; }
    }

    function escapeHtml(str) {
      return String(str)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }

    function showDefinition(span) {
      dismissPopover();
      const def = span.dataset.definition;
      if (!def) return;

      // Walk up to the speech bubble this word sits in and read its gloss.
      const bubble = span.closest('[data-gloss]');
      const bubbleGloss = bubble ? bubble.dataset.gloss : '';

      const popover = document.createElement('div');
      popover.className = 'definition-popover';

      // For swap-rendered name spans (.is-name has both .zh-text and .en-text
      // children), textContent concatenates both regardless of CSS visibility.
      // Always use the Chinese hanzi for the popover header so the user sees
      // the canonical underlying word, not "成振宇Sung Jin Woo" mashed together.
      const zhInner = span.classList.contains('is-name') ? span.querySelector('.zh-text') : null;
      // Two distinct things, deliberately not collapsed:
      //   displayWord — what is on screen, so the popover header matches the
      //                 word the reader actually tapped (Traditional in
      //                 Traditional mode).
      //   word        — canonical Simplified, from the baked dataset. This is
      //                 the flashcard STORAGE key. Keying on the displayed form
      //                 instead would file 發 and 发 as two separate cards for
      //                 the same word, and the ⭐ "already saved" check would
      //                 miss whenever the reader flipped scripts.
      const displayWord = zhInner ? zhInner.textContent : span.textContent;
      const word = span.dataset.simplified || displayWord;
      const tradWord = span.dataset.trad || word;
      const pinyinText = span.dataset.pinyin || '';
      const level = span.dataset.hskLevel || '';
      // Word audio, best available source:
      // 1. Citation clip — the word synthesized ALONE. Short words inside a
      //    sentence are spoken 2-2.6× faster with coarticulated edges
      //    (measured: 我 147ms in-sentence vs 387ms isolated); no slice of
      //    the bubble recording can sound like a dictionary pronunciation.
      //    Keyed by Simplified text, so it works at any display level.
      // 2. In-context slice — only for words with no citation clip
      //    (polyphones), where playing the occurrence's actual reading is
      //    worth the rougher edges. End clamped to the next word's onset so
      //    the neighbor never bleeds in.
      // Citation clips live at a root path the site serves; a double-clicked
      // file:// copy can't reach it (and fetch is blocked there anyway), so
      // those readers go straight to the slice fallback.
      const cite = (location.protocol !== 'file:' && hmAudio.manifest && hmAudio.manifest.words)
        ? hmAudio.manifest.words[word] || null
        : null;
      const audioBubbleEl = span.closest('.bubble-text');
      const audioEntry = (!cite && audioBubbleEl) ? bubbleAudioOf(audioBubbleEl) : null;
      let wordSpan = (audioEntry && audioEntry.ph && span.dataset.phIdx != null)
        ? audioEntry.ph[+span.dataset.phIdx] || null
        : null;
      if (wordSpan) {
        let nextS = null;
        for (let j = +span.dataset.phIdx + 1; j < audioEntry.ph.length; j++) {
          if (audioEntry.ph[j]) { nextS = audioEntry.ph[j][0]; break; }
        }
        wordSpan = [wordSpan[0],
                    nextS != null ? Math.min(wordSpan[1] + 0.04, nextS - 0.005)
                                  : wordSpan[1] + 0.04];
      }
      const playWord = function (btn) {
        if (cite) playWordSlice(cite, cite.s, cite.e, { btn: btn, tail: 0 });
        else if (wordSpan) playWordSlice(audioEntry, wordSpan[0], wordSpan[1], { btn: btn, tail: 0 });
      };
      // Phase 3.2: cedict is set ONLY when CC-CEDICT had an entry. Story-only
      // names (made-up character names, place names) lack an entry → ⭐ disabled.
      // We save the CC-CEDICT meaning specifically — not the popup's possibly-
      // overridden chapter-context version — so flashcards generalize across chapters.
      const cedictMeaning = span.dataset.cedict || '';
      const canSave = cedictMeaning.length > 0;
      const cards = loadFlashcards();
      const isSaved = !!cards[word];
      const saveTitle = canSave
        ? (isSaved ? 'Remove from flashcards' : 'Save to flashcards')
        : "No dictionary entry. Can't add as a flashcard.";

      popover.innerHTML =
        '<div class="def-header">' +
          '<div class="def-header-text">' +
            '<span class="def-word">' + displayWord + '</span>' +
            '<span class="def-pinyin">' + pinyinText + '</span>' +
            (level ? '<span class="def-level">HSK ' + (level === '7' ? '7-9' : level) + '</span>' : '') +
          '</div>' +
          ((cite || wordSpan)
            ? '<button class="hm-speak" id="popover-speak-btn" title="Play this word"' +
              ' aria-label="Play this word">' + SPEAKER_SVG + '</button>'
            : '') +
          '<button class="def-save' + (isSaved ? ' saved' : '') + '" id="popover-save-btn"' +
            (canSave ? '' : ' disabled') +
            ' title="' + saveTitle + '" aria-label="' + saveTitle + '">' +
            STAR_SVG +
          '</button>' +
        '</div>' +
        '<div class="def-meaning">' + def + '</div>' +
        // The whole-bubble translation used to live here, nested inside the
        // WORD popup — so learning what a sentence meant required first picking
        // a word, an unrelated action. It now has its own panel under the
        // bubble (showBubblePanel), which is the larger unit getting the larger
        // surface rather than being a footnote inside the smaller one.
        (!canSave ? '<div class="def-save-note">No dictionary entry. Story-specific term.</div>' : '');

      document.body.appendChild(popover);
      positionPopover(popover, span);
      activePopover = popover;

      // Identical sound whichever script is on screen — the recording is of
      // the Simplified text either way.
      const speakBtn = popover.querySelector('#popover-speak-btn');
      if (speakBtn) {
        speakBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          playWord(speakBtn);
          hmBeacon('audio_word', { w: word.slice(0, 16), cite: cite ? 1 : 0 });
        });
      }

      // ⭐ click → toggle saved state in localStorage. Always saves with the
      // CC-CEDICT meaning (cedictMeaning), not the popup's displayed
      // chapter-context override, so flashcards stay context-free.
      const saveBtn = popover.querySelector('#popover-save-btn');
      if (saveBtn && canSave) {
        saveBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          const cards = loadFlashcards();
          if (cards[word]) {
            delete cards[word];
            saveBtn.classList.remove('saved');
            saveBtn.title = 'Save to flashcards';
            saveBtn.setAttribute('aria-label', 'Save to flashcards');
          } else {
            cards[word] = {
              hanzi: word,
              // Traditional form rides along on the card. The flashcards page is
              // a static page with no per-chapter data and no OpenCC bundle, so
              // carrying the converted form here is what lets it offer the same
              // script toggle. Omitted when identical to hanzi (most words).
              trad: tradWord !== word ? tradWord : undefined,
              pinyin: pinyinText,
              meaning: cedictMeaning,
              hsk: parseInt(level, 10) || 0,
              savedAt: Date.now(),
            };
            saveBtn.classList.add('saved');
            saveBtn.title = 'Remove from flashcards';
            saveBtn.setAttribute('aria-label', 'Remove from flashcards');
          }
          try { localStorage.setItem(FLASHCARDS_KEY, JSON.stringify(cards)); } catch (e) {}
          hmBeacon('flashcard_save', { w: word, on: !!cards[word] });
        });
      }
    }

    // Stage one, in the CAPTURE phase so it runs before the word handler below
    // and can stop the event reaching it: the first tap on a bubble only
    // selects it (pinyin panel, speak button); words open on the next tap.
    // The English gloss stays veiled until asked for (applyGlossVeil), so the
    // first tap never spoils the sentence. Only intercepts when the bubble is
    // not yet selected and there is actually something to show.
    document.addEventListener('click', (e) => {
      if (!e.target.closest) return;
      const bubble = e.target.closest('.bubble-text');
      if (!bubble) return;
      if (!document.body.classList.contains('two-stage')) return;  // translations off
      if (bubble.classList.contains('is-active')) return;   // already selected: let it through
      if (!bubbleGlossOf(bubble)) return;                   // nothing to show; behave as before
      e.stopPropagation();
      e.preventDefault();
      dismissPopover();
      selectBubble(bubble);
      hmBeacon('bubble_select', { p: +bubble.dataset.pageIdx, b: +bubble.dataset.blockIdx });
    }, true);

    document.addEventListener('click', (e) => {
      const phrase = e.target.closest('.phrase:not(.punct)');
      if (phrase && phrase.dataset.definition) {
        e.stopPropagation();
        showDefinition(phrase);
        // Report the Simplified form regardless of display script, so word-click
        // analytics don't split one word across two spellings. (textContent is
        // also wrong for .is-name spans — it concatenates both child spans.)
        const beaconWord = phrase.dataset.simplified || (phrase.textContent || '');
        hmBeacon('word_click', { w: beaconWord.slice(0, 16), hsk: phrase.dataset.hskLevel || null });
      } else {
        // Two-level teardown, mirroring the two-level selection. If a word
        // popover is open, tapping away closes just that — the bubble stays
        // selected, because you are still working on that sentence. Only a
        // second tap, with no popover left to close, drops the selection.
        const hadPopover = !!activePopover;
        dismissPopover();
        if (hadPopover) return;
        if (!e.target.closest('.bubble-panel, .controls, .settings-popup, .tooltip-hint, .reader-toast')) {
          deselectBubble();
        }
      }
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      if (e.key === 'o' || e.key === 'O') toggleMode();
      // Escape unwinds the same two levels, in the same order.
      if (e.key === 'Escape') {
        if (activePopover) dismissPopover(); else deselectBubble();
      }
    });

    init();

    // Reflect a persisted Traditional preference. init() already painted using
    // currentScript, so only the font-stack class and the checkbox need
    // syncing — no second render pass.
    document.body.classList.toggle('script-trad', currentScript === 'trad');
    const scriptBox = document.getElementById('scriptCheckbox');
    if (scriptBox) scriptBox.checked = currentScript === 'trad';

    // Sync prefs from storage into the checkboxes AND the body classes. Without
    // this the JS defaults silently win until the reader touches a toggle.
    loadReaderPrefs();
    applyReaderPrefs();

    hmBeacon('reader_open', Object.assign({
      level: readerData.targetLevel || currentLevel,
      pages: readerData.pages.length,
      script: currentScript,
    }, hmDevice()));

    // Scale fonts after images load
    document.querySelectorAll('.page img').forEach(img => {
      if (img.complete && img.naturalWidth) scaleFonts();
      else img.addEventListener('load', scaleFonts);
    });
    // Re-scale on window resize
    window.addEventListener('resize', scaleFonts);

    // ============ First-visit tooltip hint ============
    const TOOLTIP_HINT_KEY = 'seenTooltipHint';
    const tooltipHint = document.getElementById('tooltipHint');

    function dismissTooltipHint() {
      tooltipHint.classList.remove('show');
      localStorage.setItem(TOOLTIP_HINT_KEY, '1');
    }

    // Written from the device's real pointer type. "Hover" is meaningless on a
    // phone, and it was the first thing every mobile reader was told.
    function buildTooltipHint() {
      const body = tooltipHint.querySelector('.tooltip-hint-body');
      if (!body) return;
      // Three short lines, once per browser, auto-dismissing. Deliberately
      // describes both taps as one idea — "tap the text" covers selecting a
      // bubble AND opening a word, so the two-stage rule never has to be
      // spelled out. A seven-step guided tour was tried and removed: it stood
      // between the reader and the comic they came for, and everything it
      // taught except that one rule was already conventional.
      const touch = window.matchMedia('(pointer: coarse)').matches;
      const verb = touch ? 'Tap' : 'Click';
      const rows = [[touch ? '📖' : '💡', verb + ' the text for pinyin, meaning and audio']];
      rows.push(['⚙', 'Turn any of it off in settings']);
      rows.push(['🟠', 'Orange words are above your HSK level']);
      body.innerHTML = rows.map(r =>
        '<div class="tooltip-hint-row"><span class="tooltip-hint-icon">' + r[0] +
        '</span><span>' + r[1] + '</span></div>').join('');
    }

    function showTooltipHint() {
      if (localStorage.getItem(TOOLTIP_HINT_KEY)) return;
      buildTooltipHint();
      setTimeout(() => tooltipHint.classList.add('show'), 800);
      setTimeout(() => dismissTooltipHint(), 10000);
    }

    showTooltipHint();
  </script>
</body>
</html>`;

  return html;
}
