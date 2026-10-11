declare module 'pdf-poppler' {
  export interface Options {
    format?: 'png' | 'jpeg' | 'tiff' | 'pdf' | 'ps' | 'eps' | 'svg';
    out_dir?: string;
    out_prefix?: string;
    page?: number | null;
    scale?: number;
    crop_x?: number;
    crop_y?: number;
    crop_width?: number;
    crop_height?: number;
  }

  export function convert(pdfPath: string, options: Options): Promise<void>;
  export function info(pdfPath: string): Promise<{
    pages: number;
    encrypted: boolean;
    page_size: string;
    file_size: string;
    optimized: boolean;
    pdf_version: string;
  }>;
}
