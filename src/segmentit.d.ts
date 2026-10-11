declare module 'segmentit' {
  interface SegmentResult {
    w: string;
    p: number;
  }

  interface SegmentOptions {
    simple?: boolean;
  }

  class Segment {
    constructor();
    useDefault(): this;
    doSegment(text: string, options?: SegmentOptions): string[] | SegmentResult[];
  }

  export = Segment;
}
