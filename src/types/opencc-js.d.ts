declare module 'opencc-js' {
  interface ConverterOptions {
    from: 'tw' | 'hk' | 'cn' | 'twp' | 'jp' | 't' | 's';
    to: 'tw' | 'hk' | 'cn' | 'twp' | 'jp' | 't' | 's';
  }

  function Converter(options: ConverterOptions): (text: string) => string;

  export { Converter };
}
