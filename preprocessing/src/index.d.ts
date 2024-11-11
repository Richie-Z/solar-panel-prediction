declare module 'pdf2doi' {

  export function fromFile(url: string): Promise<{ doi: string }>;
}
