export function slugify(text: string): string {
  return text
    .normalize('NFD')                     // 1) split accented chars
    .replace(/[\u0300-\u036f]/g, '')      // 2) remove the accents
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')          // 3) anything not a–z, 0–9 → “-”
    .replace(/^-+|-+$/g, '')              // 4) trim leading/trailing “-”
}
