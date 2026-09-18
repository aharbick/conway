/**
 * Rebuild the "Strip Completion B64" sheet contents from a backup.
 *
 * The Apps Script deliberately refuses to create that sheet if it goes missing - an empty
 * bitmap reads as zero completed intervals and would restart the whole search - so this is
 * the restore path.
 *
 *   node javascript/tools/bitmap-to-chunk-csv.js data/strip-completion-bitmap-20260918.csv out.csv
 *   node javascript/tools/bitmap-to-chunk-csv.js cache-response.json out.csv
 *
 * Input is either a CSV export of the old "Strip Completion" sheet (one 64-bit decimal
 * value per row) or a saved getCompleteStripCache response (JSON with a base64 "bitmap").
 * Output is a CSV to import into a sheet named exactly "Strip Completion B64", with
 * "Convert text to numbers, dates, and formulas" turned OFF.
 *
 * Every conversion is verified by rejoining the chunks and comparing to the source bitmap.
 */
const fs = require('fs');

const TOTAL_CENTERS = 8548;
const MIDDLE_IDX_COUNT = 512;
const TOTAL_INTERVALS = TOTAL_CENTERS * MIDDLE_IDX_COUNT; // 4,376,576
const BITMAP_BYTES = Math.ceil(TOTAL_INTERVALS / 8); // 547,072
const CHUNK_BYTES = 3072; // must match STRIP_CHUNK_BYTES in progress-api.js
const CHUNK_ROWS = Math.ceil(BITMAP_BYTES / CHUNK_BYTES); // 179
const PREFIX = 'b64:'; // must match STRIP_CHUNK_PREFIX

const [inPath, outPath] = process.argv.slice(2);
if (!inPath || !outPath) {
  console.error('usage: node javascript/tools/bitmap-to-chunk-csv.js <backup.csv|cache.json> <out.csv>');
  process.exit(2);
}

const popcount = (buf) => {
  let n = 0;
  for (const b of buf) {
    let v = b;
    while (v) {
      n += v & 1;
      v >>= 1;
    }
  }
  return n;
};

let bitmap;
if (inPath.endsWith('.json')) {
  const body = JSON.parse(fs.readFileSync(inPath, 'utf8'));
  if (!body.bitmap) throw new Error('no "bitmap" field in that JSON');
  bitmap = Buffer.from(body.bitmap, 'base64');
  console.log(`read base64 bitmap from ${inPath}`);
} else {
  const lines = fs.readFileSync(inPath, 'utf8').split(/\r?\n/);
  const header = lines.shift();
  while (lines.length && lines[lines.length - 1] === '') lines.pop();
  console.log(`read ${lines.length} legacy rows from ${inPath} (header ${JSON.stringify(header)})`);
  if (lines.length > Math.ceil(TOTAL_INTERVALS / 64)) throw new Error('more rows than the bitmap holds');

  bitmap = Buffer.alloc(BITMAP_BYTES);
  lines.forEach((line, rowIdx) => {
    const cell = line.trim().replace(/^"|"$/g, '');
    if (!cell || cell === '0') return;
    const value = BigInt(cell);
    for (let b = 0; b < 8; b++) {
      const byteIdx = rowIdx * 8 + b;
      if (byteIdx >= BITMAP_BYTES) break;
      bitmap[byteIdx] = Number((value >> BigInt(b * 8)) & 0xffn);
    }
  });
}

if (bitmap.length !== BITMAP_BYTES) {
  throw new Error(`bitmap is ${bitmap.length} bytes, expected ${BITMAP_BYTES}`);
}

let highest = -1;
for (let i = TOTAL_INTERVALS - 1; i >= 0; i--) {
  if ((bitmap[i >> 3] >> (i & 7)) & 1) {
    highest = i;
    break;
  }
}
console.log(`${popcount(bitmap)} intervals complete, highest ${Math.floor(highest / MIDDLE_IDX_COUNT)}:${highest % MIDDLE_IDX_COUNT}`);

const rows = ['stripBitmapBase64'];
for (let c = 0; c < CHUNK_ROWS; c++) {
  const start = c * CHUNK_BYTES;
  rows.push(PREFIX + bitmap.subarray(start, Math.min(start + CHUNK_BYTES, BITMAP_BYTES)).toString('base64'));
}

// The chunks must rejoin into the identical bitmap, and none may look like a formula
const joined = rows.slice(1).map((r) => r.substring(PREFIX.length)).join('');
if (!Buffer.from(joined, 'base64').equals(bitmap)) throw new Error('chunk round-trip FAILED');
if (rows.slice(1).some((r) => /^[=+\-@]/.test(r))) throw new Error('a chunk would be parsed as a formula');
console.log(`round-trip verified: ${CHUNK_ROWS} chunks rejoin to the identical ${BITMAP_BYTES}-byte bitmap`);

fs.writeFileSync(outPath, rows.join('\n') + '\n');
console.log(`wrote ${outPath} (header + ${CHUNK_ROWS} chunk rows)`);
