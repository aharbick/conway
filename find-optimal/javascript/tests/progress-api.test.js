/**
 * Tests for the strip completion bitmap logic in progress-api.js.
 *
 *   node javascript/tests/progress-api.test.js              # synthetic bitmap
 *   node javascript/tests/progress-api.test.js cache.json   # a real getCompleteStripCache response
 *
 * The Apps Script globals the bitmap code touches (Utilities, Sheet ranges) are stubbed
 * here, so this exercises the real functions without deploying. Worth running before any
 * change to the chunk layout: a mistake there corrupts the completion bitmap, which is
 * what stops the search from redoing finished work.
 */
const fs = require('fs');
const path = require('path');

// ---------------------------------------------------------- Apps Script stubs --
global.Utilities = {
  base64Encode: (bytes) => {
    for (const b of bytes) {
      if (b > 127 || b < -128) throw new Error(`byte out of signed range: ${b}`);
    }
    return Buffer.from(bytes.map((b) => b & 0xff)).toString('base64');
  },
  // Apps Script hands back SIGNED bytes; the code under test has to cope with that
  base64Decode: (s) => Array.from(Buffer.from(s, 'base64')).map((b) => (b > 127 ? b - 256 : b)),
};
global.SpreadsheetApp = {};
global.ContentService = {
  MimeType: { JSON: 'json' },
  createTextOutput: (t) => ({ setMimeType: () => t }),
};
global.LockService = { getScriptLock: () => ({ tryLock: () => true, releaseLock: () => {} }) };

/** A single-column sheet; row 1 is the header. */
function FakeSheet() {
  this.cells = [];
  this.getRange = (row, col, numRows = 1) => ({
    getValue: () => this.cells[row - 1] || '',
    setValue: (v) => {
      this.cells[row - 1] = v;
    },
    getValues: () => Array.from({ length: numRows }, (_, i) => [this.cells[row - 1 + i] || '']),
    setValues: (vals) => {
      vals.forEach((r, i) => {
        this.cells[row - 1 + i] = r[0];
      });
    },
  });
}

// ------------------------------------------------------- load the real script --
const apiPath = path.join(__dirname, '..', 'progress-api.js');
eval(
  fs.readFileSync(apiPath, 'utf8') +
    '\nglobal.__api = {stripChunkLength, writeStripBitmapBytes, readStripBitmapBytes, countStripBits,' +
    ' setStripIntervalComplete, ensureStripBitmapSheet, STRIP_BITMAP_BYTES, STRIP_CHUNK_ROWS,' +
    ' STRIP_CHUNK_BYTES, STRIP_CHUNK_PREFIX, STRIP_BITMAP_SHEET_NAME, STRIP_MIDDLE_IDX_COUNT,' +
    ' STRIP_TOTAL_CENTERS};'
);
const A = global.__api;

let failures = 0;
function check(name, cond, extra = '') {
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}${extra ? '  ' + extra : ''}`);
  if (!cond) failures++;
}

// --------------------------------------------------------------- the fixture --
// Either a real getCompleteStripCache response, or a bitmap shaped like one: every
// interval complete up to a frontier, with a scattering of gaps behind it.
const fixturePath = process.argv[2];
let bitmap;
if (fixturePath) {
  bitmap = Buffer.from(JSON.parse(fs.readFileSync(fixturePath, 'utf8')).bitmap, 'base64');
  console.log(`fixture: ${fixturePath}\n`);
} else {
  bitmap = Buffer.alloc(A.STRIP_BITMAP_BYTES);
  const frontier = 839 * A.STRIP_MIDDLE_IDX_COUNT + 130;
  for (let i = 0; i < frontier; i++) bitmap[i >> 3] |= 1 << (i & 7);
  for (const i of [61966, 62446, 67094, 122527, 122719, 155375, 298519, 337438, 337439, 337487]) {
    bitmap[i >> 3] &= ~(1 << (i & 7));
  }
  console.log('fixture: synthetic (pass a getCompleteStripCache response to use real data)\n');
}
const signed = Array.from(bitmap).map((b) => (b > 127 ? b - 256 : b));
const totalBits = A.countStripBits(signed);

const isSet = (bytes, c, m) => {
  const i = c * A.STRIP_MIDDLE_IDX_COUNT + m;
  return ((bytes[i >> 3] & 0xff) >> (i & 7)) & 1;
};

// ------------------------------------------------------------------- layout ---
check('fixture size matches STRIP_BITMAP_BYTES', bitmap.length === A.STRIP_BITMAP_BYTES,
      `${bitmap.length} vs ${A.STRIP_BITMAP_BYTES}`);
check('chunk size is a multiple of 3 (so chunk base64 concatenates)', A.STRIP_CHUNK_BYTES % 3 === 0);
check('chunk rows cover the bitmap with no spare row',
      A.STRIP_CHUNK_ROWS * A.STRIP_CHUNK_BYTES >= A.STRIP_BITMAP_BYTES &&
        (A.STRIP_CHUNK_ROWS - 1) * A.STRIP_CHUNK_BYTES < A.STRIP_BITMAP_BYTES,
      `${A.STRIP_CHUNK_ROWS} rows x ${A.STRIP_CHUNK_BYTES}B`);
check('last chunk length is the remainder',
      A.stripChunkLength(A.STRIP_CHUNK_ROWS - 1) ===
        A.STRIP_BITMAP_BYTES - (A.STRIP_CHUNK_ROWS - 1) * A.STRIP_CHUNK_BYTES,
      `${A.stripChunkLength(A.STRIP_CHUNK_ROWS - 1)}B`);

// --------------------------------------------- a missing sheet must throw -----
// Silently creating an empty bitmap would report zero completed intervals and restart
// the whole search, so an absent sheet has to be an error.
let threw = false;
try {
  A.ensureStripBitmapSheet({ getSheetByName: () => null });
} catch (err) {
  threw = /missing/i.test(err.message);
}
check('a missing bitmap sheet throws instead of being created empty', threw);
check('an existing bitmap sheet is returned as-is',
      A.ensureStripBitmapSheet({ getSheetByName: (n) => (n === A.STRIP_BITMAP_SHEET_NAME ? 'sheet' : null) }) === 'sheet');

const migrated = signed;

// ------------------------------------------------------- chunk round trip -----
const sheet = new FakeSheet();
A.writeStripBitmapBytes(sheet, migrated);
const back = A.readStripBitmapBytes(sheet);
check('chunk write/read round-trips',
      back.length === signed.length && back.every((b, i) => b === signed[i]));

const cells = sheet.getRange(2, 1, A.STRIP_CHUNK_ROWS).getValues();
const joined = cells.map((r) => r[0].substring(A.STRIP_CHUNK_PREFIX.length)).join('');
check('joined chunk base64 decodes to the exact bitmap',
      Buffer.from(joined, 'base64').equals(bitmap), `${joined.length} chars`);
check('no chunk cell would be parsed as a formula by Sheets',
      cells.every((r) => !/^[=+\-@]/.test(r[0])));

// ------------------------------------------------------------ setting bits ----
const fresh = new FakeSheet();
A.writeStripBitmapBytes(fresh, new Array(A.STRIP_BITMAP_BYTES).fill(0));
const targets = [
  [0, 0], [0, 1], [121, 14], [121, 494], [659, 191], [786, 487],
  [4096, 256], [8547, 510], [A.STRIP_TOTAL_CENTERS - 1, A.STRIP_MIDDLE_IDX_COUNT - 1],
];
for (const [c, m] of targets) A.setStripIntervalComplete(fresh, c, m);
let afterSet = A.readStripBitmapBytes(fresh);
check('every set interval reads back complete', targets.every(([c, m]) => isSet(afterSet, c, m)));
check('and nothing else got set', A.countStripBits(afterSet) === targets.length,
      `${A.countStripBits(afterSet)} bits`);

A.setStripIntervalComplete(fresh, 121, 14);
check('setting an already-complete interval is idempotent',
      A.countStripBits(A.readStripBitmapBytes(fresh)) === targets.length);

A.setStripIntervalComplete(fresh, A.STRIP_TOTAL_CENTERS, 0);
A.setStripIntervalComplete(fresh, 0, A.STRIP_MIDDLE_IDX_COUNT);
A.setStripIntervalComplete(fresh, -1, 0);
check('out-of-range indices are ignored',
      A.countStripBits(A.readStripBitmapBytes(fresh)) === targets.length);

// -------------------------------------------- one completion on live data -----
const live = new FakeSheet();
A.writeStripBitmapBytes(live, migrated.slice());
A.setStripIntervalComplete(live, 839, 200);
const after = A.readStripBitmapBytes(live);
check('one completion adds exactly one bit and leaves the rest alone',
      A.countStripBits(after) === totalBits + 1 && isSet(after, 839, 200) &&
        after.every((b, i) => b === migrated[i] || i === (839 * A.STRIP_MIDDLE_IDX_COUNT + 200) >> 3));

console.log(failures ? `\n${failures} FAILURE(S)` : '\nall checks passed');
process.exit(failures ? 1 : 0);
