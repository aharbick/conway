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
global.SpreadsheetApp = { flush: () => lockEvents.push('flush') };
global.ContentService = {
  MimeType: { JSON: 'json' },
  createTextOutput: (t) => ({ setMimeType: () => t }),
};
// Record the order of flush vs releaseLock: releasing a lock with writes still buffered
// lets the next execution read a stale cell and overwrite it.
const lockEvents = [];
global.LockService = {
  getScriptLock: () => ({ tryLock: () => true, releaseLock: () => lockEvents.push('release') }),
};

/** A sheet addressed by (row, col); column A holds the header plus the chunks. */
function FakeSheet() {
  this.cells = new Map();
  const key = (r, c) => `${r},${c}`;
  this.get = (r, c) => this.cells.get(key(r, c)) ?? '';
  this.getRange = (row, col, numRows = 1) => ({
    getValue: () => this.cells.get(key(row, col)) ?? '',
    setValue: (v) => this.cells.set(key(row, col), v),
    getValues: () => Array.from({ length: numRows }, (_, i) => [this.cells.get(key(row + i, col)) ?? '']),
    setValues: (vals) => vals.forEach((r, i) => this.cells.set(key(row + i, col), r[0])),
  });
}

// ------------------------------------------------------- load the real script --
// Loaded through new Function so it gets its own scope: with a plain eval the script would
// see this file's locals, and a variable here could stand in for one the script failed to
// declare - exactly the bug that shipped in e05ccb6. Only the stubbed globals resolve.
const apiPath = path.join(__dirname, '..', 'progress-api.js');
const A = new Function(
  fs.readFileSync(apiPath, 'utf8') +
    '\nreturn {stripChunkLength, writeStripBitmapBytes, readStripBitmapBytes, countStripBits,' +
    ' setStripIntervalComplete, ensureStripBitmapSheet, readStripCompletedCount, handleRequest,' +
    ' countBitsInBase64,' +
    ' writeStripCompletedCount, STRIP_BITMAP_BYTES, STRIP_CHUNK_ROWS, STRIP_COUNT_COL,' +
    ' STRIP_CHUNK_BYTES, STRIP_CHUNK_PREFIX, STRIP_BITMAP_SHEET_NAME, STRIP_MIDDLE_IDX_COUNT,' +
    ' STRIP_TOTAL_CENTERS, getOrCreateSheet,' +
    ' STRIP_BESTS_SHEET_NAME, STRIP_BESTS_LEGACY_NAMES, STRIP_BESTS_HEADERS};'
)();

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

// ------------------------------------------------------ completion counter ----
const counted = new FakeSheet();
A.writeStripBitmapBytes(counted, new Array(A.STRIP_BITMAP_BYTES).fill(0));
A.writeStripCompletedCount(counted, 0);

A.setStripIntervalComplete(counted, 100, 5);
A.setStripIntervalComplete(counted, 100, 6);
check('the counter tracks new completions', A.readStripCompletedCount(counted) === 2,
      `C1=${A.readStripCompletedCount(counted)}`);

A.setStripIntervalComplete(counted, 100, 5);
A.setStripIntervalComplete(counted, 100, 6);
check('re-running an interval does not inflate the counter', A.readStripCompletedCount(counted) === 2,
      `C1=${A.readStripCompletedCount(counted)}`);

A.setStripIntervalComplete(counted, 9999, 0); // out of range
check('an ignored interval does not touch the counter', A.readStripCompletedCount(counted) === 2);

check('the counter agrees with a popcount of the bitmap',
      A.readStripCompletedCount(counted) === A.countStripBits(A.readStripBitmapBytes(counted)));

check('the counter is labelled in B1', counted.get(1, 2) === 'completedIntervals');
check('the counter does not overwrite the chunk header', counted.get(1, 1) === '');

// A restore wipes C1; the next completion must rebuild it from the bitmap rather than
// starting over at 1.
const wiped = new FakeSheet();
A.writeStripBitmapBytes(wiped, migrated.slice());
check('a wiped counter reads as missing', A.readStripCompletedCount(wiped) === null);
A.setStripIntervalComplete(wiped, 839, 300);
check('a wiped counter is rebuilt from the bitmap, not reset to 1',
      A.readStripCompletedCount(wiped) === totalBits + 1,
      `C1=${A.readStripCompletedCount(wiped)}, expected ${totalBits + 1}`);

// ---------------------------------------------------------- request path ------
// Call the handlers the way a deployed web app does. A unit test of the bitmap helpers
// cannot catch a bad reference inside a request handler; this can.
global.AUTHORIZED_API_KEY = 'test-key';
const served = new FakeSheet();
A.writeStripBitmapBytes(served, migrated.slice());
A.writeStripCompletedCount(served, totalBits);

const flushStub = () => lockEvents.push('flush');
const withSheet = {
  flush: flushStub,
  openById: () => ({ getSheetByName: (n) => (n === A.STRIP_BITMAP_SHEET_NAME ? served : null) }),
};
const withoutSheet = { flush: flushStub, openById: () => ({ getSheetByName: () => null }) };
const request = (params, spreadsheetApp = withSheet) => {
  global.SpreadsheetApp = spreadsheetApp;
  return JSON.parse(A.handleRequest({ parameter: { apiKey: 'test-key', ...params } }));
};

const cacheResp = request({ action: 'getCompleteStripCache' });
check('getCompleteStripCache succeeds', cacheResp.success === true,
      cacheResp.error ? `error=${cacheResp.error}` : '');
check('it returns the bitmap intact',
      cacheResp.success && Buffer.from(cacheResp.bitmap, 'base64').equals(bitmap));
check('it reports the cached completion count', cacheResp.completedIntervals === totalBits,
      `${cacheResp.completedIntervals}`);

// The server counts bits straight from base64; this must agree with decoding the bytes.
check('base64 popcount matches a byte-level popcount',
      A.countBitsInBase64(bitmap.toString('base64')) === totalBits,
      `${A.countBitsInBase64(bitmap.toString('base64'))} vs ${totalBits}`);
check('base64 popcount handles an empty string', A.countBitsInBase64('') === 0);
check('base64 popcount ignores padding',
      A.countBitsInBase64(Buffer.from([0xff]).toString('base64')) === 8,
      Buffer.from([0xff]).toString('base64'));

// A drifted C1 must be corrected by the next read rather than persisting
served.getRange(1, A.STRIP_COUNT_COL).setValue(totalBits - 5);
const healResp = request({ action: 'getCompleteStripCache' });
check('a drifted C1 is reported as it was found', healResp.cachedCount === totalBits - 5,
      `${healResp.cachedCount}`);
check('the response count is derived, not the drifted cache', healResp.completedIntervals === totalBits);
check('and C1 is corrected in place', A.readStripCompletedCount(served) === totalBits,
      `C1=${A.readStripCompletedCount(served)}`);

const incResp = request({ action: 'incrementStripCompletion', centerIdx: '839', middleIdx: '400' });
check('incrementStripCompletion succeeds', incResp.success === true,
      incResp.error ? `error=${incResp.error}` : '');
check('it sets the bit and bumps the count',
      isSet(A.readStripBitmapBytes(served), 839, 400) &&
        A.readStripCompletedCount(served) === totalBits + 1);

const badIdx = request({ action: 'incrementStripCompletion', centerIdx: '9999', middleIdx: '0' });
check('an out-of-range completion is rejected', badIdx.success === false);

const missing = request({ action: 'getCompleteStripCache' }, withoutSheet);
check('a missing bitmap sheet becomes a JSON error, not an empty bitmap',
      missing.success === false && /missing/i.test(missing.error || ''),
      missing.success ? 'returned success!' : '');
check('and no bitmap is handed back in that case', missing.bitmap === undefined);

// ------------------------------------------- a renamed sheet must not fork -----
// Renaming the constant from 'Strip Progress' to 'Strip Bests' made the writer create an
// empty tab beside the full one, and every row since went to the new one while the
// original froze. A legacy tab has to be adopted, not left behind.
function FakeBook(names) {
  this.sheets = {};
  this.inserted = [];
  names.forEach((n) => { this.sheets[n] = { name: n, setName: (v) => { this.sheets[n].name = v; } }; });
  this.getSheetByName = (n) => {
    const hit = Object.keys(this.sheets).find((k) => this.sheets[k].name === n);
    return hit ? this.sheets[hit] : null;
  };
  this.insertSheet = (n) => {
    this.inserted.push(n);
    const made = { name: n, setName: () => {}, getRange: () => ({ setValues: () => {} }) };
    this.sheets[n] = made;
    return made;
  };
}

const bookLegacy = new FakeBook(['Strip Progress']);
const adopted = A.getOrCreateSheet(bookLegacy, A.STRIP_BESTS_SHEET_NAME,
                                   A.STRIP_BESTS_LEGACY_NAMES, A.STRIP_BESTS_HEADERS);
check('a sheet under the old name is adopted, not forked',
      adopted.name === A.STRIP_BESTS_SHEET_NAME && bookLegacy.inserted.length === 0,
      bookLegacy.inserted.length ? `created ${bookLegacy.inserted.join(',')}` : '');

const bookBoth = new FakeBook(['Strip Bests', 'Strip Progress']);
check('the canonical sheet wins when both exist',
      A.getOrCreateSheet(bookBoth, A.STRIP_BESTS_SHEET_NAME, A.STRIP_BESTS_LEGACY_NAMES,
                         A.STRIP_BESTS_HEADERS).name === 'Strip Bests' &&
        bookBoth.inserted.length === 0);

const bookNone = new FakeBook([]);
A.getOrCreateSheet(bookNone, A.STRIP_BESTS_SHEET_NAME, A.STRIP_BESTS_LEGACY_NAMES,
                   A.STRIP_BESTS_HEADERS);
check('with neither name present a fresh sheet is still created',
      bookNone.inserted.join(',') === A.STRIP_BESTS_SHEET_NAME);

// ----------------------------------------------------------- sendMail ---------
// The endpoint can send mail from the owner's Google account, so the recipient guard is
// the part worth testing: the API key rides in a query string and is not secret enough to
// back an open relay.
const sent = [];
global.Session = { getEffectiveUser: () => ({ getEmail: () => 'owner@example.com' }) };
global.MailApp = {
  getRemainingDailyQuota: () => mailQuota,
  sendEmail: (opts) => sent.push(opts),
};
let mailQuota = 100;

const ok = request({ action: 'sendMail', subject: 'hello', body: 'world' });
check('sendMail succeeds', ok.success === true, ok.error || '');
check('it defaults to the script owner', sent.length === 1 && sent[0].to === 'owner@example.com',
      sent.length ? sent[0].to : '(nothing sent)');
check('subject and body are passed through',
      sent[0].subject === 'hello' && sent[0].body === 'world');

const stranger = request({ action: 'sendMail', subject: 's', body: 'b', to: 'someone@else.com' });
check('an unlisted recipient is refused',
      stranger.success === false && /not allowed/i.test(stranger.error || ''),
      stranger.success ? 'it sent!' : '');
check('and nothing was sent in that case', sent.length === 1);

check('the owner may be named explicitly',
      request({ action: 'sendMail', subject: 's', body: 'b', to: 'OWNER@example.com' }).success === true,
      'case-insensitive match');

const empty = request({ action: 'sendMail' });
check('an empty message is refused', empty.success === false);

mailQuota = 0;
const over = request({ action: 'sendMail', subject: 's', body: 'b' });
check('an exhausted quota is reported, not swallowed',
      over.success === false && /quota/i.test(over.error || ''));
mailQuota = 100;

check('a bad API key never reaches MailApp',
      JSON.parse(A.handleRequest({ parameter: { apiKey: 'wrong', action: 'sendMail',
                                                subject: 's', body: 'b' } })).success === false &&
        sent.length === 2);

// ------------------------------------------------- flush before unlocking -----
lockEvents.length = 0;
request({ action: 'incrementStripCompletion', centerIdx: '500', middleIdx: '7' });
check('a locked write flushes before releasing the lock',
      lockEvents.join(',') === 'flush,release', lockEvents.join(',') || '(nothing recorded)');

console.log(failures ? `\n${failures} FAILURE(S)` : '\nall checks passed');
process.exit(failures ? 1 : 0);
