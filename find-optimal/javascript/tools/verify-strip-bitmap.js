/**
 * Verify that getCompleteStripCache is serving the right completion data.
 *
 *   source .envrc
 *   node javascript/tools/verify-strip-bitmap.js
 *   node javascript/tools/verify-strip-bitmap.js --csv data/strip-completion-bitmap-20260918.csv
 *
 * With --csv, the live bitmap is compared bit for bit against a CSV export of the legacy
 * "Strip Completion" sheet (one 64-bit decimal value per row). The check that matters is
 * that no interval marked complete in the backup is missing from the API: that would mean
 * lost progress, and the search would redo the work. Intervals present only in the API are
 * expected - those finished after the export.
 *
 * Exits non-zero if anything is wrong, so it is safe to use in a script.
 */
const fs = require('fs');

const TOTAL_CENTERS = 8548;
const MIDDLE_IDX_COUNT = 512;
const TOTAL_INTERVALS = TOTAL_CENTERS * MIDDLE_IDX_COUNT; // 4,376,576
const BITMAP_BYTES = Math.ceil(TOTAL_INTERVALS / 8); // 547,072

const args = process.argv.slice(2);
const csvPath = args.includes('--csv') ? args[args.indexOf('--csv') + 1] : null;

const label = (i) => `${Math.floor(i / MIDDLE_IDX_COUNT)}:${i % MIDDLE_IDX_COUNT}`;
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
const isSet = (buf, i) => ((buf[i >> 3] >> (i & 7)) & 1) === 1;

let failures = 0;
function check(name, cond, extra = '') {
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}${extra ? '  ' + extra : ''}`);
  if (!cond) failures++;
}

function legacyCsvToBitmap(path) {
  const lines = fs.readFileSync(path, 'utf8').split(/\r?\n/);
  lines.shift(); // header
  while (lines.length && lines[lines.length - 1] === '') lines.pop();

  const bitmap = Buffer.alloc(BITMAP_BYTES);
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
  return { bitmap, rows: lines.length };
}

(async () => {
  const url = process.env.GOOGLE_WEBAPP_URL;
  const key = process.env.GOOGLE_API_KEY;
  if (!url || !key) {
    console.error('GOOGLE_WEBAPP_URL and GOOGLE_API_KEY must be set (source .envrc)');
    process.exit(2);
  }

  const started = Date.now();
  const res = await fetch(`${url}?action=getCompleteStripCache&apiKey=${encodeURIComponent(key)}`);
  const body = await res.json();
  const elapsed = ((Date.now() - started) / 1000).toFixed(1);

  check('API responded 200', res.status === 200, `http=${res.status} in ${elapsed}s`);
  check('response reports success', body.success === true, body.error ? `error=${body.error}` : '');
  if (!body.success) process.exit(1);

  const live = Buffer.from(body.bitmap, 'base64');
  check('bitmap is the expected size', live.length === BITMAP_BYTES, `${live.length} bytes`);
  check('totalIntervals matches the search space', body.totalIntervals === TOTAL_INTERVALS,
        `${body.totalIntervals}`);
  check('bitmapSize field agrees with the payload', body.bitmapSize === live.length,
        `${body.bitmapSize}`);

  // A bitmap of all zeros is what a silently-reinitialized sheet looks like
  const liveBits = popcount(live);
  check('bitmap is not empty', liveBits > 0, `${liveBits} intervals complete`);

  let first = -1;
  for (let i = 0; i < TOTAL_INTERVALS; i++) {
    if (!isSet(live, i)) {
      first = i;
      break;
    }
  }
  let highest = -1;
  for (let i = TOTAL_INTERVALS - 1; i >= 0; i--) {
    if (isSet(live, i)) {
      highest = i;
      break;
    }
  }
  // Which count the response carries depends on the deployed version, so branch on the
  // fields present rather than assuming.
  if (body.cachedCount !== undefined) {
    // Current version: completedIntervals is derived from the base64 on the server, and the
    // read corrects C1. Counting the decoded bytes here checks the server's arithmetic by a
    // different route, so the two should agree exactly.
    check("server's base64 popcount matches an independent byte-level popcount",
          body.completedIntervals === liveBits,
          body.completedIntervals === liveBits ? `${liveBits}` : `server=${body.completedIntervals} vs ${liveBits}`);
    if (body.cachedCount !== liveBits) {
      console.log(`\nNOTE: C1 held ${body.cachedCount === null ? '(unset)' : body.cachedCount} vs ` +
                  `${liveBits} in the bitmap - this read corrected it`);
    }
  } else if (body.completedIntervals !== undefined) {
    // Older version: completedIntervals *is* the cached C1 value, with nothing correcting
    // it, so a mismatch is drift to report rather than a failure.
    if (body.completedIntervals !== liveBits) {
      console.log(`\nNOTE: cached count in C1 is ${body.completedIntervals} vs ${liveBits} in the ` +
                  `bitmap. The deployed script does not self-correct it - deploy the current ` +
                  `progress-api.js, or run recountStripCompletions().`);
    } else {
      console.log(`\ncached count in C1 agrees with the bitmap: ${liveBits}`);
    }
  } else {
    console.log(`\nNOTE: the deployed script returns no completion count`);
  }

  console.log(`\nlive: ${liveBits} intervals complete, first incomplete ${label(first)}, highest complete ${label(highest)}`);
  console.log(`      (the search should resume at ${label(first)})\n`);

  if (!csvPath) {
    console.log('no --csv given, so nothing to compare against');
  } else {
    const { bitmap: backup, rows } = legacyCsvToBitmap(csvPath);
    const backupBits = popcount(backup);
    console.log(`backup ${csvPath}: ${rows} rows, ${backupBits} intervals complete\n`);

    const missing = [];
    const extra = [];
    for (let i = 0; i < TOTAL_INTERVALS; i++) {
      const inLive = isSet(live, i);
      const inBackup = isSet(backup, i);
      if (inBackup && !inLive && missing.length < 50) missing.push(i);
      else if (inBackup && !inLive) missing.push(i);
      if (inLive && !inBackup) extra.push(i);
    }

    check('every interval in the backup is still complete in the API', missing.length === 0,
          missing.length ? `${missing.length} MISSING, e.g. ${missing.slice(0, 10).map(label).join(', ')}` : '');
    console.log(`      intervals complete in the API but not in the backup: ${extra.length}` +
                (extra.length && extra.length <= 10 ? ` (${extra.map(label).join(', ')})` : '') +
                ' - expected, these finished after the export');

    if (extra.length) {
      const contiguous = extra.every((v, idx) => idx === 0 || v === extra[idx - 1] + 1);
      check('those extra intervals are a contiguous run from the old frontier', contiguous,
            `${label(extra[0])} .. ${label(extra[extra.length - 1])}`);
    }
  }

  console.log(failures ? `\n${failures} FAILURE(S)` : '\nall checks passed');
  process.exit(failures ? 1 : 0);
})();
