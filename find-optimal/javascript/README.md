# Google Apps Script progress tracking

The search records its progress in a Google Sheet through a small REST API implemented in
Apps Script. `find-optimal` uses it to know which work is already done, so the data here is
what stops the search from repeating finished intervals.

```
javascript/
  progress-api.js   Apps Script web app: the REST API the C++ code calls
  utils.js          Apps Script spreadsheet helper functions
  tools/            local Node scripts for verifying and restoring the progress data
  tests/            local Node tests for the Apps Script logic
```

`progress-api.js` and `utils.js` run *inside* Apps Script and are deployed by pasting them
into the editor. Everything under `tools/` and `tests/` runs locally with Node and is never
deployed.

## Installation

1. Create the Apps Script:
  - Go to https://script.google.com
  - Create a new project
  - Replace the default code with the script above
  - Save the project
2. Deploy as Web App:
  - Click "Deploy" → "New deployment"
  - Choose type: "Web app"
  - Set execute as: "Me"
  - Set access: "Anyone" (or "Anyone with Google account" for more security)
  - Click "Deploy" and copy the Web App URL

Note that saving the editor and *deploying* are separate steps. Functions you run from the
editor use the saved code immediately, but the web app keeps serving the deployed version
until you publish a new one ("Deploy" → "Manage deployments" → edit → new version).

## Calling the API (with curl)

Google doesn't work nicely with POST... So all APIs are GET.

Send Progress (GET):
curl -L "https://script.google.com/macros/s/AKfycbxU11EoXYKwgckXVfONFJdvM_QOVvTKwxzTzbwwwTUsqWMDpOx67yOfH5RsKhfTekpiow/exec?action=sendProgress&apiKey=SECRET_KEY_VALUE&frameComplete=false&frameIdx=12345&kernelIdx=2&bestGenerations=250&bestPattern=123456789ABCDEF0&bestPatternBin=0001001100110011001100110011001100110011001100110011001100110011"

Get Best Result (GET) - unchanged:
curl -L "https://script.google.com/macros/s/AKfycbxU11EoXYKwgckXVfONFJdvM_QOVvTKwxzTzbwwwTUsqWMDpOx67yOfH5RsKhfTekpiow/exec?action=getBestResult&apiKey=SECRET_KEY_VALUE"

## Strip completion bitmap

Strip search progress is one bit per `centerIdx:middleIdx` interval, 8548 × 512 =
4,376,576 bits = 547,072 bytes, indexed as `centerIdx * 512 + middleIdx`. It lives in the
**`Strip Completion B64`** sheet as 179 rows of base64, 3,072 bytes per row.

Two details make that layout work, and both will break the bitmap if changed carelessly:

* **3,072 is a multiple of 3.** Base64 encodes 3 bytes into 4 characters, so a chunk of that
  size encodes with no padding, which means concatenating the rows' base64 is itself valid
  base64 for the whole bitmap. `getCompleteStripCache` is therefore just a `getValues()` and
  a string join - no decoding, no BigInt. The previous layout (one 64-bit decimal per row,
  68,384 rows) took ~19s to read and held the script lock the whole time, which starved the
  completion and summary writes queued behind it.
* **Each cell is prefixed with `b64:`.** Sheets parses a leading `=` or `+` as a formula and
  base64 can start with `+`, which would silently corrupt a chunk.

`getCompleteStripCache` runs *outside* `withLock` on purpose. Completions only ever set
bits, each write touches a single chunk row, and a cell is read whole, so a concurrent write
can at worst leave a just-set bit out of the response - costing one interval that gets
searched and re-marked, not corruption.

Nothing creates the sheet automatically. If it is missing, the API errors and the search
stops with an explanation, because an empty bitmap looks like zero completed intervals and
would restart the entire search. To set up a genuinely new spreadsheet, run
`initializeStripBitmapSheet()` from the Apps Script editor; to recover a lost sheet, use
`tools/bitmap-to-chunk-csv.js` below.

### The completion count in B1/C1

Row 1 of columns B and C holds a cached count of completed intervals:

| | A | B | C |
|---|---|---|---|
| **1** | `stripBitmapBase64` | `completedIntervals` | `429736` |
| **2** | `b64:////…` | | |

`setStripIntervalComplete` bumps C1 only when a bit actually flips 0→1, so re-running an
interval cannot inflate it, and if C1 is missing or not a number - importing a CSV over the
sheet wipes it - the next completion rebuilds it from the bitmap rather than restarting the
count at 1.

C1 is a cache, not the source of truth. `getCompleteStripCache` derives the count from the
bitmap it is already returning and rewrites C1 when the two disagree, so a lost increment
corrects itself on the next read instead of persisting. The count is derived by summing bit
counts per base64 symbol rather than decoding 547KB: each symbol carries 6 bits and
encoders zero-fill the tail, so the sum is exact. The response carries both
`completedIntervals` (derived, always exact) and `cachedCount` (what C1 held beforehand),
which is how `tools/verify-strip-bitmap.js` reports drift. `recountStripCompletions()` is
still there for use right after a restore.

Note that `withLock` calls `SpreadsheetApp.flush()` before releasing the lock. Apps Script
buffers spreadsheet writes and does *not* flush them on release, so without it the next
execution can read a stale cell and overwrite it - for a bitmap chunk that means silently
dropping a completed interval. C1 was observed drifting 3 behind this way before the flush
was added.

Use it for dashboard formulas, e.g. in Strip Analysis:

```
=TEXT('Strip Completion B64'!C1/512,"#,###.##") & " centers ✅ (" & TEXT('Strip Completion B64'!C1/(8548*512),"0.00%") & ")"
```

Do **not** derive progress from `SUM('Strip Summary'!B:B)`. That sums the histogram's count
column, which is the number of summary submissions: intervals that find nothing never file
one (21 so far), and any interval processed twice files a second (114 duplicates in the logs
to date). The two errors pull in opposite directions and never reconcile. Note `/512` gives
center-equivalents, not completed centers - progress is not aligned to center boundaries.

## Tools

Both need the API credentials for the live checks:

```sh
source .envrc   # GOOGLE_WEBAPP_URL and GOOGLE_API_KEY
```

### tools/verify-strip-bitmap.js

Confirms the API is serving real completion data rather than an empty or partial bitmap.
Checks the payload size, `totalIntervals`, a non-zero completion count, and reports the
first incomplete interval - which is exactly where the search will resume, so it should
match the `Strip search: C:M to 8547:511` line the search prints at startup.

```sh
node javascript/tools/verify-strip-bitmap.js
node javascript/tools/verify-strip-bitmap.js --csv data/strip-completion-bitmap-20260918.csv
```

With `--csv` it also compares bit for bit against a CSV export of the legacy sheet and fails
if any interval marked complete in the backup is missing from the API - i.e. lost progress.
Intervals present only in the API are expected; those finished after the export. Exits
non-zero on failure, so it works as a pre-restart check.

### tools/bitmap-to-chunk-csv.js

Rebuilds the contents of `Strip Completion B64` from a backup - the restore path, since the
API refuses to recreate the sheet itself.

```sh
node javascript/tools/bitmap-to-chunk-csv.js data/strip-completion-bitmap-20260918.csv out.csv
node javascript/tools/bitmap-to-chunk-csv.js saved-cache-response.json out.csv
```

Input is either a CSV export of the old `Strip Completion` sheet (one 64-bit decimal per
row) or a saved `getCompleteStripCache` response. Each run verifies the conversion by
rejoining the chunks and comparing against the source bitmap.

To restore: create a sheet named exactly `Strip Completion B64`, then **File → Import →**
upload the generated CSV with *Import location* "Replace current sheet" and **"Convert text
to numbers, dates, and formulas" turned OFF**. The import wipes the count in C1, so run
`recountStripCompletions()` afterwards, then verify with `tools/verify-strip-bitmap.js`.

## Tests

### tests/progress-api.test.js

Runs the real functions out of `progress-api.js` against stubbed Apps Script globals
(`Utilities`, `LockService`, `SpreadsheetApp`, sheet ranges), so no deployment is involved.
Covers the chunk layout, the write/read round trip, the base64 join, bit setting at the
boundaries, out-of-range indices, idempotency, formula-safety of every stored cell, the
completion counter (including rebuilding a wiped C1), and that a missing sheet throws
rather than being created empty.

It also drives `handleRequest` the way a deployed web app does, for both
`getCompleteStripCache` and `incrementStripCompletion`. That part matters: a bad reference
inside a request handler is invisible to `node --check` and to unit tests of the helpers,
and one shipped that way once - `getCompleteStripCache` referenced a variable that had been
removed, so the deployed API returned `ReferenceError: sheet is not defined` for every
cache read. The script is loaded through `new Function` rather than `eval` so it cannot see
this file's locals; with a plain `eval`, a same-named variable in the test masked that very
bug.

```sh
node javascript/tests/progress-api.test.js              # synthetic bitmap
node javascript/tests/progress-api.test.js cache.json   # a saved API response as the fixture
```

Passing a real `getCompleteStripCache` response exercises the logic against actual data.
Run this before changing anything about the chunk layout.

# Utilities

The code in utils.js can be added to an app script similar to above and provide a variety of utilities usable in Google Sheets:

* bestPatternBinToDecimal(binaryString) - converts a string like 0011010110101001111101000010000000011101110000010011110111000110 to 3866890173849615814
* bestPatternBinToRLE(binaryString) - converts the string to RLE format that works in Golly, etc.
