/**
 * Google Apps Script WebApp for Conway's Game of Life Optimization Progress Tracking
 */

// Frame search sheet names
const FRAME_BESTS_SHEET_NAME = 'Frame Bests';
const FRAME_COMPLETION_SHEET_NAME = 'Frame Completion';
const FRAME_SUMMARY_SHEET_NAME = 'Frame Summary';

// Strip search sheet names
const STRIP_BESTS_SHEET_NAME = 'Strip Bests';
const STRIP_SUMMARY_SHEET_NAME = 'Strip Summary';

// Names these sheets used to have. Renaming the constant alone is not enough: the writers
// look a sheet up by name and create it when it is missing, so the 2025-12-16 rename of
// Frame/Strip Progress to Frame/Strip Bests silently started an empty tab beside the real
// one and the original stopped growing. getOrCreateSheet adopts a legacy tab by renaming
// it in place, so the next rename migrates the data instead of forking it.
const FRAME_BESTS_LEGACY_NAMES = ['Frame Progress'];
const STRIP_BESTS_LEGACY_NAMES = ['Strip Progress'];

const FRAME_BESTS_HEADERS = ['frameIdx', 'kernelIdx', 'bestGenerations', 'bestPattern'];
const STRIP_BESTS_HEADERS = ['centerIdx', 'middleIdx', 'bestGenerations', 'bestPattern'];

// The display name on the From line, so these are easy to spot and to filter on.
const MAIL_SENDER_NAME = 'find-optimal';

// Where sendMail delivers. Nothing is sent until this has an address.
//
// It must not be the account that owns this script, nor anything that routes back to it.
// Gmail files a message it considers self-sent under Sent and delivers no inbox copy, and
// it counts every verified "Send mail as" identity as the account - so sending from an
// alias does not help, which was tried. The failure gives every sign of success: the script
// reports sent, the quota decrements, and Delivered-To names the right mailbox, because the
// message really was delivered. It is just labelled Sent rather than Inbox.
//
// A Google Group with the account as its only member is how the reports still reach that
// mailbox: the group relays the message, so it arrives from the group and is delivered
// normally. Any mailbox outside the account works too.
//
// It is a constant rather than a lookup of the script owner because Session.getEffectiveUser
// needs the userinfo.email scope, which a web app deployment does not carry - asking for it
// would mean re-authorizing the whole script to learn an address that is known anyway.
const MAIL_DEFAULT_RECIPIENT = 'gol-alerts@aharbick.com';

// Extra addresses sendMail may deliver to. The default recipient is always allowed; anything
// else has to be listed here, so a leaked API key cannot turn this into an open relay.
const MAIL_ALLOWED_RECIPIENTS = [];

const LOCK_TIMEOUT_MS = 30000; // 30 seconds timeout for locks

// Spreadsheet ID for our Progress data
const SPREADSHEET_ID = '1bXt22T9cyv1A1vcdozR54n-imEFdIhsMcvEhEUMlsmY';

// Frame completion constants
const FRAME_TOTAL_FRAMES = 2102800;
const FRAME_BITS_PER_ROW = 64; // 64 bits per cell
const FRAME_COMPLETION_ROWS = Math.ceil(FRAME_TOTAL_FRAMES / FRAME_BITS_PER_ROW); // 32857 rows

// Strip completion constants
const STRIP_TOTAL_CENTERS = 8548;
const STRIP_MIDDLE_IDX_COUNT = 512;
const STRIP_TOTAL_INTERVALS = STRIP_TOTAL_CENTERS * STRIP_MIDDLE_IDX_COUNT; // 4,376,576
// Strip completion bitmap, stored as base64 chunks instead of 68,384 BigInt rows.
//
// The old layout cost ~19s to read: 68,384 rows plus 8 BigInt shift-and-mask operations
// per row (~547,000 BigInt ops) to rebuild 547,072 bytes, all of it inside the script
// lock, which blocked every completion and summary write behind it. Here the bitmap lives
// in 179 cells of base64 and a read is a single getValues() plus a join - no BigInt at
// all - so getCompleteStripCache stops hogging the lock.
//
// STRIP_CHUNK_BYTES must stay a multiple of 3: base64 then encodes each full chunk with
// no padding, so concatenating the cells' base64 is itself valid base64 for the whole
// bitmap and the server never has to decode it to answer a read.
const STRIP_BITMAP_SHEET_NAME = 'Strip Completion B64';
const STRIP_BITMAP_BYTES = Math.ceil(STRIP_TOTAL_INTERVALS / 8);                  // 547,072
const STRIP_CHUNK_BYTES = 3072;                                                    // 4,096 base64 chars
const STRIP_CHUNK_ROWS = Math.ceil(STRIP_BITMAP_BYTES / STRIP_CHUNK_BYTES);        // 179 rows
// Sheets parses a leading '=' or '+' as a formula, and base64 can start with '+', so the
// stored text carries a prefix that is stripped on read.
const STRIP_CHUNK_PREFIX = 'b64:';
// Column A row 1 is the chunk header and A2:A180 the chunks, so row 1 of columns B and C
// is free. C1 caches the number of completed intervals for dashboard formulas: summing the
// Strip Summary histogram instead gets this wrong, because intervals that find nothing
// never file a summary and re-run intervals file a second one.
const STRIP_COUNT_LABEL_COL = 2; // B1: 'completedIntervals'
const STRIP_COUNT_COL = 3;       // C1: the count itself

/**
 * Execute a function with script-level locking for concurrency safety
 * @param {Function} operation - The function to execute under lock
 * @returns {Object} ContentService response object
 */
/**
 * The sheet by this name, adopting one left under an older name, or a new one with headers.
 *
 * Creating a fresh sheet is the last resort on purpose. An empty tab appearing beside a
 * full one looks like a working script while the history stops accumulating, and nothing
 * in the write path would ever notice.
 */
function getOrCreateSheet(spreadsheet, name, legacyNames, headers) {
  const sheet = spreadsheet.getSheetByName(name);
  if (sheet) return sheet;

  for (let i = 0; i < legacyNames.length; i++) {
    const legacy = spreadsheet.getSheetByName(legacyNames[i]);
    if (legacy) {
      console.log(`Adopting '${legacyNames[i]}' as '${name}': same data under a new name`);
      legacy.setName(name);
      return legacy;
    }
  }

  const created = spreadsheet.insertSheet(name);
  created.getRange(1, 1, 1, headers.length).setValues([headers]);
  return created;
}

function withLock(operation) {
  const lock = LockService.getScriptLock();

  try {
    // Acquire lock with timeout
    if (!lock.tryLock(LOCK_TIMEOUT_MS)) {
      return sendJsonResponse(false, 'Could not acquire lock - operation timed out');
    }

    // Execute the operation
    return operation();

  } catch (error) {
    return sendJsonResponse(false, error.toString());
  } finally {
    // Spreadsheet writes are buffered, and Apps Script does not flush them when a lock is
    // released. Without this, the next execution can read a stale cell and overwrite it -
    // which for a bitmap chunk means silently dropping a completed interval's bit.
    try {
      SpreadsheetApp.flush();
    } catch (flushError) {
      console.error(`flush before releasing the lock failed: ${flushError}`);
    }
    lock.releaseLock();
  }
}

/**
 * Helper function to create consistent JSON responses
 * @param {boolean} success - Whether the operation was successful
 * @param {string} message - The message to return
 * @param {Object} additionalData - Optional additional data to include in response
 * @returns {ContentService.TextOutput} The formatted response
 */
function sendJsonResponse(success, message, additionalData = {}) {
  const responseData = {
    success: success,
    ...additionalData
  };

  if (success) {
    responseData.message = message;
  } else {
    responseData.error = message;
  }

  return ContentService
    .createTextOutput(JSON.stringify(responseData))
    .setMimeType(ContentService.MimeType.JSON);
}

/**
 * Ensure Frame Completion sheet exists and is properly initialized
 * @param {Spreadsheet} spreadsheet - The spreadsheet object
 * @returns {Sheet} The Frame Completion sheet
 */
function ensureFrameCompletionSheet(spreadsheet) {
  let sheet = spreadsheet.getSheetByName(FRAME_COMPLETION_SHEET_NAME);

  if (!sheet) {
    sheet = spreadsheet.insertSheet(FRAME_COMPLETION_SHEET_NAME);
    // Add header
    sheet.getRange(1, 1).setValue('frameBitmap');

    // Initialize all rows with 0 (no frames completed)
    const initData = Array(FRAME_COMPLETION_ROWS).fill(['0']);
    sheet.getRange(2, 1, FRAME_COMPLETION_ROWS, 1).setValues(initData);
  }

  return sheet;
}

/**
 * Set a frame as completed in the Frame Completion sheet
 * @param {Sheet} sheet - The Frame Completion sheet
 * @param {number} frameIdx - The frame index to mark as complete
 */
function setFrameComplete(sheet, frameIdx) {
  if (frameIdx < 0 || frameIdx >= FRAME_TOTAL_FRAMES) {
    return;
  }

  const rowIndex = Math.floor(frameIdx / FRAME_BITS_PER_ROW) + 2; // +2 for header and 1-based indexing
  const bitIndex = frameIdx % FRAME_BITS_PER_ROW;

  // Get current value
  const currentValue = sheet.getRange(rowIndex, 1).getValue() || '0';
  const currentBitmap = BigInt(currentValue);

  // Set the bit
  const newBitmap = currentBitmap | (BigInt(1) << BigInt(bitIndex));

  // Write back as string to preserve precision
  sheet.getRange(rowIndex, 1).setValue(newBitmap.toString());
}

/**
 * Main entry point for the WebApp
 * Handles both GET and POST requests
 */
function doGet(e) {
  return handleRequest(e);
}

function doPost(e) {
  return handleRequest(e);
}

function handleRequest(e) {
  try {
    // Use GET parameters only
    const data = e.parameter;

    // Extract and validate API key
    const apiKey = data.apiKey;
    if (!apiKey) {
      return sendJsonResponse(false, 'Missing required parameter: apiKey');
    }

    // Validate API key
    if (apiKey !== AUTHORIZED_API_KEY) {
      return sendJsonResponse(false, 'Invalid API key');
    }

    const action = data.action;

    switch (action) {
      // Frame search actions
      case 'sendProgress':
        return googleSendProgress(e, SPREADSHEET_ID);
      case 'sendSummaryData':
        return googleSendSummaryData(e, SPREADSHEET_ID);
      case 'getBestResult':
        return googleGetBestResult(e, SPREADSHEET_ID);
      case 'getCompleteFrameCache':
        return googleGetCompleteFrameCache(e, SPREADSHEET_ID);
      // Strip search actions
      case 'sendStripProgress':
        return googleSendStripProgress(e, SPREADSHEET_ID);
      case 'sendStripSummaryData':
        return googleSendStripSummaryData(e, SPREADSHEET_ID);
      case 'getCompleteStripCache':
        return googleGetCompleteStripCache(e, SPREADSHEET_ID);
      case 'incrementStripCompletion':
        return googleIncrementStripCompletion(e, SPREADSHEET_ID);
      case 'sendMail':
        return googleSendMail(e);
      default:
        return sendJsonResponse(false, 'Invalid action. Valid actions: sendProgress, sendSummaryData, getBestResult, getCompleteFrameCache, sendStripProgress, sendStripSummaryData, getCompleteStripCache, incrementStripCompletion, sendMail');
    }
  } catch (error) {
    return sendJsonResponse(false, error.toString());
  }
}

/**
 * Sends a notification mail on behalf of the account that owns this script.
 *
 * Recipients are restricted deliberately. The API key travels in a query string, is stored
 * in a .envrc on a workstation and is shared by every caller, so it is not a secret worth
 * betting an open relay on: anyone holding it could otherwise send mail from this Google
 * account to anywhere. Leaving `to` off sends to MAIL_DEFAULT_RECIPIENT, which is all the
 * search needs, and MAIL_ALLOWED_RECIPIENTS is the only way to widen that.
 */
function googleSendMail(e) {
  const data = e.parameter;
  const subject = data.subject || '';
  const body = data.body || '';

  if (!subject && !body) {
    return sendJsonResponse(false, 'Missing required parameters: subject and/or body');
  }

  const allowed = [MAIL_DEFAULT_RECIPIENT].concat(MAIL_ALLOWED_RECIPIENTS)
                    .filter((a) => a).map((a) => a.toLowerCase());
  if (allowed.length === 0) {
    return sendJsonResponse(false,
      'No recipient configured: set MAIL_DEFAULT_RECIPIENT in progress-api.js and redeploy');
  }

  const to = data.to || MAIL_DEFAULT_RECIPIENT;
  if (allowed.indexOf(to.toLowerCase()) === -1) {
    return sendJsonResponse(false, 'Recipient not allowed: add it to MAIL_ALLOWED_RECIPIENTS');
  }

  // A quota failure is the interesting case - silence would look like a working notifier
  const remaining = MailApp.getRemainingDailyQuota();
  if (remaining <= 0) {
    return sendJsonResponse(false, 'Daily mail quota exhausted');
  }

  const message = { to: to, subject: subject, body: body, name: MAIL_SENDER_NAME };
  // The report is a monospace table, so the readable version is the HTML one; the plain
  // body stays as the fallback for clients that ask for it.
  if (data.htmlBody) message.htmlBody = data.htmlBody;
  MailApp.sendEmail(message);
  return sendJsonResponse(true, 'Mail sent', { to: to, quotaRemaining: remaining - 1 });
}

/**
 * SETUP: run this once from the Apps Script editor, before using sendMail.
 *
 * Apps Script grants OAuth scopes when a function runs in the editor and you accept the
 * consent prompt - not when a web app is deployed. A deployment made before MailApp
 * appeared in this file therefore carries no script.send_mail scope, and every sendMail
 * fails with a permissions exception however many times it is redeployed.
 *
 * Running this asks for the missing scope and proves the address works. Redeploy as a new
 * version afterwards so /exec runs with it.
 */
function authorizeMail(to) {
  const recipient = to || MAIL_DEFAULT_RECIPIENT;
  if (!recipient) {
    throw new Error('Set MAIL_DEFAULT_RECIPIENT at the top of this file first');
  }

  // Quota is the only evidence available here that Google took the message. MailApp keeps
  // no copy in Sent and reports nothing about delivery, so a send that is accepted and then
  // filtered at the far end looks exactly like a send that worked.
  const before = MailApp.getRemainingDailyQuota();
  const stamp = new Date().toISOString();
  const message = {
    to: recipient,
    subject: 'find-optimal: mail authorized ' + stamp,
    body: 'progress-api.js can send mail now.\n\nSent at ' + stamp + ' to ' + recipient,
    name: MAIL_SENDER_NAME,
  };
  MailApp.sendEmail(message);
  const after = MailApp.getRemainingDailyQuota();

  console.log('Sent to ' + recipient);
  console.log('Quota ' + before + ' -> ' + after +
              (after < before ? ' (accepted by Google)'
                              : ' (UNCHANGED - it was not actually sent)'));
  console.log('Search for it with: in:anywhere subject:"find-optimal: mail authorized ' +
              stamp + '"');
  console.log('If it is not in the inbox: landing under Sent means this recipient routes' +
              ' back to the sending account, so send to a group that relays for it or to a' +
              ' mailbox outside the account. A different From address does not help.');
}

/**
 * Adds progress data to the Google Sheet
 */
function googleSendProgress(e, spreadsheetId) {
  return withLock(() => {
    // Use GET parameters only
    const data = e.parameter;

    const frameIdx = parseInt(data.frameIdx) || 0;
    const kernelIdx = parseInt(data.kernelIdx) || 0;
    const bestGenerations = parseInt(data.bestGenerations) || 0;
    const bestPattern = data.bestPattern || '';

    // Get the spreadsheet and worksheet
    const spreadsheet = SpreadsheetApp.openById(spreadsheetId);
    const sheet = getOrCreateSheet(spreadsheet, FRAME_BESTS_SHEET_NAME,
                                   FRAME_BESTS_LEGACY_NAMES, FRAME_BESTS_HEADERS);

    const newRow = [frameIdx, kernelIdx, bestGenerations, bestPattern];

    sheet.appendRow(newRow);

    // Update Frame Completion sheet if this frame is complete (kernelIdx == 15)
    if (kernelIdx === 15) {
      const frameCompletionSheet = ensureFrameCompletionSheet(spreadsheet);
      setFrameComplete(frameCompletionSheet, frameIdx);
    }

    return sendJsonResponse(true, 'Progress data saved successfully');
  });
}

/**
 * Returns the highest bestGenerations value from Summary Data sheet
 * @param {Object} e - Event object with parameters
 * @param {string} spreadsheetId - Spreadsheet ID
 * @param {string} e.parameter.searchType - Optional: 'frame' (default) or 'strip'
 */
function googleGetBestResult(e, spreadsheetId) {
  return withLock(() => {
    const params = e.parameter;
    const searchType = (params.searchType || 'frame').toLowerCase();

    const spreadsheet = SpreadsheetApp.openById(spreadsheetId);
    const sheetName = searchType === 'strip' ? STRIP_SUMMARY_SHEET_NAME : FRAME_SUMMARY_SHEET_NAME;
    const sheet = spreadsheet.getSheetByName(sheetName);

    if (!sheet) {
      return sendJsonResponse(true, `No ${sheetName} sheet found`, { bestGenerations: 0 });
    }

    // Get all data from the sheet
    const data = sheet.getDataRange().getValues();

    if (data.length <= 1) { // Only header row or empty
      return sendJsonResponse(true, 'No data found', { bestGenerations: 0 });
    }

    // Find the column indices (assuming first row contains headers)
    const headers = data[0];
    const bestGenerationsCol = headers.indexOf('bestGenerations');

    if (bestGenerationsCol === -1) {
      return sendJsonResponse(false, 'bestGenerations column not found in Summary Data sheet');
    }

    // Find the maximum bestGenerations value (sheet should be sorted, so we can take the last row)
    // But let's iterate to be safe in case sorting failed somewhere
    let maxGenerations = 0;

    for (let i = 1; i < data.length; i++) {
      const row = data[i];
      const generations = parseInt(row[bestGenerationsCol]) || 0;

      if (generations > maxGenerations) {
        maxGenerations = generations;
      }
    }

    return sendJsonResponse(true, 'Best result retrieved successfully', { bestGenerations: maxGenerations });
  });
}

/**
 * Get a bitmap of all completed frames for efficient caching
 * Returns a base64-encoded bitmap where each bit represents a frame's completion status
 * Reads directly from the Frame Completion sheet for fast access
 */
function googleGetCompleteFrameCache(e, spreadsheetId) {
  return withLock(() => {
    const spreadsheet = SpreadsheetApp.openById(spreadsheetId);
    const frameCompletionSheet = ensureFrameCompletionSheet(spreadsheet);

    // Read all completion data from Frame Completion sheet
    const dataRange = frameCompletionSheet.getRange(2, 1, FRAME_COMPLETION_ROWS, 1).getValues();

    // Convert 64-bit values to 8-bit bitmap
    const bitmapBytes = Math.ceil(FRAME_TOTAL_FRAMES / 8);
    const bitmap = new Uint8Array(bitmapBytes);

    for (let rowIdx = 0; rowIdx < dataRange.length && rowIdx < FRAME_COMPLETION_ROWS; rowIdx++) {
      const bitmapValue = BigInt(dataRange[rowIdx][0] || '0');

      // Each row contains 64 bits, convert to 8 bytes
      for (let byteInRow = 0; byteInRow < 8; byteInRow++) {
        const globalByteIdx = rowIdx * 8 + byteInRow;
        if (globalByteIdx >= bitmapBytes) break;

        // Extract 8 bits from the 64-bit value
        const byteValue = Number((bitmapValue >> BigInt(byteInRow * 8)) & BigInt(0xFF));
        bitmap[globalByteIdx] = byteValue;
      }
    }

    // Convert bitmap to base64 for transmission
    const bitmapBase64 = Utilities.base64Encode(bitmap);

    return sendJsonResponse(true, 'Frame cache retrieved successfully', {
      bitmap: bitmapBase64,
      totalFrames: FRAME_TOTAL_FRAMES,
      bitmapSize: bitmapBytes
    });
  });
}

/**
 * Updates summary data for histogram tracking
 * Increments count if bestGenerations exists, otherwise creates new row
 */
function googleSendSummaryData(e, spreadsheetId) {
  return withLock(() => {
    // Use GET parameters only
    const data = e.parameter;

    // Validate required parameters
    if (!data.bestGenerations || !data.bestPattern || !data.bestPatternBin) {
      return sendJsonResponse(false, 'Missing required parameters: bestGenerations, bestPattern, bestPatternBin');
    }

    const bestGenerations = parseInt(data.bestGenerations);
    if (isNaN(bestGenerations) || bestGenerations < 0) {
      return sendJsonResponse(false, 'Invalid bestGenerations parameter: must be a non-negative integer');
    }

    const bestPattern = data.bestPattern;
    const bestPatternBin = data.bestPatternBin;

    // Optional parameter for frame completion tracking
    const completedFrameIdx = data.completedFrameIdx ? parseInt(data.completedFrameIdx) : null;

    // Get the spreadsheet and worksheet
    const spreadsheet = SpreadsheetApp.openById(spreadsheetId);
    const sheet = spreadsheet.getSheetByName(FRAME_SUMMARY_SHEET_NAME);

    if (!sheet) {
      return sendJsonResponse(false, 'Frame Summary sheet not found');
    }

    // Get all data to search for existing row
    const dataRange = sheet.getDataRange();
    const data_values = dataRange.getValues();

    if (data_values.length === 0) {
      return sendJsonResponse(false, 'Summary Data sheet is empty');
    }

    // Find column indices
    const headers = data_values[0];
    const bestGenerationsCol = headers.indexOf('bestGenerations');
    const countCol = headers.indexOf('count');
    const bestPatternCol = headers.indexOf('bestPattern');
    const bestPatternBinCol = headers.indexOf('bestPatternBin');

    if (bestGenerationsCol === -1 || countCol === -1 || bestPatternCol === -1 || bestPatternBinCol === -1) {
      return sendJsonResponse(false, 'Required columns not found in Summary Data sheet: bestGenerations, count, bestPattern, bestPatternBin');
    }

    // Search for existing row with same bestGenerations
    let foundRow = -1;
    for (let i = 1; i < data_values.length; i++) {
      const rowBestGenerations = parseInt(data_values[i][bestGenerationsCol]) || 0;
      if (rowBestGenerations === bestGenerations) {
        foundRow = i + 1; // Convert to 1-based row index
        break;
      }
    }

    if (foundRow > 0) {
      // Increment count in existing row
      const currentCount = parseInt(data_values[foundRow - 1][countCol]) || 0;
      sheet.getRange(foundRow, countCol + 1).setValue(currentCount + 1);
    } else {
      // Add new row
      sheet.appendRow([bestGenerations, 1, bestPattern, bestPatternBin]);

      // Sort the sheet by bestGenerations column (ascending order)
      // Get the range of all data (excluding header row)
      const lastRow = sheet.getLastRow();
      if (lastRow > 1) {
        const sortRange = sheet.getRange(2, 1, lastRow - 1, 4);
        sortRange.sort({ column: bestGenerationsCol + 1, ascending: true });
      }
    }

    // Update Frame Completion sheet if completedFrameIdx is provided
    if (completedFrameIdx !== null && completedFrameIdx >= 0 && completedFrameIdx < FRAME_TOTAL_FRAMES) {
      const frameCompletionSheet = ensureFrameCompletionSheet(spreadsheet);
      setFrameComplete(frameCompletionSheet, completedFrameIdx);
    }

    return sendJsonResponse(true, 'Summary data saved successfully');
  });
}

/**
 * UTILITY FUNCTION: Backfill Frame Completion sheet from Progress sheet data
 * Call this manually in Apps Script console: backfillFrameCompletionFromProgress()
 * Reads Progress sheet for kernelIdx=15 entries and populates Frame Completion bitmap
 */
function backfillFrameCompletionFromProgress() {
  const spreadsheet = SpreadsheetApp.openById(SPREADSHEET_ID);

  console.log('Starting Frame Completion backfill...');

  // Get Progress sheet
  const progressSheet = spreadsheet.getSheetByName(FRAME_BESTS_SHEET_NAME);
  if (!progressSheet) {
    console.error('Frame Progress sheet not found');
    return;
  }

  // Ensure Frame Completion sheet exists
  const frameCompletionSheet = ensureFrameCompletionSheet(spreadsheet);

  // Get all progress data
  console.log('Reading Progress sheet data...');
  const progressData = progressSheet.getDataRange().getValues();
  if (progressData.length <= 1) {
    console.log('No data found in Progress sheet');
    return;
  }

  // Find column indices in Progress sheet
  const progressHeaders = progressData[0];
  const frameIdxCol = progressHeaders.indexOf('frameIdx');
  const kernelIdxCol = progressHeaders.indexOf('kernelIdx');

  if (frameIdxCol === -1 || kernelIdxCol === -1) {
    console.error('Required columns (frameIdx, kernelIdx) not found in Progress sheet');
    return;
  }

  console.log(`Found ${progressData.length - 1} total progress entries`);

  // Collect all completed frames (kernelIdx == 15)
  console.log('Collecting completed frames (kernelIdx=15)...');
  const completedFrames = new Set();
  for (let i = 1; i < progressData.length; i++) {
    const row = progressData[i];
    const frameIdx = parseInt(row[frameIdxCol]) || 0;
    const kernelIdx = parseInt(row[kernelIdxCol]) || 0;

    if (kernelIdx === 15 && frameIdx >= 0 && frameIdx < FRAME_TOTAL_FRAMES) {
      completedFrames.add(frameIdx);
    }

    // Progress report every 100k entries
    if (i % 100000 === 0) {
      console.log(`Processed ${i}/${progressData.length - 1} entries, found ${completedFrames.size} completed frames`);
    }
  }

  console.log(`Found ${completedFrames.size} completed frames`);

  // Read current Frame Completion data
  console.log('Reading current Frame Completion data...');
  const frameCompletionData = frameCompletionSheet.getRange(2, 1, FRAME_COMPLETION_ROWS, 1).getValues();

  // Build new bitmap data
  console.log('Building new bitmap data...');
  const newBitmapData = [];
  for (let rowIdx = 0; rowIdx < FRAME_COMPLETION_ROWS; rowIdx++) {
    let rowBitmap = BigInt(frameCompletionData[rowIdx] ? frameCompletionData[rowIdx][0] || '0' : '0');

    // Check each bit position in this row (64 frames per row)
    for (let bitIdx = 0; bitIdx < FRAME_BITS_PER_ROW; bitIdx++) {
      const frameIdx = rowIdx * FRAME_BITS_PER_ROW + bitIdx;
      if (frameIdx >= FRAME_TOTAL_FRAMES) break;

      if (completedFrames.has(frameIdx)) {
        // Set the bit for this completed frame
        rowBitmap |= (BigInt(1) << BigInt(bitIdx));
      }
    }

    newBitmapData.push([rowBitmap.toString()]);

    // Progress report every 1000 rows
    if (rowIdx % 1000 === 0) {
      console.log(`Built bitmap for row ${rowIdx}/${FRAME_COMPLETION_ROWS}`);
    }
  }

  // Write all the new bitmap data at once
  console.log('Writing Frame Completion data...');
  frameCompletionSheet.getRange(2, 1, FRAME_COMPLETION_ROWS, 1).setValues(newBitmapData);

  console.log(`✅ Frame completion cache backfilled successfully!`);
  console.log(`📊 Processed ${completedFrames.size} completed frames from ${progressData.length - 1} progress entries`);
}

// ============================================================================
// STRIP SEARCH APIs
// ============================================================================

/**
 * Number of bytes in a given chunk row (the last one is short).
 */
function stripChunkLength(chunkIdx) {
  const start = chunkIdx * STRIP_CHUNK_BYTES;
  return Math.min(STRIP_CHUNK_BYTES, STRIP_BITMAP_BYTES - start);
}

/**
 * Base64 for a run of zero bytes, used to initialize the sheet.
 */
function zeroChunkBase64(length) {
  return Utilities.base64Encode(new Array(length).fill(0));
}

/**
 * Get the bitmap sheet, or throw if it is missing.
 *
 * Deliberately does NOT create it. An absent sheet used to fall back to the legacy
 * row-format sheet and, failing that, to a freshly zeroed bitmap - which would report
 * "0 completed intervals" and send the search back to 0:0 to redo everything. Failing
 * here instead surfaces as a failed cache load, which the client treats as fatal.
 * Use initializeStripBitmapSheet() to set up a new spreadsheet on purpose.
 *
 * @param {Spreadsheet} spreadsheet - The spreadsheet object
 * @returns {Sheet} The chunked bitmap sheet
 */
function ensureStripBitmapSheet(spreadsheet) {
  const sheet = spreadsheet.getSheetByName(STRIP_BITMAP_SHEET_NAME);
  if (!sheet) {
    throw new Error(
        `${STRIP_BITMAP_SHEET_NAME} sheet is missing. Refusing to create an empty bitmap: ` +
        `that would look like zero completed intervals and restart the whole search. ` +
        `Restore the sheet from a backup, or run initializeStripBitmapSheet() if this is ` +
        `genuinely a new spreadsheet.`);
  }
  return sheet;
}

/**
 * Write a whole byte array out as base64 chunk rows.
 * @param {Sheet} sheet - The chunked bitmap sheet
 * @param {number[]} bytes - Signed byte values, STRIP_BITMAP_BYTES long
 */
function writeStripBitmapBytes(sheet, bytes) {
  const rows = [];
  for (let chunkIdx = 0; chunkIdx < STRIP_CHUNK_ROWS; chunkIdx++) {
    const start = chunkIdx * STRIP_CHUNK_BYTES;
    const slice = bytes.slice(start, start + stripChunkLength(chunkIdx));
    rows.push([STRIP_CHUNK_PREFIX + Utilities.base64Encode(slice)]);
  }
  sheet.getRange(2, 1, STRIP_CHUNK_ROWS, 1).setValues(rows);
}

// Base64 symbols in value order; index = the 6 bits the symbol encodes.
const STRIP_B64_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
let stripB64BitCounts = null;

/**
 * Count the set bits in a base64 string without decoding it.
 *
 * Each symbol carries 6 bits of the byte stream and encoders zero-fill the tail, so summing
 * the symbols' bit counts is exactly the popcount of the encoded bytes; '=' padding adds
 * nothing. That keeps the completion count derivable on the read path, where decoding
 * 547KB would be far too slow.
 *
 * @param {string} encoded - Base64 text
 * @returns {number} Number of set bits
 */
function countBitsInBase64(encoded) {
  if (!stripB64BitCounts) {
    stripB64BitCounts = new Uint8Array(128);
    for (let value = 0; value < 64; value++) {
      let bits = 0;
      for (let b = 0; b < 6; b++) {
        if (value & (1 << b)) bits++;
      }
      stripB64BitCounts[STRIP_B64_ALPHABET.charCodeAt(value)] = bits;
    }
  }

  let count = 0;
  for (let i = 0; i < encoded.length; i++) {
    count += stripB64BitCounts[encoded.charCodeAt(i)];
  }
  return count;
}

/**
 * Read the cached completion count.
 * @param {Sheet} sheet - The chunked bitmap sheet
 * @returns {?number} The count, or null if the cell is missing or not a number
 */
function readStripCompletedCount(sheet) {
  const raw = sheet.getRange(1, STRIP_COUNT_COL).getValue();
  const value = typeof raw === 'number' ? raw : parseInt(String(raw), 10);
  return Number.isFinite(value) && value >= 0 ? value : null;
}

/**
 * Write the completion count and its label.
 * @param {Sheet} sheet - The chunked bitmap sheet
 * @param {number} count - Number of completed intervals
 */
function writeStripCompletedCount(sheet, count) {
  sheet.getRange(1, STRIP_COUNT_LABEL_COL).setValue('completedIntervals');
  sheet.getRange(1, STRIP_COUNT_COL).setValue(count);
}

/**
 * Set one interval's bit by rewriting just its chunk row.
 * Uses linear indexing: linearIdx = centerIdx * 512 + middleIdx
 * @param {Sheet} sheet - The chunked bitmap sheet
 * @param {number} centerIdx - The center index
 * @param {number} middleIdx - The middleIdx to mark as complete
 */
function setStripIntervalComplete(sheet, centerIdx, middleIdx) {
  if (centerIdx < 0 || centerIdx >= STRIP_TOTAL_CENTERS) {
    return;
  }
  if (middleIdx < 0 || middleIdx >= STRIP_MIDDLE_IDX_COUNT) {
    return;
  }

  const linearIdx = centerIdx * STRIP_MIDDLE_IDX_COUNT + middleIdx;
  const byteIdx = Math.floor(linearIdx / 8);
  const bitIdx = linearIdx % 8;
  const chunkIdx = Math.floor(byteIdx / STRIP_CHUNK_BYTES);
  const byteInChunk = byteIdx % STRIP_CHUNK_BYTES;
  const cell = sheet.getRange(chunkIdx + 2, 1); // +2 for header and 1-based indexing

  const stored = String(cell.getValue() || '');
  const encoded = stored.startsWith(STRIP_CHUNK_PREFIX) ? stored.substring(STRIP_CHUNK_PREFIX.length) : stored;
  const bytes = encoded ? Utilities.base64Decode(encoded) : new Array(stripChunkLength(chunkIdx)).fill(0);

  const updated = (bytes[byteInChunk] & 0xFF) | (1 << bitIdx);
  if (updated === (bytes[byteInChunk] & 0xFF)) {
    return; // already complete, leave the cell and the count alone
  }
  bytes[byteInChunk] = updated > 127 ? updated - 256 : updated;

  cell.setValue(STRIP_CHUNK_PREFIX + Utilities.base64Encode(bytes));

  // Only reached on a real 0 -> 1 flip, so re-running an interval cannot inflate this.
  // Callers hold the script lock, which makes the read-modify-write safe.
  const count = readStripCompletedCount(sheet);
  if (count === null) {
    // No usable count - a restore that replaced the sheet wipes C1. Derive it from the
    // bitmap, which is authoritative, rather than starting a plausible-looking count at 1.
    writeStripCompletedCount(sheet, countStripBits(readStripBitmapBytes(sheet)));
  } else {
    writeStripCompletedCount(sheet, count + 1);
  }
}

/**
 * Adds strip progress data to the Google Sheet
 */
function googleSendStripProgress(e, spreadsheetId) {
  return withLock(() => {
    const data = e.parameter;

    const centerIdx = parseInt(data.centerIdx) || 0;
    const middleIdx = parseInt(data.middleIdx) || 0;
    const bestGenerations = parseInt(data.bestGenerations) || 0;
    const bestPattern = data.bestPattern || '';

    // Get the spreadsheet and worksheet
    const spreadsheet = SpreadsheetApp.openById(spreadsheetId);
    const sheet = getOrCreateSheet(spreadsheet, STRIP_BESTS_SHEET_NAME,
                                   STRIP_BESTS_LEGACY_NAMES, STRIP_BESTS_HEADERS);

    const newRow = [centerIdx, middleIdx, bestGenerations, bestPattern];
    sheet.appendRow(newRow);

    return sendJsonResponse(true, 'Strip progress data saved successfully');
  });
}

/**
 * Updates strip summary data for histogram tracking
 * Increments count if bestGenerations exists, otherwise creates new row
 */
function googleSendStripSummaryData(e, spreadsheetId) {
  return withLock(() => {
    const data = e.parameter;

    // Validate required parameters
    if (!data.bestGenerations || !data.bestPattern || !data.bestPatternBin) {
      return sendJsonResponse(false, 'Missing required parameters: bestGenerations, bestPattern, bestPatternBin');
    }

    const bestGenerations = parseInt(data.bestGenerations);
    if (isNaN(bestGenerations) || bestGenerations < 0) {
      return sendJsonResponse(false, 'Invalid bestGenerations parameter: must be a non-negative integer');
    }

    const bestPattern = data.bestPattern;
    const bestPatternBin = data.bestPatternBin;

    // Get the spreadsheet and worksheet
    const spreadsheet = SpreadsheetApp.openById(spreadsheetId);
    let sheet = spreadsheet.getSheetByName(STRIP_SUMMARY_SHEET_NAME);

    // Create sheet if it doesn't exist
    if (!sheet) {
      sheet = spreadsheet.insertSheet(STRIP_SUMMARY_SHEET_NAME);
      // Add headers
      sheet.getRange(1, 1, 1, 4).setValues([['bestGenerations', 'count', 'bestPattern', 'bestPatternBin']]);
    }

    // Get all data to search for existing row
    const dataRange = sheet.getDataRange();
    const data_values = dataRange.getValues();

    // Find column indices
    const headers = data_values[0];
    const bestGenerationsCol = headers.indexOf('bestGenerations');
    const countCol = headers.indexOf('count');

    // Search for existing row with same bestGenerations
    let foundRow = -1;
    for (let i = 1; i < data_values.length; i++) {
      const rowBestGenerations = parseInt(data_values[i][bestGenerationsCol]) || 0;
      if (rowBestGenerations === bestGenerations) {
        foundRow = i + 1; // Convert to 1-based row index
        break;
      }
    }

    if (foundRow > 0) {
      // Increment count in existing row
      const currentCount = parseInt(data_values[foundRow - 1][countCol]) || 0;
      sheet.getRange(foundRow, countCol + 1).setValue(currentCount + 1);
    } else {
      // Add new row
      sheet.appendRow([bestGenerations, 1, bestPattern, bestPatternBin]);

      // Sort the sheet by bestGenerations column (ascending order)
      const lastRow = sheet.getLastRow();
      if (lastRow > 1) {
        const sortRange = sheet.getRange(2, 1, lastRow - 1, 4);
        sortRange.sort({ column: bestGenerationsCol + 1, ascending: true });
      }
    }

    return sendJsonResponse(true, 'Strip summary data saved successfully');
  });
}

/**
 * Get strip completion cache as base64-encoded bitmap
 * Same format as frame completion for consistency
 */
function googleGetCompleteStripCache(e, spreadsheetId) {
  const spreadsheet = SpreadsheetApp.openById(spreadsheetId);

  // Deliberately NOT under withLock. Reading used to hold the script lock for ~19s, which
  // queued every completion and summary write behind it and pushed some past their own
  // 30s lock timeout. Completions only ever set bits and each write touches a single chunk
  // row, so the worst a concurrent write can do is leave a just-set bit out of this
  // response - which costs one interval being searched again and re-marked, not
  // corruption. Torn bits are impossible: a cell is read whole.
  return stripCacheResponse(ensureStripBitmapSheet(spreadsheet));
}

/**
 * Build the cache response by concatenating the stored base64 chunks.
 * No decode, no BigInt: STRIP_CHUNK_BYTES is a multiple of 3, so the chunks' base64
 * joins into valid base64 for the whole bitmap.
 * @param {Sheet} sheet - The chunked bitmap sheet
 * @returns {ContentService.TextOutput} The formatted response
 */
function stripCacheResponse(sheet) {
  const rows = sheet.getRange(2, 1, STRIP_CHUNK_ROWS, 1).getValues();

  let bitmapBase64 = '';
  for (let i = 0; i < rows.length; i++) {
    const stored = String(rows[i][0] || '');
    if (!stored) {
      return sendJsonResponse(false, `Strip bitmap chunk ${i} of ${STRIP_CHUNK_ROWS} is empty - the ` +
          `${STRIP_BITMAP_SHEET_NAME} sheet is damaged, restore it from a backup`);
    }
    bitmapBase64 += stored.startsWith(STRIP_CHUNK_PREFIX) ? stored.substring(STRIP_CHUNK_PREFIX.length) : stored;
  }

  // Derive the count from the bitmap we are already returning, and correct C1 if it has
  // drifted. C1 exists for dashboard formulas; deriving the number here means a lost
  // increment fixes itself on the next read instead of persisting.
  const completedIntervals = countBitsInBase64(bitmapBase64);
  const cachedCount = readStripCompletedCount(sheet);
  if (cachedCount !== completedIntervals) {
    writeStripCompletedCount(sheet, completedIntervals);
  }

  return sendJsonResponse(true, 'Strip cache retrieved successfully', {
    bitmap: bitmapBase64,
    totalIntervals: STRIP_TOTAL_INTERVALS,
    bitmapSize: STRIP_BITMAP_BYTES,
    completedIntervals: completedIntervals,  // derived from the bitmap, always exact
    cachedCount: cachedCount                 // what C1 held before this call, for drift checks
  });
}

/**
 * Read the chunked bitmap sheet back into a byte array.
 * @param {Sheet} sheet - The chunked bitmap sheet
 * @returns {number[]} Signed byte values, STRIP_BITMAP_BYTES long
 */
function readStripBitmapBytes(sheet) {
  const rows = sheet.getRange(2, 1, STRIP_CHUNK_ROWS, 1).getValues();
  let bytes = [];
  for (let i = 0; i < STRIP_CHUNK_ROWS; i++) {
    const stored = String(rows[i][0] || '');
    const encoded = stored.startsWith(STRIP_CHUNK_PREFIX) ? stored.substring(STRIP_CHUNK_PREFIX.length) : stored;
    bytes = bytes.concat(encoded ? Utilities.base64Decode(encoded) : new Array(stripChunkLength(i)).fill(0));
  }
  return bytes;
}

/**
 * Count the set bits in a byte array (how many intervals are complete).
 * @param {number[]} bytes - Signed byte values
 * @returns {number} Number of set bits
 */
function countStripBits(bytes) {
  let count = 0;
  for (let i = 0; i < bytes.length; i++) {
    let v = bytes[i] & 0xFF;
    while (v) {
      count += v & 1;
      v >>= 1;
    }
  }
  return count;
}

/**
 * UTILITY FUNCTION: create the strip bitmap sheet with nothing marked complete.
 * Call this manually in the Apps Script console for a brand new spreadsheet only:
 * initializeStripBitmapSheet()
 *
 * Nothing calls this automatically - an empty bitmap means the search starts over from
 * 0:0, so creating one is always a deliberate act.
 */
function initializeStripBitmapSheet() {
  const spreadsheet = SpreadsheetApp.openById(SPREADSHEET_ID);

  const existing = spreadsheet.getSheetByName(STRIP_BITMAP_SHEET_NAME);
  if (existing) {
    const bits = countStripBits(readStripBitmapBytes(existing));
    console.error(`${STRIP_BITMAP_SHEET_NAME} already exists with ${bits} completed intervals - ` +
                  `refusing to overwrite it. Delete the sheet first if you really mean to reset.`);
    return;
  }

  const sheet = spreadsheet.insertSheet(STRIP_BITMAP_SHEET_NAME);
  sheet.getRange(1, 1).setValue('stripBitmapBase64');
  writeStripBitmapBytes(sheet, new Array(STRIP_BITMAP_BYTES).fill(0));
  writeStripCompletedCount(sheet, 0);
  console.log(`✅ Created ${STRIP_BITMAP_SHEET_NAME} with ${STRIP_CHUNK_ROWS} empty chunk rows`);
}

/**
 * UTILITY FUNCTION: recompute the cached completion count in C1 from the bitmap.
 * Call this manually in the Apps Script console: recountStripCompletions()
 *
 * Rarely needed now that getCompleteStripCache corrects C1 on every read. Still useful
 * right after restoring the sheet from a backup, since importing over the sheet wipes C1.
 */
function recountStripCompletions() {
  const spreadsheet = SpreadsheetApp.openById(SPREADSHEET_ID);
  withLock(() => {
    const sheet = ensureStripBitmapSheet(spreadsheet);
    const actual = countStripBits(readStripBitmapBytes(sheet));
    const stored = readStripCompletedCount(sheet);
    writeStripCompletedCount(sheet, actual);
    if (stored === actual) {
      console.log(`✅ Count in C1 was already correct: ${actual} completed intervals`);
    } else {
      console.log(`✅ Updated C1: ${stored === null ? '(unset)' : stored} -> ${actual} completed intervals`);
    }
  });
}

/**
 * Set a specific strip interval as complete
 * Lightweight API that only updates the completion bitmap without logging progress
 */
function googleIncrementStripCompletion(e, spreadsheetId) {
  return withLock(() => {
    const data = e.parameter;

    const centerIdx = parseInt(data.centerIdx);
    if (isNaN(centerIdx) || centerIdx < 0 || centerIdx >= STRIP_TOTAL_CENTERS) {
      return sendJsonResponse(false, 'Invalid centerIdx parameter');
    }

    const middleIdx = parseInt(data.middleIdx);
    if (isNaN(middleIdx) || middleIdx < 0 || middleIdx >= STRIP_MIDDLE_IDX_COUNT) {
      return sendJsonResponse(false, 'Invalid middleIdx parameter');
    }

    const spreadsheet = SpreadsheetApp.openById(spreadsheetId);
    const stripBitmapSheet = ensureStripBitmapSheet(spreadsheet);
    setStripIntervalComplete(stripBitmapSheet, centerIdx, middleIdx);

    return sendJsonResponse(true, 'Strip interval marked complete');
  });
}

