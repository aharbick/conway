/**
 * Google Apps Script WebApp for Conway's Game of Life Optimization Progress Tracking
 */

const PROGRESS_SHEET_NAME = 'Progress';
const FRAME_COMPLETION_SHEET_NAME = 'Frame Completion';
const SUMMARY_DATA_SHEET_NAME = 'Summary Data';
const LOCK_TIMEOUT_MS = 30000; // 30 seconds timeout for locks

// Spreadsheet ID for our Progress data
const SPREADSHEET_ID = '1bXt22T9cyv1A1vcdozR54n-imEFdIhsMcvEhEUMlsmY';

// Frame completion constants
const TOTAL_FRAMES = 2102800;
const FRAMES_PER_ROW = 64; // 64 bits per cell
const FRAME_COMPLETION_ROWS = Math.ceil(TOTAL_FRAMES / FRAMES_PER_ROW); // 32857 rows

/**
 * Execute a function with script-level locking for concurrency safety
 * @param {Function} operation - The function to execute under lock
 * @returns {Object} ContentService response object
 */
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
    // Always release the lock
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
  if (frameIdx < 0 || frameIdx >= TOTAL_FRAMES) {
    return;
  }

  const rowIndex = Math.floor(frameIdx / FRAMES_PER_ROW) + 2; // +2 for header and 1-based indexing
  const bitIndex = frameIdx % FRAMES_PER_ROW;

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
      case 'sendProgress':
        return googleSendProgress(e, SPREADSHEET_ID);
      case 'sendSummaryData':
        return googleSendSummaryData(e, SPREADSHEET_ID);
      case 'getBestResult':
        return googleGetBestResult(e, SPREADSHEET_ID);
      case 'getCompleteFrameCache':
        return googleGetCompleteFrameCache(e, SPREADSHEET_ID);
      default:
        return sendJsonResponse(false, 'Invalid action. Use "sendProgress", "sendSummaryData", "getBestResult", or "getCompleteFrameCache"');
    }
  } catch (error) {
    return sendJsonResponse(false, error.toString());
  }
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
    let sheet = spreadsheet.getSheetByName(PROGRESS_SHEET_NAME);

    // Create sheet if it doesn't exist
    if (!sheet) {
      sheet = spreadsheet.insertSheet(PROGRESS_SHEET_NAME);
      // Add headers
      sheet.getRange(1, 1, 1, 4).setValues([['frameIdx', 'kernelIdx', 'bestGenerations', 'bestPattern']]);
    }

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
 */
function googleGetBestResult(e, spreadsheetId) {
  return withLock(() => {
    const spreadsheet = SpreadsheetApp.openById(spreadsheetId);
    const sheet = spreadsheet.getSheetByName(SUMMARY_DATA_SHEET_NAME);

    if (!sheet) {
      return sendJsonResponse(true, 'No Summary Data sheet found', { bestGenerations: 0 });
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
    const bitmapBytes = Math.ceil(TOTAL_FRAMES / 8);
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
      totalFrames: TOTAL_FRAMES,
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

    // Get the spreadsheet and worksheet
    const spreadsheet = SpreadsheetApp.openById(spreadsheetId);
    const sheet = spreadsheet.getSheetByName(SUMMARY_DATA_SHEET_NAME);

    if (!sheet) {
      return sendJsonResponse(false, 'Summary Data sheet not found');
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

    return sendJsonResponse(true, 'Summary data saved successfully');
  });
}

