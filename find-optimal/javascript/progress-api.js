/**
 * Google Apps Script WebApp for Conway's Game of Life Optimization Progress Tracking
 */

const PROGRESS_SHEET_NAME = 'Progress';
const LOCK_TIMEOUT_MS = 30000; // 30 seconds timeout for locks

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
      return ContentService
        .createTextOutput(JSON.stringify({
          success: false,
          error: 'Could not acquire lock - operation timed out'
        }))
        .setMimeType(ContentService.MimeType.JSON);
    }

    // Execute the operation
    return operation();

  } catch (error) {
    return ContentService
      .createTextOutput(JSON.stringify({
        success: false,
        error: error.toString()
      }))
      .setMimeType(ContentService.MimeType.JSON);
  } finally {
    // Always release the lock
    lock.releaseLock();
  }
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

    // Extract spreadsheetId (acts as API key)
    const spreadsheetId = data.spreadsheetId;
    if (!spreadsheetId) {
      return ContentService
        .createTextOutput(JSON.stringify({
          error: 'Missing required parameter: spreadsheetId'
        }))
        .setMimeType(ContentService.MimeType.JSON);
    }

    const action = data.action;

    switch (action) {
      case 'sendProgress':
        return googleSendProgress(e, spreadsheetId);
      case 'getBestResult':
        return googleGetBestResult(e, spreadsheetId);
      case 'getBestCompleteFrame':
        return googleGetBestCompleteFrame(e, spreadsheetId);
      case 'getIsFrameComplete':
        return googleGetIsFrameComplete(e, spreadsheetId);
      case 'getCompleteFrameCache':
        return googleGetCompleteFrameCache(e, spreadsheetId);
      default:
        return ContentService
          .createTextOutput(JSON.stringify({
            error: 'Invalid action. Use "sendProgress", "getBestResult", "getBestCompleteFrame", "getIsFrameComplete", or "getCompleteFrameCache"'
          }))
          .setMimeType(ContentService.MimeType.JSON);
    }
  } catch (error) {
    return ContentService
      .createTextOutput(JSON.stringify({
        error: error.toString()
      }))
      .setMimeType(ContentService.MimeType.JSON);
  }
}

/**
 * Adds progress data to the Google Sheet
 */
function googleSendProgress(e, spreadsheetId) {
  return withLock(() => {
    // Use GET parameters only
    const data = e.parameter;

    // Extract parameters (matching the C++ function signature)
    const frameComplete = data.frameComplete === 'true' ? true : null;
    const frameIdx = parseInt(data.frameIdx) || 0;
    const kernelIdx = parseInt(data.kernelIdx) || 0;
    const chunkIdx = parseInt(data.chunkIdx) || 0;
    const patternsPerSecond = parseInt(data.patternsPerSecond) || 0;
    const bestGenerations = parseInt(data.bestGenerations) || 0;
    const bestPattern = data.bestPattern ? "'" + data.bestPattern : '';  // Prefix with ' to force text format
    const bestPatternBin = data.bestPatternBin || '';
    const isTest = data.test === 'true' ? true : null;
    const randomFrame = data.randomFrame === 'true' ? true : (data.randomFrame === 'false' ? false : null);

    // Get the spreadsheet and worksheet
    const spreadsheet = SpreadsheetApp.openById(spreadsheetId);
    let sheet = spreadsheet.getSheetByName(PROGRESS_SHEET_NAME);

    // Create sheet if it doesn't exist
    if (!sheet) {
      sheet = spreadsheet.insertSheet(PROGRESS_SHEET_NAME);
      // Add headers
      sheet.getRange(1, 1, 1, 11).setValues([[
        'timestamp', 'frameComplete', 'frameIdx', 'kernelIdx', 'chunkIdx',
        'patternsPerSecond', 'bestGenerations', 'bestPattern', 'bestPatternBin', 'test', 'randomFrame'
      ]]);
      // Format bestPattern column (column H, index 8) as text to prevent scientific notation
      sheet.getRange('H:H').setNumberFormat('@');
    }

    // Add the new row
    const timestamp = Math.floor(Date.now() / 1000); // Unix timestamp
    const newRow = [
      timestamp,
      frameComplete,
      frameIdx,
      kernelIdx,
      chunkIdx,
      patternsPerSecond,
      bestGenerations,
      bestPattern,
      bestPatternBin,
      isTest,
      randomFrame
    ];

    sheet.appendRow(newRow);

    // Ensure the bestPattern cell is formatted as text to prevent scientific notation
    const lastRow = sheet.getLastRow();
    sheet.getRange(lastRow, 8).setNumberFormat('@');  // Column H (bestPattern)

    return ContentService
      .createTextOutput(JSON.stringify({
        success: true,
        message: 'Progress data saved successfully',
        timestamp: timestamp
      }))
      .setMimeType(ContentService.MimeType.JSON);
  });
}

/**
 * Returns the highest bestGenerations value from non-test records
 */
function googleGetBestResult(e, spreadsheetId) {
  return withLock(() => {
    const spreadsheet = SpreadsheetApp.openById(spreadsheetId);
    const sheet = spreadsheet.getSheetByName(PROGRESS_SHEET_NAME);

    if (!sheet) {
      return ContentService
        .createTextOutput(JSON.stringify({
          bestGenerations: 0,
          message: 'No progress sheet found'
        }))
        .setMimeType(ContentService.MimeType.JSON);
    }

    // Get all data from the sheet
    const data = sheet.getDataRange().getValues();

    if (data.length <= 1) { // Only header row or empty
      return ContentService
        .createTextOutput(JSON.stringify({
          bestGenerations: 0,
          message: 'No data found'
        }))
        .setMimeType(ContentService.MimeType.JSON);
    }

    // Find the column indices (assuming first row contains headers)
    const headers = data[0];
    const bestGenerationsCol = headers.indexOf('bestGenerations');
    const testCol = headers.indexOf('test');

    if (bestGenerationsCol === -1) {
      throw new Error('bestGenerations column not found');
    }

    // Find the maximum bestGenerations value from non-test records
    let maxGenerations = 0;

    for (let i = 1; i < data.length; i++) {
      const row = data[i];
      const isTest = testCol !== -1 ? (row[testCol] === true || row[testCol] === 'true') : false;
      const generations = parseInt(row[bestGenerationsCol]) || 0;

      // Skip test records (where test is true) and null/empty test values are considered non-test
      if (!isTest && generations > maxGenerations) {
        maxGenerations = generations;
      }
    }

    return ContentService
      .createTextOutput(JSON.stringify({
        bestGenerations: maxGenerations
      }))
      .setMimeType(ContentService.MimeType.JSON);
  });
}

/**
 * Returns the highest frameIdx value from completed, non-test records
 */
function googleGetBestCompleteFrame(e, spreadsheetId) {
  return withLock(() => {
    const spreadsheet = SpreadsheetApp.openById(spreadsheetId);
    const sheet = spreadsheet.getSheetByName(PROGRESS_SHEET_NAME);

    if (!sheet) {
      return ContentService
        .createTextOutput(JSON.stringify({
          bestFrameIdx: null,
          message: 'No progress sheet found'
        }))
        .setMimeType(ContentService.MimeType.JSON);
    }

    const data = sheet.getDataRange().getValues();

    if (data.length <= 1) {
      return ContentService
        .createTextOutput(JSON.stringify({
          bestFrameIdx: null,
          message: 'No data found'
        }))
        .setMimeType(ContentService.MimeType.JSON);
    }

    // Find column indices
    const headers = data[0];
    const frameIdxCol = headers.indexOf('frameIdx');
    const frameCompleteCol = headers.indexOf('frameComplete');
    const testCol = headers.indexOf('test');
    const randomFrameCol = headers.indexOf('randomFrame');

    if (frameIdxCol === -1 || frameCompleteCol === -1) {
      throw new Error('Required columns not found');
    }

    // Find the maximum frameIdx value from completed, non-test, sequential (non-random) records
    let maxFrameIdx = null;

    for (let i = 1; i < data.length; i++) {
      const row = data[i];
      const isTest = testCol !== -1 ? (row[testCol] === true || row[testCol] === 'true') : false;
      const frameComplete = row[frameCompleteCol] === true || row[frameCompleteCol] === 'true';
      const frameIdx = parseInt(row[frameIdxCol]) || 0;
      const isRandomFrame = randomFrameCol !== -1 ? (row[randomFrameCol] === true || row[randomFrameCol] === 'true') : false;

      // Skip test records and random frame records - only consider sequential processing
      if (!isTest && !isRandomFrame && frameComplete && (maxFrameIdx === null || frameIdx > maxFrameIdx)) {
        maxFrameIdx = frameIdx;
      }
    }

    return ContentService
      .createTextOutput(JSON.stringify({
        bestFrameIdx: maxFrameIdx
      }))
      .setMimeType(ContentService.MimeType.JSON);
  });
}

/**
 * Check if a specific frame is complete
 * Returns true if there's a row with the given frameIdx and frameComplete=true
 */
function googleGetIsFrameComplete(e, spreadsheetId) {
  return withLock(() => {
    const data = e.parameter;
    const frameIdx = parseInt(data.frameIdx);

    if (isNaN(frameIdx)) {
      return ContentService
        .createTextOutput(JSON.stringify({
          isComplete: false,
          error: 'Invalid or missing frameIdx parameter'
        }))
        .setMimeType(ContentService.MimeType.JSON);
    }

    const spreadsheet = SpreadsheetApp.openById(spreadsheetId);
    const sheet = spreadsheet.getSheetByName(PROGRESS_SHEET_NAME);

    if (!sheet) {
      return ContentService
        .createTextOutput(JSON.stringify({
          isComplete: false,
          message: 'No progress sheet found'
        }))
        .setMimeType(ContentService.MimeType.JSON);
    }

    const dataRange = sheet.getDataRange().getValues();

    if (dataRange.length <= 1) {
      return ContentService
        .createTextOutput(JSON.stringify({
          isComplete: false,
          message: 'No data found'
        }))
        .setMimeType(ContentService.MimeType.JSON);
    }

    // Find column indices
    const headers = dataRange[0];
    const frameIdxCol = headers.indexOf('frameIdx');
    const frameCompleteCol = headers.indexOf('frameComplete');
    const testCol = headers.indexOf('test');

    if (frameIdxCol === -1 || frameCompleteCol === -1) {
      throw new Error('Required columns not found');
    }

    // Check if frame is complete
    for (let i = 1; i < dataRange.length; i++) {
      const row = dataRange[i];
      const rowFrameIdx = parseInt(row[frameIdxCol]) || 0;
      const isTest = testCol !== -1 ? (row[testCol] === true || row[testCol] === 'true') : false;
      const frameComplete = row[frameCompleteCol] === true || row[frameCompleteCol] === 'true';

      // Skip test records and check for matching frameIdx with frameComplete=true
      if (!isTest && rowFrameIdx === frameIdx && frameComplete) {
        return ContentService
          .createTextOutput(JSON.stringify({
            isComplete: true,
            frameIdx: frameIdx
          }))
          .setMimeType(ContentService.MimeType.JSON);
      }
    }

    // Frame not found as complete
    return ContentService
      .createTextOutput(JSON.stringify({
        isComplete: false,
        frameIdx: frameIdx,
        message: 'Frame not found or not complete'
      }))
      .setMimeType(ContentService.MimeType.JSON);
  });
}

/**
 * Get a bitmap of all completed frames for efficient caching
 * Returns a base64-encoded bitmap where each bit represents a frame's completion status
 */
function googleGetCompleteFrameCache(e, spreadsheetId) {
  return withLock(() => {
    const spreadsheet = SpreadsheetApp.openById(spreadsheetId);
    const sheet = spreadsheet.getSheetByName(PROGRESS_SHEET_NAME);

    if (!sheet) {
      return ContentService
        .createTextOutput(JSON.stringify({
          bitmap: "",
          totalFrames: 0,
          message: 'No progress sheet found'
        }))
        .setMimeType(ContentService.MimeType.JSON);
    }

    const dataRange = sheet.getDataRange().getValues();

    if (dataRange.length <= 1) {
      return ContentService
        .createTextOutput(JSON.stringify({
          bitmap: "",
          totalFrames: 0,
          message: 'No data found'
        }))
        .setMimeType(ContentService.MimeType.JSON);
    }

    // Find column indices
    const headers = dataRange[0];
    const frameIdxCol = headers.indexOf('frameIdx');
    const frameCompleteCol = headers.indexOf('frameComplete');
    const testCol = headers.indexOf('test');
    const randomFrameCol = headers.indexOf('randomFrame');

    if (frameIdxCol === -1 || frameCompleteCol === -1) {
      throw new Error('Required columns not found');
    }

    // Create bitmap for 2,102,800 frames (262,850 bytes)
    const TOTAL_FRAMES = 2102800;
    const bitmapBytes = Math.ceil(TOTAL_FRAMES / 8);
    const bitmap = new Uint8Array(bitmapBytes);

    // Process all rows and set bits for completed frames
    for (let i = 1; i < dataRange.length; i++) {
      const row = dataRange[i];
      const isTest = testCol !== -1 ? (row[testCol] === true || row[testCol] === 'true') : false;
      const isRandomFrame = randomFrameCol !== -1 ? (row[randomFrameCol] === true || row[randomFrameCol] === 'true') : false;
      const frameComplete = row[frameCompleteCol] === true || row[frameCompleteCol] === 'true';
      const frameIdx = parseInt(row[frameIdxCol]) || 0;

      // Skip test records - but include both sequential and random frame records
      if (!isTest && frameComplete && frameIdx >= 0 && frameIdx < TOTAL_FRAMES) {
        const byteIdx = Math.floor(frameIdx / 8);
        const bitIdx = frameIdx % 8;
        bitmap[byteIdx] |= (1 << bitIdx);
      }
    }

    // Convert bitmap to base64 for transmission
    const bitmapBase64 = Utilities.base64Encode(bitmap);

    return ContentService
      .createTextOutput(JSON.stringify({
        bitmap: bitmapBase64,
        totalFrames: TOTAL_FRAMES,
        bitmapSize: bitmapBytes
      }))
      .setMimeType(ContentService.MimeType.JSON);
  });
}
