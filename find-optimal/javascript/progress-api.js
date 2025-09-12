/**
 * Google Apps Script WebApp for Conway's Game of Life Optimization Progress Tracking
 */

const PROGRESS_SHEET_NAME = 'Progress';
const LOCK_TIMEOUT_MS = 30000; // 30 seconds timeout for locks

// Spreadsheet ID for our Progress data
const SPREADSHEET_ID = '1bXt22T9cyv1A1vcdozR54n-imEFdIhsMcvEhEUMlsmY';

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

    // Extract and validate API key
    const apiKey = data.apiKey;
    if (!apiKey) {
      return ContentService
        .createTextOutput(JSON.stringify({
          error: 'Missing required parameter: apiKey'
        }))
        .setMimeType(ContentService.MimeType.JSON);
    }

    // Validate API key
    if (apiKey !== AUTHORIZED_API_KEY) {
      return ContentService
        .createTextOutput(JSON.stringify({
          error: 'Invalid API key'
        }))
        .setMimeType(ContentService.MimeType.JSON);
    }

    const action = data.action;

    switch (action) {
      case 'sendProgress':
        return googleSendProgress(e, SPREADSHEET_ID);
      case 'getBestResult':
        return googleGetBestResult(e, SPREADSHEET_ID);
      case 'getCompleteFrameCache':
        return googleGetCompleteFrameCache(e, SPREADSHEET_ID);
      case 'getIncompleteFrames':
        return googleGetIncompleteFrames(e, SPREADSHEET_ID);
      default:
        return ContentService
          .createTextOutput(JSON.stringify({
            error: 'Invalid action. Use "sendProgress", "getBestResult", "getCompleteFrameCache", or "getIncompleteFrames"'
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

    return ContentService
      .createTextOutput(JSON.stringify({
        success: true,
        message: 'Progress data saved successfully'
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

    if (bestGenerationsCol === -1) {
      throw new Error('bestGenerations column not found');
    }

    // Find the maximum bestGenerations value
    let maxGenerations = 0;

    for (let i = 1; i < data.length; i++) {
      const row = data[i];
      const generations = parseInt(row[bestGenerationsCol]) || 0;

      if (generations > maxGenerations) {
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
    const kernelIdxCol = headers.indexOf('kernelIdx');

    if (frameIdxCol === -1 || kernelIdxCol === -1) {
      throw new Error('Required columns not found');
    }

    // Create bitmap for 2,102,800 frames (262,850 bytes)
    const TOTAL_FRAMES = 2102800;
    const bitmapBytes = Math.ceil(TOTAL_FRAMES / 8);
    const bitmap = new Uint8Array(bitmapBytes);

    // Process all rows and set bits for completed frames
    for (let i = 1; i < dataRange.length; i++) {
      const row = dataRange[i];
      const kernelIdx = parseInt(row[kernelIdxCol]) || 0;
      const frameIdx = parseInt(row[frameIdxCol]) || 0;

      // Frame is complete when kernelIdx == 15
      if (kernelIdx === 15 && frameIdx >= 0 && frameIdx < TOTAL_FRAMES) {
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

/**
 * Get frames that do NOT have all 16 kernelIds (0-15) completed
 * Uses the "Kernel Counts" pivot table sheet for efficient lookup
 * Returns array of frameIdx values
 */
function googleGetIncompleteFrames(e, spreadsheetId) {
  return withLock(() => {
    const spreadsheet = SpreadsheetApp.openById(spreadsheetId);
    const sheet = spreadsheet.getSheetByName('Kernel Counts');

    if (!sheet) {
      return ContentService
        .createTextOutput(JSON.stringify([]))
        .setMimeType(ContentService.MimeType.JSON);
    }

    const dataRange = sheet.getDataRange().getValues();

    if (dataRange.length <= 1) {
      return ContentService
        .createTextOutput(JSON.stringify([]))
        .setMimeType(ContentService.MimeType.JSON);
    }

    // Find column indices
    const headers = dataRange[0];
    const frameIdxCol = headers.indexOf('frameIdx');
    const kernelCountCol = headers.indexOf('kernelCount');

    if (frameIdxCol === -1 || kernelCountCol === -1) {
      throw new Error('Required columns (frameIdx, kernelCount) not found in Kernel Counts sheet');
    }

    // Find frames with incomplete kernel counts (< 16)
    const incompleteFrames = [];
    const expectedKernels = 16;

    for (let i = 1; i < dataRange.length; i++) {
      const row = dataRange[i];
      const frameIdx = parseInt(row[frameIdxCol]) || 0;
      const kernelCount = parseInt(row[kernelCountCol]) || 0;

      // Frame is incomplete if it has fewer than 16 kernels
      if (kernelCount < expectedKernels && frameIdx >= 0) {
        incompleteFrames.push(frameIdx);
      }
    }

    return ContentService
      .createTextOutput(JSON.stringify(incompleteFrames))
      .setMimeType(ContentService.MimeType.JSON);
  });
}
