/**
 * Google Apps Script WebApp for Conway's Game of Life Optimization Progress Tracking
 */

const PROGRESS_SHEET_NAME = 'Progress';

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
      default:
        return ContentService
          .createTextOutput(JSON.stringify({
            error: 'Invalid action. Use "sendProgress", "getBestResult", or "getBestCompleteFrame"'
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
  try {
    // Use GET parameters only
    const data = e.parameter;
    
    // Extract parameters (matching the C++ function signature)
    const frameComplete = data.frameComplete === 'true';
    const frameIdx = parseInt(data.frameIdx) || 0;
    const kernelIdx = parseInt(data.kernelIdx) || 0;
    const chunkIdx = parseInt(data.chunkIdx) || 0;
    const patternsPerSecond = parseInt(data.patternsPerSecond) || 0;
    const bestGenerations = parseInt(data.bestGenerations) || 0;
    const bestPattern = data.bestPattern || '';
    const bestPatternBin = data.bestPatternBin || '';
    const isTest = data.test === 'true';
    
    // Get the spreadsheet and worksheet
    const spreadsheet = SpreadsheetApp.openById(spreadsheetId);
    let sheet = spreadsheet.getSheetByName(PROGRESS_SHEET_NAME);
    
    // Create sheet if it doesn't exist
    if (!sheet) {
      sheet = spreadsheet.insertSheet(PROGRESS_SHEET_NAME);
      // Add headers
      sheet.getRange(1, 1, 1, 10).setValues([[
        'timestamp', 'frameComplete', 'frameIdx', 'kernelIdx', 'chunkIdx',
        'patternsPerSecond', 'bestGenerations', 'bestPattern', 'bestPatternBin', 'test'
      ]]);
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
      isTest
    ];
    
    sheet.appendRow(newRow);
    
    return ContentService
      .createTextOutput(JSON.stringify({
        success: true,
        message: 'Progress data saved successfully',
        timestamp: timestamp
      }))
      .setMimeType(ContentService.MimeType.JSON);
      
  } catch (error) {
    return ContentService
      .createTextOutput(JSON.stringify({
        success: false,
        error: error.toString()
      }))
      .setMimeType(ContentService.MimeType.JSON);
  }
}

/**
 * Returns the highest bestGenerations value from non-test records
 */
function googleGetBestResult(e, spreadsheetId) {
  try {
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
      
      if (!isTest && generations > maxGenerations) {
        maxGenerations = generations;
      }
    }
    
    return ContentService
      .createTextOutput(JSON.stringify({
        bestGenerations: maxGenerations
      }))
      .setMimeType(ContentService.MimeType.JSON);
      
  } catch (error) {
    return ContentService
      .createTextOutput(JSON.stringify({
        bestGenerations: -1,
        error: error.toString()
      }))
      .setMimeType(ContentService.MimeType.JSON);
  }
}

/**
 * Returns the highest frameIdx value from completed, non-test records
 */
function googleGetBestCompleteFrame(e, spreadsheetId) {
  try {
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
    
    if (frameIdxCol === -1 || frameCompleteCol === -1) {
      throw new Error('Required columns not found');
    }
    
    // Find the maximum frameIdx value from completed, non-test records
    let maxFrameIdx = null;
    
    for (let i = 1; i < data.length; i++) {
      const row = data[i];
      const isTest = testCol !== -1 ? (row[testCol] === true || row[testCol] === 'true') : false;
      const frameComplete = row[frameCompleteCol] === true || row[frameCompleteCol] === 'true';
      const frameIdx = parseInt(row[frameIdxCol]) || 0;
      
      if (!isTest && frameComplete && (maxFrameIdx === null || frameIdx > maxFrameIdx)) {
        maxFrameIdx = frameIdx;
      }
    }
    
    return ContentService
      .createTextOutput(JSON.stringify({
        bestFrameIdx: maxFrameIdx
      }))
      .setMimeType(ContentService.MimeType.JSON);
      
  } catch (error) {
    return ContentService
      .createTextOutput(JSON.stringify({
        bestFrameIdx: null,
        error: error.toString()
      }))
      .setMimeType(ContentService.MimeType.JSON);
  }
}
