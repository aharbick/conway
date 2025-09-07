/**
 * Convert a 64-bit binary string (8x8 Conway's Game of Life pattern) into RLE format
 * suitable for copying and pasting into Conway's Game of Life simulators.
 * 
 * @param {string} binstring - 64-character binary string (0s and 1s) representing an 8x8 pattern
 * @returns {string} RLE formatted string with proper header
 */
function bestPatternBinToRLE(binstring) {
  if (!binstring || typeof binstring !== 'string') {
    return '#CXRLE Pos=-4,-4\nx = 8, y = 8, rule = B3/S23:P8,8\n!';
  }
  
  // Remove any spaces or non-binary characters and ensure it's exactly 64 characters
  const cleanBin = binstring.replace(/[^01]/g, '');
  if (cleanBin.length !== 64) {
    throw new Error('Binary string must be exactly 64 characters of 0/1 (8x8).');
  }

  // Split into 8 rows of 8 characters each
  const rows = [];
  for (let i = 0; i < 8; i++) {
    rows.push(cleanBin.slice(i * 8, (i + 1) * 8));
  }

  const rleRows = [];
  for (const row of rows) {
    // Trim trailing zeros so we don't emit trailing 'b' runs
    const lastOneIndex = row.lastIndexOf('1');
    if (lastOneIndex === -1) {
      // Entire row is blank → empty line; the '$' separator will represent it
      rleRows.push('');
      continue;
    }

    const segment = row.slice(0, lastOneIndex + 1);
    const out = [];
    let i = 0;
    while (i < segment.length) {
      const ch = segment[i];
      let j = i;
      while (j < segment.length && segment[j] === ch) {
        j++;
      }
      const runLength = j - i;
      if (runLength > 1) {
        out.push(runLength.toString());
      }
      out.push(ch === '1' ? 'o' : 'b');
      i = j;
    }
    rleRows.push(out.join(''));
  }

  const rleBody = rleRows.join('$') + '!';
  
  // Return with proper RLE header
  return '#CXRLE Pos=-4,-4\nx = 8, y = 8, rule = B3/S23:P8,8\n' + rleBody;
}

// For Google Apps Script usage, you might want a wrapper that works with sheet cell values
function PATTERN_TO_RLE(binstring) {
  try {
    return bestPatternBinToRLE(binstring);
  } catch (error) {
    return 'ERROR: ' + error.message;
  }
}

// Function that turns the binary pattern string into the ulong64 value
function PRECISE_BIN2DEC(binaryString) {
  if (!binaryString || binaryString.length === 0) return "";

  let result = BigInt(0);
  const binary = binaryString.toString().trim();

  for (let i = 0; i < binary.length; i++) {
    if (binary[i] === '1') {
      result += BigInt(2) ** BigInt(binary.length - 1 - i);
    }
  }

  return result.toString();
}

