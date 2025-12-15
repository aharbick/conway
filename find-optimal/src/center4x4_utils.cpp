// Center 4x4 symmetry reduction utilities for strip search
#include "cuda_utils.h"

#include "center4x4_utils.h"
#include "symmetry_utils.h"

// Global array of unique 4x4 center patterns
uint16_t g_unique4x4Centers[CENTER_4X4_TOTAL_UNIQUE];

// Initialize the g_unique4x4Centers array by enumerating all 2^16 patterns
// and keeping only those that are minimal under D4 symmetry.
void initializeUnique4x4Centers() {
  uint32_t count = 0;
  for (uint32_t pattern = 0; pattern < 65536 && count < CENTER_4X4_TOTAL_UNIQUE; pattern++) {
    if (isMinimal4x4((uint16_t)pattern)) {
      g_unique4x4Centers[count++] = (uint16_t)pattern;
    }
  }
}

// Get the 4x4 center pattern for a given index (0 to 8547)
uint16_t get4x4CenterByIndex(uint32_t idx) {
  return g_unique4x4Centers[idx];
}

// Extract the center 4x4 block from a 32-bit middle block
// The center is columns 2-5 (bits 2-5 per row), packed into 16 bits.
//
// Middle block layout (4 rows × 8 bits each):
//   Row 2: bits 24-31 (bits 26-29 are center columns 2-5)
//   Row 3: bits 16-23 (bits 18-21 are center columns 2-5)
//   Row 4: bits 8-15  (bits 10-13 are center columns 2-5)
//   Row 5: bits 0-7   (bits 2-5 are center columns 2-5)
//
// 4x4 center output (4 rows × 4 bits each):
//   Row 0: bits 0-3
//   Row 1: bits 4-7
//   Row 2: bits 8-11
//   Row 3: bits 12-15
uint16_t extractCenter4x4(uint32_t middleBlock) {
  // Extract center bits (columns 2-5) from each row and pack into 16 bits
  uint16_t row0 = (middleBlock >> 26) & 0x0F;  // Middle row 2 → center row 0
  uint16_t row1 = (middleBlock >> 18) & 0x0F;  // Middle row 3 → center row 1
  uint16_t row2 = (middleBlock >> 10) & 0x0F;  // Middle row 4 → center row 2
  uint16_t row3 = (middleBlock >> 2) & 0x0F;   // Middle row 5 → center row 3

  return row0 | (row1 << 4) | (row2 << 8) | (row3 << 12);
}

// Extract the left ear (columns 6-7) from a 32-bit middle block
// Returns 8 bits: 2 bits per row × 4 rows
//
// Left ear output:
//   Row 0: bits 0-1
//   Row 1: bits 2-3
//   Row 2: bits 4-5
//   Row 3: bits 6-7
uint8_t extractLeftEar(uint32_t middleBlock) {
  uint8_t row0 = (middleBlock >> 30) & 0x03;  // Middle row 2, columns 6-7
  uint8_t row1 = (middleBlock >> 22) & 0x03;  // Middle row 3, columns 6-7
  uint8_t row2 = (middleBlock >> 14) & 0x03;  // Middle row 4, columns 6-7
  uint8_t row3 = (middleBlock >> 6) & 0x03;   // Middle row 5, columns 6-7

  return row0 | (row1 << 2) | (row2 << 4) | (row3 << 6);
}

// Extract the right ear (columns 0-1) from a 32-bit middle block
// Returns 8 bits: 2 bits per row × 4 rows
//
// Right ear output:
//   Row 0: bits 0-1
//   Row 1: bits 2-3
//   Row 2: bits 4-5
//   Row 3: bits 6-7
uint8_t extractRightEar(uint32_t middleBlock) {
  uint8_t row0 = (middleBlock >> 24) & 0x03;  // Middle row 2, columns 0-1
  uint8_t row1 = (middleBlock >> 16) & 0x03;  // Middle row 3, columns 0-1
  uint8_t row2 = (middleBlock >> 8) & 0x03;   // Middle row 4, columns 0-1
  uint8_t row3 = middleBlock & 0x03;          // Middle row 5, columns 0-1

  return row0 | (row1 << 2) | (row2 << 4) | (row3 << 6);
}

// Reconstruct a 32-bit middle block from its components
//
// Input formats:
//   center4x4: 16 bits (4 rows × 4 bits, row 0 in bits 0-3)
//   leftEar: 8 bits (4 rows × 2 bits, row 0 in bits 0-1)
//   rightEar: 8 bits (4 rows × 2 bits, row 0 in bits 0-1)
//
// Output format (middle block):
//   Row 2: bits 24-31 (MSB)
//   Row 3: bits 16-23
//   Row 4: bits 8-15
//   Row 5: bits 0-7 (LSB)
//
// Within each 8-bit row:
//   7  6  5  4  3  2  1  0   (bit positions)
//  [L][L][C][C][C][C][R][R]
uint32_t reconstructMiddleBlock(uint16_t center4x4, uint8_t leftEar, uint8_t rightEar) {
  uint32_t result = 0;

  // Process each of the 4 rows
  for (int row = 0; row < 4; row++) {
    uint32_t centerBits = (center4x4 >> (row * 4)) & 0x0F;  // 4 bits for this row
    uint32_t leftBits = (leftEar >> (row * 2)) & 0x03;      // 2 bits for this row
    uint32_t rightBits = (rightEar >> (row * 2)) & 0x03;    // 2 bits for this row

    // Combine: [left 2 bits][center 4 bits][right 2 bits]
    uint32_t rowValue = (leftBits << 6) | (centerBits << 2) | rightBits;

    // Place in correct position (row 0 → bits 24-31, row 3 → bits 0-7)
    result |= rowValue << ((3 - row) * 8);
  }

  return result;
}
