#ifndef _CENTER4X4_UTILS_H_
#define _CENTER4X4_UTILS_H_

#include <cstdint>
#include "cuda_utils.h"
#include "gol_core.h"

// Center 4x4 symmetry reduction for strip search
//
// The middle 8x4 block (rows 2-5) is split into:
//   - Left ear:  columns 6-7 (bits 6-7 per row) = 8 bits total
//   - Center:    columns 2-5 (bits 2-5 per row) = 16 bits total
//   - Right ear: columns 0-1 (bits 0-1 per row) = 8 bits total
//
// By applying D4 symmetry reduction to the center 4x4 block, we reduce
// the search space from 2^32 to 8548 × 256 × 256 = ~560M combinations.
//
// Bit layout within 32-bit middle block (rows 2-5):
//   Row 2: bits 24-31 (MSB)
//   Row 3: bits 16-23
//   Row 4: bits 8-15
//   Row 5: bits 0-7 (LSB)
//
// Within each 8-bit row:
//   7  6  5  4  3  2  1  0   (bit positions)
//  [L][L][C][C][C][C][R][R]

// Number of unique 4x4 binary matrices under D4 symmetry (rotations + reflections)
// From Burnside's lemma: (2^16 + 2×2^8 + 2×2^10 + 2×2^4 + 2^8) / 8 = 8548
// Reference: https://oeis.org/A054247
#define CENTER_4X4_TOTAL_UNIQUE 8548

// Extraction masks for middle block components
#define MIDDLE_LEFT_EAR_MASK   0xC0C0C0C0UL  // bits 6-7 of each row
#define MIDDLE_CENTER_MASK     0x3C3C3C3CUL  // bits 2-5 of each row
#define MIDDLE_RIGHT_EAR_MASK  0x03030303UL  // bits 0-1 of each row

// Global array of unique 4x4 center patterns (populated at startup)
extern uint16_t g_unique4x4Centers[CENTER_4X4_TOTAL_UNIQUE];

// Initialize the g_unique4x4Centers array by enumerating all 2^16 patterns
// and keeping only those that are minimal under D4 symmetry.
// Must be called before any strip search operations.
void initializeUnique4x4Centers();

// Get the 4x4 center pattern for a given index (0 to 8547)
// Returns the pattern from g_unique4x4Centers array.
uint16_t get4x4CenterByIndex(uint32_t idx);

// Extract the center 4x4 block from a 32-bit middle block
// The center is columns 2-5 (bits 2-5 per row), packed into 16 bits.
uint16_t extractCenter4x4(uint32_t middleBlock);

// Extract the left ear (columns 6-7) from a 32-bit middle block
// Returns 8 bits: 2 bits per row × 4 rows
uint8_t extractLeftEar(uint32_t middleBlock);

// Extract the right ear (columns 0-1) from a 32-bit middle block
// Returns 8 bits: 2 bits per row × 4 rows
uint8_t extractRightEar(uint32_t middleBlock);

// Reconstruct a 32-bit middle block from its components
// center4x4: 16-bit center block (columns 2-5)
// leftEar: 8-bit left ear (columns 6-7)
// rightEar: 8-bit right ear (columns 0-1)
uint32_t reconstructMiddleBlock(uint16_t center4x4, uint8_t leftEar, uint8_t rightEar);

// ============================================================================
// Strip Search Signature Functions
// ============================================================================

// CityHash-inspired hash function for 32-bit signatures
// Provides much better distribution than simple modulo
__host__ __device__ static inline uint32_t hashSignature(uint32_t sig) {
  sig *= 0x9ddfea08U;  // CityHash-style multiply
  sig ^= sig >> 16;    // Mix high bits into low bits
  return sig;
}

// Compute 2-generation signature for strip deduplication
// Strips that produce the same signature are equivalent for search purposes.
// The signature captures how the strip affects the "relevant half" of the grid
// after 2 generations (due to speed-of-light causality).
//
// Parameters:
//   pattern: Full 64-bit pattern with strip placed (strip + middle block)
//   isTop: true = top strip (signature from bottom 4 rows of gen2)
//          false = bottom strip (signature from top 4 rows of gen2)
//
// Returns: 32-bit signature for deduplication
__host__ __device__ static inline uint32_t computeStripSignature(uint64_t pattern, bool isTop) {
  uint64_t gen1 = computeNextGeneration8x8(pattern);
  uint64_t gen2 = computeNextGeneration8x8(gen1);
  return isTop ? (uint32_t)(gen2 & 0xFFFFFFFFULL)   // Bottom 4 rows (affects middle)
               : (uint32_t)(gen2 >> 32);            // Top 4 rows (affects middle)
}

// Assemble a full 64-bit pattern from strip search components
// Layout: [top strip 16 bits][middle block 32 bits][bottom strip 16 bits]
//         bits 0-15          bits 16-47            bits 48-63
__host__ __device__ static inline uint64_t assembleStripPattern(
    uint16_t topStrip,
    uint32_t middleBlock,
    uint16_t bottomStrip
) {
  return (uint64_t)topStrip |
         ((uint64_t)middleBlock << 16) |
         ((uint64_t)bottomStrip << 48);
}

// Extract top strip from assembled pattern
__host__ __device__ static inline uint16_t extractTopStrip(uint64_t pattern) {
  return (uint16_t)(pattern & 0xFFFF);
}

// Extract middle block from assembled pattern
__host__ __device__ static inline uint32_t extractMiddleBlock(uint64_t pattern) {
  return (uint32_t)((pattern >> 16) & 0xFFFFFFFF);
}

// Extract bottom strip from assembled pattern
__host__ __device__ static inline uint16_t extractBottomStrip(uint64_t pattern) {
  return (uint16_t)(pattern >> 48);
}

#endif
