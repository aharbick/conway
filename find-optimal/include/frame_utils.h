#ifndef _FRAME_UTILS_H_
#define _FRAME_UTILS_H_

#include <stdbool.h>

#include <cstdint>

#include "constants.h"

// See the algorithm described in PERFORMANCE under "Eliminating Rotations"

// Optimized bit manipulation functions using bit tricks instead of loops
// Ultra-fast 8x8 transpose using bit manipulation (18 operations total)
__host__ __device__ inline uint64_t transpose8x8(uint64_t x) {
  uint64_t t;
  t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AA;
  x = x ^ t ^ (t << 7);
  t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCC;
  x = x ^ t ^ (t << 14);
  t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0;
  x = x ^ t ^ (t << 28);
  return x;
}

// Optimized horizontal flip using bit reversal (12 operations total)
__host__ __device__ inline uint64_t flipHorizontal8x8(uint64_t x) {
  // Reverse bits in each 8-bit row using bit manipulation
  x = ((x & 0x0F0F0F0F0F0F0F0F) << 4) | ((x & 0xF0F0F0F0F0F0F0F0) >> 4);
  x = ((x & 0x3333333333333333) << 2) | ((x & 0xCCCCCCCCCCCCCCCC) >> 2);
  x = ((x & 0x5555555555555555) << 1) | ((x & 0xAAAAAAAAAAAAAAAA) >> 1);
  return x;
}

// Optimized vertical flip using bit manipulation (6 operations total)
__host__ __device__ inline uint64_t flipVertical8x8(uint64_t x) {
  // Swap rows: 0↔7, 1↔6, 2↔5, 3↔4
  // First swap 4-row blocks (top 4 rows ↔ bottom 4 rows)
  x = ((x & 0x00000000FFFFFFFF) << 32) | ((x & 0xFFFFFFFF00000000) >> 32);

  // Then swap 2-row blocks within each 4-row block
  x = ((x & 0x0000FFFF0000FFFF) << 16) | ((x & 0xFFFF0000FFFF0000) >> 16);

  // Finally swap 1-row blocks within each 2-row block
  x = ((x & 0x00FF00FF00FF00FF) << 8) | ((x & 0xFF00FF00FF00FF00) >> 8);

  return x;
}

// Rotates a pattern 90 degrees clockwise using optimized bit manipulation
__host__ __device__ inline uint64_t rotate90(uint64_t pattern) {
  return flipVertical8x8(transpose8x8(pattern));
}

// Reflects a pattern horizontally using optimized bit manipulation
__host__ __device__ inline uint64_t reflectHorizontal(uint64_t pattern) {
  return flipHorizontal8x8(pattern);
}

// Extract frame bits from a pattern
__host__ __device__ inline uint64_t extractFrame(uint64_t pattern) {
  // Frame mask - 1s for frame positions, 0s elsewhere
  static constexpr uint64_t FRAME_MASK = 0xE7ULL << 56 |  // FFFooFFF
                                         0xC3ULL << 48 |  // FFooooFF
                                         0x81ULL << 40 |  // FooooooF
                                                          // oooooooo
                                                          // oooooooo
                                         0x81ULL << 16 |  // FooooooF
                                         0xC3ULL << 8 |   // FFooooFF
                                         0xE7ULL;         // FFFooFFF

  return pattern & FRAME_MASK;
}

// Spreads 24 bits into frame positions
__host__ __device__ inline uint64_t spreadBitsToFrame(uint64_t bits) {
  uint64_t result = 0;
  int bitPos = 0;

  // Row offsets and patterns
  static const int offsets[] = {56, 48, 40, 16, 8, 0};  // Row bit offsets
  static const uint64_t patterns[] = {
      0xE7,  // FFFooFFF - 11100111
      0xC3,  // FFooooFF - 11000011
      0x81,  // FooooooF - 10000001
      0x81,  // FooooooF - 10000001
      0xC3,  // FFooooFF - 11000011
      0xE7   // FFFooFFF - 11100111
  };

  // Process each row
  for (int row = 0; row < 6; row++) {
    uint64_t pattern = patterns[row];
    int offset = offsets[row];

    // Process each bit in the pattern
    for (int pos = 7; pos >= 0; pos--) {
      if (pattern & (1ULL << pos)) {
        result |= ((bits >> bitPos++) & 1ULL) << (offset + pos);
      }
    }
  }

  return result;
}

// Returns true if this frame is the lexicographically minimal version
// among all its rotations and reflections
__host__ __device__ inline bool isMinimalFrame(uint64_t frame) {
  uint64_t min = frame;

  // Check all rotations
  uint64_t rotated = frame;
  for (int i = 0; i < 3; i++) {
    rotated = rotate90(rotated);
    if (rotated < min)
      return false;
  }

  // Check horizontal reflection and its rotations
  uint64_t reflected = reflectHorizontal(frame);
  if (reflected < min)
    return false;

  rotated = reflected;
  for (int i = 0; i < 3; i++) {
    rotated = rotate90(rotated);
    if (rotated < min)
      return false;
  }

  return true;
}

// Get the actual frame value for a given frame index
__host__ inline uint64_t getFrameByIndex(uint64_t frameIdx) {
  uint64_t currentIdx = 0;
  for (uint64_t i = 0; i < FRAME_SEARCH_MAX_FRAMES; ++i) {
    const uint64_t frame = spreadBitsToFrame(i);
    if (isMinimalFrame(frame)) {
      if (currentIdx == frameIdx) {
        return frame;
      }
      ++currentIdx;
    }
  }
  return 0;  // Frame not found
}

#endif