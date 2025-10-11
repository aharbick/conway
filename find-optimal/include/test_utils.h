#ifndef _TEST_UTILS_H_
#define _TEST_UTILS_H_

#include <cstdint>
#include <gtest/gtest.h>

// Shared test utility functions

// Helper function to create a pattern from a 2D char array
inline uint64_t createPattern(const char grid[8][8]) {
  uint64_t pattern = 0;
  for (int row = 0; row < 8; row++) {
    for (int col = 0; col < 8; col++) {
      if (grid[row][col] == '1') {
        pattern |= 1ULL << (row * 8 + col);
      }
    }
  }
  return pattern;
}

// Helper function to verify a pattern matches expected grid
inline void verifyPattern(uint64_t pattern, const char expected[8][8]) {
  for (int row = 0; row < 8; row++) {
    for (int col = 0; col < 8; col++) {
      bool bitSet = (pattern & (1ULL << (row * 8 + col))) != 0;
      bool expectedSet = (expected[row][col] == '1');
      EXPECT_EQ(bitSet, expectedSet) << "Mismatch at (" << row << "," << col << ")";
    }
  }
}

// Helper function to create a 7x7 pattern from a 2D char array
inline uint64_t createPattern7x7(const char grid[7][7]) {
  uint64_t pattern = 0;
  for (int row = 0; row < 7; row++) {
    for (int col = 0; col < 7; col++) {
      if (grid[row][col] == '1') {
        pattern |= 1ULL << (row * 7 + col);
      }
    }
  }
  return pattern;
}

#endif // _TEST_UTILS_H_