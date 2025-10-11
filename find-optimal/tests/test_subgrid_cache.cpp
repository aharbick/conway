#include <gtest/gtest.h>

#include <climits>
#include <vector>

// Define CUDA decorators as empty for CPU compilation
#ifndef __NVCC__
#define __host__
#define __device__
#define __global__
#endif

#include "constants.h"
#include "gol_core.h"
#include "subgrid_cache.h"
#include "test_utils.h"

class SubgridCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Any setup needed before each test
  }
};

// ============================================================================
// expand7x7To8x8 Tests
// ============================================================================

TEST_F(SubgridCacheTest, Expand7x7To8x8_EmptyPattern) {
  uint64_t pattern7x7 = 0;

  // Test all 4 positions
  for (int pos = 0; pos < 4; pos++) {
    int rowOffset = (pos >= 2) ? 1 : 0;
    int colOffset = (pos & 1) ? 1 : 0;

    uint64_t result = expand7x7To8x8(pattern7x7, rowOffset, colOffset);
    EXPECT_EQ(result, 0ULL) << "Empty pattern should expand to empty at position " << pos;
  }
}

TEST_F(SubgridCacheTest, Expand7x7To8x8_Position0_TopLeftCorner) {
  // 7x7 pattern with single bit in top-left corner
  // clang-format off
  const char pattern7x7_grid[7][7] = {
    {'1', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0'}
  };

  const char expected8x8[8][8] = {
    {'1', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'}
  };
  // clang-format on

  uint64_t pattern7x7 = createPattern7x7(pattern7x7_grid);
  uint64_t result = expand7x7To8x8(pattern7x7, 0, 0);  // Position 0: rows 0-6, cols 0-6

  verifyPattern(result, expected8x8);
}

TEST_F(SubgridCacheTest, Expand7x7To8x8_Position1_TopRightCorner) {
  // 7x7 pattern with single bit in top-right corner
  // clang-format off
  const char pattern7x7_grid[7][7] = {
    {'0', '0', '0', '0', '0', '0', '1'},
    {'0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0'}
  };

  const char expected8x8[8][8] = {
    {'0', '0', '0', '0', '0', '0', '0', '1'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'}
  };
  // clang-format on

  uint64_t pattern7x7 = createPattern7x7(pattern7x7_grid);
  uint64_t result = expand7x7To8x8(pattern7x7, 0, 1);  // Position 1: rows 0-6, cols 1-7

  verifyPattern(result, expected8x8);
}

TEST_F(SubgridCacheTest, Expand7x7To8x8_Position2_BottomLeftCorner) {
  // 7x7 pattern with single bit in bottom-left corner
  // clang-format off
  const char pattern7x7_grid[7][7] = {
    {'0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0'},
    {'1', '0', '0', '0', '0', '0', '0'}
  };

  const char expected8x8[8][8] = {
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'1', '0', '0', '0', '0', '0', '0', '0'}
  };
  // clang-format on

  uint64_t pattern7x7 = createPattern7x7(pattern7x7_grid);
  uint64_t result = expand7x7To8x8(pattern7x7, 1, 0);  // Position 2: rows 1-7, cols 0-6

  verifyPattern(result, expected8x8);
}

TEST_F(SubgridCacheTest, Expand7x7To8x8_Position3_BottomRightCorner) {
  // 7x7 pattern with single bit in bottom-right corner
  // clang-format off
  const char pattern7x7_grid[7][7] = {
    {'0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '1'}
  };

  const char expected8x8[8][8] = {
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '1'}
  };
  // clang-format on

  uint64_t pattern7x7 = createPattern7x7(pattern7x7_grid);
  uint64_t result = expand7x7To8x8(pattern7x7, 1, 1);  // Position 3: rows 1-7, cols 1-7

  verifyPattern(result, expected8x8);
}

TEST_F(SubgridCacheTest, Expand7x7To8x8_Glider) {
  // 7x7 glider pattern
  // clang-format off
  const char pattern7x7_grid[7][7] = {
    {'0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '1', '0', '0', '0', '0'},
    {'0', '0', '0', '1', '0', '0', '0'},
    {'0', '1', '1', '1', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0'}
  };

  const char expected8x8[8][8] = {
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '1', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '1', '0', '0', '0', '0'},
    {'0', '1', '1', '1', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'}
  };
  // clang-format on

  uint64_t pattern7x7 = createPattern7x7(pattern7x7_grid);
  uint64_t result = expand7x7To8x8(pattern7x7, 0, 0);

  verifyPattern(result, expected8x8);
}

TEST_F(SubgridCacheTest, Expand7x7To8x8_FullPattern) {
  // All 1s in 7x7
  uint64_t pattern7x7 = (1ULL << 49) - 1;  // Lower 49 bits all set

  // Position 0: rows 0-6, cols 0-6
  // clang-format off
  const char expected_pos0[8][8] = {
    {'1', '1', '1', '1', '1', '1', '1', '0'},
    {'1', '1', '1', '1', '1', '1', '1', '0'},
    {'1', '1', '1', '1', '1', '1', '1', '0'},
    {'1', '1', '1', '1', '1', '1', '1', '0'},
    {'1', '1', '1', '1', '1', '1', '1', '0'},
    {'1', '1', '1', '1', '1', '1', '1', '0'},
    {'1', '1', '1', '1', '1', '1', '1', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'}
  };
  // clang-format on

  uint64_t result = expand7x7To8x8(pattern7x7, 0, 0);
  verifyPattern(result, expected_pos0);

  // Position 3: rows 1-7, cols 1-7
  // clang-format off
  const char expected_pos3[8][8] = {
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '1', '1', '1', '1', '1', '1', '1'},
    {'0', '1', '1', '1', '1', '1', '1', '1'},
    {'0', '1', '1', '1', '1', '1', '1', '1'},
    {'0', '1', '1', '1', '1', '1', '1', '1'},
    {'0', '1', '1', '1', '1', '1', '1', '1'},
    {'0', '1', '1', '1', '1', '1', '1', '1'},
    {'0', '1', '1', '1', '1', '1', '1', '1'}
  };
  // clang-format on

  result = expand7x7To8x8(pattern7x7, 1, 1);
  verifyPattern(result, expected_pos3);
}

TEST_F(SubgridCacheTest, Expand7x7To8x8_IgnoresHighBits) {
  // Pattern with bits set in the high bits (beyond 49 bits) should be ignored
  // Set bit 0 (valid) and bits 50-63 (invalid - should be ignored)
  uint64_t pattern7x7_with_junk = 1ULL | (0xFFFFULL << 50);

  // Should only see the bit at position 0 (top-left of 7x7 grid)
  // clang-format off
  const char expected8x8[8][8] = {
    {'1', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'}
  };
  // clang-format on

  uint64_t result = expand7x7To8x8(pattern7x7_with_junk, 0, 0);
  verifyPattern(result, expected8x8);

  // Test with a more complex pattern - valid 7x7 pattern plus junk in high bits
  // clang-format off
  const char valid7x7_grid[7][7] = {
    {'0', '0', '1', '0', '0', '0', '0'},
    {'0', '0', '0', '1', '0', '0', '0'},
    {'0', '1', '1', '1', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '0'}
  };
  // clang-format on

  uint64_t valid_pattern = createPattern7x7(valid7x7_grid);
  uint64_t pattern_with_junk = valid_pattern | (0xFFFFULL << 50);

  // Should produce identical result to pattern without junk
  uint64_t result_with_junk = expand7x7To8x8(pattern_with_junk, 0, 0);
  uint64_t result_clean = expand7x7To8x8(valid_pattern, 0, 0);

  EXPECT_EQ(result_with_junk, result_clean)
      << "Pattern with high bits set should produce same result as clean pattern";
}

TEST_F(SubgridCacheTest, Expand7x7To8x8_SpecificPattern_AllPositions) {
  // Test a specific 8x8 pattern extracted as 7x7 at all 4 positions
  // Original 8x8 pattern:
  // 00010010
  // 11111000
  // 00000100
  // 01010000
  // 11000100
  // 11100100
  // 11111010
  // 00000000

  // clang-format off
  const char pattern7x7_grid[7][7] = {
    {'0', '0', '0', '1', '0', '0', '1'},
    {'1', '1', '1', '1', '1', '0', '0'},
    {'0', '0', '0', '0', '0', '1', '0'},
    {'0', '1', '0', '1', '0', '0', '0'},
    {'1', '1', '0', '0', '0', '1', '0'},
    {'1', '1', '1', '0', '0', '1', '0'},
    {'1', '1', '1', '1', '1', '0', '1'}
  };

  const char expected8x8_pos0[8][8] = {
    {'0', '0', '0', '1', '0', '0', '1', '0'},
    {'1', '1', '1', '1', '1', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '1', '0', '0'},
    {'0', '1', '0', '1', '0', '0', '0', '0'},
    {'1', '1', '0', '0', '0', '1', '0', '0'},
    {'1', '1', '1', '0', '0', '1', '0', '0'},
    {'1', '1', '1', '1', '1', '0', '1', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'}
  };
  // clang-format on

  uint64_t pattern7x7 = createPattern7x7(pattern7x7_grid);
  uint64_t result = expand7x7To8x8(pattern7x7, 0, 0);
  verifyPattern(result, expected8x8_pos0);

  // clang-format off
  const char expected8x8_pos1[8][8] = {
    {'0', '0', '0', '1', '0', '0', '1', '0'},
    {'0', '1', '1', '1', '1', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '1', '0', '0'},
    {'0', '1', '0', '1', '0', '0', '0', '0'},
    {'0', '1', '0', '0', '0', '1', '0', '0'},
    {'0', '1', '1', '0', '0', '1', '0', '0'},
    {'0', '1', '1', '1', '1', '0', '1', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'}
  };
  // clang-format on

  result = expand7x7To8x8(pattern7x7, 0, 1);
  verifyPattern(result, expected8x8_pos1);

  // clang-format off
  const char expected8x8_pos2[8][8] = {
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'1', '1', '1', '1', '1', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '1', '0', '0'},
    {'0', '1', '0', '1', '0', '0', '0', '0'},
    {'1', '1', '0', '0', '0', '1', '0', '0'},
    {'1', '1', '1', '0', '0', '1', '0', '0'},
    {'1', '1', '1', '1', '1', '0', '1', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'}
  };
  // clang-format on

  result = expand7x7To8x8(pattern7x7, 1, 0);
  verifyPattern(result, expected8x8_pos2);

  // clang-format off
  const char expected8x8_pos3[8][8] = {
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '1', '1', '1', '1', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '1', '0', '0'},
    {'0', '1', '0', '1', '0', '0', '0', '0'},
    {'0', '1', '0', '0', '0', '1', '0', '0'},
    {'0', '1', '1', '0', '0', '1', '0', '0'},
    {'0', '1', '1', '1', '1', '0', '1', '0'},
    {'0', '0', '0', '0', '0', '0', '0', '0'}
  };
  // clang-format on

  result = expand7x7To8x8(pattern7x7, 1, 1);
  verifyPattern(result, expected8x8_pos3);
}

