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
    {'0', '0', '0', '0', '1', '0', '0', '1'},
    {'0', '1', '1', '1', '1', '1', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '1', '0'},
    {'0', '0', '1', '0', '1', '0', '0', '0'},
    {'0', '1', '1', '0', '0', '0', '1', '0'},
    {'0', '1', '1', '1', '0', '0', '1', '0'},
    {'0', '1', '1', '1', '1', '1', '0', '1'},
    {'0', '0', '0', '0', '0', '0', '0', '0'}
  };
  // clang-format on

  result = expand7x7To8x8(pattern7x7, 0, 1);
  verifyPattern(result, expected8x8_pos1);

  // clang-format off
  const char expected8x8_pos2[8][8] = {
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '1', '0', '0', '1', '0'},
    {'1', '1', '1', '1', '1', '0', '0', '0'},
    {'0', '0', '0', '0', '0', '1', '0', '0'},
    {'0', '1', '0', '1', '0', '0', '0', '0'},
    {'1', '1', '0', '0', '0', '1', '0', '0'},
    {'1', '1', '1', '0', '0', '1', '0', '0'},
    {'1', '1', '1', '1', '1', '0', '1', '0'}
  };
  // clang-format on

  result = expand7x7To8x8(pattern7x7, 1, 0);
  verifyPattern(result, expected8x8_pos2);

  // clang-format off
  const char expected8x8_pos3[8][8] = {
    {'0', '0', '0', '0', '0', '0', '0', '0'},
    {'0', '0', '0', '0', '1', '0', '0', '1'},
    {'0', '1', '1', '1', '1', '1', '0', '0'},
    {'0', '0', '0', '0', '0', '0', '1', '0'},
    {'0', '0', '1', '0', '1', '0', '0', '0'},
    {'0', '1', '1', '0', '0', '0', '1', '0'},
    {'0', '1', '1', '1', '0', '0', '1', '0'},
    {'0', '1', '1', '1', '1', '1', '0', '1'}
  };
  // clang-format on

  result = expand7x7To8x8(pattern7x7, 1, 1);
  verifyPattern(result, expected8x8_pos3);
}

// ============================================================================
// findSubgridCandidates Tests
// ============================================================================

// CPU wrapper that mimics the CUDA kernel behavior for testing
void findSubgridCandidatesCPU(uint64_t rangeStart, uint64_t rangeEnd,
                              std::vector<SubgridCacheEntry>& candidates,
                              CycleDetectionAlgorithm algorithm, int minGenerations = SUBGRID_MIN_GENERATIONS) {
  candidates.clear();

  // Process each 7x7 pattern in range
  for (uint64_t pattern7x7 = rangeStart; pattern7x7 < rangeEnd; pattern7x7++) {
    // Test all 4 possible positions of 7x7 within 8x8
    for (int pos = 0; pos < 4; pos++) {
      int rowOffset = (pos >= 2) ? 1 : 0;
      int colOffset = (pos & 1) ? 1 : 0;

      uint64_t pattern8x8 = expand7x7To8x8(pattern7x7, rowOffset, colOffset);

      // Count generations for this 8x8 pattern
      int gens = countGenerations(pattern8x8, algorithm);

      // Save each 8x8 pattern that meets the threshold
      if (gens >= minGenerations) {
        SubgridCacheEntry entry;
        entry.pattern = pattern8x8;
        entry.generations = gens;
        candidates.push_back(entry);
      }
    }
  }
}

TEST_F(SubgridCacheTest, FindSubgridCandidates_SinglePattern_All4Positions) {
  // Pattern 1: single bit in top-left of 7x7, should fit in all 4 positions
  std::vector<SubgridCacheEntry> candidates;
  findSubgridCandidatesCPU(1, 2, candidates, CYCLE_DETECTION_FLOYD, 1);

  // Should produce exactly 4 candidates (one for each position)
  EXPECT_EQ(candidates.size(), 4) << "Pattern 1 should produce 4 candidates";

  // Verify each candidate has correct expansion
  std::vector<uint64_t> expectedPatterns;
  for (int pos = 0; pos < 4; pos++) {
    int rowOffset = (pos >= 2) ? 1 : 0;
    int colOffset = (pos & 1) ? 1 : 0;
    expectedPatterns.push_back(expand7x7To8x8(1, rowOffset, colOffset));
  }

  for (size_t i = 0; i < candidates.size(); i++) {
    EXPECT_TRUE(std::find(expectedPatterns.begin(), expectedPatterns.end(),
                          candidates[i].pattern) != expectedPatterns.end())
        << "Candidate " << i << " pattern not found in expected patterns";
    EXPECT_GE(candidates[i].generations, 1);
  }
}

TEST_F(SubgridCacheTest, FindSubgridCandidates_RangeWithLowThreshold) {
  // Test range 1-256 with minGenerations=1
  // Should produce <= 256*4 candidates (some patterns die or don't fit)
  std::vector<SubgridCacheEntry> candidates;
  findSubgridCandidatesCPU(1, 256, candidates, CYCLE_DETECTION_FLOYD, 1);

  EXPECT_LE(candidates.size(), 256 * 4)
      << "Should not exceed maximum possible candidates";

  // Verify all candidates meet minimum threshold
  for (const auto& entry : candidates) {
    EXPECT_GE(entry.generations, 1);
  }
}

TEST_F(SubgridCacheTest, FindSubgridCandidates_EmptyPattern) {
  // Pattern 0 (all zeros) should die immediately
  std::vector<SubgridCacheEntry> candidates;
  findSubgridCandidatesCPU(0, 1, candidates, CYCLE_DETECTION_FLOYD, 1);

  EXPECT_EQ(candidates.size(), 0)
      << "Empty pattern should produce 0 candidates";
}

TEST_F(SubgridCacheTest, FindSubgridCandidates_EdgePatternFiltering) {
  // Create a 7x7 pattern with rightmost column set (bit 6 of each row)
  // This pattern should only fit in positions 0 and 2 (not 1 and 3)
  uint64_t pattern7x7 = 0;
  for (int row = 0; row < 7; row++) {
    pattern7x7 |= (1ULL << (row * 7 + 6));  // Set rightmost bit of each row
  }

  // Get the actual pattern number by encoding
  uint64_t patternNum = pattern7x7;

  std::vector<SubgridCacheEntry> candidates;
  findSubgridCandidatesCPU(patternNum, patternNum + 1, candidates,
                          CYCLE_DETECTION_FLOYD, 1);

  // Should produce <= 4 candidates (some positions won't work due to edge bits)
  EXPECT_LE(candidates.size(), 4);

  // Verify candidates are valid expansions
  for (const auto& entry : candidates) {
    bool found = false;
    for (int pos = 0; pos < 4; pos++) {
      int rowOffset = (pos >= 2) ? 1 : 0;
      int colOffset = (pos & 1) ? 1 : 0;
      if (entry.pattern == expand7x7To8x8(patternNum, rowOffset, colOffset)) {
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found) << "Candidate pattern must match one of the 4 expansions";
  }
}

TEST_F(SubgridCacheTest, FindSubgridCandidates_HighGenerationThreshold) {
  // Known pattern: 7x7 binary 0100001101011100110011101010110000010001111011101
  // This equals 148131766871005 in decimal
  // When expanded to 8x8 gives 18768881803890362 with 206 generations
  uint64_t pattern7x7 = 148131766871005ULL;
  uint64_t target8x8 = 18768881803890362ULL;

  // Test with minGenerations=207 (> 206) - should NOT find the pattern
  std::vector<SubgridCacheEntry> candidates;
  findSubgridCandidatesCPU(pattern7x7, pattern7x7 + 1, candidates,
                          CYCLE_DETECTION_FLOYD, 207);

  bool foundTarget = false;
  for (const auto& entry : candidates) {
    if (entry.pattern == target8x8) {
      foundTarget = true;
    }
  }
  EXPECT_FALSE(foundTarget)
      << "Pattern with 206 generations should NOT be found with minGenerations=207";

  // Test with minGenerations=206 - SHOULD find the pattern
  candidates.clear();
  findSubgridCandidatesCPU(pattern7x7, pattern7x7 + 1, candidates,
                          CYCLE_DETECTION_FLOYD, 206);

  foundTarget = false;
  for (const auto& entry : candidates) {
    if (entry.pattern == target8x8) {
      foundTarget = true;
      EXPECT_EQ(entry.generations, 206)
          << "Generation count should be exactly 206";
    }
  }
  EXPECT_TRUE(foundTarget)
      << "Pattern with 206 generations should be found with minGenerations=206";

  // Test with minGenerations=1 - should definitely find it
  candidates.clear();
  findSubgridCandidatesCPU(pattern7x7, pattern7x7 + 1, candidates,
                          CYCLE_DETECTION_FLOYD, 1);

  foundTarget = false;
  for (const auto& entry : candidates) {
    if (entry.pattern == target8x8) {
      foundTarget = true;
      EXPECT_EQ(entry.generations, 206);
    }
  }
  EXPECT_TRUE(foundTarget)
      << "Pattern should be found with low threshold";
}

TEST_F(SubgridCacheTest, FindSubgridCandidates_ExactPatternVerification) {
  // Test that stored patterns exactly match expand7x7To8x8 output
  std::vector<SubgridCacheEntry> candidates;
  findSubgridCandidatesCPU(1, 10, candidates, CYCLE_DETECTION_FLOYD, 1);

  // Verify each candidate matches an expansion
  for (const auto& entry : candidates) {
    bool foundMatch = false;

    // Check all possible 7x7 patterns in range and all positions
    for (uint64_t pattern7x7 = 1; pattern7x7 < 10; pattern7x7++) {
      for (int pos = 0; pos < 4; pos++) {
        int rowOffset = (pos >= 2) ? 1 : 0;
        int colOffset = (pos & 1) ? 1 : 0;

        uint64_t expected = expand7x7To8x8(pattern7x7, rowOffset, colOffset);
        if (entry.pattern == expected) {
          foundMatch = true;
          // Verify generation count matches
          int expectedGens = countGenerations(expected, CYCLE_DETECTION_FLOYD);
          EXPECT_EQ(entry.generations, expectedGens)
              << "Generation count mismatch for pattern " << pattern7x7
              << " at position " << pos;
          break;
        }
      }
      if (foundMatch) break;
    }

    EXPECT_TRUE(foundMatch)
        << "Each candidate must match an expand7x7To8x8 output";
  }
}

TEST_F(SubgridCacheTest, FindSubgridCandidates_EmptyRange) {
  // Test empty range (rangeStart == rangeEnd)
  std::vector<SubgridCacheEntry> candidates;
  findSubgridCandidatesCPU(100, 100, candidates, CYCLE_DETECTION_FLOYD, 1);

  EXPECT_EQ(candidates.size(), 0)
      << "Empty range should produce 0 candidates";
}

