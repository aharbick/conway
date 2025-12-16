#include <gtest/gtest.h>
#include <set>
#include <vector>

#include "cuda_utils.h"
#include "center4x4_utils.h"
#include "symmetry_utils.h"
#include "constants.h"

// Test strip search utilities
class StripSearchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize the center lookup table before tests
    initializeUnique4x4Centers();
  }
};

// ============================================================================
// Middle Block Reconstruction Tests
// ============================================================================

TEST_F(StripSearchTest, ReconstructMiddleBlockZeros) {
  // All zeros should produce zero
  uint32_t result = reconstructMiddleBlock(0, 0, 0);
  EXPECT_EQ(result, 0);
}

TEST_F(StripSearchTest, ReconstructMiddleBlockAllOnes) {
  // All ones should produce all ones
  uint32_t result = reconstructMiddleBlock(0xFFFF, 0xFF, 0xFF);
  EXPECT_EQ(result, 0xFFFFFFFF);
}

TEST_F(StripSearchTest, ReconstructMiddleBlockCenterOnly) {
  // Only center bits set - check that they land in columns 2-5
  uint32_t result = reconstructMiddleBlock(0xFFFF, 0, 0);
  EXPECT_EQ(result & MIDDLE_CENTER_MASK, result);  // Only center bits should be set
  EXPECT_EQ(result & MIDDLE_LEFT_EAR_MASK, 0);     // No left ear bits
  EXPECT_EQ(result & MIDDLE_RIGHT_EAR_MASK, 0);    // No right ear bits
}

TEST_F(StripSearchTest, ReconstructMiddleBlockLeftEarOnly) {
  // Only left ear bits set - check that they land in columns 6-7
  uint32_t result = reconstructMiddleBlock(0, 0xFF, 0);
  EXPECT_EQ(result & MIDDLE_LEFT_EAR_MASK, result);  // Only left ear bits should be set
  EXPECT_EQ(result & MIDDLE_CENTER_MASK, 0);         // No center bits
  EXPECT_EQ(result & MIDDLE_RIGHT_EAR_MASK, 0);      // No right ear bits
}

TEST_F(StripSearchTest, ReconstructMiddleBlockRightEarOnly) {
  // Only right ear bits set - check that they land in columns 0-1
  uint32_t result = reconstructMiddleBlock(0, 0, 0xFF);
  EXPECT_EQ(result & MIDDLE_RIGHT_EAR_MASK, result);  // Only right ear bits should be set
  EXPECT_EQ(result & MIDDLE_CENTER_MASK, 0);          // No center bits
  EXPECT_EQ(result & MIDDLE_LEFT_EAR_MASK, 0);        // No left ear bits
}

TEST_F(StripSearchTest, ExtractCenterRoundTrip) {
  // Test that extract/reconstruct round-trips correctly for all 2^16 center values
  for (uint32_t center = 0; center < STRIP_SEARCH_TOTAL_STRIPS; center++) {
    uint32_t middleBlock = reconstructMiddleBlock((uint16_t)center, 0, 0);
    uint16_t extracted = extractCenter4x4(middleBlock);
    EXPECT_EQ(extracted, center) << "Center round-trip failed for " << center;
  }
}

TEST_F(StripSearchTest, FullReconstructExtractRoundTrip) {
  // Exhaustively test all combinations: 8548 unique centers × 256 left ears × 256 right ears
  // This matches our actual search space (~560M combinations)
  for (uint32_t centerIdx = 0; centerIdx < CENTER_4X4_TOTAL_UNIQUE; centerIdx++) {
    uint16_t center = get4x4CenterByIndex(centerIdx);

    for (uint32_t leftEar = 0; leftEar < CENTER_4X4_TOTAL_EAR_VALUES; leftEar++) {
      for (uint32_t rightEar = 0; rightEar < CENTER_4X4_TOTAL_EAR_VALUES; rightEar++) {
        uint32_t middleBlock = reconstructMiddleBlock(center, (uint8_t)leftEar, (uint8_t)rightEar);

        EXPECT_EQ(extractCenter4x4(middleBlock), center)
            << "Center extraction failed for centerIdx=" << centerIdx
            << " leftEar=" << leftEar << " rightEar=" << rightEar;
        EXPECT_EQ(extractLeftEar(middleBlock), (uint8_t)leftEar)
            << "Left ear extraction failed for centerIdx=" << centerIdx
            << " leftEar=" << leftEar << " rightEar=" << rightEar;
        EXPECT_EQ(extractRightEar(middleBlock), (uint8_t)rightEar)
            << "Right ear extraction failed for centerIdx=" << centerIdx
            << " leftEar=" << leftEar << " rightEar=" << rightEar;
      }
    }
  }
}

// ============================================================================
// Center Index Tests
// ============================================================================

TEST_F(StripSearchTest, Get4x4CenterByIndexBounds) {
  // Test boundary indices
  uint16_t first = get4x4CenterByIndex(0);
  uint16_t last = get4x4CenterByIndex(CENTER_4X4_TOTAL_UNIQUE - 1);

  // First should be 0 (empty pattern is minimal)
  EXPECT_EQ(first, 0);

  // Last should be non-zero and less than 2^16
  EXPECT_LT(last, STRIP_SEARCH_TOTAL_STRIPS);
}

TEST_F(StripSearchTest, Get4x4CenterByIndexAllUnique) {
  // Verify that all 8548 indices return unique patterns
  std::set<uint16_t> seen;

  for (uint32_t idx = 0; idx < CENTER_4X4_TOTAL_UNIQUE; idx++) {
    uint16_t center = get4x4CenterByIndex(idx);

    EXPECT_EQ(seen.count(center), 0)
        << "Duplicate center pattern " << center << " at index " << idx;
    seen.insert(center);
  }

  EXPECT_EQ(seen.size(), CENTER_4X4_TOTAL_UNIQUE)
      << "Should have exactly " << CENTER_4X4_TOTAL_UNIQUE << " unique patterns";
}

TEST_F(StripSearchTest, Get4x4CenterByIndexAllMinimal) {
  // Verify that all returned patterns are minimal under D4 symmetry
  for (uint32_t idx = 0; idx < CENTER_4X4_TOTAL_UNIQUE; idx++) {
    uint16_t center = get4x4CenterByIndex(idx);
    EXPECT_TRUE(isMinimal4x4(center))
        << "Pattern at index " << idx << " (value " << center << ") is not minimal";
  }
}

TEST_F(StripSearchTest, Get4x4CenterByIndexMonotonic) {
  // Verify that indices are in ascending order of pattern value
  uint16_t prev = get4x4CenterByIndex(0);
  for (uint32_t idx = 1; idx < CENTER_4X4_TOTAL_UNIQUE; idx++) {
    uint16_t curr = get4x4CenterByIndex(idx);
    EXPECT_LT(prev, curr)
        << "Patterns not in ascending order at index " << idx;
    prev = curr;
  }
}

// ============================================================================
// Coverage Tests
// ============================================================================

TEST_F(StripSearchTest, AllMinimalPatternsAreCovered) {
  // Verify that every minimal 4x4 pattern appears in the lookup table
  std::set<uint16_t> inTable;
  for (uint32_t idx = 0; idx < CENTER_4X4_TOTAL_UNIQUE; idx++) {
    inTable.insert(get4x4CenterByIndex(idx));
  }

  uint32_t missedCount = 0;
  for (uint32_t pattern = 0; pattern < STRIP_SEARCH_TOTAL_STRIPS; pattern++) {
    if (isMinimal4x4((uint16_t)pattern)) {
      if (inTable.count((uint16_t)pattern) == 0) {
        missedCount++;
        if (missedCount <= 5) {
          ADD_FAILURE() << "Minimal pattern " << pattern << " not found in lookup table";
        }
      }
    }
  }

  EXPECT_EQ(missedCount, 0)
      << "Lookup table is missing " << missedCount << " minimal patterns";
}

TEST_F(StripSearchTest, IterationCoversTotalMiddleBlocks) {
  // Verify that the total iteration count matches expected
  // 8548 centers × 256 left ears × 256 right ears = 560,201,728
  const uint64_t expected = (uint64_t)CENTER_4X4_TOTAL_UNIQUE * CENTER_4X4_TOTAL_EAR_VALUES * CENTER_4X4_TOTAL_EAR_VALUES;
  EXPECT_EQ(expected, 560201728ULL);

  // Also verify via constants
  EXPECT_EQ((uint64_t)CENTER_4X4_TOTAL_UNIQUE * CENTER_4X4_EARS_PER_CENTER, expected);
}

// ============================================================================
// Specific Pattern Tests
// ============================================================================

TEST_F(StripSearchTest, KnownPatternMiddleBlockReconstruction) {
  // Test with a known pattern from actual search
  // Pattern: 0001001111111001000001000101000011000100111001011111101001101001
  // This is a 64-bit pattern, extract the middle 32 bits (rows 2-5)

  // Binary: 0001001111111001 000001000101000 0 11000100111001011111101001101001
  // Middle block (bits 16-47): extract by shifting and masking
  uint64_t fullPattern = 0x13F9051184E5FA69ULL;
  uint32_t middleBlock = (uint32_t)((fullPattern >> 16) & 0xFFFFFFFF);

  // Extract and verify components
  uint16_t center = extractCenter4x4(middleBlock);
  uint8_t leftEar = extractLeftEar(middleBlock);
  uint8_t rightEar = extractRightEar(middleBlock);

  // Reconstruct and verify
  uint32_t reconstructed = reconstructMiddleBlock(center, leftEar, rightEar);
  EXPECT_EQ(reconstructed, middleBlock)
      << "Known pattern reconstruction failed";
}
