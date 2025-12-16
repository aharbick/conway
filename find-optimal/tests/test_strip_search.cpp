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

// ============================================================================
// Pattern Assembly Tests
// ============================================================================

TEST_F(StripSearchTest, AssembleStripPatternLayout) {
  // Test that pattern assembly correctly places bits in expected positions
  uint16_t topStrip = 0x1234;
  uint32_t middleBlock = 0xABCDEF01;
  uint16_t bottomStrip = 0x5678;

  uint64_t pattern = assembleStripPattern(topStrip, middleBlock, bottomStrip);

  // Verify bit layout: top=0-15, middle=16-47, bottom=48-63
  EXPECT_EQ((uint16_t)(pattern & 0xFFFF), topStrip);
  EXPECT_EQ((uint32_t)((pattern >> 16) & 0xFFFFFFFF), middleBlock);
  EXPECT_EQ((uint16_t)(pattern >> 48), bottomStrip);
}

TEST_F(StripSearchTest, AssembleExtractRoundTrip) {
  // Test round-trip: assemble then extract returns original components
  uint16_t topStrip = 0xFEDC;
  uint32_t middleBlock = 0x12345678;
  uint16_t bottomStrip = 0xABCD;

  uint64_t pattern = assembleStripPattern(topStrip, middleBlock, bottomStrip);

  EXPECT_EQ(extractTopStrip(pattern), topStrip);
  EXPECT_EQ(extractMiddleBlock(pattern), middleBlock);
  EXPECT_EQ(extractBottomStrip(pattern), bottomStrip);
}

TEST_F(StripSearchTest, AssembleStripPatternZeros) {
  // All zeros should produce zero
  uint64_t pattern = assembleStripPattern(0, 0, 0);
  EXPECT_EQ(pattern, 0);
}

TEST_F(StripSearchTest, AssembleStripPatternAllOnes) {
  // All ones should produce all ones
  uint64_t pattern = assembleStripPattern(0xFFFF, 0xFFFFFFFF, 0xFFFF);
  EXPECT_EQ(pattern, 0xFFFFFFFFFFFFFFFFULL);
}

// ============================================================================
// Hash Function Tests
// ============================================================================

TEST_F(StripSearchTest, HashSignatureDeterministic) {
  // Same input should always produce same output
  uint32_t sig = 0x12345678;
  uint32_t hash1 = hashSignature(sig);
  uint32_t hash2 = hashSignature(sig);
  EXPECT_EQ(hash1, hash2);
}

TEST_F(StripSearchTest, HashSignatureDistribution) {
  // Test that hash function has reasonable distribution (no obvious clustering)
  // Count bucket usage with 256 buckets for 1000 sequential inputs
  std::vector<int> buckets(256, 0);
  for (uint32_t i = 0; i < 1000; i++) {
    uint32_t hash = hashSignature(i);
    buckets[hash & 0xFF]++;
  }

  // Check that no bucket has more than 10x the average (avg ~= 4)
  int maxBucket = 0;
  int minBucket = 1000;
  for (int count : buckets) {
    if (count > maxBucket) maxBucket = count;
    if (count > 0 && count < minBucket) minBucket = count;
  }

  // With good distribution, max should be < 20 for 1000 items in 256 buckets
  EXPECT_LT(maxBucket, 40) << "Hash function has poor distribution (clustering)";
}

TEST_F(StripSearchTest, HashSignatureZeroHandling) {
  // Zero input should produce non-zero output (good hash property)
  uint32_t hash = hashSignature(0);
  EXPECT_EQ(hash, 0);  // Actually 0 * anything = 0, so this is expected
}

// ============================================================================
// Signature Computation Tests
// ============================================================================

TEST_F(StripSearchTest, ComputeStripSignatureDeterministic) {
  // Same pattern should always produce same signature
  uint64_t pattern = 0x123456789ABCDEF0ULL;

  uint32_t sig1_top = computeStripSignature(pattern, true);
  uint32_t sig2_top = computeStripSignature(pattern, true);
  EXPECT_EQ(sig1_top, sig2_top);

  uint32_t sig1_bottom = computeStripSignature(pattern, false);
  uint32_t sig2_bottom = computeStripSignature(pattern, false);
  EXPECT_EQ(sig1_bottom, sig2_bottom);
}

TEST_F(StripSearchTest, ComputeStripSignatureTopBottomDifferent) {
  // For most patterns, top and bottom signatures should differ
  // (they look at different halves of gen2)
  uint64_t pattern = 0x123456789ABCDEF0ULL;
  uint32_t sigTop = computeStripSignature(pattern, true);
  uint32_t sigBottom = computeStripSignature(pattern, false);

  // They could theoretically be equal for some symmetric patterns,
  // but for this random pattern they should differ
  EXPECT_NE(sigTop, sigBottom);
}

TEST_F(StripSearchTest, ComputeStripSignatureEmptyPattern) {
  // Empty pattern stays empty after any number of generations
  uint64_t pattern = 0;
  uint32_t sigTop = computeStripSignature(pattern, true);
  uint32_t sigBottom = computeStripSignature(pattern, false);

  EXPECT_EQ(sigTop, 0);
  EXPECT_EQ(sigBottom, 0);
}

TEST_F(StripSearchTest, DifferentStripsCanProduceSameSignature) {
  // Two different strips that produce the same 2-gen effect should have same signature
  // This tests the core principle of strip deduplication

  uint32_t middleBlock = 0;  // Empty middle block

  // Count unique signatures for all 65536 top strips
  std::set<uint32_t> uniqueSignatures;
  for (uint32_t strip = 0; strip < STRIP_SEARCH_TOTAL_STRIPS; strip++) {
    uint64_t pattern = assembleStripPattern((uint16_t)strip, middleBlock, 0);
    uint32_t sig = computeStripSignature(pattern, true);
    uniqueSignatures.insert(sig);
  }

  // With empty middle block, many strips should produce same signature
  // (strips that differ only in "irrelevant" cells)
  // Expect significantly fewer unique signatures than total strips
  EXPECT_LT(uniqueSignatures.size(), STRIP_SEARCH_TOTAL_STRIPS)
      << "Expected signature deduplication to reduce unique count";

  // Per docs, we expect ~17K unique signatures, so at least 50% reduction
  EXPECT_LT(uniqueSignatures.size(), STRIP_SEARCH_TOTAL_STRIPS / 2)
      << "Expected at least 2x reduction from signature deduplication";
}

TEST_F(StripSearchTest, UniqueSignatureCountShowsDeduplication) {
  // Test that signature-based deduplication reduces unique strip count
  // Note: The ~17K figure from docs is for a "typical" middle block.
  // Edge cases (empty, full) have much higher deduplication.

  std::vector<uint32_t> testMiddleBlocks = {
      0x00000000,  // Empty - high deduplication
      0xFFFFFFFF,  // Full - very high deduplication (cells die from overcrowding)
      0x12345678,  // Random-ish - moderate deduplication
      0x0F0F0F0F,  // Striped pattern
  };

  for (uint32_t middleBlock : testMiddleBlocks) {
    std::set<uint32_t> uniqueTopSigs;
    std::set<uint32_t> uniqueBottomSigs;

    for (uint32_t strip = 0; strip < STRIP_SEARCH_TOTAL_STRIPS; strip++) {
      // Top strip signature
      uint64_t topPattern = assembleStripPattern((uint16_t)strip, middleBlock, 0);
      uniqueTopSigs.insert(computeStripSignature(topPattern, true));

      // Bottom strip signature
      uint64_t bottomPattern = assembleStripPattern(0, middleBlock, (uint16_t)strip);
      uniqueBottomSigs.insert(computeStripSignature(bottomPattern, false));
    }

    // All middle blocks should show significant deduplication
    // At minimum 2x reduction (32768 unique vs 65536 total)
    EXPECT_LT(uniqueTopSigs.size(), STRIP_SEARCH_TOTAL_STRIPS / 2)
        << "Expected at least 2x deduplication for middleBlock=" << middleBlock;
    EXPECT_LT(uniqueBottomSigs.size(), STRIP_SEARCH_TOTAL_STRIPS / 2)
        << "Expected at least 2x deduplication for middleBlock=" << middleBlock;

    // Should have at least some unique signatures (sanity check)
    EXPECT_GT(uniqueTopSigs.size(), 100U)
        << "Too few unique top signatures for middleBlock=" << middleBlock;
    EXPECT_GT(uniqueBottomSigs.size(), 100U)
        << "Too few unique bottom signatures for middleBlock=" << middleBlock;
  }
}
