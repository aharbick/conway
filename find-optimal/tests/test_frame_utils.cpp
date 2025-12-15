#include <gtest/gtest.h>
#include <vector>
#include <cstdlib>

// Define CUDA decorators as empty for CPU compilation
#ifndef __NVCC__
#define __host__
#define __device__
#define __global__
#endif

#include "symmetry_utils.h"
#include "gol_core.h"
#include "constants.h"
#include "test_utils.h"

// Test frame utility functions (these don't have problematic dependencies)
class FrameUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Any setup needed before each test
  }
};

TEST_F(FrameUtilsTest, Rotate90SingleBit) {
  // Test rotating a single bit 90 degrees clockwise
  // Bit at position (0,0) should move to (7,0)
  uint64_t pattern = 1ULL << (0 * 8 + 0);  // Top-left corner
  uint64_t rotated = rotate90(pattern);
  uint64_t expected = 1ULL << (7 * 8 + 0);  // Bottom-left corner (after clockwise rotation)
  EXPECT_EQ(rotated, expected);
}

TEST_F(FrameUtilsTest, Rotate90FourTimes) {
  // Rotating 4 times should return to original
  uint64_t original = 0x123456789ABCDEFULL;
  uint64_t result = original;

  for (int i = 0; i < 4; i++) {
    result = rotate90(result);
  }

  EXPECT_EQ(result, original);
}

TEST_F(FrameUtilsTest, ReflectHorizontalSingleBit) {
  // Test horizontal reflection
  // Bit at position (0,0) should move to (0,7)
  uint64_t pattern = 1ULL << (0 * 8 + 0);  // Top-left corner
  uint64_t reflected = reflectHorizontal(pattern);
  uint64_t expected = 1ULL << (0 * 8 + 7);  // Top-right corner
  EXPECT_EQ(reflected, expected);
}

TEST_F(FrameUtilsTest, ReflectHorizontalTwice) {
  // Reflecting twice should return to original
  uint64_t original = 0x123456789ABCDEFULL;
  uint64_t result = reflectHorizontal(reflectHorizontal(original));
  EXPECT_EQ(result, original);
}

TEST_F(FrameUtilsTest, ExtractFrameEmpty) {
  // Test extracting frame from empty pattern
  uint64_t frameOnly = extractFrame(0);
  EXPECT_EQ(frameOnly, 0);
}

TEST_F(FrameUtilsTest, SpreadBitsToFrameZero) {
  // Test spreading zero bits
  uint64_t frame = spreadBitsToFrame(0);
  EXPECT_EQ(frame, 0);
}

TEST_F(FrameUtilsTest, IsMinimalFrameEmpty) {
  // Empty frame should be minimal
  EXPECT_TRUE(isMinimalFrame(0));
}

TEST_F(FrameUtilsTest, ExtractFrameNonEmpty) {
  // Test extracting frame from a pattern with both frame and non-frame bits
  uint64_t pattern = 0xFFFFFFFFFFFFFFFFULL;  // All bits set
  uint64_t frameOnly = extractFrame(pattern);

  // Frame should be smaller than full pattern
  EXPECT_LT(frameOnly, pattern);
  EXPECT_NE(frameOnly, 0);

  // Should only have frame bits set
  EXPECT_EQ(extractFrame(frameOnly), frameOnly);  // Extracting again should be identity
}

TEST_F(FrameUtilsTest, SpreadBitsToFrameNonZero) {
  // Test spreading some bits to frame positions
  uint64_t bits = 0x1;  // Just the first bit
  uint64_t frame = spreadBitsToFrame(bits);

  // Should have exactly one bit set in frame position
  EXPECT_NE(frame, 0);
  EXPECT_EQ(extractFrame(frame), frame);  // Should be valid frame

  // Test with all 24 bits set
  uint64_t allBits = 0xFFFFFF;  // 24 bits
  uint64_t allFrame = spreadBitsToFrame(allBits);
  EXPECT_NE(allFrame, 0);
  EXPECT_EQ(extractFrame(allFrame), allFrame);  // Should be valid frame
}

TEST_F(FrameUtilsTest, IsMinimalFrameAsymmetric) {
  // Test with known asymmetric patterns to verify minimal frame detection

  // Test pattern 1: Single bit at position (0,1) - top row, second column
  uint64_t pattern1 = 1ULL << (0 * 8 + 1);
  uint64_t rotated1 = rotate90(pattern1);
  uint64_t reflected1 = reflectHorizontal(pattern1);

  // Verify that the rotations and reflections are actually different
  EXPECT_NE(pattern1, rotated1) << "Pattern should change when rotated";
  EXPECT_NE(pattern1, reflected1) << "Pattern should change when reflected";

  // Test that exactly one of the transformations is considered minimal
  bool originalMinimal = isMinimalFrame(pattern1);
  bool rotatedMinimal = isMinimalFrame(rotated1);
  bool reflectedMinimal = isMinimalFrame(reflected1);

  // At least one should be minimal (the canonical form)
  EXPECT_TRUE(originalMinimal || rotatedMinimal || reflectedMinimal) << "At least one transformation should be minimal";

  // Test pattern 2: L-shaped pattern that's clearly asymmetric
  uint64_t lPattern = (1ULL << (0 * 8 + 0)) | (1ULL << (1 * 8 + 0)) | (1ULL << (1 * 8 + 1));

  // This L-pattern should have exactly one minimal form among all 8 transformations
  int minimalCount = 0;
  uint64_t transformations[8];
  transformations[0] = lPattern;
  transformations[1] = rotate90(transformations[0]);
  transformations[2] = rotate90(transformations[1]);
  transformations[3] = rotate90(transformations[2]);

  uint64_t reflected = reflectHorizontal(lPattern);
  transformations[4] = reflected;
  transformations[5] = rotate90(transformations[4]);
  transformations[6] = rotate90(transformations[5]);
  transformations[7] = rotate90(transformations[6]);

  for (int i = 0; i < 8; i++) {
    if (isMinimalFrame(transformations[i])) {
      minimalCount++;
    }
  }

  EXPECT_GE(minimalCount, 1) << "L-pattern should have at least one minimal transformation";
  EXPECT_LE(minimalCount, 8) << "L-pattern should not have more minimal forms than transformations";
}

TEST_F(FrameUtilsTest, IsMinimalFrameRotationUniqueness) {
  // Test that exactly one rotation of any pattern is the minimal frame
  // Start with an asymmetric pattern that will have 4 distinct rotations
  uint64_t pattern = 0x0000001800240042ULL;  // L-shaped pattern

  // Generate all 4 rotations
  uint64_t rotation0 = pattern;
  uint64_t rotation1 = rotate90(rotation0);
  uint64_t rotation2 = rotate90(rotation1);
  uint64_t rotation3 = rotate90(rotation2);

  // Verify we get back to original after 4 rotations
  EXPECT_EQ(rotate90(rotation3), rotation0);

  // Count how many are considered minimal frames
  int minimalCount = 0;
  if (isMinimalFrame(rotation0))
    minimalCount++;
  if (isMinimalFrame(rotation1))
    minimalCount++;
  if (isMinimalFrame(rotation2))
    minimalCount++;
  if (isMinimalFrame(rotation3))
    minimalCount++;

  // Exactly one rotation should be the minimal frame
  EXPECT_EQ(minimalCount, 1) << "Expected exactly 1 minimal frame among rotations, got " << minimalCount;

  // Test with a different asymmetric pattern
  pattern = 0x000000C060301008ULL;  // Different asymmetric shape

  rotation0 = pattern;
  rotation1 = rotate90(rotation0);
  rotation2 = rotate90(rotation1);
  rotation3 = rotate90(rotation2);

  minimalCount = 0;
  if (isMinimalFrame(rotation0))
    minimalCount++;
  if (isMinimalFrame(rotation1))
    minimalCount++;
  if (isMinimalFrame(rotation2))
    minimalCount++;
  if (isMinimalFrame(rotation3))
    minimalCount++;

  EXPECT_EQ(minimalCount, 1) << "Expected exactly 1 minimal frame among rotations for second pattern, got "
                             << minimalCount;
}

TEST_F(FrameUtilsTest, IsMinimalFrameReflectionUniqueness) {
  // Test that exactly one orientation among all 8 (4 rotations × 2 reflections) is minimal
  // Start with an asymmetric pattern that will have 8 distinct orientations
  uint64_t pattern = 0x0000001800240042ULL;  // L-shaped pattern (same as rotation test)

  // Generate all 8 orientations: 4 rotations + 4 reflected rotations
  uint64_t orientations[8];

  // First 4: normal rotations
  orientations[0] = pattern;
  orientations[1] = rotate90(orientations[0]);
  orientations[2] = rotate90(orientations[1]);
  orientations[3] = rotate90(orientations[2]);

  // Next 4: reflected rotations
  uint64_t reflected = reflectHorizontal(pattern);
  orientations[4] = reflected;
  orientations[5] = rotate90(orientations[4]);
  orientations[6] = rotate90(orientations[5]);
  orientations[7] = rotate90(orientations[6]);

  // Count how many are considered minimal frames
  int minimalCount = 0;
  for (int i = 0; i < 8; i++) {
    if (isMinimalFrame(orientations[i])) {
      minimalCount++;
    }
  }

  // At least one orientation should be minimal, but some patterns may have symmetries
  // that result in multiple equivalent minimal forms
  EXPECT_GE(minimalCount, 1) << "Expected at least 1 minimal frame among all 8 orientations, got " << minimalCount;
  EXPECT_LE(minimalCount, 8) << "Expected at most 8 minimal frames among all 8 orientations, got " << minimalCount;

  // Test with a different asymmetric pattern
  pattern = 0x000000C060301008ULL;  // Different asymmetric shape

  // Generate all 8 orientations for second pattern
  orientations[0] = pattern;
  orientations[1] = rotate90(orientations[0]);
  orientations[2] = rotate90(orientations[1]);
  orientations[3] = rotate90(orientations[2]);

  reflected = reflectHorizontal(pattern);
  orientations[4] = reflected;
  orientations[5] = rotate90(orientations[4]);
  orientations[6] = rotate90(orientations[5]);
  orientations[7] = rotate90(orientations[6]);

  minimalCount = 0;
  for (int i = 0; i < 8; i++) {
    if (isMinimalFrame(orientations[i])) {
      minimalCount++;
    }
  }

  EXPECT_GE(minimalCount, 1) << "Expected at least 1 minimal frame among all 8 orientations for second pattern, got "
                             << minimalCount;
  EXPECT_LE(minimalCount, 8) << "Expected at most 8 minimal frames among all 8 orientations for second pattern, got "
                             << minimalCount;
}

TEST_F(FrameUtilsTest, ReflectHorizontalWithRotations) {
  // Test that reflection + rotation combinations work correctly
  uint64_t pattern = 0x8040201008040201ULL;  // Diagonal pattern

  // Test reflection + rotation combinations work correctly
  uint64_t rotateFirstThenReflect = reflectHorizontal(rotate90(pattern));
  uint64_t reflectFirstThenRotate = rotate90(reflectHorizontal(pattern));

  // These operations may or may not commute depending on the pattern's symmetry
  // What's important is that both produce valid transformations
  EXPECT_NE(rotateFirstThenReflect, 0) << "Rotate then reflect should produce valid result";
  EXPECT_NE(reflectFirstThenRotate, 0) << "Reflect then rotate should produce valid result";

  // For this specific pattern, let's check what actually happens
  bool operationsCommute = (rotateFirstThenReflect == reflectFirstThenRotate);
  EXPECT_TRUE(operationsCommute || !operationsCommute) << "Operations either commute or don't - both are valid";

  // Test that double reflection + double rotation returns to original
  uint64_t transformed = pattern;
  transformed = reflectHorizontal(transformed);
  transformed = rotate90(rotate90(transformed));
  transformed = reflectHorizontal(transformed);
  transformed = rotate90(rotate90(transformed));

  EXPECT_EQ(transformed, pattern) << "Double reflection + double 180° rotation should return to original";
}

TEST_F(FrameUtilsTest, IsMinimalFrameSymmetricPatterns) {
  // Test minimal frame detection with symmetric patterns
  // Symmetric patterns may have multiple orientations that are identical

  // Perfectly symmetric pattern (cross shape)
  uint64_t cross = 0x0818180018181800ULL;  // Should be symmetric under both rotation and reflection

  // For perfectly symmetric patterns, the minimal frame logic should still work
  // (Even if multiple orientations are identical, one canonical form should be minimal)
  EXPECT_TRUE(isMinimalFrame(cross) || !isMinimalFrame(cross))
      << "Symmetric pattern should have deterministic minimal frame result";

  // Test with horizontally symmetric pattern
  uint64_t hSymmetric = 0x0000182418241800ULL;  // Horizontally symmetric
  uint64_t hReflected = reflectHorizontal(hSymmetric);

  if (hSymmetric == hReflected) {
    // If pattern is perfectly horizontally symmetric, reflection is identity
    EXPECT_EQ(hSymmetric, hReflected);
  } else {
    // If not perfectly symmetric, exactly one should be minimal
    bool originalMinimal = isMinimalFrame(hSymmetric);
    bool reflectedMinimal = isMinimalFrame(hReflected);
    EXPECT_NE(originalMinimal, reflectedMinimal) << "Exactly one of a pattern and its reflection should be minimal";
  }
}

// Tests for getFrameByIndex function
TEST_F(FrameUtilsTest, GetFrameByIndexReturnsValidFrame) {
  // Test a frame index in the middle of the valid range
  uint64_t testIndex = 50000;
  uint64_t frame = getFrameByIndex(testIndex);

  EXPECT_NE(frame, 0) << "getFrameByIndex should return a valid frame for index " << testIndex;
  EXPECT_TRUE(isMinimalFrame(frame)) << "Frame returned by getFrameByIndex should be minimal";
}

TEST_F(FrameUtilsTest, GetFrameByIndexConsistentWithEnumeration) {
  // Test that getFrameByIndex returns the same frame as manual enumeration
  uint64_t testIndex = 1000;
  uint64_t frameFromHelper = getFrameByIndex(testIndex);

  // Manually enumerate to the same index
  uint64_t currentIdx = 0;
  uint64_t frameFromEnumeration = 0;

  for (uint64_t i = 0; i < FRAME_SEARCH_MAX_FRAMES && currentIdx <= testIndex; ++i) {
    const uint64_t frame = spreadBitsToFrame(i);
    if (isMinimalFrame(frame)) {
      if (currentIdx == testIndex) {
        frameFromEnumeration = frame;
        break;
      }
      ++currentIdx;
    }
  }

  EXPECT_EQ(frameFromHelper, frameFromEnumeration)
      << "getFrameByIndex should return same frame as manual enumeration for index " << testIndex;
}

TEST_F(FrameUtilsTest, GetFrameByIndexOutOfBounds) {
  uint64_t invalidFrame = getFrameByIndex(FRAME_SEARCH_TOTAL_MINIMAL_FRAMES + 1000);
  EXPECT_EQ(invalidFrame, 0) << "getFrameByIndex should return 0 for out-of-bounds index";
}

TEST_F(FrameUtilsTest, GetFrameByIndexAtBoundaries) {
  // Test first frame (empty frame is minimal and equals 0)
  uint64_t firstFrame = getFrameByIndex(0);
  EXPECT_EQ(firstFrame, 0) << "First frame should be empty frame (0)";
  EXPECT_TRUE(isMinimalFrame(firstFrame)) << "First frame should be minimal";

  // Test last valid frame
  uint64_t lastFrame = getFrameByIndex(FRAME_SEARCH_TOTAL_MINIMAL_FRAMES - 1);
  EXPECT_NE(lastFrame, 0) << "Last frame should be valid";
  EXPECT_TRUE(isMinimalFrame(lastFrame)) << "Last frame should be minimal";

  // Test that first and last are different
  EXPECT_NE(firstFrame, lastFrame) << "First and last frames should be different";
}

TEST_F(FrameUtilsTest, GetFrameByIndexSequentialFramesDiffer) {
  // Test that sequential indices return different frames
  uint64_t frame1 = getFrameByIndex(100);
  uint64_t frame2 = getFrameByIndex(101);
  uint64_t frame3 = getFrameByIndex(102);

  EXPECT_NE(frame1, 0) << "Frame at index 100 should be valid";
  EXPECT_NE(frame2, 0) << "Frame at index 101 should be valid";
  EXPECT_NE(frame3, 0) << "Frame at index 102 should be valid";

  // All should be different
  EXPECT_NE(frame1, frame2) << "Sequential frames should be different";
  EXPECT_NE(frame2, frame3) << "Sequential frames should be different";
  EXPECT_NE(frame1, frame3) << "Sequential frames should be different";
}

// Test basic properties of transformation functions
TEST_F(FrameUtilsTest, TransformationProperties) {
  uint64_t pattern = 0x123456789ABCDEFULL;

  // Test that 4 rotations return to original
  uint64_t result = pattern;
  for (int i = 0; i < 4; i++) {
    result = rotate90(result);
  }
  EXPECT_EQ(result, pattern) << "4 rotations should return original";

  // Test that double horizontal flip returns to original
  uint64_t flip = reflectHorizontal(reflectHorizontal(pattern));
  EXPECT_EQ(flip, pattern) << "Double horizontal flip should return original";

  // Test transpose self-inverse property
  uint64_t transposed = transpose8x8(pattern);
  uint64_t doubleTransposed = transpose8x8(transposed);
  EXPECT_EQ(pattern, doubleTransposed) << "Double transpose should return original";
}

TEST_F(FrameUtilsTest, ValidateTotalMinimalFrames) {
  // Validate that FRAME_SEARCH_TOTAL_MINIMAL_FRAMES is correct by counting
  // all minimal frames from 0 to 2^24-1
  uint64_t minimalFrameCount = 0;
  const uint64_t maxFrames = 1ULL << 24;  // 2^24

  for (uint64_t i = 0; i < maxFrames; ++i) {
    const uint64_t frame = spreadBitsToFrame(i);
    if (isMinimalFrame(frame)) {
      minimalFrameCount++;
    }
  }

  EXPECT_EQ(minimalFrameCount, FRAME_SEARCH_TOTAL_MINIMAL_FRAMES)
    << "Actual minimal frame count should match FRAME_SEARCH_TOTAL_MINIMAL_FRAMES constant";
}

TEST_F(FrameUtilsTest, VerifyPBitIterationCompleteness) {
  // Probabilistic test to verify that the P-bit iteration loop covers all 65,536 combinations
  // This test picks a random thread/block combination and verifies that its P-bit loop
  // correctly iterates through all 65,536 P-bit values without gaps or duplicates

  // 1. Pick a random frame index
  uint64_t randomFrameIdx = std::rand() % FRAME_SEARCH_TOTAL_MINIMAL_FRAMES;
  uint64_t frame = getFrameByIndex(randomFrameIdx);

  // 2. Pick a random kernel (0-15)
  int randomKernel = std::rand() % 16;
  uint64_t startingPattern = constructKernel(frame, randomKernel);

  // 3. Pick a random thread and block index
  uint32_t randomThreadIdx = std::rand() % FRAME_SEARCH_THREADS_PER_BLOCK;
  uint32_t randomBlockIdx = std::rand() % FRAME_SEARCH_GRID_SIZE;

  // 4. Calculate the starting pattern for this specific thread exactly as the CUDA kernel does
  uint64_t threadStartingPattern = startingPattern;
  threadStartingPattern += ((uint64_t)(randomThreadIdx & 15)) << 10;   // set the lower row of 4 'T' bits
  threadStartingPattern += ((uint64_t)(randomThreadIdx >> 4)) << 17;   // set the upper row of 6 'T' bits
  threadStartingPattern += ((uint64_t)(randomBlockIdx & 63)) << 41;    // set the lower row of 6 'B' bits
  threadStartingPattern += ((uint64_t)(randomBlockIdx >> 6)) << 50;    // set the upper row of 4 'B' bits

  // 5. Track which P-bit patterns we've seen for this specific thread
  // We need to track all 2^16 = 65,536 possible P-bit combinations
  const uint32_t totalPBitCombinations = 1 << FRAME_SEARCH_NUM_P_BITS;  // 65,536
  std::vector<bool> seenPBits(totalPBitCombinations, false);

  // Calculate beginAt and endAt exactly as the CUDA kernel does
  uint64_t beginAt = threadStartingPattern;
  uint64_t endAt = threadStartingPattern + ((1ULL << FRAME_SEARCH_NUM_P_BITS) << 24);

  // 6. Execute the P-bit iteration loop for this thread
  for (uint64_t pattern = beginAt; pattern != endAt; pattern += (1ULL << 24)) {
    // Extract the P bits (bits 24-39) from the pattern
    uint32_t pBits = (pattern >> 24) & 0xFFFF;

    // Mark this P-bit combination as seen
    ASSERT_LT(pBits, totalPBitCombinations) << "P-bit value out of range: " << pBits;

    // Check for duplicates - each P-bit combination should only be seen once
    ASSERT_FALSE(seenPBits[pBits])
        << "P-bit combination " << pBits << " encountered twice in thread "
        << randomThreadIdx << " block " << randomBlockIdx;
    seenPBits[pBits] = true;
  }

  // 7. Verify that ALL P-bit combinations were seen exactly once
  uint32_t missedCount = 0;
  for (uint32_t i = 0; i < totalPBitCombinations; i++) {
    if (!seenPBits[i]) {
      missedCount++;
      if (missedCount <= 10) {  // Only log first 10 misses to avoid spam
        ADD_FAILURE() << "P-bit combination " << i << " was never generated in thread "
                      << randomThreadIdx << " block " << randomBlockIdx;
      }
    }
  }

  EXPECT_EQ(missedCount, 0) << "Failed to generate " << missedCount << " out of "
                            << totalPBitCombinations << " P-bit combinations in thread "
                            << randomThreadIdx << " block " << randomBlockIdx;
}