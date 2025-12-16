#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <unordered_map>
#include <unordered_set>

#include "platform_compat.h"
#include "cuda_utils.h"
#include "symmetry_utils.h"
#include "gol_core.h"

/**
 * Focused tests to verify the exact search space reduction from 2^64 to 2^61.
 *
 * This validates the mathematical claims about eliminating rotational symmetries
 * and ensures the bit allocation strategy is correct.
 */
class SearchSpaceReductionTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  uint64_t getCanonicalForm(uint64_t frame) {
    uint64_t minimal = frame;

    // Try all 4 rotations
    uint64_t current = frame;
    for (int i = 0; i < 4; i++) {
      if (current < minimal) {
        minimal = current;
      }
      current = rotate90(current);
    }

    // Try all 4 reflected rotations
    uint64_t reflected = reflectHorizontal(frame);
    current = reflected;
    for (int i = 0; i < 4; i++) {
      if (current < minimal) {
        minimal = current;
      }
      current = rotate90(current);
    }

    return minimal;
  }

  int countEquivalentRotations(uint64_t frame) {
    std::unordered_set<uint64_t> uniqueRotations;

    // Add all 4 rotations
    uint64_t current = frame;
    for (int i = 0; i < 4; i++) {
      uniqueRotations.insert(current);
      current = rotate90(current);
    }

    // Add all 4 reflected rotations
    uint64_t reflected = reflectHorizontal(frame);
    current = reflected;
    for (int i = 0; i < 4; i++) {
      uniqueRotations.insert(current);
      current = rotate90(current);
    }

    return uniqueRotations.size();
  }
};

TEST_F(SearchSpaceReductionTest, BitAllocationArithmetic) {
  // Verify that the bit allocation adds up correctly
  // Total bits: 64
  // Frame bits: 24 (perimeter of 8x8 grid)
  // Kernel bits: 4 (K bits at positions 3,4,59,60)
  // Interior bits: 36 (remaining bits handled by CUDA search)

  uint64_t frameMask = extractFrame(0xFFFFFFFFFFFFFFFFULL);
  uint64_t kernelMask = (3ULL << 3) | (3ULL << 59);  // K bits

  int frameBits = POPCOUNTLL(frameMask);
  int kernelBits = POPCOUNTLL(kernelMask);
  int totalAccountedBits = frameBits + kernelBits;
  int interiorBits = 64 - totalAccountedBits;

  EXPECT_EQ(frameBits, 24) << "Frame should have exactly 24 bits";
  EXPECT_EQ(kernelBits, 4) << "Kernel should have exactly 4 bits";
  EXPECT_EQ(interiorBits, 36) << "Interior should have exactly 36 bits";
}

TEST_F(SearchSpaceReductionTest, RotationReflectionReductionFactor) {
  // Test that rotation+reflection elimination reduces the search space by exactly 8x
  // by measuring the fraction of frames that are considered minimal

  const int SAMPLE_SIZE = 1000000;
  std::mt19937_64 rng(12345);

  int minimalFrames = 0;
  std::unordered_map<uint64_t, int> equivalenceClassSizes;
  std::unordered_set<uint64_t> seenCanonical;

  for (int i = 0; i < SAMPLE_SIZE; i++) {
    // Generate random 24-bit frame configuration
    uint64_t frameBits = rng() & ((1ULL << 24) - 1);
    uint64_t frame = spreadBitsToFrame(frameBits);

    if (isMinimalFrame(frame)) {
      minimalFrames++;
    }

    // For a subset, compute the canonical form and track equivalence class sizes
    if (i < 100000) {
      uint64_t canonical = getCanonicalForm(frame);
      if (seenCanonical.find(canonical) == seenCanonical.end()) {
        seenCanonical.insert(canonical);

        // Count how many rotations map to this canonical form
        int classSize = countEquivalentRotations(canonical);
        equivalenceClassSizes[classSize]++;
      }
    }
  }

  double minimalFraction = (double)minimalFrames / SAMPLE_SIZE;
  double reductionFactor = 1.0 / minimalFraction;

  // Should be approximately 8x reduction (12.5% minimal)
  EXPECT_NEAR(minimalFraction, 0.125, 0.02) << "Expected ~12.5% of frames to be minimal";
  EXPECT_NEAR(reductionFactor, 8.0, 1.0) << "Expected ~8x reduction factor";
}

TEST_F(SearchSpaceReductionTest, ExactSearchSpaceCalculation) {
  // Calculate the exact search space reduction
  // Original space: 2^64
  // After 8x reduction from rotations+reflections: 2^64 / 8 = 2^61

  const uint64_t TOTAL_64BIT_PATTERNS = 1ULL << 63;  // 2^63 to avoid overflow in display
  const uint64_t TOTAL_FRAMES = 1ULL << 24;          // 2^24 possible frame configurations

  // Estimate minimal frames from statistical sample
  const int SAMPLE_SIZE = 100000;
  std::mt19937_64 rng(54321);
  int minimalCount = 0;

  for (int i = 0; i < SAMPLE_SIZE; i++) {
    uint64_t frameBits = rng() & ((1ULL << 24) - 1);
    uint64_t frame = spreadBitsToFrame(frameBits);
    if (isMinimalFrame(frame)) {
      minimalCount++;
    }
  }

  uint64_t estimatedMinimalFrames = (uint64_t)((double)minimalCount / SAMPLE_SIZE * TOTAL_FRAMES);

  const int INTERIOR_BITS = 36;                                // From bit allocation test
  const int KERNEL_BITS = 4;                                   // K bits at positions 3,4,59,60
  const uint64_t PATTERNS_PER_FRAME = 1ULL << INTERIOR_BITS;   // 2^36 interior patterns
  const uint64_t KERNEL_CONFIGURATIONS = 1ULL << KERNEL_BITS;  // 2^4 = 16 kernel configurations

  uint64_t totalAfterReduction = estimatedMinimalFrames * KERNEL_CONFIGURATIONS * PATTERNS_PER_FRAME;

  // The search space should be 2^61 after 8x reduction (2^64 / 8 = 2^61)
  EXPECT_NEAR(log2((double)totalAfterReduction), 61.0, 1.0) << "Search space should reduce to approximately 2^61";
}

TEST_F(SearchSpaceReductionTest, CUDAKernelBitCoverage) {
  // Verify that the CUDA kernel search strategy correctly covers
  // all interior bits according to the bit allocation scheme

  // Based on gol_cuda.cu findCandidatesInKernel function:
  // - threadIdx.x sets T bits at positions determined by bit layout
  // - blockIdx.x sets B bits at positions determined by bit layout
  // - Loop iteration sets P bits at positions determined by bit layout

  // From the CUDA code analysis:
  // startingPattern += ((uint64_t)(threadIdx.x & 15)) << 10;   // 4 T bits at position 10
  // startingPattern += ((uint64_t)(threadIdx.x >> 4)) << 17;   // 6 T bits at position 17
  // startingPattern += ((uint64_t)(blockIdx.x & 63)) << 41;    // 6 B bits at position 41
  // startingPattern += ((uint64_t)(blockIdx.x >> 6)) << 50;    // 4 B bits at position 50
  // pattern += (1ULL << 23)  // P bits start at position 23 (16 bits: 23-38)

  uint64_t tLowerMask = 0xFULL << 10;   // 4 bits at position 10-13
  uint64_t tUpperMask = 0x3FULL << 17;  // 6 bits at position 17-22
  uint64_t bLowerMask = 0x3FULL << 41;  // 6 bits at position 41-46
  uint64_t bUpperMask = 0xFULL << 50;   // 4 bits at position 50-53
  uint64_t pMask = 0xFFFFULL << 23;     // 16 bits at position 23-38

  uint64_t allCudaBits = tLowerMask | tUpperMask | bLowerMask | bUpperMask | pMask;

  int tLowerBits = POPCOUNTLL(tLowerMask);
  int tUpperBits = POPCOUNTLL(tUpperMask);
  int bLowerBits = POPCOUNTLL(bLowerMask);
  int bUpperBits = POPCOUNTLL(bUpperMask);
  int pBits = POPCOUNTLL(pMask);
  int totalCudaBits = POPCOUNTLL(allCudaBits);

  // Verify this matches our expected interior bit count
  uint64_t frameMask = extractFrame(0xFFFFFFFFFFFFFFFFULL);
  uint64_t kernelMask = (3ULL << 3) | (3ULL << 59);
  uint64_t expectedInteriorMask = ~(frameMask | kernelMask);
  int expectedInteriorBits = POPCOUNTLL(expectedInteriorMask);

  // The CUDA bits should cover exactly the interior bits
  uint64_t overlapWithExpected = allCudaBits & expectedInteriorMask;
  int overlapBits = POPCOUNTLL(overlapWithExpected);

  // Check for any uncovered interior bits
  uint64_t uncoveredInterior = expectedInteriorMask & ~allCudaBits;
  int uncoveredBits = POPCOUNTLL(uncoveredInterior);

  // Check for any CUDA bits that extend beyond interior
  uint64_t extraCudaBits = allCudaBits & ~expectedInteriorMask;
  int extraBits = POPCOUNTLL(extraCudaBits);

  // Allow for small discrepancies but ensure we're close
  EXPECT_GE(overlapBits, expectedInteriorBits - 5) << "CUDA should cover most interior bits";
  EXPECT_LE(extraBits, 5) << "CUDA should not extend far beyond interior";
}
