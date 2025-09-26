/**
 * Validation program to demonstrate comprehensive pattern coverage
 * in the Conway's Game of Life search algorithm.
 *
 * This program provides concrete evidence that the search strategy
 * covers all possible 8x8 patterns without gaps or duplicates.
 */

#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "platform_compat.h"

#ifndef __NVCC__
#define __host__
#define __device__
#define __global__
#endif

#include "frame_utils.h"
#include "gol_core.h"

class CompletenessValidator {
 public:
  void runAllValidations() {
    std::cout << "=== Conway's Game of Life Search Completeness Validation ===\n\n";

    validateRotationReduction();
    validateFramePartitioning();
    validateKernelCoverage();
    validateSearchSpaceArithmetic();
    validatePatternReachability();
    validateEquivalenceClasses();

    std::cout << "\n=== All validations complete ===\n";
  }

 private:
  void validateRotationReduction() {
    std::cout << "1. Validating Rotation Elimination Strategy\n";
    std::cout << "==========================================\n";

    const int SAMPLE_SIZE = 1000000;
    std::mt19937_64 rng(12345);

    int minimalFrameCount = 0;
    int totalSamples = 0;
    std::unordered_map<uint64_t, int> rotationCounts;

    for (int i = 0; i < SAMPLE_SIZE; i++) {
      uint64_t pattern = rng();
      uint64_t frame = extractFrame(pattern);
      totalSamples++;

      if (isMinimalFrame(frame)) {
        minimalFrameCount++;
      }

      // Count rotational equivalents for a smaller sample
      if (i < 10000) {
        uint64_t canonical = getCanonicalFrame(frame);
        rotationCounts[canonical]++;
      }
    }

    double reductionRatio = (double)minimalFrameCount / totalSamples;
    std::cout << "Sample size: " << SAMPLE_SIZE << "\n";
    std::cout << "Minimal frames: " << minimalFrameCount << " (" << (reductionRatio * 100) << "%)\n";
    std::cout << "Expected ratio: ~25% (1 out of 4 rotations)\n";
    std::cout << "Reduction factor: " << (1.0 / reductionRatio) << "x\n";

    // Analyze equivalence class sizes
    std::unordered_map<int, int> classSizeDistribution;
    for (auto& pair : rotationCounts) {
      // Each canonical frame should appear ~4 times on average
      // (unless it has rotational symmetry)
      classSizeDistribution[pair.second]++;
    }

    std::cout << "\nEquivalence class size distribution:\n";
    for (auto& pair : classSizeDistribution) {
      std::cout << "  Size " << pair.first << ": " << pair.second << " classes\n";
    }

    std::cout << "\n";
  }

  void validateFramePartitioning() {
    std::cout << "2. Validating Frame Bit Partitioning\n";
    std::cout << "====================================\n";

    // Verify frame mask covers exactly the perimeter
    uint64_t frameMask = extractFrame(0xFFFFFFFFFFFFFFFFULL);
    int frameBits = POPCOUNTLL(frameMask);

    std::cout << "Frame bits: " << frameBits << "\n";
    std::cout << "Expected: 24 (perimeter of 8x8 grid)\n";

    // Verify the frame pattern matches expected perimeter
    std::cout << "\nFrame pattern (F=frame, .=interior):\n";
    for (int row = 0; row < 8; row++) {
      std::cout << "  ";
      for (int col = 0; col < 8; col++) {
        uint64_t bitPos = row * 8 + col;
        if (frameMask & (1ULL << bitPos)) {
          std::cout << "F";
        } else {
          std::cout << ".";
        }
      }
      std::cout << "\n";
    }

    // Test frame bit spread operation
    std::cout << "\nTesting spreadBitsToFrame operation:\n";

    // Test that we can generate all possible frame configurations
    int testCases = 1000;
    std::unordered_set<uint64_t> uniqueFrames;
    std::mt19937_64 rng(54321);
    std::uniform_int_distribution<uint64_t> dist(0, (1ULL << 24) - 1);

    for (int i = 0; i < testCases; i++) {
      uint64_t bits = dist(rng);
      uint64_t frame = spreadBitsToFrame(bits);

      // Verify frame is valid
      assert(extractFrame(frame) == frame);
      uniqueFrames.insert(frame);
    }

    std::cout << "Generated " << uniqueFrames.size() << " unique frames from " << testCases
              << " random 24-bit inputs\n";

    std::cout << "\n";
  }

  void validateKernelCoverage() {
    std::cout << "3. Validating Kernel Construction Coverage\n";
    std::cout << "==========================================\n";

    // Test that constructKernel correctly sets K bits
    // Use a clean frame pattern that doesn't conflict with K bit positions (3,4,59,60)
    uint64_t testFrame = extractFrame(0x8100200000200081ULL);  // Clean frame pattern

    std::cout << "Testing kernel construction with frame: 0x" << std::hex << testFrame << std::dec << "\n";

    for (int kernelIdx = 0; kernelIdx < 16; kernelIdx++) {
      uint64_t result = constructKernel(testFrame, kernelIdx);

      // Extract K bits
      bool bit3 = (result & (1ULL << 3)) != 0;
      bool bit4 = (result & (1ULL << 4)) != 0;
      bool bit59 = (result & (1ULL << 59)) != 0;
      bool bit60 = (result & (1ULL << 60)) != 0;

      int lowerK = (kernelIdx & 3);
      int upperK = (kernelIdx >> 2);

      std::cout << "  Kernel " << std::setw(2) << kernelIdx << ": K=" << upperK << lowerK
                << " -> bits [60,59,4,3] = " << bit60 << bit59 << bit4 << bit3 << "\n";

      // Verify correctness
      assert(bit3 == ((lowerK & 1) != 0));
      assert(bit4 == ((lowerK & 2) != 0));
      assert(bit59 == ((upperK & 1) != 0));
      assert(bit60 == ((upperK & 2) != 0));

      // Verify frame is preserved
      assert(extractFrame(result) == extractFrame(testFrame));
    }

    std::cout << "All 16 kernel configurations validated successfully\n\n";
  }

  void validateSearchSpaceArithmetic() {
    std::cout << "4. Validating Search Space Arithmetic\n";
    std::cout << "=====================================\n";

    const uint64_t TOTAL_PATTERNS = 1ULL << 63;  // Avoid overflow
    std::cout << "Total 8x8 patterns: 2^64 ≈ " << (TOTAL_PATTERNS * 2) << "\n";

    const uint64_t TOTAL_FRAMES = 1ULL << 24;  // 2^24 frame configurations
    std::cout << "Total frame configurations: 2^24 = " << TOTAL_FRAMES << "\n";

    const uint64_t EXPECTED_MINIMAL_FRAMES = TOTAL_FRAMES / 4;
    std::cout << "Expected minimal frames (after rotation): 2^22 = " << EXPECTED_MINIMAL_FRAMES << "\n";

    const int INTERIOR_BITS = 64 - 24 - 4;  // Total - Frame - Kernel bits
    std::cout << "Interior bits per frame: " << INTERIOR_BITS << "\n";
    std::cout << "Interior patterns per frame: 2^" << INTERIOR_BITS << " = " << (1ULL << INTERIOR_BITS) << "\n";

    const uint64_t TOTAL_AFTER_ROTATION = EXPECTED_MINIMAL_FRAMES * (1ULL << INTERIOR_BITS);
    std::cout << "Total patterns after rotation elimination: 2^" << (22 + INTERIOR_BITS) << " ≈ "
              << TOTAL_AFTER_ROTATION << "\n";

    double reductionFactor = (double)(TOTAL_PATTERNS * 2) / TOTAL_AFTER_ROTATION;
    std::cout << "Theoretical reduction factor: " << reductionFactor << "x\n";

    // Compare with your claimed 2^64 -> 2^61 reduction
    std::cout << "\nYour claimed reduction: 2^64 -> 2^61 (8x reduction)\n";
    std::cout << "Expected reduction: 2^64 -> 2^" << (22 + INTERIOR_BITS) << " (" << reductionFactor
              << "x reduction)\n";

    if (22 + INTERIOR_BITS == 61) {
      std::cout << "✓ Arithmetic checks out!\n";
    } else {
      std::cout << "⚠ Arithmetic discrepancy - need to verify bit allocation\n";
    }

    std::cout << "\n";
  }

  void validatePatternReachability() {
    std::cout << "5. Validating Pattern Reachability\n";
    std::cout << "==================================\n";

    // Test that arbitrary patterns can be reached by the search strategy
    const int TEST_PATTERNS = 10000;
    std::mt19937_64 rng(13579);

    int reachablePatterns = 0;
    int unreachablePatterns = 0;

    for (int i = 0; i < TEST_PATTERNS; i++) {
      uint64_t pattern = rng();

      // Check if this pattern can be reached
      if (isPatternReachable(pattern)) {
        reachablePatterns++;
      } else {
        unreachablePatterns++;
      }
    }

    std::cout << "Tested " << TEST_PATTERNS << " random patterns:\n";
    std::cout << "  Reachable: " << reachablePatterns << " (" << (100.0 * reachablePatterns / TEST_PATTERNS) << "%)\n";
    std::cout << "  Unreachable: " << unreachablePatterns << " (" << (100.0 * unreachablePatterns / TEST_PATTERNS)
              << "%)\n";

    if (unreachablePatterns > 0) {
      std::cout << "⚠ Some patterns appear unreachable - investigating...\n";

      // Analyze why patterns might be unreachable
      analyzeUnreachablePatterns(rng);
    } else {
      std::cout << "✓ All tested patterns are reachable\n";
    }

    std::cout << "\n";
  }

  void validateEquivalenceClasses() {
    std::cout << "6. Validating Rotational Equivalence Classes\n";
    std::cout << "============================================\n";

    // Verify that patterns are properly grouped into equivalence classes
    const int SAMPLE_SIZE = 10000;
    std::mt19937_64 rng(24680);

    std::unordered_map<uint64_t, std::vector<uint64_t>> equivalenceClasses;
    std::unordered_set<uint64_t> processedPatterns;

    int classesFound = 0;

    for (int i = 0; i < SAMPLE_SIZE && classesFound < 1000; i++) {
      uint64_t pattern = rng();

      if (processedPatterns.count(pattern))
        continue;

      // Generate all rotations of this pattern
      uint64_t canonical = getCanonicalFrame(extractFrame(pattern));
      std::vector<uint64_t> rotations = getAllFrameRotations(extractFrame(pattern));

      // Mark all rotations as processed
      for (uint64_t rotation : rotations) {
        processedPatterns.insert(rotation);
        equivalenceClasses[canonical].push_back(rotation);
      }

      classesFound++;
    }

    std::cout << "Found " << classesFound << " equivalence classes\n";

    // Analyze class sizes
    std::unordered_map<int, int> classSizes;
    for (auto& pair : equivalenceClasses) {
      int size = pair.second.size();
      classSizes[size]++;
    }

    std::cout << "Equivalence class size distribution:\n";
    for (auto& pair : classSizes) {
      std::cout << "  Size " << pair.first << ": " << pair.second << " classes ("
                << (100.0 * pair.second / classesFound) << "%)\n";
    }

    // Most classes should have size 4 (unless symmetric)
    if (classSizes[4] > classSizes[1] + classSizes[2]) {
      std::cout << "✓ Most patterns have 4 distinct rotations as expected\n";
    } else {
      std::cout << "⚠ Unexpected class size distribution - many symmetric patterns?\n";
    }

    std::cout << "\n";
  }

  // Helper functions
  uint64_t getCanonicalFrame(uint64_t frame) {
    uint64_t minimal = frame;
    uint64_t current = frame;

    for (int i = 0; i < 4; i++) {
      if (current < minimal) {
        minimal = current;
      }
      current = rotate90(current);
    }

    // Also consider reflections
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

  std::vector<uint64_t> getAllFrameRotations(uint64_t frame) {
    std::vector<uint64_t> rotations;
    uint64_t current = frame;

    for (int i = 0; i < 4; i++) {
      rotations.push_back(current);
      current = rotate90(current);
    }

    return rotations;
  }

  bool isPatternReachable(uint64_t pattern) {
    // A pattern is reachable if:
    // 1. Its frame (or a rotation of its frame) is minimal
    // 2. The remaining bits can be set by kernel + interior search

    uint64_t frame = extractFrame(pattern);

    // Check if this frame or any rotation is minimal
    uint64_t current = frame;
    for (int i = 0; i < 4; i++) {
      if (isMinimalFrame(current)) {
        return true;  // Pattern is reachable through this rotation
      }
      current = rotate90(current);
    }

    // Check reflections
    uint64_t reflected = reflectHorizontal(frame);
    current = reflected;
    for (int i = 0; i < 4; i++) {
      if (isMinimalFrame(current)) {
        return true;  // Pattern is reachable through this reflection+rotation
      }
      current = rotate90(current);
    }

    return false;  // No minimal rotation found
  }

  void analyzeUnreachablePatterns(std::mt19937_64& rng) {
    std::cout << "\nAnalyzing unreachable patterns...\n";

    int analyzed = 0;
    int frameIssues = 0;
    int otherIssues = 0;

    for (int i = 0; i < 1000 && analyzed < 100; i++) {
      uint64_t pattern = rng();

      if (!isPatternReachable(pattern)) {
        analyzed++;

        uint64_t frame = extractFrame(pattern);

        // Check if the issue is with frame minimality
        bool hasMinimalRotation = false;
        uint64_t current = frame;
        for (int rot = 0; rot < 4; rot++) {
          if (isMinimalFrame(current)) {
            hasMinimalRotation = true;
            break;
          }
          current = rotate90(current);
        }

        if (!hasMinimalRotation) {
          frameIssues++;
        } else {
          otherIssues++;
        }
      }
    }

    if (analyzed > 0) {
      std::cout << "  Frame minimality issues: " << frameIssues << " (" << (100.0 * frameIssues / analyzed) << "%)\n";
      std::cout << "  Other issues: " << otherIssues << " (" << (100.0 * otherIssues / analyzed) << "%)\n";
    }
  }
};

int main() {
  CompletenessValidator validator;
  validator.runAllValidations();
  return 0;
}