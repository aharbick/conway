#include <cstdint>
#include <iostream>
#include <iomanip>
#include <random>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>

#include "gol_core.h"
#include "platform_compat.h"

/*
  REVERSIBILITY-BASED SEARCH SPACE REDUCTION

  This program explores optimizing our 8x8 exhaustive search by exploiting the
  irreversibility property of Conway's Game of Life.

  Key Insight from conwaylife.com forums (rocknlol):
  ========================================================
  Life is NOT reversible - many different initial patterns can produce the same
  pattern after 1 generation. This means we can potentially skip computing
  patterns that will produce duplicate results.

  ALGORITHM 1: Corner Cell Irrelevance Detection
  -----------------------------------------------
  Some partial patterns make certain cells irrelevant to future evolution.
  Example: If top-left 3x3 corner is "o.." / "..." / "..." then changing that
  top-left cell doesn't affect generation 1.

  Testing all 256 possible 3x3 corners, rocknlol found 18 arrangements where
  the corner cell is irrelevant. Applied to all 4 corners:
  - Without optimization: 2^64 total patterns
  - With corner reduction: (512-18)^4 / 512^4 = 494^4 / 512^4 ≈ 0.87x
  - Space reduction factor: ~1.15x

  ALGORITHM 2: Edge Strip Deduplication (1 generation)
  ----------------------------------------------------
  For the top 3x8 strip, only 11,510,370 of 2^24 possible patterns produce
  unique results after 1 generation. Applied to top+bottom strips:
  - Reduction factor: (2^24 / 11,510,370)^2 ≈ 2.12x

  ALGORITHM 3: Two-Generation Causality (Advanced)
  -------------------------------------------------
  Due to speed-of-light limit (1 cell/generation), regions separated by 4+ cells
  cannot interact for 2 generations.

  Strategy:
  1. Iterate over middle 4x4 block: ~2^13 after symmetry deduplication
  2. Iterate over side 2x4 blocks: ~2^29 middle 8x4 blocks
  3. For each 8x4 block:
     a. Test all 2^16 top 2x8 strips, run 2 gens, keep unique top-4-rows results
        (~17,000 unique patterns per block instead of 65,536)
     b. Do same for bottom 2x8 strips (~17,000 unique)
     c. Combine: 17,000 * 17,000 ≈ 300M patterns per block
  4. Total: 2^29 * 300M ≈ 1.6×10^17 patterns
     vs original 2^64 ≈ 1.8×10^19 patterns

  Reduction factor: ~14x
  At 10 Gpps: ~185 days instead of 7+ years

  This program will empirically measure these reduction factors by:
  1. Sampling random 8x8 patterns
  2. Testing corner irrelevance
  3. Measuring edge strip deduplication rates
  4. Estimating two-generation causality savings
  5. Computing combined speedup potential
*/

// Statistics for corner irrelevance analysis
struct CornerStats {
  uint64_t totalPatterns = 0;
  uint64_t patternsWithIrrelevantCorners = 0;
  std::map<int, uint64_t> irrelevantCornersHistogram; // num irrelevant corners -> count

  void recordPattern(int numIrrelevantCorners) {
    totalPatterns++;
    if (numIrrelevantCorners > 0) {
      patternsWithIrrelevantCorners++;
    }
    irrelevantCornersHistogram[numIrrelevantCorners]++;
  }

  void printReport() const {
    std::cout << "\n=== CORNER IRRELEVANCE ANALYSIS ===\n\n";
    std::cout << "Total patterns analyzed: " << totalPatterns << "\n";
    std::cout << "Patterns with >=1 irrelevant corner: " << patternsWithIrrelevantCorners
              << " (" << std::fixed << std::setprecision(1)
              << (100.0 * patternsWithIrrelevantCorners / totalPatterns) << "%)\n\n";

    std::cout << "Histogram of irrelevant corners per pattern:\n";
    std::cout << std::setw(10) << "# Corners" << " | " << std::setw(15) << "Count" << " | "
              << std::setw(10) << "Percentage" << "\n";
    std::cout << std::string(45, '-') << "\n";

    for (const auto& [corners, count] : irrelevantCornersHistogram) {
      std::cout << std::setw(10) << corners << " | "
                << std::setw(15) << count << " | "
                << std::setw(9) << std::setprecision(2)
                << (100.0 * count / totalPatterns) << "%\n";
    }

    // Estimate reduction factor
    // If k corners are irrelevant, those corners have 512 patterns but only 494 are unique
    // Reduction = (494/512)^k for each pattern
    double avgReduction = 0.0;
    for (const auto& [corners, count] : irrelevantCornersHistogram) {
      double reduction = 1.0;
      for (int i = 0; i < corners; i++) {
        reduction *= (494.0 / 512.0);
      }
      avgReduction += reduction * count;
    }
    avgReduction /= totalPatterns;

    double speedupFactor = 1.0 / avgReduction;
    std::cout << "\nEstimated average search space reduction: " << std::setprecision(3)
              << avgReduction << "x\n";
    std::cout << "Estimated speedup factor: " << speedupFactor << "x\n";
  }
};

// Statistics for edge strip deduplication
struct EdgeStripStats {
  uint64_t totalPatterns = 0;
  uint64_t uniqueTopStrips = 0;
  uint64_t uniqueBottomStrips = 0;
  std::unordered_map<uint64_t, uint64_t> topStripGen1Map;    // gen1 top 3 rows -> count
  std::unordered_map<uint64_t, uint64_t> bottomStripGen1Map; // gen1 bottom 3 rows -> count

  void recordPattern(uint64_t pattern) {
    totalPatterns++;

    // Run 1 generation
    uint64_t gen1 = computeNextGeneration8x8(pattern);

    // Extract top 3 rows (bits 0-23)
    uint64_t topGen1 = gen1 & 0xFFFFFFULL;
    if (topStripGen1Map.find(topGen1) == topStripGen1Map.end()) {
      uniqueTopStrips++;
    }
    topStripGen1Map[topGen1]++;

    // Extract bottom 3 rows (bits 40-63)
    uint64_t bottomGen1 = (gen1 >> 40) & 0xFFFFFFULL;
    if (bottomStripGen1Map.find(bottomGen1) == bottomStripGen1Map.end()) {
      uniqueBottomStrips++;
    }
    bottomStripGen1Map[bottomGen1]++;
  }

  void printReport() const {
    std::cout << "\n=== EDGE STRIP DEDUPLICATION ANALYSIS ===\n\n";
    std::cout << "Total patterns tested: " << totalPatterns << "\n";
    std::cout << "Unique top-3-row results after gen 1: " << uniqueTopStrips << "\n";
    std::cout << "Unique bottom-3-row results after gen 1: " << uniqueBottomStrips << "\n\n";

    // Calculate reduction factor
    uint64_t maxPossibleTop = (1ULL << 24); // 2^24 possible top strips
    double topReduction = (double)maxPossibleTop / uniqueTopStrips;
    double bottomReduction = (double)maxPossibleTop / uniqueBottomStrips;
    double combinedReduction = topReduction * bottomReduction;

    std::cout << "Top strip reduction: " << std::setprecision(2) << topReduction << "x\n";
    std::cout << "Bottom strip reduction: " << bottomReduction << "x\n";
    std::cout << "Combined reduction (top * bottom): " << combinedReduction << "x\n";
  }
};

// Statistics for two-generation causality analysis
struct TwoGenCausalityStats {
  uint64_t totalBlocks = 0;
  uint64_t totalUniqueTop2Rows = 0;
  uint64_t totalUniqueBottom2Rows = 0;
  std::vector<uint64_t> uniqueTop2RowsPerBlock;
  std::vector<uint64_t> uniqueBottom2RowsPerBlock;

  void recordBlock(uint64_t middle4x4, uint64_t side2x4Left, uint64_t side2x4Right) {
    totalBlocks++;

    // Build the 8x4 middle block
    uint64_t middle8x4 = 0;
    // This is simplified - in real implementation we'd properly construct the 8x4 block
    // For now, use the inputs as-is for demonstration
    middle8x4 = middle4x4 | (side2x4Left << 32) | (side2x4Right << 48);

    // Test all 2^16 possible top 2x8 strips
    std::unordered_map<uint32_t, bool> seenTopResults;
    for (uint64_t topStrip = 0; topStrip < 65536; topStrip++) {
      // Construct full pattern with this top strip
      uint64_t pattern = topStrip | (middle8x4 << 16);

      // Run 2 generations
      uint64_t gen1 = computeNextGeneration8x8(pattern);
      uint64_t gen2 = computeNextGeneration8x8(gen1);

      // Extract top 4 rows of gen2
      uint32_t topResult = (uint32_t)(gen2 & 0xFFFFFFFFULL);
      seenTopResults[topResult] = true;
    }

    uint64_t uniqueTop = seenTopResults.size();
    uniqueTop2RowsPerBlock.push_back(uniqueTop);
    totalUniqueTop2Rows += uniqueTop;

    // Test all 2^16 possible bottom 2x8 strips
    std::unordered_map<uint32_t, bool> seenBottomResults;
    for (uint64_t bottomStrip = 0; bottomStrip < 65536; bottomStrip++) {
      // Construct full pattern with this bottom strip
      uint64_t pattern = (middle8x4 << 16) | (bottomStrip << 48);

      // Run 2 generations
      uint64_t gen1 = computeNextGeneration8x8(pattern);
      uint64_t gen2 = computeNextGeneration8x8(gen1);

      // Extract bottom 4 rows of gen2
      uint32_t bottomResult = (uint32_t)(gen2 >> 32);
      seenBottomResults[bottomResult] = true;
    }

    uint64_t uniqueBottom = seenBottomResults.size();
    uniqueBottom2RowsPerBlock.push_back(uniqueBottom);
    totalUniqueBottom2Rows += uniqueBottom;
  }

  void printReport() const {
    if (totalBlocks == 0) {
      std::cout << "\n=== TWO-GENERATION CAUSALITY ANALYSIS ===\n\n";
      std::cout << "No blocks analyzed.\n";
      return;
    }

    double avgUniqueTop = (double)totalUniqueTop2Rows / totalBlocks;
    double avgUniqueBottom = (double)totalUniqueBottom2Rows / totalBlocks;

    std::cout << "\n=== TWO-GENERATION CAUSALITY ANALYSIS ===\n\n";
    std::cout << "Blocks analyzed: " << totalBlocks << "\n";
    std::cout << "Average unique top 2-row patterns: " << std::setprecision(0) << avgUniqueTop
              << " (out of 65536 possible)\n";
    std::cout << "Average unique bottom 2-row patterns: " << avgUniqueBottom
              << " (out of 65536 possible)\n\n";

    double topReduction = 65536.0 / avgUniqueTop;
    double bottomReduction = 65536.0 / avgUniqueBottom;
    double combinedReduction = topReduction * bottomReduction;

    std::cout << "Top strip reduction: " << std::setprecision(2) << topReduction << "x\n";
    std::cout << "Bottom strip reduction: " << bottomReduction << "x\n";
    std::cout << "Combined per-block reduction: " << combinedReduction << "x\n\n";

    // Estimate total search space reduction
    // Original: 2^64 patterns
    // With this method:
    //   - 2^13 middle 4x4 blocks (after symmetry)
    //   - 2^16 combinations of side blocks = 2^29 total 8x4 blocks
    //   - Each block: avgUniqueTop * avgUniqueBottom patterns
    uint64_t middle4x4Blocks = (1ULL << 13);
    uint64_t total8x4Blocks = (1ULL << 29);
    double patternsPerBlock = avgUniqueTop * avgUniqueBottom;
    double totalPatternsToTest = total8x4Blocks * patternsPerBlock;

    double originalPatterns = 1.844674e19; // 2^64
    double overallReduction = originalPatterns / totalPatternsToTest;

    std::cout << "=== OVERALL REDUCTION ESTIMATE ===\n\n";
    std::cout << "Original search space: 2^64 ≈ 1.8e19 patterns\n";
    std::cout << "With 2-gen causality optimization:\n";
    std::cout << "  - Middle 4x4 blocks: 2^13 = " << middle4x4Blocks << "\n";
    std::cout << "  - Side combinations: 2^16 per side = 2^29 total 8x4 blocks\n";
    std::cout << "  - Patterns per block: " << std::scientific << std::setprecision(2)
              << patternsPerBlock << "\n";
    std::cout << "  - Total patterns to test: " << totalPatternsToTest << "\n\n";
    std::cout << std::fixed << "Overall reduction factor: " << std::setprecision(1)
              << overallReduction << "x\n";

    double gpps = 10.0; // 10 billion patterns per second
    double daysOriginal = originalPatterns / gpps / 86400;
    double daysOptimized = totalPatternsToTest / gpps / 86400;

    std::cout << "\nAt 10 Gpps:\n";
    std::cout << "  Original time: " << std::setprecision(0) << (daysOriginal / 365.25)
              << " years\n";
    std::cout << "  Optimized time: " << std::setprecision(0) << daysOptimized << " days\n";
  }
};

// Check if a cell in a 3x3 region is irrelevant to generation 1
bool isCellIrrelevant(uint64_t pattern, int row, int col) {
  // Extract 3x3 region around the cell
  uint64_t region = 0;
  for (int r = -1; r <= 1; r++) {
    for (int c = -1; c <= 1; c++) {
      int tr = row + r;
      int tc = col + c;
      if (tr >= 0 && tr < 8 && tc >= 0 && tc < 8) {
        uint64_t bit = (pattern >> (tr * 8 + tc)) & 1;
        region |= bit << ((r + 1) * 3 + (c + 1));
      }
    }
  }

  // Test if flipping the center cell changes gen1 result in neighborhood
  uint64_t pattern1 = pattern;
  uint64_t pattern2 = pattern ^ (1ULL << (row * 8 + col)); // Flip center cell

  uint64_t gen1_1 = computeNextGeneration8x8(pattern1);
  uint64_t gen1_2 = computeNextGeneration8x8(pattern2);

  // Check if any cells in the affected region changed
  // Affected region is the 3x3 around the cell in gen1
  for (int r = -1; r <= 1; r++) {
    for (int c = -1; c <= 1; c++) {
      int tr = row + r;
      int tc = col + c;
      if (tr >= 0 && tr < 8 && tc >= 0 && tc < 8) {
        uint64_t bit1 = (gen1_1 >> (tr * 8 + tc)) & 1;
        uint64_t bit2 = (gen1_2 >> (tr * 8 + tc)) & 1;
        if (bit1 != bit2) return false;
      }
    }
  }

  return true;
}

// Analyze corner irrelevance for a pattern
int countIrrelevantCorners(uint64_t pattern) {
  int count = 0;

  // Top-left corner (0,0)
  if (isCellIrrelevant(pattern, 0, 0)) count++;

  // Top-right corner (0,7)
  if (isCellIrrelevant(pattern, 0, 7)) count++;

  // Bottom-left corner (7,0)
  if (isCellIrrelevant(pattern, 7, 0)) count++;

  // Bottom-right corner (7,7)
  if (isCellIrrelevant(pattern, 7, 7)) count++;

  return count;
}

int main(int argc, char** argv) {
  uint64_t numSamples = 10000; // Default samples
  uint64_t seed = static_cast<uint64_t>(time(nullptr));
  bool runTwoGenTest = false; // Expensive test

  if (argc >= 2) {
    numSamples = std::stoull(argv[1]);
  }
  if (argc >= 3) {
    seed = std::stoull(argv[2]);
  }
  if (argc >= 4 && std::string(argv[3]) == "--two-gen") {
    runTwoGenTest = true;
  }

  std::cout << "=== REVERSIBILITY-BASED SEARCH SPACE REDUCTION ANALYSIS ===\n\n";
  std::cout << "Analyzing " << numSamples << " random 8x8 patterns\n";
  std::cout << "Random seed: " << seed << "\n";
  if (runTwoGenTest) {
    std::cout << "Including expensive two-generation causality test\n";
  }
  std::cout << "\n";

  std::mt19937_64 rng(seed);
  CornerStats cornerStats;
  EdgeStripStats edgeStats;
  TwoGenCausalityStats twoGenStats;

  // Progress reporting
  uint64_t reportInterval = numSamples / 20;
  if (reportInterval == 0) reportInterval = 1;

  for (uint64_t i = 0; i < numSamples; i++) {
    if (i > 0 && i % reportInterval == 0) {
      std::cout << "Progress: " << i << "/" << numSamples
                << " (" << (100 * i / numSamples) << "%)\\r" << std::flush;
    }

    uint64_t pattern = rng();

    // Test corner irrelevance
    int irrelevantCorners = countIrrelevantCorners(pattern);
    cornerStats.recordPattern(irrelevantCorners);

    // Test edge strip deduplication
    edgeStats.recordPattern(pattern);
  }

  // Two-generation causality test (much more expensive, limited samples)
  if (runTwoGenTest) {
    std::cout << "\n\nRunning two-generation causality test (this may take a while)...\n";
    uint64_t twoGenSamples = std::min(numSamples / 100, (uint64_t)100); // Much smaller sample

    for (uint64_t i = 0; i < twoGenSamples; i++) {
      if (i > 0 && i % 10 == 0) {
        std::cout << "Two-gen progress: " << i << "/" << twoGenSamples << "\\r" << std::flush;
      }

      uint64_t middle4x4 = rng() & 0xFFFFULL;
      uint64_t side2x4Left = rng() & 0xFFULL;
      uint64_t side2x4Right = rng() & 0xFFULL;

      twoGenStats.recordBlock(middle4x4, side2x4Left, side2x4Right);
    }
  }

  std::cout << "\n\nAnalysis complete!\n";

  // Print all reports
  cornerStats.printReport();
  edgeStats.printReport();

  if (runTwoGenTest) {
    twoGenStats.printReport();
  }

  return 0;
}
