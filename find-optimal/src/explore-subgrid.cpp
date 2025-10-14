#include <cstdint>
#include <iostream>
#include <iomanip>
#include <random>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

#include "gol_core.h"
#include "platform_compat.h"

/*
  The question that drives this entire project is:

    "What is the most generations that an initial pattern following traditional
     rules for Conway's Game of Life on an 8x8 planar (non-torus) can be simulated
     and reach a terminal (die-out) state"

  There are 2^64 possible initial patterns for an 8x8 grid.  We tuned a CUDA
  algorithm that can eliminate rotations and reflections and search all 2^61
  remaining solutions at around 4-5 billion patterns/second.  But even still that
  will take over 14 years to complete.

  This program explores an idea..
     1. There are 2^51 initial patterns on an 8x8 grid that only have bits set
        in a 7x7 subgrid (for which there are 4 possible subgrids)
     2. Search 2^51 such 8x8 grids at 4-5 billiion patterns/second would take less
        than a week.
     3. A random 8x8 initial pattern certainly passes through an interation where
        the only set bits are coverable by a 7x7 grid.
     4. If we can compute all 7x7 possible solutions we could use that knowledge to
        speed up processing 8x8 grids.

  We cannot cache ALL possible answers for our 7x7/2^51 solution because a pattern
  is 8 bytes and the generation count is another 2 bytes.  So you would need a massive
  amount of memory to store 2^51 * 10bytes.

  BUT... Imagine that we only stored ALL patterns lasting longer than N generations.

  If we then simiulate a random 8x8 pattern for M generations before it has overlap
  with a 7x7 grid.  Then:
     * If the current state matches a cached pattern X we know the initial pattern has
       a total generation of M+X and we can end early.
     * If the current state does NOT match a cached pattern then we expect full
       simulation to be LESS than M+150 ...
        * If M+150 > than our observed best generations we can let the simulation finish
        * Otherwise we can end early
  
  This program helps quantify the feasibility of this idea.  Specifically it will:
     1. Randomly sample some number of 8x8 initial patterns
     2. Count the number of generations up to FAST_SEARCH_MAX_GENERATIONS for that pattern
     3. Count the number of generations before a 7x7 grid overlap is detected
     4. Count the number of generations for the full pattern using slow cycle detection.
     5. Compute:
        * Avg. fast search generations
        * Avg. saved generations (with 7x7 grid overlap)
        * Estimated speed up from the savings
        * Histogram of total generations so we can know how big our 7x7 cache needs to be

  This is a run on 10,000,000,000 random patterns:

    Analyzing 10000000000 random 8x8 patterns...
    Random seed: 1759682179
    Looking for 7x7 subgrid coverage

    Progress: 9500000000/10000000000 (95%)
    Analysis complete!

    === 7x7 SUBGRID EARLY TERMINATION ANALYSIS ===

    Total patterns analyzed: 10000000000
    Patterns that became 7x7 coverable: 9709135565 (97.1%)

    Average fast search generations per pattern: 24.81
    Average generations saved per pattern: 19.08

    === COST MODEL ===  

    Operations per generation: 18
    Operations per 7x7 check: 4 (4 mask ops)
    Operations per hash lookup: 20 (estimate)

    === NET SAVINGS CALCULATION ===

    WITHOUT optimization:
      24.81 generations * 18 ops = 446 ops/pattern

    WITH 7x7 cache optimization:
      5.73 generations * (18 + 4) ops + 20 lookup = 146 ops/pattern

    RESULTS:
      Time saved: 67.3%
      Speedup: 3.06x faster
      Net savings: 300 ops/pattern

    === TOTAL GENERATIONS HISTOGRAM ===

    Generations |           Count |   Running Total
    --------------------------------------------------
            198 |               1 |               1
            197 |               1 |               2
            194 |               1 |               3
            189 |               1 |               4
            187 |               5 |               9
            186 |               4 |              13
            185 |               4 |              17
            184 |               6 |              23
            183 |              11 |              34
            182 |               6 |              40
            181 |               8 |              48
            180 |              13 |              61
            179 |              14 |              75
            178 |              17 |              92
            177 |              21 |             113
            176 |              19 |             132
            175 |              31 |             163
            174 |              31 |             194
            173 |              43 |             237
            172 |              57 |             294
            171 |             114 |             408
            170 |              64 |             472
            169 |              68 |             540
            168 |              54 |             594
            167 |              67 |             661
            166 |              92 |             753
            165 |             130 |             883
            164 |             146 |           1,029
            163 |             171 |           1,200
            162 |             193 |           1,393
            161 |             231 |           1,624
            160 |             298 |           1,922
            159 |             367 |           2,289
            158 |             408 |           2,697
            157 |             549 |           3,246
            156 |             652 |           3,898
            155 |             750 |           4,648
            154 |             958 |           5,606
            153 |           1,180 |           6,786
            152 |           1,359 |           8,145
            151 |           1,668 |           9,813
            150 |           2,018 |          11,831
          < 150 |   9,999,988,169 |  10,000,000,000
  
  In SUMMARY:
    1. We found 61 patterns >= 180  If there's a similar distribution among all
       7x7 patterns we only need (2^51 / 10B) * 61patterns * 10bytes = ~130MB to 
       store our cache.
    2. With a functional cache we see a 3x speedup.  So we should be able to process
       over 12B patterns per second on the same hardware.
*/

// Statistics structure to track subgrid coverage behavior
struct SubgridStats {
  uint64_t totalPatterns = 0;
  uint64_t patternsWithCoverage = 0;

  uint64_t totalFastSearchGenerations = 0;
  uint64_t totalGenerationsSaved = 0;

  // Histogram: totalGenerations -> count
  std::map<int, uint64_t> generationHistogram;

  void recordPattern(int fastSearchGens, int coverableAtGen, int totalGens) {
    totalPatterns++;
    totalFastSearchGenerations += fastSearchGens;
    generationHistogram[totalGens]++;

    if (coverableAtGen >= 0) {
      patternsWithCoverage++;

      // Savings = fast search generations we didn't need to compute
      int saved = fastSearchGens - coverableAtGen;
      if (saved > 0) {
        totalGenerationsSaved += saved;
      }
    }
  }

  void printReport() const {
    std::cout << "\n=== 7x7 SUBGRID EARLY TERMINATION ANALYSIS ===\n\n";

    std::cout << "Total patterns analyzed: " << totalPatterns << "\n";
    std::cout << "Patterns that became 7x7 coverable: " << patternsWithCoverage
              << " (" << std::fixed << std::setprecision(1)
              << (100.0 * patternsWithCoverage / totalPatterns) << "%)\n\n";

    double avgFastSearch = (double)totalFastSearchGenerations / totalPatterns;
    double avgSaved = (double)totalGenerationsSaved / totalPatterns;

    std::cout << "Average fast search generations per pattern: "
              << std::setprecision(2) << avgFastSearch << "\n";
    std::cout << "Average generations saved per pattern: "
              << avgSaved << "\n\n";

    std::cout << "=== COST MODEL ===\n\n";
    const int OPS_PER_GENERATION = 18;
    const int OPS_PER_7x7_CHECK = 4;
    const int OPS_PER_HASH_LOOKUP = 20;

    std::cout << "Operations per generation: " << OPS_PER_GENERATION << "\n";
    std::cout << "Operations per 7x7 check: " << OPS_PER_7x7_CHECK << " (4 mask ops)\n";
    std::cout << "Operations per hash lookup: " << OPS_PER_HASH_LOOKUP << " (estimate)\n\n";

    std::cout << "=== NET SAVINGS CALCULATION ===\n\n";

    // Without optimization: avgFastSearch * OPS_PER_GENERATION
    double costWithout = avgFastSearch * OPS_PER_GENERATION;

    // With optimization:
    // - We compute until we find 7x7 coverage: (avgFastSearch - avgSaved) generations
    // - Each generation costs: OPS_PER_GENERATION + OPS_PER_7x7_CHECK
    // - Then we do one hash lookup: OPS_PER_HASH_LOOKUP
    double generationsUntilCoverage = avgFastSearch - avgSaved;
    double costWith = generationsUntilCoverage * (OPS_PER_GENERATION + OPS_PER_7x7_CHECK) + OPS_PER_HASH_LOOKUP;

    double netSavings = costWithout - costWith;
    double savingsPercent = (netSavings / costWithout) * 100.0;
    double speedupMultiplier = costWithout / costWith;

    std::cout << "WITHOUT optimization:\n";
    std::cout << "  " << std::setprecision(2) << avgFastSearch << " generations * "
              << OPS_PER_GENERATION << " ops = "
              << (int)costWithout << " ops/pattern\n\n";

    std::cout << "WITH 7x7 cache optimization:\n";
    std::cout << "  " << std::setprecision(2) << generationsUntilCoverage
              << " generations * (" << OPS_PER_GENERATION << " + " << OPS_PER_7x7_CHECK << ") ops + "
              << OPS_PER_HASH_LOOKUP << " lookup = "
              << (int)costWith << " ops/pattern\n\n";

    std::cout << "RESULTS:\n";
    std::cout << "  Time saved: " << std::setprecision(1) << savingsPercent << "%\n";
    std::cout << "  Speedup: " << std::setprecision(2) << speedupMultiplier << "x faster\n";
    std::cout << "  Net savings: " << (int)netSavings << " ops/pattern\n\n";

    // Generation histogram - group <150 together, sort descending
    std::cout << "=== TOTAL GENERATIONS HISTOGRAM ===\n\n";
    std::cout << std::setw(11) << "Generations" << " | "
              << std::setw(15) << "Count" << " | "
              << std::setw(15) << "Running Total" << "\n";
    std::cout << std::string(50, '-') << "\n";

    // Aggregate generations < 150
    uint64_t countUnder150 = 0;
    std::map<int, uint64_t, std::greater<int>> sortedHist; // descending order

    for (const auto& [gens, count] : generationHistogram) {
      if (gens < 150) {
        countUnder150 += count;
      } else {
        sortedHist[gens] = count;
      }
    }

    // Helper to format numbers with commas
    auto formatWithCommas = [](uint64_t num) -> std::string {
      std::string s = std::to_string(num);
      std::string result;
      int count = 0;
      for (auto it = s.rbegin(); it != s.rend(); ++it) {
        if (count > 0 && count % 3 == 0) result = ',' + result;
        result = *it + result;
        count++;
      }
      return result;
    };

    uint64_t runningCount = 0;

    // Print sorted histogram (descending)
    for (const auto& [gens, count] : sortedHist) {
      runningCount += count;
      std::cout << std::setw(11) << gens << " | "
                << std::setw(15) << formatWithCommas(count) << " | "
                << std::setw(15) << formatWithCommas(runningCount) << "\n";
    }

    // Print <150 group at the end
    if (countUnder150 > 0) {
      runningCount += countUnder150;
      std::cout << std::setw(11) << "< 150" << " | "
                << std::setw(15) << formatWithCommas(countUnder150) << " | "
                << std::setw(15) << formatWithCommas(runningCount) << "\n";
    }
  }
};

// Simulate a single pattern and track when it becomes coverable
void analyzePattern(uint64_t pattern, SubgridStats& stats) {
  int coverableAtGeneration = -1;

  uint64_t g1 = pattern;
  uint64_t fastSearchGenerations = 0;

  // Check if initially coverable
  if (isCoverableBy7x7(g1)) {
    coverableAtGeneration = 0;
  }

  // Simulate using the same logic as findCandidatesInKernel
  while (fastSearchGenerations < FAST_SEARCH_MAX_GENERATIONS) {
    // Step through 6 generations and check for termination/cycles
    fastSearchGenerations += 6;
    uint64_t g2 = computeNextGeneration(g1);
    uint64_t g3 = computeNextGeneration(g2);
    uint64_t g4 = computeNextGeneration(g3);
    uint64_t g5 = computeNextGeneration(g4);
    uint64_t g6 = computeNextGeneration(g5);
    g1 = computeNextGeneration(g6);

    // After computing all 6 generations, check for coverability at each step
    // (only if we haven't found it yet)
    if (coverableAtGeneration == -1) {
      if (isCoverableBy7x7(g2)) {
        coverableAtGeneration = fastSearchGenerations - 5;
      } else if (isCoverableBy7x7(g3)) {
        coverableAtGeneration = fastSearchGenerations - 4;
      } else if (isCoverableBy7x7(g4)) {
        coverableAtGeneration = fastSearchGenerations - 3;
      } else if (isCoverableBy7x7(g5)) {
        coverableAtGeneration = fastSearchGenerations - 2;
      } else if (isCoverableBy7x7(g6)) {
        coverableAtGeneration = fastSearchGenerations - 1;
      } else if (isCoverableBy7x7(g1)) {
        coverableAtGeneration = fastSearchGenerations;
      }
    }

    // Check for cycles (same as step6GenerationsAndCheck)
    if ((g1 == g2) || (g1 == g3) || (g1 == g4)) {
      // Pattern ended in a cycle
      break;
    }

    // Check for death (g1 == 0 means all cells died)
    if (g1 == 0) {
      break;
    }
  }

  // Get accurate total generations using countGenerations
  int totalGenerations = countGenerations(pattern);

  stats.recordPattern(fastSearchGenerations, coverableAtGeneration, totalGenerations);
}

int main(int argc, char** argv) {
  // Parse arguments
  uint64_t numSamples = 1000000; // Default 1M samples
  uint64_t seed = static_cast<uint64_t>(time(nullptr));

  if (argc >= 2) {
    numSamples = std::stoull(argv[1]);
  }
  if (argc >= 3) {
    seed = std::stoull(argv[2]);
  }

  std::cout << "Analyzing " << numSamples << " random 8x8 patterns...\n";
  std::cout << "Random seed: " << seed << "\n";
  std::cout << "Looking for 7x7 subgrid coverage\n\n";

  std::mt19937_64 rng(seed);
  SubgridStats stats;

  // Progress reporting
  uint64_t reportInterval = numSamples / 20;
  if (reportInterval == 0) reportInterval = 1;

  for (uint64_t i = 0; i < numSamples; i++) {
    if (i > 0 && i % reportInterval == 0) {
      std::cout << "Progress: " << i << "/" << numSamples
                << " (" << (100 * i / numSamples) << "%)\r" << std::flush;
    }

    uint64_t pattern = rng();
    analyzePattern(pattern, stats);
  }

  std::cout << "\nAnalysis complete!\n";
  stats.printReport();

  return 0;
}