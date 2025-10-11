// CUDA kernels for 7x7 subgrid cache computation
#include <iostream>

#include "constants.h"
#include "gol_core.h"
#include "subgrid_cache.h"
#include "cuda_utils.h"

// Find candidates in a range of 7x7 patterns
// Each thread processes SUBGRID_PATTERNS_PER_THREAD patterns across 4 translations
__global__ void findSubgridCandidates(uint64_t rangeStart, uint64_t rangeEnd,
                                      SubgridCacheEntry* candidates, uint64_t* numCandidates,
                                      CycleDetectionAlgorithm algorithm, int minGenerations) {
  // Calculate which 7x7 base pattern this thread is responsible for
  uint64_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t patternsPerThread = SUBGRID_PATTERNS_PER_THREAD;
  uint64_t basePatternStart = rangeStart + (threadId * patternsPerThread);

  if (basePatternStart >= rangeEnd) return;

  uint64_t basePatternEnd = basePatternStart + patternsPerThread;
  if (basePatternEnd > rangeEnd) {
    basePatternEnd = rangeEnd;
  }

  // Process each 7x7 pattern assigned to this thread
  for (uint64_t pattern7x7 = basePatternStart; pattern7x7 < basePatternEnd; pattern7x7++) {
    // Test all 4 possible positions of 7x7 within 8x8
    // Position 0: rows 0-6, cols 0-6
    // Position 1: rows 0-6, cols 1-7
    // Position 2: rows 1-7, cols 0-6
    // Position 3: rows 1-7, cols 1-7

    for (int pos = 0; pos < 4; pos++) {
      int rowOffset = (pos >= 2) ? 1 : 0;
      int colOffset = (pos & 1) ? 1 : 0;

      uint64_t pattern8x8 = expand7x7To8x8(pattern7x7, rowOffset, colOffset);

      // Count generations for this 8x8 pattern
      int gens = countGenerations(pattern8x8, algorithm);

      // Save each 8x8 pattern that meets the threshold
      if (gens >= minGenerations) {
        uint64_t idx = atomicAdd((unsigned long long*)numCandidates, 1ULL);
        candidates[idx].pattern = pattern8x8;  // Store the 8x8 pattern
        candidates[idx].generations = gens;
      }
    }
  }
}
