// CUDA kernels for 7x7 subgrid cache computation
#include <iostream>

#include "constants.h"
#include "gol_core.h"
#include "subgrid_cache.h"
#include "cuda_utils.h"

// Expand a compact 7x7 pattern (7 bits/row) to 8x8 grid format (8 bits/row) at given position
// pattern7x7: compact format with 7 consecutive bits per row (49 bits total)
// rowOffset: 0 for rows 0-6, 1 for rows 1-7
// colOffset: 0 for cols 0-6, 1 for cols 1-7
__device__ static inline uint64_t expand7x7To8x8(uint64_t pattern7x7, int rowOffset, int colOffset) {
  uint64_t result = 0;

  // Extract each row from compact format and place in 8x8 grid
  for (int row = 0; row < 7; row++) {
    // Extract 7 bits for this row from compact pattern
    uint64_t rowBits = (pattern7x7 >> (row * 7)) & 0x7F;

    // Place in 8x8 grid at the appropriate row and column offset
    result |= (rowBits << colOffset) << ((row + rowOffset) * 8);
  }

  return result;
}

// Find candidates in a range of 7x7 patterns
// Each thread processes SUBGRID_PATTERNS_PER_THREAD patterns across 4 translations
__global__ void findSubgridCandidates(uint64_t rangeStart, uint64_t rangeEnd,
                                      SubgridCacheEntry* candidates, uint64_t* numCandidates,
                                      CycleDetectionAlgorithm algorithm) {
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

    uint16_t maxGenerations = 0;

    for (int pos = 0; pos < 4; pos++) {
      int rowOffset = (pos >= 2) ? 1 : 0;
      int colOffset = (pos & 1) ? 1 : 0;

      uint64_t pattern8x8 = expand7x7To8x8(pattern7x7, rowOffset, colOffset);

      // Count generations for this 8x8 pattern
      int gens = countGenerations(pattern8x8, algorithm);

      if (gens > maxGenerations) {
        maxGenerations = gens;
      }
    }

    // If any translation had >= 180 generations, save it
    if (maxGenerations >= SUBGRID_MIN_GENERATIONS) {
      uint64_t idx = atomicAdd((unsigned long long*)numCandidates, 1ULL);
      candidates[idx].pattern = pattern7x7;
      candidates[idx].generations = maxGenerations;
    }
  }
}
