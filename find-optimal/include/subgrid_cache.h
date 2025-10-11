#ifndef _SUBGRID_CACHE_H_
#define _SUBGRID_CACHE_H_

#include <cstdint>
#include <string>

#include "cli_parser.h"
#include "constants.h"

// Entry in the subgrid cache file
struct SubgridCacheEntry {
  uint64_t pattern;      // The 8x8 pattern (includes position information)
  uint16_t generations;  // Number of generations (180-65535)
};

// Compute the 7x7 subgrid cache and save to disk
int computeSubgridCache(ProgramArgs* cli);

// Expand a compact 7x7 pattern (7 bits/row) to 8x8 grid format (8 bits/row) at given position
// pattern7x7: compact format with 7 consecutive bits per row (49 bits total)
// rowOffset: 0 for rows 0-6, 1 for rows 1-7
// colOffset: 0 for cols 0-6, 1 for cols 1-7
__device__ __host__ inline uint64_t expand7x7To8x8(uint64_t pattern7x7, int rowOffset, int colOffset) {
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

// CUDA kernel declaration
#ifdef __NVCC__
__global__ void findSubgridCandidates(uint64_t rangeStart, uint64_t rangeEnd,
                                      SubgridCacheEntry* candidates, uint64_t* numCandidates,
                                      CycleDetectionAlgorithm algorithm, int minGenerations = SUBGRID_MIN_GENERATIONS);
#endif

#endif
