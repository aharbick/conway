#ifndef _SUBGRID_CACHE_H_
#define _SUBGRID_CACHE_H_

#include <cstdint>
#include <string>

#include "cli_parser.h"
#include "constants.h"

// Entry in the subgrid cache file
struct SubgridCacheEntry {
  uint64_t pattern;      // The 7x7 pattern (stored in lower 49 bits)
  uint16_t generations;  // Number of generations (180-65535)
};

// Compute the 7x7 subgrid cache and save to disk
int computeSubgridCache(ProgramArgs* cli);

// CUDA kernel declaration
#ifdef __NVCC__
__global__ void findSubgridCandidates(uint64_t rangeStart, uint64_t rangeEnd,
                                      SubgridCacheEntry* candidates, uint64_t* numCandidates,
                                      CycleDetectionAlgorithm algorithm);
#endif

#endif
