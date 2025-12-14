// Shared CUDA kernels used by multiple search algorithms
#include "constants.h"
#include "gol.h"

#ifdef __NVCC__

// Process candidates to find the best pattern (used by frame search and strip search)
__global__ void processCandidates(uint64_t *candidates, uint64_t *numCandidates, uint64_t *bestPattern,
                                  uint64_t *bestGenerations, CycleDetectionAlgorithm algorithm) {
  for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < *numCandidates; i += blockDim.x * gridDim.x) {
    uint64_t generations = countGenerations(candidates[i], computeNextGeneration8x8, algorithm);
    if (generations > 0) {  // Only process if it actually ended
      // Check to see if it's higher and emit it in best(Pattern|Generations)
      uint64_t old = atomicMax((unsigned long long *)bestGenerations, (unsigned long long)generations);
      if (old < generations) {
        *bestPattern = candidates[i];
      }
    }
  }
}

// Process 7x7 candidates (used by 7x7 search)
__global__ void processCandidates7x7(uint64_t *candidates, uint64_t *numCandidates, uint64_t *bestPattern,
                                     uint64_t *bestGenerations, CycleDetectionAlgorithm algorithm) {
  for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < *numCandidates; i += blockDim.x * gridDim.x) {
    // Unpack once, then use unpacked format for all iterations in countGenerations
    uint64_t unpacked = unpack7x7(candidates[i]);
    uint64_t generations = countGenerations(unpacked, computeNextGeneration7x7, algorithm);
    if (generations > 0) {  // Only process if it actually ended
      // Check to see if it's higher and emit it in best(Pattern|Generations)
      uint64_t old = atomicMax((unsigned long long *)bestGenerations, (unsigned long long)generations);
      if (old < generations) {
        *bestPattern = candidates[i];  // Store compact 7x7 format
      }
    }
  }
}

#endif
