#include <cuda.h>
#include "gol_cuda.h"

__device__ ulong64 computeNextGeneration(ulong64 currentGeneration) {
  ulong64 nextGeneration = currentGeneration;
  for (int i = 0; i < 64; i++) {
    ulong64 neighbors = __popcll(currentGeneration & gNeighborFilters[i]);
    if (currentGeneration & (1UL << i)) {
      // Alive... should die if 1 or fewer or 4 or more neighbors
      if (neighbors <= 1 || neighbors >= 4) {
        nextGeneration &= ~(1UL << i);
      }
    }
    else {
      // Dead... Come alive if exactly 3 neighbors
      if (neighbors == 3) {
        nextGeneration |= 1UL << i;
      }
    }
  }
  return nextGeneration;
}

__device__ ulong64 countGenerations(ulong64 pattern) {
  // Using a set/map/hash to spot cycles should be faster in general for this
  // problem since the number of generations is relatively small.  However on a
  // CUDA core we don't have easy access to such data structures so instead we
  // use Floyd's algorithm for cycle detection:
  // https://en.wikipedia.org/wiki/Cycle_detection#Floyd's_tortoise_and_hare
  bool ended = false;
  ulong64 generations = 0;
  ulong64 slow = pattern;
  ulong64 fast = computeNextGeneration(slow);
  do {
    generations++;
    ulong64 nextSlow = computeNextGeneration(slow);

    if (slow == nextSlow) {
      ended = true; // If we didn't change then we ended
      break;
    }
    slow = nextSlow;
    fast = computeNextGeneration(computeNextGeneration(fast));
  }
  while (slow != fast);
  ended = slow == 0; // If we died out then we ended

  return ended ? generations : 0;
}

__global__ void evaluateRange(ulong64 beginAt, ulong64 endAt,
                              ulong64 *bestPattern, ulong64 *bestGenerations) {
  for (ulong64 pattern = beginAt + (blockIdx.x * blockDim.x + threadIdx.x);
       pattern < endAt;
       pattern += blockDim.x * gridDim.x) {
    ulong64 generations = countGenerations(pattern);
    ulong64 old = atomicMax(bestGenerations, generations);
    if (old < generations) {
      *bestPattern = pattern;
    }
  }
}
