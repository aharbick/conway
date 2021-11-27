#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

typedef unsigned char ubyte;
typedef unsigned long long ulong64;

__constant__ ulong64 gNeighborFilters[64] = {
  // Row 0 pixels
  (ulong64) 770,
  (ulong64) 1797 << 0,
  (ulong64) 1797 << 1,
  (ulong64) 1797 << 2,
  (ulong64) 1797 << 3,
  (ulong64) 1797 << 4,
  (ulong64) 1797 << 5,
  (ulong64) 49216,

  // Row 1 pixels
  (ulong64) 197123,
  (ulong64) 460039 << 0,
  (ulong64) 460039 << 1,
  (ulong64) 460039 << 2,
  (ulong64) 460039 << 3,
  (ulong64) 460039 << 4,
  (ulong64) 460039 << 5,
  (ulong64) 12599488,

  // Row 2 pixels
  (ulong64) 197123 << 8,
  (ulong64) 460039 << 8 << 0,
  (ulong64) 460039 << 8 << 1,
  (ulong64) 460039 << 8 << 2,
  (ulong64) 460039 << 8 << 3,
  (ulong64) 460039 << 8 << 4,
  (ulong64) 460039 << 8 << 5,
  (ulong64) 12599488 << 8,

  // Row 3 pixels
  (ulong64) 197123 << 16,
  (ulong64) 460039 << 16 << 0,
  (ulong64) 460039 << 16 << 1,
  (ulong64) 460039 << 16 << 2,
  (ulong64) 460039 << 16 << 3,
  (ulong64) 460039 << 16 << 4,
  (ulong64) 460039 << 16 << 5,
  (ulong64) 12599488 << 16,

  // Row 4 pixels
  (ulong64) 197123 << 24,
  (ulong64) 460039 << 24 << 0,
  (ulong64) 460039 << 24 << 1,
  (ulong64) 460039 << 24 << 2,
  (ulong64) 460039 << 24 << 3,
  (ulong64) 460039 << 24 << 4,
  (ulong64) 460039 << 24 << 5,
  (ulong64) 12599488 << 24,

  // Row 5 pixels
  (ulong64) 197123 << 32,
  (ulong64) 460039 << 32 << 0,
  (ulong64) 460039 << 32 << 1,
  (ulong64) 460039 << 32 << 2,
  (ulong64) 460039 << 32 << 3,
  (ulong64) 460039 << 32 << 4,
  (ulong64) 460039 << 32 << 5,
  (ulong64) 12599488 << 32,

  // Row 6 pixels
  (ulong64) 197123 << 40,
  (ulong64) 460039 << 40 << 0,
  (ulong64) 460039 << 40 << 1,
  (ulong64) 460039 << 40 << 2,
  (ulong64) 460039 << 40 << 3,
  (ulong64) 460039 << 40 << 4,
  (ulong64) 460039 << 40 << 5,
  (ulong64) 12599488 << 40,

  // Row 7 pixels
  (ulong64) 515 << 48,
  (ulong64) 1287 << 48 << 0,
  (ulong64) 1287 << 48 << 1,
  (ulong64) 1287 << 48 << 2,
  (ulong64) 1287 << 48 << 3,
  (ulong64) 1287 << 48 << 4,
  (ulong64) 1287 << 48 << 5,
  (ulong64) 16576 << 48
};

__device__ ulong64 computeNextGeneration(ulong64 currentGeneration) {
  ulong64 nextGeneration = currentGeneration;
  for (int i = 0; i < 64; i++) {
    ulong64 neighbors = __popcll(currentGeneration & gNeighborFilters[i]);
    if (currentGeneration & (1UL << i)) {
      // Currently alive...
      if (neighbors <= 1) {
        // DIE - lonely
        nextGeneration &= ~(1UL << i);
      }
      else if (neighbors >= 4) {
        // DIE - too crowded
        nextGeneration &= ~(1UL << i);
      }
    }
    else {
      // Currently dead
      if (neighbors == 3) {
        // BIRTH - perfect number of neighbors
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
  for (int pattern = beginAt + (blockIdx.x * blockDim.x + threadIdx.x);
       pattern < endAt;
       pattern += blockDim.x * gridDim.x) {
    ulong64 generations = countGenerations(pattern);
    ulong64 old = atomicMax(bestGenerations, generations);
    if (old < generations) {
      *bestPattern = pattern;
    }
  }
}

void asBinary(ulong64 number, char *buf) {
  for (int i = 63; i >= 0; i--) {
    buf[-i+63] = (number >> i) & 1 ? '1' : '0';
  }
}

#define CHUNKSIZE 1024*1024

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);

  //  Figure out which range we're processing
  ulong64 beginAt = 1;
  ulong64 endAt = ULONG_MAX;
  if (argc == 3) {
    char *end;
    beginAt = strtoul(argv[1], &end, 10);
    endAt = strtoul(argv[2], &end, 10);
  }

  // Allocate memory on CUDA device and locally on host to get the best answers
  ulong64 *devBestPattern, *hostBestPattern;
  hostBestPattern = (ulong64 *)malloc(sizeof(ulong64));
  cudaMalloc((void**)&devBestPattern, sizeof(ulong64));

  ulong64 *devBestGenerations, *hostBestGenerations;
  hostBestGenerations = (ulong64 *)malloc(sizeof(ulong64));
  *hostBestGenerations = 0;
  cudaMalloc((void**)&devBestGenerations, sizeof(ulong64));

  ulong64 i = beginAt;
  while (i < endAt) {
    unsigned j = (i+CHUNKSIZE) > endAt ? endAt : i+CHUNKSIZE;

    cudaMemcpy(devBestGenerations, hostBestGenerations, sizeof(ulong64), cudaMemcpyHostToDevice);
    evaluateRange<<<4096, 256>>>(i, j, devBestPattern, devBestGenerations);

    // Copy device answer to host and emit
    ulong64 prevGen = *hostBestGenerations;
		cudaMemcpy(hostBestPattern, devBestPattern, sizeof(ulong64), cudaMemcpyDeviceToHost);
		cudaMemcpy(hostBestGenerations, devBestGenerations, sizeof(ulong64), cudaMemcpyDeviceToHost);
    char bin[65] = {'\0'};
    if (*hostBestGenerations != prevGen) {
      asBinary(*hostBestPattern, bin);
      printf("\nNEW best! %lu generations, %s (%lu) in range %lu-%lu\n",
             *hostBestGenerations, bin, *hostBestPattern, i, j);
    }
    else {
      printf(".");
    }

    i += CHUNKSIZE;
  }

  return 0;
}
