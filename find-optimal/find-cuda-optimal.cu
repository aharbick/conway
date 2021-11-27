#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdbool.h>
#include <pthread.h>
#include <limits.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include "mt.h"


#define ROWS 8
#define COLS 8

typedef unsigned char ubyte;

__constant__ unsigned long gNeighborFilters[64] = {
  // Row 0 pixels
  (unsigned long) 770,
  (unsigned long) 1797 << 0,
  (unsigned long) 1797 << 1,
  (unsigned long) 1797 << 2,
  (unsigned long) 1797 << 3,
  (unsigned long) 1797 << 4,
  (unsigned long) 1797 << 5,
  (unsigned long) 49216,

  // Row 1 pixels
  (unsigned long) 197123,
  (unsigned long) 460039 << 0,
  (unsigned long) 460039 << 1,
  (unsigned long) 460039 << 2,
  (unsigned long) 460039 << 3,
  (unsigned long) 460039 << 4,
  (unsigned long) 460039 << 5,
  (unsigned long) 12599488,

  // Row 2 pixels
  (unsigned long) 197123 << 8,
  (unsigned long) 460039 << 8 << 0,
  (unsigned long) 460039 << 8 << 1,
  (unsigned long) 460039 << 8 << 2,
  (unsigned long) 460039 << 8 << 3,
  (unsigned long) 460039 << 8 << 4,
  (unsigned long) 460039 << 8 << 5,
  (unsigned long) 12599488 << 8,

  // Row 3 pixels
  (unsigned long) 197123 << 16,
  (unsigned long) 460039 << 16 << 0,
  (unsigned long) 460039 << 16 << 1,
  (unsigned long) 460039 << 16 << 2,
  (unsigned long) 460039 << 16 << 3,
  (unsigned long) 460039 << 16 << 4,
  (unsigned long) 460039 << 16 << 5,
  (unsigned long) 12599488 << 16,

  // Row 4 pixels
  (unsigned long) 197123 << 24,
  (unsigned long) 460039 << 24 << 0,
  (unsigned long) 460039 << 24 << 1,
  (unsigned long) 460039 << 24 << 2,
  (unsigned long) 460039 << 24 << 3,
  (unsigned long) 460039 << 24 << 4,
  (unsigned long) 460039 << 24 << 5,
  (unsigned long) 12599488 << 24,

  // Row 5 pixels
  (unsigned long) 197123 << 32,
  (unsigned long) 460039 << 32 << 0,
  (unsigned long) 460039 << 32 << 1,
  (unsigned long) 460039 << 32 << 2,
  (unsigned long) 460039 << 32 << 3,
  (unsigned long) 460039 << 32 << 4,
  (unsigned long) 460039 << 32 << 5,
  (unsigned long) 12599488 << 32,

  // Row 6 pixels
  (unsigned long) 197123 << 40,
  (unsigned long) 460039 << 40 << 0,
  (unsigned long) 460039 << 40 << 1,
  (unsigned long) 460039 << 40 << 2,
  (unsigned long) 460039 << 40 << 3,
  (unsigned long) 460039 << 40 << 4,
  (unsigned long) 460039 << 40 << 5,
  (unsigned long) 12599488 << 40,

  // Row 7 pixels
  (unsigned long) 515 << 48,
  (unsigned long) 1287 << 48 << 0,
  (unsigned long) 1287 << 48 << 1,
  (unsigned long) 1287 << 48 << 2,
  (unsigned long) 1287 << 48 << 3,
  (unsigned long) 1287 << 48 << 4,
  (unsigned long) 1287 << 48 << 5,
  (unsigned long) 16576 << 48
};

__device__ unsigned long computeNextGeneration(unsigned long currentGeneration) {
  unsigned long nextGeneration = currentGeneration;
  for (int i = 0; i < 64; i++) {
    unsigned long neighbors = __popcll(currentGeneration & gNeighborFilters[i]);
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

__device__ unsigned long countGenerations(unsigned long pattern) {
  // Using a set/map/hash to spot cycles should be faster in general for this
  // problem since the number of generations is relatively small.  However on a
  // CUDA core we don't have easy access to such data structures so instead we
  // use Floyd's algorithm for cycle detection:
  // https://en.wikipedia.org/wiki/Cycle_detection#Floyd's_tortoise_and_hare
  bool ended = false;
  unsigned long generations = 0;
  unsigned long slow = pattern;
  unsigned long fast = computeNextGeneration(slow);
  do {
    generations++;
    unsigned long nextSlow = computeNextGeneration(slow);

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

__device__ __host__ void asBinary(unsigned long number, char *buf) {
  for (int i = 63; i >= 0; i--) {
    buf[-i+63] = (number >> i) & 1 ? '1' : '0';
  }
}

__global__ void evaluateRange(unsigned long beginAt, unsigned long endAt,
                              unsigned long *bestPattern, unsigned long *bestGenerations) {
  for (int pattern = beginAt + (blockIdx.x * blockDim.x + threadIdx.x);
       pattern < endAt;
       pattern += blockDim.x * gridDim.x) {
    /*
    if ((pattern & (pattern - 1)) == 0)  {
      printf("%lu ---- begin: %lu, end: %lu, block: %d, blockdim: %d, thread: %d, griddim: %d\n", pattern, beginAt, endAt, blockIdx.x, blockDim.x, threadIdx.x, gridDim.x);
    }
    */

    unsigned long generations = countGenerations(pattern);
    if (generations > *bestGenerations) {
      *bestGenerations = generations;
      *bestPattern = pattern;
    }
  }
}

#define CHUNKSIZE 1024*1024

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);

  //  Figure out which range we're processing
  unsigned long beginAt = 1;
  unsigned long endAt = ULONG_MAX;
  if (argc == 3) {
    char *end;
    beginAt = strtoul(argv[1], &end, 10);
    endAt = strtoul(argv[2], &end, 10);
  }

  // Allocate memory on CUDA device and locally on host to get the best answers
  unsigned long *devBestPattern, *hostBestPattern;
  hostBestPattern = (unsigned long *)malloc(sizeof(unsigned long));
  cudaMalloc((void**)&devBestPattern, sizeof(unsigned long));

  unsigned long *devBestGenerations, *hostBestGenerations;
  hostBestGenerations = (unsigned long *)malloc(sizeof(unsigned long));
  cudaMalloc((void**)&devBestGenerations, sizeof(unsigned long));

  unsigned long i = beginAt;
  while (i < endAt) {
    unsigned j = (i+CHUNKSIZE) > endAt ? endAt : i+CHUNKSIZE;

    cudaMemset(devBestPattern, 0x0, sizeof(unsigned long));
    cudaMemset(devBestGenerations, 0x0, sizeof(unsigned long));
    evaluateRange<<<4096, 256>>>(i, j, devBestPattern, devBestGenerations);

    // Copy device answer to host and emit
		cudaMemcpy(hostBestPattern, devBestPattern, sizeof(unsigned long), cudaMemcpyDeviceToHost);
		cudaMemcpy(hostBestGenerations, devBestGenerations, sizeof(unsigned long), cudaMemcpyDeviceToHost);
    char bin[65] = {'\0'};
    asBinary(*hostBestPattern, bin);
    printf("generations - %lu, %s in range %lu-%lu\n", *hostBestGenerations, bin, i, j);

    i += CHUNKSIZE;
  }

  return 0;
}
