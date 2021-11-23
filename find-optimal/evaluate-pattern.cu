#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

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

__device__ int numNeighbors(int bitIdx, unsigned long pattern) {
  return __popcll(pattern & gNeighborFilters[bitIdx]);
}

__global__ void evaluateRange(unsigned long beginAt, unsigned long endAt,
                              unsigned long *bestPattern, unsigned long *bestGenerations) {
  for (int pattern = beginAt + (blockIdx.x * blockDim.x + threadIdx.x);
       pattern < endAt;
       pattern += blockDim.x * gridDim.x) {

  }
}