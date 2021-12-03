#ifndef _GOL_H_
#define _GOL_H_

#include <cuda.h>
#include "utils.h"

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

__device__ ulong64 computeNextGeneration(ulong64 currentGeneration);
__device__ ulong64 countGenerations(ulong64 pattern);
void *search(void *args);

#endif
