#ifndef _GOL_H_
#define _GOL_H_

#include <pthread.h>

#include "utils.h"

static const int INFINITE = -1;

static const ulong64 gNeighborFilters[64] = {
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


/* The following const arrays were precomputed with this code:

int main(int argc, char **argv) {
  for (int y = 1; i <= 8; y++) {
    for (int x = 1; j <= 8; x++) {
      int conv = (8-x+1) * (8-y+1);
      unsigned long mask = 0;
      for (int i = 0; i < y; i++) {
        mask <<= 8;
        mask |= (1<<x)-1;
      }
      printf("%d, %d, 0x%lX\n", (y-1)*8 + x, conv, mask);
    }
  }
  return 0;
}

It basically figures out the convolutions and bit filter for all possible 8x8 filters (e.g. 1x1, 5x7, 8x4, etc)

*/
static const int gCoverableConvolutions[64] = {
  64,
  56,
  48,
  40,
  32,
  24,
  16,
  8,
  56,
  49,
  42,
  35,
  28,
  21,
  14,
  7,
  48,
  42,
  36,
  30,
  24,
  18,
  12,
  6,
  40,
  35,
  30,
  25,
  20,
  15,
  10,
  5,
  32,
  28,
  24,
  20,
  16,
  12,
  8,
  4,
  24,
  21,
  18,
  15,
  12,
  9,
  6,
  3,
  16,
  14,
  12,
  10,
  8,
  6,
  4,
  2,
  8,
  7,
  6,
  5,
  4,
  3,
  2,
  1};

static const ulong64 gCoverableFilters[64] = {
  (ulong64) 0x1,
  (ulong64) 0x3,
  (ulong64) 0x7,
  (ulong64) 0xF,
  (ulong64) 0x1F,
  (ulong64) 0x3F,
  (ulong64) 0x7F,
  (ulong64) 0xFF,
  (ulong64) 0x101,
  (ulong64) 0x303,
  (ulong64) 0x707,
  (ulong64) 0xF0F,
  (ulong64) 0x1F1F,
  (ulong64) 0x3F3F,
  (ulong64) 0x7F7F,
  (ulong64) 0xFFFF,
  (ulong64) 0x10101,
  (ulong64) 0x30303,
  (ulong64) 0x70707,
  (ulong64) 0xF0F0F,
  (ulong64) 0x1F1F1F,
  (ulong64) 0x3F3F3F,
  (ulong64) 0x7F7F7F,
  (ulong64) 0xFFFFFF,
  (ulong64) 0x1010101,
  (ulong64) 0x3030303,
  (ulong64) 0x7070707,
  (ulong64) 0xF0F0F0F,
  (ulong64) 0x1F1F1F1F,
  (ulong64) 0x3F3F3F3F,
  (ulong64) 0x7F7F7F7F,
  (ulong64) 0xFFFFFFFF,
  (ulong64) 0x101010101,
  (ulong64) 0x303030303,
  (ulong64) 0x707070707,
  (ulong64) 0xF0F0F0F0F,
  (ulong64) 0x1F1F1F1F1F,
  (ulong64) 0x3F3F3F3F3F,
  (ulong64) 0x7F7F7F7F7F,
  (ulong64) 0xFFFFFFFFFF,
  (ulong64) 0x10101010101,
  (ulong64) 0x30303030303,
  (ulong64) 0x70707070707,
  (ulong64) 0xF0F0F0F0F0F,
  (ulong64) 0x1F1F1F1F1F1F,
  (ulong64) 0x3F3F3F3F3F3F,
  (ulong64) 0x7F7F7F7F7F7F,
  (ulong64) 0xFFFFFFFFFFFF,
  (ulong64) 0x1010101010101,
  (ulong64) 0x3030303030303,
  (ulong64) 0x7070707070707,
  (ulong64) 0xF0F0F0F0F0F0F,
  (ulong64) 0x1F1F1F1F1F1F1F,
  (ulong64) 0x3F3F3F3F3F3F3F,
  (ulong64) 0x7F7F7F7F7F7F7F,
  (ulong64) 0xFFFFFFFFFFFFFF,
  (ulong64) 0x101010101010101,
  (ulong64) 0x303030303030303,
  (ulong64) 0x707070707070707,
  (ulong64) 0xF0F0F0F0F0F0F0F,
  (ulong64) 0x1F1F1F1F1F1F1F1F,
  (ulong64) 0x3F3F3F3F3F3F3F3F,
  (ulong64) 0x7F7F7F7F7F7F7F7F,
  (ulong64) 0xFFFFFFFFFFFFFFFF
};

ulong64 computeNextGeneration(ulong64 currentGeneration);
int countGenerations(ulong64 pattern, struct caching_filter f);
void *search(void *args);

#endif
