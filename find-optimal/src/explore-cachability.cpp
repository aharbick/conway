#include <limits.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <cstdint>
#include <set>

#include "display_utils.h"
#include "mt.h"
using namespace std;

static const uint64_t gNeighborFilters[64] = {
    // Row 0 pixels
    (uint64_t)770, (uint64_t)1797 << 0, (uint64_t)1797 << 1, (uint64_t)1797 << 2, (uint64_t)1797 << 3,
    (uint64_t)1797 << 4, (uint64_t)1797 << 5, (uint64_t)49216,

    // Row 1 pixels
    (uint64_t)197123, (uint64_t)460039 << 0, (uint64_t)460039 << 1, (uint64_t)460039 << 2, (uint64_t)460039 << 3,
    (uint64_t)460039 << 4, (uint64_t)460039 << 5, (uint64_t)12599488,


    // Row 2 pixels
    (uint64_t)197123 << 8, (uint64_t)460039 << 8 << 0, (uint64_t)460039 << 8 << 1, (uint64_t)460039 << 8 << 2,
    (uint64_t)460039 << 8 << 3, (uint64_t)460039 << 8 << 4, (uint64_t)460039 << 8 << 5, (uint64_t)12599488 << 8,

    // Row 3 pixels
    (uint64_t)197123 << 16, (uint64_t)460039 << 16 << 0, (uint64_t)460039 << 16 << 1, (uint64_t)460039 << 16 << 2,
    (uint64_t)460039 << 16 << 3, (uint64_t)460039 << 16 << 4, (uint64_t)460039 << 16 << 5, (uint64_t)12599488 << 16,

    // Row 4 pixels
    (uint64_t)197123 << 24, (uint64_t)460039 << 24 << 0, (uint64_t)460039 << 24 << 1, (uint64_t)460039 << 24 << 2,
    (uint64_t)460039 << 24 << 3, (uint64_t)460039 << 24 << 4, (uint64_t)460039 << 24 << 5, (uint64_t)12599488 << 24,

    // Row 5 pixels
    (uint64_t)197123 << 32, (uint64_t)460039 << 32 << 0, (uint64_t)460039 << 32 << 1, (uint64_t)460039 << 32 << 2,
    (uint64_t)460039 << 32 << 3, (uint64_t)460039 << 32 << 4, (uint64_t)460039 << 32 << 5, (uint64_t)12599488 << 32,

    // Row 6 pixels
    (uint64_t)197123 << 40, (uint64_t)460039 << 40 << 0, (uint64_t)460039 << 40 << 1, (uint64_t)460039 << 40 << 2,
    (uint64_t)460039 << 40 << 3, (uint64_t)460039 << 40 << 4, (uint64_t)460039 << 40 << 5, (uint64_t)12599488 << 40,

    // Row 7 pixels
    (uint64_t)515 << 48, (uint64_t)1287 << 48 << 0, (uint64_t)1287 << 48 << 1, (uint64_t)1287 << 48 << 2,
    (uint64_t)1287 << 48 << 3, (uint64_t)1287 << 48 << 4, (uint64_t)1287 << 48 << 5, (uint64_t)16576 << 48};

uint64_t computeNextGeneration(uint64_t currentGeneration) {
  uint64_t nextGeneration = currentGeneration;
  for (int i = 0; i < 64; i++) {
    uint64_t neighbors = __builtin_popcountll(currentGeneration & gNeighborFilters[i]);
    if (currentGeneration & (1ULL << i)) {
      // Currently alive...
      if (neighbors <= 1) {
        // DIE - lonely
        nextGeneration &= ~(1ULL << i);
      } else if (neighbors >= 4) {
        // DIE - too crowded
        nextGeneration &= ~(1ULL << i);
      }
    } else {
      // Currently dead
      if (neighbors == 3) {
        // BIRTH - perfect number of neighbors
        nextGeneration |= 1ULL << i;
      }
    }
  }
  return nextGeneration;
}

/*
  The goal is to exhaustively search the space of all possible 8x8 game of life
  boards, simulate them, and find the methusalehs that end in death or a stable
  state.  In other words, from an initial starting pattern what is the most
  generations that we can simulate before the pattern ends.

  We have a random explorer that checks random possibilities from all possible
  2^64.  We checked ~10M random to look at the distribution and there is a long
  tail with the highest observed generations to be 199 and only once and you
  get down into the single digit number of patterns when you have more than 150
  generations.

  It's possible to exhaustively search a 6x6 grid in a reasonable amount of
  time.  We can store the number of generations for any arbitrary 6x6 grid in
  1-2bytes and as a consequence without anything fancy we could just have a
  massive 70-140GB array in RAM to do instance lookups.

  The questions are:

  1. Is it possible to quickly determine if all of the live cells in an 8x8
  grid can be covered by a 6x6 grid?

  2. If so... what percentage of the 8x8 space can be covered by a 6x6 grid?
  It's 56% of the cells... is it 56% of the candidates?

  3. For the grids that cannot be covered by a 6x6 grid how many generations
  does it take before they can?  For example this 8x8 grid:
    -------------------                -------------------
    | 0 0 1 1 0 0 0 0 |                | 0 1 1 1 0 0 0 0 |
    | 1 1 0 1 0 0 0 0 |                | 1 0 0 1 0 0 0 0 |
    | 1 1 0 0 0 0 0 0 |                | 0 0 0 0 0 0 0 0 |
    | 1 1 0 1 0 0 0 0 |     becomes    | 1 0 0 1 0 0 0 0 |
    | 0 0 1 1 0 0 0 0 |                | 0 1 1 1 0 0 0 0 |
    | 0 0 0 0 0 0 0 0 |                | 0 0 0 0 0 0 0 0 |
    | 0 0 0 0 0 0 0 0 |                | 0 0 0 0 0 0 0 0 |
    | 0 0 0 0 0 0 0 1 |                | 0 0 0 0 0 0 0 0 |
    -------------------                -------------------

    after one generation...  The first one isn't coverable by the 6x6 grid but
    the second one is.

   So... the point of this program is to see how efficiently we can assess whether
   the live cells in an 8x8 grid can be covered by a 6x6 grid and how much that
   could speed things up.
*/

/* Consider the pattern above... as a 64-bit number:

   0011000011010000110000001101000000110000000000000000000000000001

   The following mask when &-ed to a pattern will retain ONLY bits that would
   be covered by an 6x6 area:

     0000000000000000000000000000000000000000000000110001010010000000
     0000000000000000001111110011111100111111001111110011111100111111 (0x3F3F3F3F3F3F in hex)

   It's complement ~mask will reveal on bits that are outside of our 6x6 area:

     1111111111111111110000001100000011000000110000001100000011000000

   If we convolve our mask over the 8x8 grid shifting either 1 or 6 places (to
   wrap to the next row) it will take us 8 shifts to have tested all possible
   placements of a 6x6 grid on our original 8x8 grid.

   Adding against both our mask and ~mask can be used to know if we've
   covered all the bits in our 8x8 grid.
*/
bool coverableByXxY(uint64_t pattern, int x, int y) {
  static uint64_t masks[64] = {0};
  static int convolutions[64] = {0};
  if (masks[x * y] == 0) {
    convolutions[x * y] = (8 - x + 1) * (8 - y + 1);
    for (int i = 0; i < y; i++) {
      masks[x * y] <<= 8;
      masks[x * y] |= (1 << x) - 1;
    }
  }

  uint64_t mask = masks[x * y];
  int iters = convolutions[x * y];
  for (int i = 0; i < iters; i++) {
    if ((pattern & mask) > 0 && (pattern & ~mask) == 0) {
      return true;
    }

    if (x != 8 && ((i + 1) % (8 - x + 1)) == 0) {
      mask <<= 1;
    } else {
      mask <<= x;
    }
  }

  return false;
}

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);

  // Initialize Random number generator
  init_genrand64((uint64_t)time(NULL));

  int x = 7;
  int y = 5;
  if (argc == 3) {
    y = atoi(argv[1]);
    x = atoi(argv[2]);
  }

  uint64_t totalGenerations = 0;
  uint64_t savedGenerations = 0;

  // Check 1m random numbers
  for (uint64_t i = 0; i < 1000 * 1000; i++) {
    uint64_t pattern = genrand64_int64() % ULONG_MAX;

    uint64_t generations = 0;
    uint64_t shortcutGenerations = 0;
    uint64_t currentGen = pattern;
    set<uint64_t> visitedPatterns;

    // Loop until we cycle to a number we've already visited
    do {
      if (shortcutGenerations == 0 && coverableByXxY(currentGen, x, y)) {
        shortcutGenerations = generations;
      }

      // Mark the pattern as seen, keep track of the number of generations, and compute the next generation
      visitedPatterns.insert(currentGen);
      generations++;
      uint64_t nextGen = computeNextGeneration(currentGen);

      // We found a terminal pattern if we found a pattern that doesn't change with computeNextGeneration()
      if (nextGen == 0 || currentGen == nextGen) {
        break;
      }

      // Advance the current gen to what we just checked with nextGen
      currentGen = nextGen;
    } while (visitedPatterns.find(currentGen) == visitedPatterns.end());

    totalGenerations += generations;
    if (shortcutGenerations > 0) {
      savedGenerations += generations - shortcutGenerations;
    }
  }

  printf("Total generations: %lu, saved generations: %lu (%2.2f%%)\n", totalGenerations, savedGenerations,
         (float)savedGenerations / totalGenerations * 100);
}
