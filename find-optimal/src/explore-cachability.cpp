#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>

#include "display_utils.h"
#include "platform_compat.h"
using namespace std;

static const uint64_t gNeighborFilters[64] = {
    // Row 0 pixels
    770ULL, 1797ULL << 0, 1797ULL << 1, 1797ULL << 2, 1797ULL << 3, 1797ULL << 4, 1797ULL << 5, 49216ULL,

    // Row 1 pixels
    197123ULL, 460039ULL << 0, 460039ULL << 1, 460039ULL << 2, 460039ULL << 3, 460039ULL << 4, 460039ULL << 5,
    12599488ULL,


    // Row 2 pixels
    197123ULL << 8, 460039ULL << 8 << 0, 460039ULL << 8 << 1, 460039ULL << 8 << 2, 460039ULL << 8 << 3,
    460039ULL << 8 << 4, 460039ULL << 8 << 5, 12599488ULL << 8,

    // Row 3 pixels
    197123ULL << 16, 460039ULL << 16 << 0, 460039ULL << 16 << 1, 460039ULL << 16 << 2, 460039ULL << 16 << 3,
    460039ULL << 16 << 4, 460039ULL << 16 << 5, 12599488ULL << 16,

    // Row 4 pixels
    197123ULL << 24, 460039ULL << 24 << 0, 460039ULL << 24 << 1, 460039ULL << 24 << 2, 460039ULL << 24 << 3,
    460039ULL << 24 << 4, 460039ULL << 24 << 5, 12599488ULL << 24,

    // Row 5 pixels
    197123ULL << 32, 460039ULL << 32 << 0, 460039ULL << 32 << 1, 460039ULL << 32 << 2, 460039ULL << 32 << 3,
    460039ULL << 32 << 4, 460039ULL << 32 << 5, 12599488ULL << 32,

    // Row 6 pixels
    197123ULL << 40, 460039ULL << 40 << 0, 460039ULL << 40 << 1, 460039ULL << 40 << 2, 460039ULL << 40 << 3,
    460039ULL << 40 << 4, 460039ULL << 40 << 5, 12599488ULL << 40,

    // Row 7 pixels
    515ULL << 48, 1287ULL << 48 << 0, 1287ULL << 48 << 1, 1287ULL << 48 << 2, 1287ULL << 48 << 3, 1287ULL << 48 << 4,
    1287ULL << 48 << 5, 16576ULL << 48};

uint64_t computeNextGeneration(uint64_t currentGeneration) {
  uint64_t nextGeneration = currentGeneration;
  for (int i = 0; i < 64; i++) {
    uint64_t neighbors = POPCOUNTLL(currentGeneration & gNeighborFilters[i]);
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
  std::mt19937_64 rng(static_cast<uint64_t>(time(nullptr)));

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
    uint64_t pattern = rng() % ULONG_MAX;

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

  std::cout << "Total generations: " << totalGenerations << ", saved generations: " << savedGenerations << " ("
            << std::fixed << std::setprecision(2) << ((float)savedGenerations / totalGenerations * 100) << "%%)\n";
}
