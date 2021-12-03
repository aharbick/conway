#include <assert.h>
#include <stdbool.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <limits.h>
#include <unistd.h> // sleep
#include <math.h> // pow

#include "cli.h"
#include "gol.h"
#include "utils.h"
#include "mt.h" // For mersenne twister random numbers

///////////////////////////////////////////////////////////////////////////
// GLOBAL variables updated across threads

pthread_mutex_t gMutex = PTHREAD_MUTEX_INITIALIZER;
int gBestGenerations = 0;

ulong64 computeNextGeneration(ulong64 currentGeneration) {
  ulong64 nextGeneration = currentGeneration;
  for (int i = 0; i < 64; i++) {
    ulong64 neighbors = __builtin_popcountll(currentGeneration & gNeighborFilters[i]);
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

int countGenerations(ulong64 pattern) {
  // Use Floyd's algorithm for cycle detection...  Perhaps slightly less
  // efficient (if it's a long period cycle) than a map structure, but when
  // implementing this on a GPU in CUDA those data structures are not readily
  // available.
  // https://en.wikipedia.org/wiki/Cycle_detection#Floyd's_tortoise_and_hare
  bool ended = false;
  int generations = 0;

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

  return ended ? generations : INFINITE;
}

void *search(void *args) {
  prog_args *cli = (prog_args *)args;

  printf("[Thread %d] %s %llu - %llu\n",
         cli->threadId, cli->random ? "RANDOMLY searching" : "searching ALL", cli->beginAt, cli->endAt);

  if (cli->random) {
    // Initialize Random number generator
    init_genrand64((ulong64) time(NULL));
  }

  for (ulong64 i = cli->beginAt; i <= cli->endAt; i++) {
    if (i % 10000000 == 0) {
      printf("."); // every 10m patterns
    }

    // Sequential or random pattern...
    ulong64 pattern = i;
    if (cli->random) {
      pattern = genrand64_int64() % (cli->endAt + 1 - cli->beginAt) + cli->beginAt;
    }

    int generations = countGenerations(pattern);
    if (generations > 0) { // it ended
      pthread_mutex_lock(&gMutex);
      if (gBestGenerations < generations) {
        char bin[65] = {'\0'};
        asBinary(pattern, bin);
        printf("[Thread %d] %d generations : %llu : %s\n", cli->threadId, generations, pattern, bin);
        gBestGenerations = generations;
      }
      pthread_mutex_unlock(&gMutex);
    }
  }

  return NULL;
}
