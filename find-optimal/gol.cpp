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

typedef struct CoverageCache {
  ulong64 size;
  int *entries;
} CoverageCache;
CoverageCache *gCache = NULL;

///////////////////////////////////////////////////////////////////////////
ulong64 coverableByXxY(ulong64 pattern, int x, int y) {
  ulong64 bitFilter = gCoverableFilters[(y-1)*8 + x-1];
  int convolutions = gCoverableConvolutions[(y-1)*8 +x-1];
  for (int i = 0; i < convolutions; i++) {
    if ((pattern & bitFilter) > 0 && (pattern & ~bitFilter) == 0) {
      // Extract the subgrid... The folloing is a pretty gnarly expression.
      // All it's doing is shifting the pattern according to which convolution
      // we're at and how wide (x) the filter is. Consider moving a 7x7
      // filter.. I has 4 possible locations and the shift expression evaluates
      // like this.
      //    i=0  ---> ((7-1)*(0/(8-7+1))+0)  --> 0
      //    i=1  ---> ((7-1)*(1/(8-7+1))+1)  --> 1
      //    i=2  ---> ((7-1)*(2/(8-7+1))+2)  --> 8
      //    i=3  ---> ((7-1)*(0/(8-7+1))+0)  --> 9
      ulong64 subgrid = pattern >> ((x-1)*(i/(8-x+1))+i);

      // Turn it into a key...
      //    |---8bytes---|------------------------------56bytes------------------------------|
      //         (i)                                   (subgrid)
      return (subgrid & (ULONG_MAX >> 8)) | ((ulong64)i << 56);
    }

    if (x < 8 && i%(8-x+1) == 0) {
      bitFilter <<= 1;
    }
    else {
      bitFilter <<= x;
    }
  }

  return 0;
}

void coverableCacheFree() {
  if (gCache != NULL) {
    pthread_mutex_lock(&gMutex);
    free(gCache->entries);
    free(gCache);
    gCache = NULL;
    pthread_mutex_unlock(&gMutex);
  }
}

void coverableCacheInit(int threadId, struct caching_filter f) {
  // Cache can't be greater than 2^56 because we reserve the top byte for offset.
  assert(f.x * f.y <= 56);

  if (gCache == NULL) {
    pthread_mutex_lock(&gMutex);
    gCache = (CoverageCache*)malloc(sizeof(CoverageCache));
    gCache->size = pow(2, 8 * f.y);
    printf("[Thread %d] allocating cache %llu entries\n", threadId, gCache->size);
    int convolutions = gCoverableConvolutions[(f.y-1)*8 + f.x-1];
    gCache->entries = (int *)calloc(gCache->size, sizeof(int) * convolutions);
    pthread_mutex_unlock(&gMutex);
  }
}

void coverableCacheSet(ulong64 key, int generations) {
  if (key > 0) {
    int offset = (int) (key >> 56);
    ulong64 entry = (key & (ULONG_MAX >> 8));
    gCache->entries[entry + offset] = generations;
  }
}

ulong64 coverableCacheGet(ulong64 key) {
  if (key > 0) {
    int offset = (int) (key >> 56);
    ulong64 entry = (key & (ULONG_MAX >> 8));
    return gCache->entries[entry + offset];
  }

  // Not cached
  return 0;
}


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

int countGenerations(ulong64 pattern, struct caching_filter f) {
  // Use Floyd's algorithm for cycle detection...  Perhaps slightly less
  // efficient (if it's a long period cycle) than a map structure, but when
  // implementing this on a GPU in CUDA those data structures are not readily
  // available.
  // https://en.wikipedia.org/wiki/Cycle_detection#Floyd's_tortoise_and_hare
  bool ended = false;
  int generations = 0;

  ulong64 coverableKey = 0;
  int coverableGenerations = 0;

  ulong64 slow = pattern;
  ulong64 fast = computeNextGeneration(slow);
  do {
    generations++;
    ulong64 nextSlow = computeNextGeneration(slow);

    // In the GOL algorithm implemented here, there are two "inner loops"...
    // The first is above inside of computeNextGeneration() where we iterate
    // the 64 bits of our 8x8 grid counting neighbors and updating the bits to
    // form the next generation.  This can be parallized in CUDA, but absent
    // parallelization it's not possible to prevent the need to evaluate each
    // bit in order to compute the next generation. The second "inner loop" is
    // here where we are looping through consecutive generations looking for a
    // subsequent generation to oscillate (see Floyd's cycle detection comment
    // above) OR become stable (totally dead or unchanging).
    //
    // The "coverable" code in this file is an attempt to shortcut this "inner
    // loop".  The idea is that if we can identify a subgrid of the full grid
    // that contains all of the live cells and that subgrid is sufficiently
    // small such that it could fit entirely in RAM then we can lazily cache
    // the results as we see them.
    //
    // An 8x8 grid has a vast amount of possibilities whereas ever possible
    // solution to a 4x7 grid could fit in 2GB of RAM.  As we're iteratively
    // computing subsequent generations we try to identify cases where all of
    // the live cells can be covered by that smaller subgrid.  When we see that
    // coverableByXxY() condition we first check to see if we've already fully
    // evaluated that subgrid if so we can shortcut our "inner loop".  If we
    // haven't cached that subgrid then we start counting coverableGenerations
    // until we reach a terminal state.  At that point we cache either the
    // number of coverableGenerations (if we ended in a stable form or totally
    // dead) OR 0 which we interpret as "coverable pattern that is infinite"
    // (and hence not relevant to us)
    if (f.x > 0 && f.y > 0 && coverableKey == 0) {
      ulong64 key = coverableByXxY(nextSlow, f.x, f.y);
      if (key > 0) {
        coverableKey = key;
        coverableGenerations = 1;
        int cachedGenerations = coverableCacheGet(key);
        if (cachedGenerations == INFINITE) {
          return INFINITE;
        }
        else if (cachedGenerations > 0)  {
          return generations + cachedGenerations;
        }
      }
    }
    else {
      // Keep track of how many generations since we identified our coverable
      // pattern... When we finally end this loop we'll cache it as a shortcut
      // in future trips through this loop.
      coverableGenerations++;
    }

    if (slow == nextSlow) {
      ended = true; // If we didn't change then we ended
      break;
    }
    slow = nextSlow;
    fast = computeNextGeneration(computeNextGeneration(fast));
  }
  while (slow != fast);
  ended = slow == 0; // If we died out then we ended


  //  While looping we found a coverable pattern that wasn't cached... Save it now
  if (coverableKey > 0) {
    coverableCacheSet(coverableKey, ended ? coverableGenerations : INFINITE);
  }

  return ended ? generations : INFINITE;
}

void *search(void *args) {
  prog_args *cli = (prog_args *)args;

  printf("[Thread %d] %s %llu - %llu\n",
         cli->threadId, cli->random ? "RANDOMLY searching" : "searching ALL", cli->beginAt, cli->endAt);

  // Initialize our coverage cache
  coverableCacheInit(cli->threadId, cli->filter);

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
      pattern = genrand64_int64() % ULONG_MAX;//(cli->endAt + 1 - cli->beginAt) + cli->beginAt;
    }

    int generations = countGenerations(pattern, cli->filter);
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
