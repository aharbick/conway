#include <cuda.h>
#include <pthread.h>
#include "cli.h"
#include "gol_cuda.h"
#include "utils.h"
#include "mt.h"

///////////////////////////////////////////////////////////////////////////
// GLOBAL variables updated across CPU threads

pthread_mutex_t gMutex = PTHREAD_MUTEX_INITIALIZER;
int gBestGenerations = 0;

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

void *search(void *args) {
  prog_args *cli = (prog_args *)args;

  printf("[Thread %d] %s %llu - %llu\n",
         cli->threadId, cli->random ? "RANDOMLY searching" : "searching ALL", cli->beginAt, cli->endAt);
  cudaSetDevice((cli->gpusToUse > 1) ? cli->threadId : 0);

  if (cli->random) {
    // Initialize Random number generator
    init_genrand64((ulong64) time(NULL));
  }

  // Allocate memory on CUDA device and locally on host to get the best answers
  ulong64 *devBestPattern, *hostBestPattern;
  hostBestPattern = (ulong64 *)malloc(sizeof(ulong64));
  cudaMalloc((void**)&devBestPattern, sizeof(ulong64));

  ulong64 *devBestGenerations, *hostBestGenerations;
  hostBestGenerations = (ulong64 *)malloc(sizeof(ulong64));
  *hostBestGenerations = 0;
  cudaMalloc((void**)&devBestGenerations, sizeof(ulong64));

  ulong64 chunk = 1;
  ulong64 chunkSize = 1024*1024;
  ulong64 i = cli->beginAt;
  while (i < cli->endAt) {
    ulong64 start = i;
    ulong64 end = (start + chunkSize) > cli->endAt ? cli->endAt : start+chunkSize;
    if (cli->random) {
      // We're randomly searching..  I didn't get cuRAND to work so we randomize our batches. Each
      // call to evaluateRange is sequential but we look at random locations across all possible.
      start = genrand64_int64() % ULONG_MAX;
      end  = start + chunkSize;
    }

    cudaMemcpy(devBestGenerations, &gBestGenerations, sizeof(ulong64), cudaMemcpyHostToDevice);
    evaluateRange<<<cli->blockSize, cli->threadsPerBlock>>>(start, end, devBestPattern, devBestGenerations);

    // Copy device answer to host and emit
    ulong64 prev = *hostBestPattern;
    cudaMemcpy(hostBestPattern, devBestPattern, sizeof(ulong64), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostBestGenerations, devBestGenerations, sizeof(ulong64), cudaMemcpyDeviceToHost);
    if (prev != *hostBestPattern) {
      pthread_mutex_lock(&gMutex);
      if (gBestGenerations < *hostBestGenerations) {
        char bin[65] = {'\0'};
        asBinary(*hostBestPattern, bin);
        printf("[Thread %d] %d generations : %llu : %s\n", cli->threadId, *hostBestGenerations, *hostBestPattern, bin);
        gBestGenerations = *hostBestGenerations;
      }
      pthread_mutex_unlock(&gMutex);
    }

    if (chunk % 1000 == 0) { // every billion
      printf("[Thread %d] Up to %lu, %2.10f%% complete\n", cli->threadId, i, (float) i/cli->endAt * 100);
    }

    chunk++;
    i += chunkSize;
  }

  return NULL;
}
