#ifndef _GOL_H_
#define _GOL_H_

#include <assert.h>
#include <stdbool.h>
#include <pthread.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>

#ifdef __NVCC__
#include <cuda.h>
#else
#define __device__
#define __host__
#define __global__
#endif

#include "mt.h" // For mersenne twister random numbers

typedef unsigned long long ulong64;
typedef struct prog_args {
  int threadId;
  int cpuThreads;
  int gpusToUse;
  int blockSize;
  int threadsPerBlock;
  bool random;
  bool unrestrictedRandom;
  ulong64 beginAt;
  ulong64 endAt;
  ulong64 perf_iterations;
} prog_args;

static const int INFINITE = -1;

void asBinary(ulong64 number, char *buf) {
  for (int i = 63; i >= 0; i--) {
    buf[-i+63] = (number >> i) & 1 ? '1' : '0';
  }
}

void printPattern(ulong64 number) {
  char pat[65] = {'\0'};
  asBinary(number, pat);
  for (int i = 0; i < 64; i++) {
    printf(" %c ", pat[i]);
    if ((i+1) % 8 == 0) {
      printf("\n");
    }
  }
  printf("\n");
}

///////////////////////////////////////////////////////////////////////////
// GLOBAL variables updated across threads

pthread_mutex_t gMutex = PTHREAD_MUTEX_INITIALIZER;
int gBestGenerations = 0;

__device__ void add2(ulong64 a, ulong64 b, ulong64 &s0, ulong64 &s1) {
  s0 = a ^ b ;
  s1 = a & b ;
}

__device__ void add3(ulong64 a, ulong64 b, ulong64 c, ulong64 &s0, ulong64 &s1) {
  ulong64 t0, t1, t2 ;
  add2(a, b, t0, t1) ;
  add2(t0, c, s0, t2) ;
  s1 = t1 ^ t2 ;
}

__device__ ulong64 computeNextGeneration(ulong64 a) {
  ulong64 s0, sh2, a0, a1, sll, slh ;
  add2((a & 0x7f7f7f7f7f7f7f7f)<<1, (a & 0xFEFEFEFEFEFEFEFE)>>1, s0, sh2) ;
  add2(s0, a, a0, a1) ;
  a1 |= sh2 ;
  add3(a0>>8, a0<<8, s0, sll, slh) ;
  ulong64 y = a1 >> 8 ;
  ulong64 x = a1 << 8 ;
  return (x^y^sh2^slh)&((x|y)^(sh2|slh))&(sll|a) ;
}

__device__ int countGenerations(ulong64 pattern) {
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

#ifdef __NVCC__
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
#endif

__host__ void *search(void *args) {
  prog_args *cli = (prog_args *)args;

  if (cli->random) {
    // Initialize Random number generator
    init_genrand64((ulong64) time(NULL));
    char range[64] = {'\0'};
    if (cli->unrestrictedRandom) {
      sprintf(range, "(1 - ULONG_MAX)");
    }
    else {
      sprintf(range, "(%llu - %llu)", cli->beginAt, cli->endAt);
    }
    printf("[Thread %d] RANDOMLY searching %llu candidates %s\n", cli->threadId, cli->endAt - cli->beginAt, range);

  }
  else {
    printf("[Thread %d] searching ALL %llu - %llu\n", cli->threadId, cli->beginAt, cli->endAt);
  }

#ifdef __NVCC__
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
    if (chunk % 1000 == 0) { //
      printf("."); // every billion patterns
    }

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
      if (gBestGenerations <= *hostBestGenerations) {
        char bin[65] = {'\0'};
        asBinary(*hostBestPattern, bin);
        printf("[Thread %d] %d generations : %llu : %s\n", cli->threadId, *hostBestGenerations, *hostBestPattern, bin);
        gBestGenerations = *hostBestGenerations;
      }
      pthread_mutex_unlock(&gMutex);
    }

    chunk++;
    i += chunkSize;
  }
#else
  for (ulong64 i = cli->beginAt; i <= cli->endAt; i++) {
    if (i % 10000000 == 0) {
      printf("."); // every 10m patterns
    }

    // Sequential or random pattern...
    ulong64 pattern = i;
    if (cli->random) {
      ulong64 max = cli->unrestrictedRandom ? ULONG_MAX : (cli->endAt + 1 - cli->beginAt) + cli->beginAt;
      pattern = genrand64_int64() % max;
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
#endif

  return NULL;
}

#endif
