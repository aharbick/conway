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

__host__ __device__ void add2(ulong64 a, ulong64 b, ulong64 &s0, ulong64 &s1) {
  s0 = a ^ b ;
  s1 = a & b ;
}

__host__ __device__ void add3(ulong64 a, ulong64 b, ulong64 c, ulong64 &s0, ulong64 &s1) {
  ulong64 t0, t1, t2 ;
  add2(a, b, t0, t1) ;
  add2(t0, c, s0, t2) ;
  s1 = t1 ^ t2 ;
}

__host__ __device__ ulong64 computeNextGeneration(ulong64 a) {
  ulong64 s0, sh2, a0, a1, sll, slh ;
  add2((a & 0x7f7f7f7f7f7f7f7f)<<1, (a & 0xFEFEFEFEFEFEFEFE)>>1, s0, sh2) ;
  add2(s0, a, a0, a1) ;
  a1 |= sh2 ;
  add3(a0>>8, a0<<8, s0, sll, slh) ;
  ulong64 y = a1 >> 8 ;
  ulong64 x = a1 << 8 ;
  return (x^y^sh2^slh)&((x|y)^(sh2|slh))&(sll|a) ;
}

__host__ __device__ int countGenerations(ulong64 pattern) {
  bool ended = false;
  int generations = 0;

  //  Run for up to  generations just checking if we cycle within 6
  //  generations or die out.
  ulong64 g1, g2, g3, g4, g5, g6;
  g1 = pattern;
  do {
    generations+=6;
    g2 = computeNextGeneration(g1);
    g3 = computeNextGeneration(g2);
    g4 = computeNextGeneration(g3);
    g5 = computeNextGeneration(g4);
    g6 = computeNextGeneration(g5);
    g1 = computeNextGeneration(g6);

    if (g1 == 0) {
      ended = true; // died out

      // Adjust the age
      if (g2 == 0) {generations-=5;}
      else if (g3 == 0) {generations-=4;}
      else if (g4 == 0) {generations-=3;}
      else if (g5 == 0) {generations-=2;}
      else if (g6 == 0) {generations-=1;}

      break;
    }

    if ((g1 == g2) || (g1 == g3) || (g1 == g4)) {
      // periodic
      break;
    }

  }
  while (generations < 300);

  // Fall back to Floyd's cycle detection algorithm if we haven't
  // we didn't exit the previous loop because of die out or cycle.
  if (!ended && generations >= 300) {
    ulong64 slow = g1;
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
  }

  return ended ? generations : 0;
}

#ifdef __NVCC__
__global__ void processCandidates(ulong64 *candidates, ulong64 *numCandidates,
                                  ulong64 *bestPattern, ulong64 *bestGenerations) {
  for (ulong64 i = blockIdx.x * blockDim.x + threadIdx.x; i < *numCandidates; i += blockDim.x * gridDim.x) {
    ulong64 generations = countGenerations(candidates[i]);
    ulong64 old = atomicMax(bestGenerations, generations);
    if (old < generations) {
      *bestPattern = candidates[i];
    }
  }
}

__global__ void findCandidates(ulong64 beginAt, ulong64 endAt,
                               ulong64 *candidates, ulong64 *numCandidates) {
  ulong64 pattern = beginAt + (blockIdx.x * blockDim.x + threadIdx.x);
  ulong64 g1, g2, g3, g4, g5, g6;
  g1 = pattern;
  ulong64 generations = 0;
  while (pattern < endAt) {
    generations += 6;
    g2 = computeNextGeneration(g1);
    g3 = computeNextGeneration(g2);
    g4 = computeNextGeneration(g3);
    g5 = computeNextGeneration(g4);
    g6 = computeNextGeneration(g5);
    g1 = computeNextGeneration(g6);

    if ((g1 == g2) || (g1 == g3) || (g1 == g4)) {
      // pattern is ended or is cyclical... reset counter ready to advance to next pattern.
      generations = 0;
    }
    else if (generations == 144) { // Must be multiple of 6
      ulong64 idx = atomicAdd(numCandidates, 1);
      candidates[idx] = pattern;
      // reset counter ready to advance to next pattern:
      generations = 0;
    }

    if (generations == 0) {
      // we reset the generation counter, so load the next pattern:
      pattern += blockDim.x * gridDim.x;
      g1 = pattern;
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
  // Allocate memory on CUDA device
  ulong64 *d_candidates, *d_numCandidates, *d_bestPattern, *d_bestGenerations;
  cudaMalloc((void**)&d_candidates, sizeof(ulong64) * 1<<30);
  cudaMalloc((void**)&d_numCandidates, sizeof(ulong64));
  cudaMalloc((void**)&d_bestPattern, sizeof(ulong64));
  cudaMalloc((void**)&d_bestGenerations, sizeof(ulong64));

  // Allocate memory on host
  ulong64 *h_candidates, *h_numCandidates, *h_bestPattern, *h_bestGenerations;
  h_candidates = (ulong64 *)calloc(1<<30, sizeof(ulong64));
  h_numCandidates = (ulong64 *)malloc(sizeof(ulong64));
  h_bestPattern = (ulong64 *)malloc(sizeof(ulong64));
  h_bestGenerations = (ulong64 *)malloc(sizeof(ulong64));

  ulong64 chunksize = 1024*1024*1024; // ~1BILLION
  ulong64 start = cli->beginAt;
  ulong64 end;
  while (start < cli->endAt) {
    if (cli->random) {
      // We're randomly searching..  I didn't get cuRAND to work so we randomize our batches. Each
      // call to findCandidates is sequential but we look at random locations across all possible.
      start = genrand64_int64() % ULONG_MAX;
    }
    end = (start + chunksize) > cli->endAt ? cli->endAt : start + chunksize;

    // Clear Initialize dev memory and launch kernel to find candidates
    *h_numCandidates = 0;
    cudaMemcpy(d_numCandidates, h_numCandidates, sizeof(ulong64), cudaMemcpyHostToDevice);
    findCandidates<<<cli->blockSize,cli->threadsPerBlock>>>(start, end, d_candidates, d_numCandidates);

    // Initialized best generations and launch kernel to process candidates
    processCandidates<<<cli->blockSize, cli->threadsPerBlock>>>(d_candidates, d_numCandidates, d_bestPattern, d_bestGenerations);

    // Copy down to host...
    cudaMemcpy(h_bestPattern, d_bestPattern, sizeof(ulong64), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bestGenerations, d_bestGenerations, sizeof(ulong64), cudaMemcpyDeviceToHost);
    pthread_mutex_lock(&gMutex);
    if (gBestGenerations < *h_bestGenerations) {
      char bin[65] = {'\0'};
      asBinary(*h_bestPattern, bin);
      printf("[Thread %d] %d generations : %llu : %s\n", cli->threadId, *h_bestGenerations, *h_bestPattern, bin);
      gBestGenerations = *h_bestGenerations;
    }
    pthread_mutex_unlock(&gMutex);

    start = (end+1) > cli->endAt ? cli->endAt : end+1;

    printf("."); // every billion patterns
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
