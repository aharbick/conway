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
__global__ void findPromisingPatterns(ulong64 beginAt, ulong64 endAt, ulong64 *promising, ulong64 *numPromising) {
  ulong64 pattern = beginAt + (blockIdx.x * blockDim.x + threadIdx.x);
  ulong64 g1, g2, g3, g4, g5, g6;
  g1 = pattern;
  int generations = 0;
  while (pattern < endAt) {
    generations += 6;
    g2 = computeNextGeneration(g1);
    g3 = computeNextGeneration(g2);
    g4 = computeNextGeneration(g3);
    g5 = computeNextGeneration(g4);
    g6 = computeNextGeneration(g5);
    g1 = computeNextGeneration(g6);

    if (g1 == 0 || (g1 == g2) || (g1 == g3) || (g1 == g4)) {
      // pattern is ended or is cyclical... reset counter ready to advance to next pattern.
      generations = 0;
    } else if (generations >= 100) {
      // Based on a sample of 10M random patterns 99.84% patterns end before 100 generations.
      // Save longer patterns for further analysis.
      ulong64 idx = atomicAdd(numPromising, 1);
      promising[idx] = pattern;
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
  // Allocate memory on CUDA device and locally on host
  ulong64 *devPromising, *hostPromising;
  hostPromising = (ulong64 *)malloc(sizeof(ulong64) * 1024 * 1024);
  cudaMalloc((void**)&devPromising, sizeof(ulong64) * 1024 * 1024);

  int *devNumPromising, *hostNumPromising;
  hostNumPromising = (int *)malloc(sizeof(int));
  cudaMalloc((void**)&devNumPromising, sizeof(int));

  ulong64 chunk = 1;
  ulong64 chunkSize = 1024*1024*1024; // 1B patterns (1024 per block per thread)
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

    // We check ~1B patterns to find any that are longer than 100 generations without simple periods.
    findPromisingPatterns<<<cli->blockSize, cli->threadsPerBlock>>>(start, end, devPromising, devNumPromising);
    cudaMemcpy(hostNumPromising, devNumPromising, sizeof(ulong64), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostPromising, devPromising, *devNumPromising * sizeof(ulong64), cudaMemcpyDeviceToHost);

    // Now iterate over the candidates doing full checks.
    for (int i = 0; i < *hostNumPromising; i++) {
      int generations = countGenerations(hostPromising[i]);
      pthread_mutex_lock(&gMutex);
      if (gBestGenerations <= generations) {
        char bin[65] = {'\0'};
        asBinary(hostPromising[i], bin);
        printf("[Thread %d] %d generations : %llu : %s\n", cli->threadId, generations, hostPromising[i], bin);
        gBestGenerations = generations;
      }
      pthread_mutex_unlock(&gMutex);
    }

    chunk++;
    i += chunkSize;

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
