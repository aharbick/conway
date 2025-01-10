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
#include "types.h"
#include "frame_utils.h"
#include "display_utils.h"

typedef struct prog_args {
  int threadId;
  int cpuThreads;
  int gpusToUse;
  int blockSize;
  int threadsPerBlock;
  bool kernelBatches;
  bool random;
  ulong64 randomSamples;
  ulong64 perfCount;
  ulong64 beginAt;
  ulong64 endAt;
} prog_args;

#ifdef __NVCC__
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}
#endif

///////////////////////////////////////////////////////////////////////////
// GLOBAL variables updated across threads

pthread_mutex_t gMutex = PTHREAD_MUTEX_INITIALIZER;
int gBestGenerations = 0;

__host__ void updateBestGenerations(int threadId, int generations, ulong64 pattern) {
  pthread_mutex_lock(&gMutex);
  if (gBestGenerations < generations) {
    char bin[65] = {'\0'};
    asBinary(pattern, bin);
    printf("[Thread %d] %d generations : %llu : %s\n", threadId, generations, pattern, bin);
    gBestGenerations = generations;
  }
  pthread_mutex_unlock(&gMutex);
}

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

  //  Run for up to 300 generations just checking if we cycle within 6
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
#ifdef JETSON_NANO
#define MAX_CANDIDATES 1<<20ULL  // 1MB for Jetson Nano (1<<20)
#define CHUNK_SIZE 1000*1000ULL  // Smaller chunks for Jetson Nano
#else
#define MAX_CANDIDATES 1<<30ULL  // 1GB for other devices (1<<30)
#define CHUNK_SIZE 1000*1000*1000ULL
#endif

__global__ void processCandidates(ulong64 *candidates, ulong64 *numCandidates,
                                  ulong64 *bestPattern, ulong64 *bestGenerations) {
  for (ulong64 i = blockIdx.x * blockDim.x + threadIdx.x; i < *numCandidates; i += blockDim.x * gridDim.x) {
    ulong64 generations = countGenerations(candidates[i]);
    if (generations > 0) {  // Only process if it actually ended
      // Check to see if it's higher and emit it in best(Pattern|Generations)
      ulong64 old = atomicMax(bestGenerations, generations);
      if (old < generations) {
        *bestPattern = candidates[i];
      }
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
    // Remove debug print to reduce memory pressure
    g1 = computeNextGeneration(g6);

    // Reduce the generations threshold for candidates
    if ((g1 == g2) || (g1 == g3) || (g1 == g4)) {
      generations = 0;
    }
    else if (generations >= 180) {
      ulong64 idx = atomicAdd(numCandidates, 1);
      if (idx < MAX_CANDIDATES) { // Add bounds check
        candidates[idx] = pattern;
      }
      generations = 0;
    }

    if (generations == 0) {
      pattern += blockDim.x * gridDim.x;
      g1 = pattern;
    }
  }
}

__global__ void searchKernel(ulong64 kernel_id, ulong64 batch, ulong64 batchSize, ulong64 *bestPattern, ulong64 *bestGenerations) {
  // Construct the starting pattern for this thread
  ulong64 pattern = kernel_id;
  pattern += ((ulong64)(threadIdx.x & 15)) << 10;  // lower row of T bits
  pattern += ((ulong64)(threadIdx.x >> 4)) << 17;  // upper row of T bits
  pattern += ((ulong64)(blockIdx.x & 63)) << 41;   // lower row of B bits
  pattern += ((ulong64)(blockIdx.x >> 6)) << 50;   // upper row of B bits
  pattern += batch * 0x1000000;                    // Batch is 0-based... start at this offset.

  for (ulong64 i = 0; i < batchSize; i++) {
    int generations = countGenerations(pattern);
    if (generations > 0) {  // Only process if it actually ended
      ulong64 old = atomicMax(bestGenerations, generations);
      if (old < generations) {
        *bestPattern = pattern;
      }
    }
    pattern += 0x1000000;  // Increment to next pattern
  }
}

// CUDA version
__host__ void searchRandom(prog_args *cli) {
  init_genrand64((ulong64) time(NULL));

  // Allocate memory on CUDA device
  ulong64 *d_candidates, *d_numCandidates, *d_bestPattern, *d_bestGenerations;
  cudaCheckError(cudaMalloc((void**)&d_candidates, sizeof(ulong64) * MAX_CANDIDATES));
  cudaCheckError(cudaMalloc((void**)&d_numCandidates, sizeof(ulong64)));
  cudaCheckError(cudaMalloc((void**)&d_bestPattern, sizeof(ulong64)));
  cudaCheckError(cudaMalloc((void**)&d_bestGenerations, sizeof(ulong64)));

  // Allocate memory on host
  ulong64 *h_candidates, *h_numCandidates, *h_bestPattern, *h_bestGenerations;
  h_candidates = (ulong64 *)calloc(MAX_CANDIDATES, sizeof(ulong64));
  h_numCandidates = (ulong64 *)malloc(sizeof(ulong64));
  h_bestPattern = (ulong64 *)malloc(sizeof(ulong64));
  h_bestGenerations = (ulong64 *)malloc(sizeof(ulong64));

  // We're randomly searching..  I didn't get cuRAND to work so we randomize our batches. Each
  // call to findCandidates is sequential but we look at random locations across all possible.
  ulong64 chunkSize = CHUNK_SIZE;
  ulong64 iterations = cli->randomSamples / chunkSize;
  for (ulong64 i = 0; i < iterations; i++) {
    ulong64 start = genrand64_int64();
    ulong64 end = start + chunkSize;

    // Clear Initialize dev memory and launch kernel to find candidates
    *h_numCandidates = 0;
    cudaCheckError(cudaMemcpy(d_numCandidates, h_numCandidates, sizeof(ulong64), cudaMemcpyHostToDevice));
    findCandidates<<<cli->blockSize,cli->threadsPerBlock>>>(start, end, d_candidates, d_numCandidates);
    cudaCheckError(cudaGetLastError()); // Check for launch errors
    cudaCheckError(cudaDeviceSynchronize()); // Wait for kernel to finish and check for errors

    // Get the number of candidates found
    cudaCheckError(cudaMemcpy(h_numCandidates, d_numCandidates, sizeof(ulong64), cudaMemcpyDeviceToHost));

    // Initialize best generations and pattern to 0 before processing and launch kernel to process candidates
    *h_bestGenerations = 0;
    *h_bestPattern = 0;
    cudaCheckError(cudaMemcpy(d_bestGenerations, h_bestGenerations, sizeof(ulong64), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_bestPattern, h_bestPattern, sizeof(ulong64), cudaMemcpyHostToDevice));
    processCandidates<<<cli->blockSize, cli->threadsPerBlock>>>(d_candidates, d_numCandidates, d_bestPattern, d_bestGenerations);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    // Copy down to host...
    cudaCheckError(cudaMemcpy(h_bestPattern, d_bestPattern, sizeof(ulong64), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(h_bestGenerations, d_bestGenerations, sizeof(ulong64), cudaMemcpyDeviceToHost));

    // Update global best and print pattern
    updateBestGenerations(cli->threadId, *h_bestGenerations, *h_bestPattern);
  }

  // Add cleanup
  cudaCheckError(cudaFree(d_candidates));
  cudaCheckError(cudaFree(d_numCandidates));
  cudaCheckError(cudaFree(d_bestPattern));
  cudaCheckError(cudaFree(d_bestGenerations));
  free(h_candidates);
  free(h_numCandidates);
  free(h_bestPattern);
  free(h_bestGenerations);
}

// See the algorithm described in PERFORMANCE under "Eliminating Rotations"
__host__ void searchAll(prog_args *cli) {
  // Allocate device memory
  ulong64 *d_bestPattern, *d_bestGenerations;
  cudaCheckError(cudaMalloc((void**)&d_bestPattern, sizeof(ulong64)));
  cudaCheckError(cudaMalloc((void**)&d_bestGenerations, sizeof(ulong64)));

  // Host variables for results
  ulong64 h_bestPattern, h_bestGenerations;

  // Keep track of how many patterns we've processed
  ulong64 processed = 0;

  // Iterate all possible 24-bit numbers and use spreadBitsToFrame to cover all 64-bit "frames"
  // see frame_util.h for more details.
  for (ulong64 i = 0; i < (1 << 24); i++) {
    ulong64 frame = spreadBitsToFrame(i);
    if (isMinimalFrame(frame)) {
      // Launch 16 kernels for this frame
      for (int i = 0; i < 16; i++) {
        ulong64 kernel_id = frame;
        kernel_id += ((ulong64)(i & 3)) << 3;     // lower pair of K bits
        kernel_id += ((ulong64)(i >> 2)) << 59;   // upper pair of K bits

        // Reset best generations for this kernel
        h_bestGenerations = 0;
        cudaCheckError(cudaMemcpy(d_bestGenerations, &h_bestGenerations, sizeof(ulong64), cudaMemcpyHostToDevice));

        // On a powerful GPU we can process 2^16 patterns per kernel invocation.
        // If kernelBatches is set, we'll process batches of 1024 for 64 iterations instead.
        //
        // This gives us the same total number of patterns (65536) but with smaller 
        // kernel invocations that are less likely to trigger the watchdog timer.
        ulong64 batchSize = cli->kernelBatches ? 1024 : 1<<16ULL;
        int batches = (1<<16ULL) / batchSize;
        for (int batchNum = 0; batchNum < batches; batchNum++) {
          searchKernel<<<cli->blockSize, cli->threadsPerBlock>>>(kernel_id, batchNum, batchSize, d_bestPattern, d_bestGenerations);
          cudaCheckError(cudaGetLastError());
          cudaCheckError(cudaDeviceSynchronize());
          processed += batchSize;
          if (processed >= cli->perfCount) {
            break;
          }
        }

        // Check results
        cudaCheckError(cudaMemcpy(&h_bestPattern, d_bestPattern, sizeof(ulong64), cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(&h_bestGenerations, d_bestGenerations, sizeof(ulong64), cudaMemcpyDeviceToHost));

        // Update global best and print pattern
        updateBestGenerations(cli->threadId, h_bestGenerations, h_bestPattern);
      }
    }
  }

  // Cleanup
  cudaCheckError(cudaFree(d_bestPattern));
  cudaCheckError(cudaFree(d_bestGenerations));
}
#else
// CPU version
__host__ void searchRandom(prog_args *cli) {
  init_genrand64((ulong64) time(NULL));

  for (ulong64 i = 0; i < cli->randomSamples; i++) {
    ulong64 pattern = genrand64_int64();
    int generations = countGenerations(pattern);
    updateBestGenerations(cli->threadId, generations, pattern);
  }
}

__host__ void searchAll(prog_args *cli) {
  for (ulong64 i = cli->beginAt; i < cli->endAt; i++) {
    int generations = countGenerations(i);
    updateBestGenerations(cli->threadId, generations, i);
  }
}
#endif

__host__ void *search(void *args) {
  prog_args *cli = (prog_args *)args;

#ifdef __NVCC__
  printf("[Thread %d] Running with CUDA enabled\n", cli->threadId);
  printf("[Thread %d] Kernel batches %s\n", cli->threadId, cli->kernelBatches ? "enabled" : "disabled");
#else
  printf("[Thread %d] Running CPU-only version\n", cli->threadId);
#endif

  if (cli->random) {
    printf("[Thread %d] searching RANDOMLY %llu candidates\n", cli->threadId, cli->randomSamples);
    searchRandom(cli);
  }
  else if (cli->perfCount > 0) {
    printf("[Thread %d] searching %llu patterns\n", cli->threadId, cli->perfCount);
    searchAll(cli);
  }
  else {
    char searchRangeMessage[64] = {'\0'};
    if (cli->beginAt > 0 && cli->endAt > 0) {
      snprintf(searchRangeMessage, sizeof(searchRangeMessage), "ALL in range (%llu - %llu)", cli->beginAt, cli->endAt);
    }
    else if (cli->beginAt == 0 && cli->endAt > 0) {
      snprintf(searchRangeMessage, sizeof(searchRangeMessage), "ALL in range (0 - %llu)", cli->endAt);
    }
    else {
      snprintf(searchRangeMessage, sizeof(searchRangeMessage), "ALL");
    }
    printf("[Thread %d] searching %s\n", cli->threadId, searchRangeMessage);
    searchAll(cli);
  }

  return NULL;
}

#endif

