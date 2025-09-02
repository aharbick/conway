#ifndef _GOL_H_
#define _GOL_H_

#include <assert.h>
#include <limits.h>
#include <locale.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <cstdint>

#ifdef __NVCC__
#include <cuda.h>
#else
#define __device__
#define __host__
#define __global__
#endif

#include <stdarg.h>

#include "airtable_client.h"
#include "display_utils.h"
#include "frame_utils.h"
#include "gol_core.h"  // Core GOL computation functions
#include "mt.h"        // For mersenne twister random numbers

// User-configurable defaults (can be overridden via command line)
#define DEFAULT_CUDA_GRID_SIZE 1024
#define DEFAULT_CUDA_THREADS_PER_BLOCK 1024

// Frame search constants
#define FRAME_SEARCH_GRID_SIZE 1024
#define FRAME_SEARCH_THREADS_PER_BLOCK 1024
#define FRAME_SEARCH_TOTAL_THREADS (FRAME_SEARCH_GRID_SIZE * FRAME_SEARCH_THREADS_PER_BLOCK)
#define FRAME_SEARCH_NUM_KERNELS 16
#define FRAME_SEARCH_DEFAULT_CHUNK_SIZE 32768
#define FRAME_SEARCH_MAX_CHUNK_SIZE 65536
#define FRAME_SEARCH_MAX_CANDIDATES (1 << 30)
#define FRAME_SEARCH_MAX_FRAMES (1 << 24)
#define FRAME_SEARCH_KERNEL_PATTERN_INCREMENT 0x1000000ULL
#define FRAME_SEARCH_TOTAL_MINIMAL_FRAMES 2102800

// Random search constants
#define RANDOM_SEARCH_CHUNK_SIZE (1 << 20)
#define RANDOM_SEARCH_MAX_CANDIDATES (1 << 20)

// Search type enumeration
typedef enum { SEARCH_TYPE_FRAME_BASED, SEARCH_TYPE_RANDOM } SearchType;

// Common buffer sizes
#define BINARY_STRING_BUFFER_SIZE 65
#define MESSAGE_BUFFER_SIZE 64

__host__ double getCurrentTime() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec / 1000000000.0;
}

typedef struct ProgramArgs {
  int threadId;
  int cpuThreads;
  int gpusToUse;
  int blockSize;
  int threadsPerBlock;
  bool random;
  bool verbose;
  bool testAirtable;
  bool resumeFromDatabase;
  uint64_t randomSamples;
  uint64_t beginAt;
  uint64_t endAt;
  uint64_t frameBeginIdx;
  uint64_t frameEndIdx;
  uint64_t chunkSize;
} ProgramArgs;

__host__ void printThreadStatus(int threadId, const char *format, ...) {
  va_list args;
  va_start(args, format);
  printf("[Thread %d - %llu] ", threadId, (uint64_t)time(NULL));
  vprintf(format, args);
  printf("\n");
  va_end(args);
}

#ifdef __NVCC__
#define cudaCheckError(ans) \
  { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}

typedef struct {
  uint64_t *d_candidates, *d_numCandidates, *d_bestPattern, *d_bestGenerations;
  uint64_t *h_candidates, *h_numCandidates, *h_bestPattern, *h_bestGenerations;
} SearchMemory;

__host__ SearchMemory *allocateSearchMemory(size_t candidateSize) {
  SearchMemory *mem = (SearchMemory *)malloc(sizeof(SearchMemory));

  // Allocate device memory
  cudaCheckError(cudaMalloc((void **)&mem->d_candidates, sizeof(uint64_t) * candidateSize));
  cudaCheckError(cudaMalloc((void **)&mem->d_numCandidates, sizeof(uint64_t)));
  cudaCheckError(cudaMalloc((void **)&mem->d_bestPattern, sizeof(uint64_t)));
  cudaCheckError(cudaMalloc((void **)&mem->d_bestGenerations, sizeof(uint64_t)));

  // Allocate host memory
  mem->h_candidates = (uint64_t *)calloc(candidateSize, sizeof(uint64_t));
  mem->h_numCandidates = (uint64_t *)malloc(sizeof(uint64_t));
  mem->h_bestPattern = (uint64_t *)malloc(sizeof(uint64_t));
  mem->h_bestGenerations = (uint64_t *)malloc(sizeof(uint64_t));

  return mem;
}

__host__ void freeSearchMemory(SearchMemory *mem) {
  if (!mem)
    return;

  // Free device memory
  cudaCheckError(cudaFree(mem->d_candidates));
  cudaCheckError(cudaFree(mem->d_numCandidates));
  cudaCheckError(cudaFree(mem->d_bestPattern));
  cudaCheckError(cudaFree(mem->d_bestGenerations));

  // Free host memory
  free(mem->h_candidates);
  free(mem->h_numCandidates);
  free(mem->h_bestPattern);
  free(mem->h_bestGenerations);

  free(mem);
}

__host__ void reportChunkResults(SearchMemory *mem, ProgramArgs *cli, double startTime, uint64_t frame,
                                 uint64_t frameIdx, int kernelIdx, int chunkIdx, bool isFrameComplete) {
  double chunkTime = getCurrentTime() - startTime;
  uint64_t patternsPerSec = (FRAME_SEARCH_TOTAL_THREADS * cli->chunkSize) / chunkTime;

  if (*mem->h_numCandidates <= 0) {
    printf("[Thread %d - %llu] WARN: NO PATTERNS FOUND frameIdx=%llu, kernelIdx=%d, chunkIdx=%d\n", cli->threadId,
           (uint64_t)time(NULL), frameIdx, kernelIdx, chunkIdx);

    airtableSendProgress(isFrameComplete, frameIdx, kernelIdx, chunkIdx, patternsPerSec, 0, 0, "ERROR", false);
    return;
  }

  char bestPatternBin[BINARY_STRING_BUFFER_SIZE] = {'\0'};
  asBinary(*mem->h_bestPattern, bestPatternBin);

  printf(
      "[Thread %d - %llu] frameIdx=%llu, kernelIdx=%d, chunkIdx=%d, bestGenerations=%d, bestPattern=%llu, "
      "bestPatternBin=%s, patternsPerSec=%llu\n",
      cli->threadId, (uint64_t)time(NULL), frameIdx, kernelIdx, chunkIdx, (int)*mem->h_bestGenerations,
      *mem->h_bestPattern, bestPatternBin, patternsPerSec);

  airtableSendProgress(isFrameComplete, frameIdx, kernelIdx, chunkIdx, patternsPerSec, (int)*mem->h_bestGenerations,
                       *mem->h_bestPattern, bestPatternBin, false);
}
#endif

///////////////////////////////////////////////////////////////////////////
// GLOBAL variables updated across threads

pthread_mutex_t gMutex = PTHREAD_MUTEX_INITIALIZER;
int gBestGenerations = 0;

__host__ bool updateBestGenerations(int generations) {
  pthread_mutex_lock(&gMutex);

  bool isNewGlobalBest = (gBestGenerations < generations);
  if (isNewGlobalBest) {
    gBestGenerations = generations;
  }

  pthread_mutex_unlock(&gMutex);
  return isNewGlobalBest;
}

// Core GOL functions now included from gol_core.h

#ifdef __NVCC__
__global__ void processCandidates(uint64_t *candidates, uint64_t *numCandidates, uint64_t *bestPattern,
                                  uint64_t *bestGenerations) {
  for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < *numCandidates; i += blockDim.x * gridDim.x) {
    uint64_t generations = countGenerations(candidates[i]);
    if (generations > 0) {  // Only process if it actually ended
      // Check to see if it's higher and emit it in best(Pattern|Generations)
      uint64_t old = atomicMax((unsigned long long *)bestGenerations, (unsigned long long)generations);
      if (old < generations) {
        *bestPattern = candidates[i];
      }
    }
  }
}

__global__ void findCandidates(uint64_t beginAt, uint64_t endAt, uint64_t *candidates, uint64_t *numCandidates) {
  uint64_t pattern = beginAt + (blockIdx.x * blockDim.x + threadIdx.x);
  uint64_t g1 = pattern;
  uint64_t generations = 0;

  while (pattern < endAt) {
    if (step6GenerationsAndCheck(&g1, pattern, &generations, candidates, numCandidates)) {
      // Pattern ended/cyclical or candidate found, advance to next pattern
      pattern += blockDim.x * gridDim.x;
      g1 = pattern;
    }
  }
}

__global__ void findCandidatesInKernel(uint64_t kernel, int chunkIdx, uint64_t chunkSize, uint64_t *candidates,
                                       uint64_t *numCandidates) {
  // Construct the starting pattern for this thread
  uint64_t pattern = kernel;
  pattern += ((uint64_t)(threadIdx.x & 15)) << 10;  // lower row of T bits
  pattern += ((uint64_t)(threadIdx.x >> 4)) << 17;  // upper row of T bits
  pattern += ((uint64_t)(blockIdx.x & 63)) << 41;   // lower row of B bits
  pattern += ((uint64_t)(blockIdx.x >> 6)) << 50;   // upper row of B bits

  // Each thread processes chunkSize patterns, starting from chunkIdx * chunkSize
  pattern += ((uint64_t)chunkIdx * chunkSize) * FRAME_SEARCH_KERNEL_PATTERN_INCREMENT;
  uint64_t endAt = pattern + (chunkSize * FRAME_SEARCH_KERNEL_PATTERN_INCREMENT);

  uint64_t g1 = pattern;
  uint64_t generations = 0;

  while (pattern != endAt) {
    if (step6GenerationsAndCheck(&g1, pattern, &generations, candidates, numCandidates)) {
      // Pattern ended/cyclical or candidate found, advance to next pattern
      pattern += FRAME_SEARCH_KERNEL_PATTERN_INCREMENT;  // Increment to next pattern
      g1 = pattern;
    }
  }
}


__host__ void executeCandidateSearch(SearchMemory *mem, ProgramArgs *cli, uint64_t start, uint64_t end) {
  // Phase 1: Find candidates
  *mem->h_numCandidates = 0;
  cudaCheckError(cudaMemcpy(mem->d_numCandidates, mem->h_numCandidates, sizeof(uint64_t), cudaMemcpyHostToDevice));
  findCandidates<<<cli->blockSize, cli->threadsPerBlock>>>(start, end, mem->d_candidates, mem->d_numCandidates);
  cudaCheckError(cudaGetLastError());
  cudaCheckError(cudaDeviceSynchronize());
  cudaCheckError(cudaMemcpy(mem->h_numCandidates, mem->d_numCandidates, sizeof(uint64_t), cudaMemcpyDeviceToHost));

  // Phase 2: Process candidates if found
  if (*mem->h_numCandidates > 0) {
    *mem->h_bestGenerations = 0;
    *mem->h_bestPattern = 0;
    cudaCheckError(
        cudaMemcpy(mem->d_bestGenerations, mem->h_bestGenerations, sizeof(uint64_t), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(mem->d_bestPattern, mem->h_bestPattern, sizeof(uint64_t), cudaMemcpyHostToDevice));
    processCandidates<<<cli->blockSize, cli->threadsPerBlock>>>(mem->d_candidates, mem->d_numCandidates,
                                                                mem->d_bestPattern, mem->d_bestGenerations);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    cudaCheckError(cudaMemcpy(mem->h_bestPattern, mem->d_bestPattern, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    cudaCheckError(
        cudaMemcpy(mem->h_bestGenerations, mem->d_bestGenerations, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    bool isNewGlobalBest = updateBestGenerations(*mem->h_bestGenerations);

    // Print result for new best
    if (isNewGlobalBest) {
      char bin[BINARY_STRING_BUFFER_SIZE] = {'\0'};
      asBinary(*mem->h_bestPattern, bin);
      printf("[Thread %d - %llu] %s%d generations : %llu : %s\n", cli->threadId, (uint64_t)time(NULL), "NEW HIGH ",
             (int)*mem->h_bestGenerations, *mem->h_bestPattern, bin);
    }
  }
}

__host__ void executeKernelSearch(SearchMemory *mem, ProgramArgs *cli, uint64_t frame, uint64_t frameIdx) {
  int numChunks = FRAME_SEARCH_MAX_CHUNK_SIZE / cli->chunkSize;

  // Loop over all kernels for this frame
  for (int kernelIdx = 0; kernelIdx < FRAME_SEARCH_NUM_KERNELS; kernelIdx++) {
    uint64_t kernel = constructKernel(frame, kernelIdx);

    // Loop over all chunks for this kernel
    for (int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
      double chunkStartTime = getCurrentTime();

      // Phase 1: Find candidates in kernel
      *mem->h_numCandidates = 0;
      cudaCheckError(cudaMemcpy(mem->d_numCandidates, mem->h_numCandidates, sizeof(uint64_t), cudaMemcpyHostToDevice));
      findCandidatesInKernel<<<FRAME_SEARCH_GRID_SIZE, FRAME_SEARCH_THREADS_PER_BLOCK>>>(
          kernel, chunkIdx, cli->chunkSize, mem->d_candidates, mem->d_numCandidates);
      cudaCheckError(cudaGetLastError());
      cudaCheckError(cudaDeviceSynchronize());
      cudaCheckError(cudaMemcpy(mem->h_numCandidates, mem->d_numCandidates, sizeof(uint64_t), cudaMemcpyDeviceToHost));

      // Phase 2: Process candidates if found
      if (*mem->h_numCandidates > 0) {
        *mem->h_bestGenerations = 0;
        *mem->h_bestPattern = 0;
        cudaCheckError(
            cudaMemcpy(mem->d_bestGenerations, mem->h_bestGenerations, sizeof(uint64_t), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(mem->d_bestPattern, mem->h_bestPattern, sizeof(uint64_t), cudaMemcpyHostToDevice));
        processCandidates<<<FRAME_SEARCH_GRID_SIZE, FRAME_SEARCH_THREADS_PER_BLOCK>>>(
            mem->d_candidates, mem->d_numCandidates, mem->d_bestPattern, mem->d_bestGenerations);
        cudaCheckError(cudaGetLastError());
        cudaCheckError(cudaDeviceSynchronize());

        cudaCheckError(cudaMemcpy(mem->h_bestPattern, mem->d_bestPattern, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        cudaCheckError(
            cudaMemcpy(mem->h_bestGenerations, mem->d_bestGenerations, sizeof(uint64_t), cudaMemcpyDeviceToHost));

        bool isNewGlobalBest = updateBestGenerations(*mem->h_bestGenerations);

        // Report chunk results if verbose output is enabled
        if (cli->verbose) {
          bool isFrameComplete = (kernelIdx == FRAME_SEARCH_NUM_KERNELS - 1) && (chunkIdx == numChunks - 1);
          reportChunkResults(mem, cli, chunkStartTime, frame, frameIdx, kernelIdx, chunkIdx, isFrameComplete);
        }
      }
    }
  }
}

__host__ void searchRandom(ProgramArgs *cli) {
  init_genrand64((uint64_t)time(NULL));

  SearchMemory *mem = allocateSearchMemory(RANDOM_SEARCH_MAX_CANDIDATES);

  // We're randomly searching..  I didn't get cuRAND to work so we randomize our batches. Each
  // call to findCandidates is sequential but we look at random locations across all possible.
  uint64_t chunkSize = RANDOM_SEARCH_CHUNK_SIZE;
  uint64_t iterations = cli->randomSamples / chunkSize;
  for (uint64_t i = 0; i < iterations; i++) {
    uint64_t start = genrand64_int64();
    uint64_t end = start + chunkSize;
    executeCandidateSearch(mem, cli, start, end);
  }

  // Add cleanup
  freeSearchMemory(mem);
}

// See the algorithm described in PERFORMANCE under "Eliminating Rotations"
__host__ void searchAll(ProgramArgs *cli) {
  SearchMemory *mem = allocateSearchMemory(FRAME_SEARCH_MAX_CANDIDATES);

  // Iterate through frame range or all possible 24-bit numbers and use spreadBitsToFrame to cover all 64-bit "frames"
  // Frames in the 8x8 grid are the 24 corner bits marked with F.

  //    FFFKKFFF
  //    FFBBBBFF
  //    FBBBBBBF
  //    PPPPPPPP
  //    PPPPPPPP
  //    FTTTTTTF
  //    FFTTTTFF
  //    FFFKKFFF

  // The 4 bits marked with K represent 16 starting patterns for 16 kernels.
  // Each kernel will have 1024 blocks and 1024 threads.  The given block/thread
  // will process the pattern with the matching B and T bits set.  Each thread
  // will process the 2^16 patterns marked with P bits.
  //
  // Because of this representation the kernel must be invoked with 1024 blocks
  // and 1024 threads otherwise we will not process all of the 2^40 patterns inside
  // the 2^24 frames.

  for (uint64_t i = 0, currentFrameIdx = 0; i < FRAME_SEARCH_MAX_FRAMES && currentFrameIdx < cli->frameEndIdx; i++) {
    uint64_t frame = spreadBitsToFrame(i);
    if (isMinimalFrame(frame)) {
      if (currentFrameIdx >= cli->frameBeginIdx) {
        executeKernelSearch(mem, cli, frame, currentFrameIdx);
      }
      currentFrameIdx++;
    }
  }

  // Cleanup
  freeSearchMemory(mem);
}
#else
// CPU version
__host__ void searchRandom(ProgramArgs *cli) {
  init_genrand64((uint64_t)time(NULL));

  for (uint64_t i = 0; i < cli->randomSamples; i++) {
    uint64_t pattern = genrand64_int64();
    int generations = countGenerations(pattern);
    bool isNewGlobalBest = updateBestGenerations(generations);

    if (isNewGlobalBest) {
      char bin[BINARY_STRING_BUFFER_SIZE] = {'\0'};
      asBinary(pattern, bin);
      printf("[Thread %d - %llu] %s%d generations : %llu : %s\n", cli->threadId, (uint64_t)time(NULL), "NEW HIGH ",
             generations, pattern, bin);
    }
  }
}

__host__ void searchAll(ProgramArgs *cli) {
  for (uint64_t i = cli->beginAt; i < cli->endAt; i++) {
    int generations = countGenerations(i);
    bool isNewGlobalBest = updateBestGenerations(generations);

    if (isNewGlobalBest) {
      char bin[BINARY_STRING_BUFFER_SIZE] = {'\0'};
      asBinary(i, bin);
      printf("[Thread %d - %llu] %s%d generations : %llu : %s\n", cli->threadId, (uint64_t)time(NULL), "NEW HIGH ",
             generations, i, bin);
    }
  }
}
#endif

__host__ const char *getSearchDescription(ProgramArgs *cli, char *buffer, size_t bufferSize) {
  if (cli->beginAt > 0 || cli->endAt > 0) {
    snprintf(buffer, bufferSize, "ALL in range (%llu - %llu)", cli->beginAt, cli->endAt);
  } else if (cli->frameBeginIdx > 0 || cli->frameEndIdx > 0) {
    snprintf(buffer, bufferSize, "ALL in frames (%llu - %llu)", cli->frameBeginIdx, cli->frameEndIdx);
  } else {
    strcpy(buffer, "ALL");
  }
  return buffer;
}

__host__ void *search(void *args) {
  ProgramArgs *cli = (ProgramArgs *)args;

#ifdef __NVCC__
  printThreadStatus(cli->threadId, "Running with CUDA enabled");
#else
  printThreadStatus(cli->threadId, "Running CPU-only version");
#endif

  if (cli->random) {
    printThreadStatus(cli->threadId, "searching RANDOMLY %llu candidates", cli->randomSamples);
    searchRandom(cli);
  } else {
    char searchRangeMessage[MESSAGE_BUFFER_SIZE] = {'\0'};
    getSearchDescription(cli, searchRangeMessage, sizeof(searchRangeMessage));
    printThreadStatus(cli->threadId, "searching %s", searchRangeMessage);
    searchAll(cli);
  }

  return NULL;
}

#endif
