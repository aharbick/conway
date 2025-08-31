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
#include <locale.h>

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
#include "airtable_client.h"
#include <stdarg.h>

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
#define FRAME_SEARCH_MAX_CANDIDATES (1<<30)
#define FRAME_SEARCH_MAX_FRAMES (1 << 24)
#define FRAME_SEARCH_KERNEL_PATTERN_INCREMENT 0x1000000ULL
#define FRAME_SEARCH_TOTAL_FRAMES 2102800

// Conway's Game of Life bit manipulation masks
#define GOL_HORIZONTAL_SHIFT_MASK 0x7f7f7f7f7f7f7f7f
#define GOL_VERTICAL_SHIFT_MASK 0xFEFEFEFEFEFEFEFE

// Random search constants
#define RANDOM_SEARCH_CHUNK_SIZE (1<<20)
#define RANDOM_SEARCH_MAX_CANDIDATES (1<<20)

// Generation counting constants
#define MIN_CANDIDATE_GENERATIONS 180
#define FAST_SEARCH_MAX_GENERATIONS 300

// Search type enumeration
typedef enum {
  SEARCH_TYPE_FRAME_BASED,
  SEARCH_TYPE_RANDOM
} search_type_t;

// Common buffer sizes
#define BINARY_STRING_BUFFER_SIZE 65
#define MESSAGE_BUFFER_SIZE 64

__host__ double getCurrentTime() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec / 1000000000.0;
}

typedef struct prog_args {
  int threadId;
  int cpuThreads;
  int gpusToUse;
  int blockSize;
  int threadsPerBlock;
  bool random;
  bool verbose;
  bool testAirtable;
  bool resumeFromDatabase;
  ulong64 randomSamples;
  ulong64 beginAt;
  ulong64 endAt;
  ulong64 frameBeginAt;
  ulong64 frameEndAt;
  ulong64 chunkSize;
} prog_args;

__host__ void printThreadStatus(int threadId, const char* format, ...) {
  va_list args;
  va_start(args, format);
  printf("[Thread %d - %llu] ", threadId, (ulong64)time(NULL));
  vprintf(format, args);
  printf("\n");
  va_end(args);
}

__host__ void printChunkStats(prog_args* cli, double startTime, ulong64 processedFrames,
                             ulong64 startFrame, int kernelId, int chunk, bool is_frame_complete) {
  double chunkEndTime = getCurrentTime();
  double chunkTime = chunkEndTime - startTime;
  double rate = (FRAME_SEARCH_TOTAL_THREADS * cli->chunkSize) / chunkTime;
  ulong64 frameId = startFrame + processedFrames;
  printThreadStatus(cli->threadId,
      "Processed frame=%llu, kernel=%d, chunk=%d (%'llu patterns/sec)",
      frameId, kernelId, chunk, (ulong64)rate);

  airtable_send_progress(frameId, kernelId, chunk, rate, is_frame_complete, false);
}

#ifdef __NVCC__
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}

typedef struct {
  ulong64 *d_candidates, *d_numCandidates, *d_bestPattern, *d_bestGenerations;
  ulong64 *h_candidates, *h_numCandidates, *h_bestPattern, *h_bestGenerations;
} SearchMemory;

__host__ SearchMemory* allocateSearchMemory(size_t candidateSize) {
  SearchMemory* mem = (SearchMemory*)malloc(sizeof(SearchMemory));

  // Allocate device memory
  cudaCheckError(cudaMalloc((void**)&mem->d_candidates, sizeof(ulong64) * candidateSize));
  cudaCheckError(cudaMalloc((void**)&mem->d_numCandidates, sizeof(ulong64)));
  cudaCheckError(cudaMalloc((void**)&mem->d_bestPattern, sizeof(ulong64)));
  cudaCheckError(cudaMalloc((void**)&mem->d_bestGenerations, sizeof(ulong64)));

  // Allocate host memory
  mem->h_candidates = (ulong64*)calloc(candidateSize, sizeof(ulong64));
  mem->h_numCandidates = (ulong64*)malloc(sizeof(ulong64));
  mem->h_bestPattern = (ulong64*)malloc(sizeof(ulong64));
  mem->h_bestGenerations = (ulong64*)malloc(sizeof(ulong64));

  return mem;
}

__host__ void freeSearchMemory(SearchMemory* mem) {
  if (!mem) return;

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
#endif

///////////////////////////////////////////////////////////////////////////
// GLOBAL variables updated across threads

pthread_mutex_t gMutex = PTHREAD_MUTEX_INITIALIZER;
int gBestGenerations = 0;

__host__ void updateBestGenerations(int threadId, int generations, ulong64 pattern) {
  pthread_mutex_lock(&gMutex);

  char bin[BINARY_STRING_BUFFER_SIZE] = {'\0'};
  asBinary(pattern, bin);

  bool isNewGlobalBest = (gBestGenerations < generations);
  if (isNewGlobalBest) {
    // Update our new best.
    gBestGenerations = generations;
  }

  // Print out result and sent to airbase
  printf("[Thread %d - %llu] %s%d generations : %llu : %s\n",
         threadId, (ulong64)time(NULL), isNewGlobalBest ? "NEW HIGH " : "", generations, pattern, bin);
  airtable_send_result(generations, pattern, bin, false);

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
  add2((a & GOL_HORIZONTAL_SHIFT_MASK)<<1, (a & GOL_VERTICAL_SHIFT_MASK)>>1, s0, sh2) ;
  add2(s0, a, a0, a1) ;
  a1 |= sh2 ;
  add3(a0>>8, a0<<8, s0, sll, slh) ;
  ulong64 y = a1 >> 8 ;
  ulong64 x = a1 << 8 ;
  return (x^y^sh2^slh)&((x|y)^(sh2|slh))&(sll|a) ;
}

__host__ __device__ int adjustGenerationsForDeadout(int generations, ulong64 g2, ulong64 g3, ulong64 g4, ulong64 g5, ulong64 g6) {
  if (g2 == 0) return generations - 5;
  if (g3 == 0) return generations - 4;
  if (g4 == 0) return generations - 3;
  if (g5 == 0) return generations - 2;
  if (g6 == 0) return generations - 1;
  return generations;
}

#ifdef __NVCC__
__device__ bool step6GenerationsAndCheck(ulong64* g1, ulong64 pattern, ulong64* generations,
                                        ulong64* candidates, ulong64* numCandidates) {
  *generations += 6;
  ulong64 g2 = computeNextGeneration(*g1);
  ulong64 g3 = computeNextGeneration(g2);
  ulong64 g4 = computeNextGeneration(g3);
  ulong64 g5 = computeNextGeneration(g4);
  ulong64 g6 = computeNextGeneration(g5);
  *g1 = computeNextGeneration(g6);

  if ((*g1 == g2) || (*g1 == g3) || (*g1 == g4)) {
    *generations = 0;
    return true; // Pattern ended/cyclical, advance to next
  }

  if (*generations >= MIN_CANDIDATE_GENERATIONS) {
    ulong64 idx = atomicAdd(numCandidates, 1);
    candidates[idx] = pattern;
    *generations = 0;
    return true; // Candidate found, advance to next
  }

  return false; // Continue with current pattern
}

#endif

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
      generations = adjustGenerationsForDeadout(generations, g2, g3, g4, g5, g6);
      break;
    }

    if ((g1 == g2) || (g1 == g3) || (g1 == g4)) {
      // periodic
      break;
    }

  }
  while (generations < FAST_SEARCH_MAX_GENERATIONS);

  // Fall back to Floyd's cycle detection algorithm if we haven't
  // we didn't exit the previous loop because of die out or cycle.
  if (!ended && generations >= FAST_SEARCH_MAX_GENERATIONS) {
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
  ulong64 g1 = pattern;
  ulong64 generations = 0;

  while (pattern < endAt) {
    if (step6GenerationsAndCheck(&g1, pattern, &generations, candidates, numCandidates)) {
      // Pattern ended/cyclical or candidate found, advance to next pattern
      pattern += blockDim.x * gridDim.x;
      g1 = pattern;
    }
  }
}

__global__ void findCandidatesInKernel(ulong64 kernel, int chunk, ulong64 chunkSize, ulong64 *candidates, ulong64 *numCandidates) {
  // Construct the starting pattern for this thread
  ulong64 pattern = kernel;
  pattern += ((ulong64)(threadIdx.x & 15)) << 10;  // lower row of T bits
  pattern += ((ulong64)(threadIdx.x >> 4)) << 17;  // upper row of T bits
  pattern += ((ulong64)(blockIdx.x & 63)) << 41;   // lower row of B bits
  pattern += ((ulong64)(blockIdx.x >> 6)) << 50;   // upper row of B bits

  // Each thread processes chunkSize patterns, starting from chunk * chunkSize
  pattern += ((ulong64)chunk * chunkSize) * FRAME_SEARCH_KERNEL_PATTERN_INCREMENT;
  ulong64 endAt = pattern + (chunkSize * FRAME_SEARCH_KERNEL_PATTERN_INCREMENT);

  ulong64 g1 = pattern;
  ulong64 generations = 0;

  while (pattern != endAt) {
    if (step6GenerationsAndCheck(&g1, pattern, &generations, candidates, numCandidates)) {
      // Pattern ended/cyclical or candidate found, advance to next pattern
      pattern += FRAME_SEARCH_KERNEL_PATTERN_INCREMENT;  // Increment to next pattern
      g1 = pattern;
    }
  }
}

__host__ ulong64 constructKernel(ulong64 frame, int kernelIndex) {
  ulong64 kernel = frame;
  kernel += ((ulong64)(kernelIndex & 3)) << 3;      // lower pair of K bits
  kernel += ((ulong64)(kernelIndex >> 2)) << 59;    // upper pair of K bits
  return kernel;
}

__host__ void executeCandidateSearch(SearchMemory* mem, prog_args* cli, ulong64 start, ulong64 end) {
  // Phase 1: Find candidates
  *mem->h_numCandidates = 0;
  cudaCheckError(cudaMemcpy(mem->d_numCandidates, mem->h_numCandidates, sizeof(ulong64), cudaMemcpyHostToDevice));
  findCandidates<<<cli->blockSize, cli->threadsPerBlock>>>(start, end, mem->d_candidates, mem->d_numCandidates);
  cudaCheckError(cudaGetLastError());
  cudaCheckError(cudaDeviceSynchronize());
  cudaCheckError(cudaMemcpy(mem->h_numCandidates, mem->d_numCandidates, sizeof(ulong64), cudaMemcpyDeviceToHost));

  // Phase 2: Process candidates if found
  if (*mem->h_numCandidates > 0) {
    *mem->h_bestGenerations = 0;
    *mem->h_bestPattern = 0;
    cudaCheckError(cudaMemcpy(mem->d_bestGenerations, mem->h_bestGenerations, sizeof(ulong64), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(mem->d_bestPattern, mem->h_bestPattern, sizeof(ulong64), cudaMemcpyHostToDevice));
    processCandidates<<<cli->blockSize, cli->threadsPerBlock>>>(mem->d_candidates, mem->d_numCandidates, mem->d_bestPattern, mem->d_bestGenerations);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    cudaCheckError(cudaMemcpy(mem->h_bestPattern, mem->d_bestPattern, sizeof(ulong64), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(mem->h_bestGenerations, mem->d_bestGenerations, sizeof(ulong64), cudaMemcpyDeviceToHost));
    updateBestGenerations(cli->threadId, *mem->h_bestGenerations, *mem->h_bestPattern);
  }
}

__host__ void executeKernelSearch(SearchMemory* mem, prog_args* cli, ulong64 kernel, int chunk) {
  // Phase 1: Find candidates in kernel
  *mem->h_numCandidates = 0;
  cudaCheckError(cudaMemcpy(mem->d_numCandidates, mem->h_numCandidates, sizeof(ulong64), cudaMemcpyHostToDevice));
  findCandidatesInKernel<<<FRAME_SEARCH_GRID_SIZE,FRAME_SEARCH_THREADS_PER_BLOCK>>>(kernel, chunk, cli->chunkSize, mem->d_candidates, mem->d_numCandidates);
  cudaCheckError(cudaGetLastError());
  cudaCheckError(cudaDeviceSynchronize());
  cudaCheckError(cudaMemcpy(mem->h_numCandidates, mem->d_numCandidates, sizeof(ulong64), cudaMemcpyDeviceToHost));

  // Phase 2: Process candidates if found
  if (*mem->h_numCandidates > 0) {
    *mem->h_bestGenerations = 0;
    *mem->h_bestPattern = 0;
    cudaCheckError(cudaMemcpy(mem->d_bestGenerations, mem->h_bestGenerations, sizeof(ulong64), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(mem->d_bestPattern, mem->h_bestPattern, sizeof(ulong64), cudaMemcpyHostToDevice));
    processCandidates<<<FRAME_SEARCH_GRID_SIZE,FRAME_SEARCH_THREADS_PER_BLOCK>>>(mem->d_candidates, mem->d_numCandidates, mem->d_bestPattern, mem->d_bestGenerations);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    cudaCheckError(cudaMemcpy(mem->h_bestPattern, mem->d_bestPattern, sizeof(ulong64), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(mem->h_bestGenerations, mem->d_bestGenerations, sizeof(ulong64), cudaMemcpyDeviceToHost));
    updateBestGenerations(cli->threadId, *mem->h_bestGenerations, *mem->h_bestPattern);
  }
}

__host__ void processKernelChunks(SearchMemory* mem, prog_args* cli, ulong64 kernel,
                                 int kernelId, ulong64 processedFrames, ulong64 startFrame) {
  int numChunks = FRAME_SEARCH_MAX_CHUNK_SIZE / cli->chunkSize;
  for (int chunk = 0; chunk < numChunks; chunk++) {
    double chunkStartTime = getCurrentTime();
    executeKernelSearch(mem, cli, kernel, chunk);

    if (cli->verbose && *mem->h_numCandidates > 0) {
      bool is_frame_complete = (kernelId == FRAME_SEARCH_NUM_KERNELS - 1) && (chunk == numChunks - 1);
      printChunkStats(cli, chunkStartTime, processedFrames, startFrame, kernelId, chunk, is_frame_complete);
    }
  }
}

__host__ void processFrameKernels(SearchMemory* mem, prog_args* cli, ulong64 frame,
                                 ulong64 processedFrames, ulong64 startFrame) {
  for (int i = 0; i < FRAME_SEARCH_NUM_KERNELS; i++) {
    ulong64 kernel = constructKernel(frame, i);
    processKernelChunks(mem, cli, kernel, i, processedFrames, startFrame);
  }
}

__host__ void searchRandom(prog_args *cli) {
  init_genrand64((ulong64) time(NULL));

  SearchMemory* mem = allocateSearchMemory(RANDOM_SEARCH_MAX_CANDIDATES);

  // We're randomly searching..  I didn't get cuRAND to work so we randomize our batches. Each
  // call to findCandidates is sequential but we look at random locations across all possible.
  ulong64 chunkSize = RANDOM_SEARCH_CHUNK_SIZE;
  ulong64 iterations = cli->randomSamples / chunkSize;
  for (ulong64 i = 0; i < iterations; i++) {
    ulong64 start = genrand64_int64();
    ulong64 end = start + chunkSize;
    executeCandidateSearch(mem, cli, start, end);
  }

  // Add cleanup
  freeSearchMemory(mem);
}

// See the algorithm described in PERFORMANCE under "Eliminating Rotations"
__host__ void searchAll(prog_args *cli) {
  SearchMemory* mem = allocateSearchMemory(FRAME_SEARCH_MAX_CANDIDATES);

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

  ulong64 startFrame = (cli->frameBeginAt > 0) ? cli->frameBeginAt : 0;
  ulong64 endFrame = (cli->frameEndAt > 0) ? cli->frameEndAt : FRAME_SEARCH_MAX_FRAMES;
  ulong64 totalFrames = endFrame - startFrame;
  ulong64 skippedFrames = 0;
  ulong64 processedFrames = 0;

  for (ulong64 i = 0; i < FRAME_SEARCH_MAX_FRAMES && processedFrames < totalFrames; i++) {
    ulong64 frame = spreadBitsToFrame(i);
    if (isMinimalFrame(frame)) {
      // Skip until the start
      if (skippedFrames < startFrame) {
	skippedFrames++;
	continue;
      }

      // Then start processing running FRAME_SEARCH_NUM_KERNELS per frame.
      processFrameKernels(mem, cli, frame, processedFrames, startFrame);
      processedFrames++;
    }
  }

  // Cleanup
  freeSearchMemory(mem);
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

__host__ const char* getSearchDescription(prog_args* cli, char* buffer, size_t bufferSize) {
  if (cli->beginAt > 0 || cli->endAt > 0) {
    snprintf(buffer, bufferSize, "ALL in range (%llu - %llu)", cli->beginAt, cli->endAt);
  } else if (cli->frameBeginAt > 0 || cli->frameEndAt > 0) {
    snprintf(buffer, bufferSize, "ALL in frames (%llu - %llu)", cli->frameBeginAt, cli->frameEndAt);
  } else {
    strcpy(buffer, "ALL");
  }
  return buffer;
}

__host__ void *search(void *args) {
  prog_args *cli = (prog_args *)args;

#ifdef __NVCC__
  printThreadStatus(cli->threadId, "Running with CUDA enabled");
#else
  printThreadStatus(cli->threadId, "Running CPU-only version");
#endif

  if (cli->random) {
    printThreadStatus(cli->threadId, "searching RANDOMLY %llu candidates", cli->randomSamples);
    searchRandom(cli);
  }
  else {
    char searchRangeMessage[MESSAGE_BUFFER_SIZE] = {'\0'};
    getSearchDescription(cli, searchRangeMessage, sizeof(searchRangeMessage));
    printThreadStatus(cli->threadId, "searching %s", searchRangeMessage);
    searchAll(cli);
  }

  return NULL;
}

#endif

