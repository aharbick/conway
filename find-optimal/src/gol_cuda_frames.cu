// CUDA kernels and functions for frame-based 8x8 search and 7x7 exhaustive search
#include <iomanip>
#include <iostream>

#include "constants.h"
#include "gol.h"
#include "logging.h"
#include "subgrid_cache.h"

#ifdef __NVCC__

// External declarations for shared kernels
extern __global__ void processCandidates(uint64_t *candidates, uint64_t *numCandidates, uint64_t *bestPattern,
                                         uint64_t *bestGenerations, CycleDetectionAlgorithm algorithm);
extern __global__ void processCandidates7x7(uint64_t *candidates, uint64_t *numCandidates, uint64_t *bestPattern,
                                            uint64_t *bestGenerations, CycleDetectionAlgorithm algorithm);

// 7x7 exhaustive search kernel
// Divides all 2^49 patterns among threads, each processing a range
__global__ void find7x7Candidates(uint64_t rangeStart, uint64_t rangeEnd,
                                   uint64_t *candidates, uint64_t *numCandidates,
                                   CycleDetectionAlgorithm algorithm) {
  // Calculate which pattern range this thread is responsible for
  uint64_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t totalThreads = gridDim.x * blockDim.x;

  uint64_t patternsPerThread = (rangeEnd - rangeStart + totalThreads - 1) / totalThreads;
  uint64_t threadStart = rangeStart + (threadId * patternsPerThread);
  uint64_t threadEnd = threadStart + patternsPerThread;

  if (threadStart >= rangeEnd) return;
  if (threadEnd > rangeEnd) threadEnd = rangeEnd;

  // Process each 7x7 pattern in this thread's range
  for (uint64_t pattern7x7 = threadStart; pattern7x7 < threadEnd; pattern7x7++) {
    // Unpack once, then use unpacked format for all iterations in countGenerations
    uint64_t unpacked = unpack7x7(pattern7x7);
    int gens = countGenerations(unpacked, computeNextGeneration7x7, algorithm);

    // Save patterns that reach minimum candidate threshold
    // Store in compact 7x7 format to save memory
    if (gens >= MIN_CANDIDATE_GENERATIONS) {
      uint64_t idx = atomicAdd((unsigned long long*)numCandidates, 1ULL);
      if (idx < FRAME_SEARCH_MAX_CANDIDATES) {
        candidates[idx] = pattern7x7;  // Store compact format
      }
    }
  }
}

// Frame search kernel (8x8 patterns)
__global__ void findCandidatesInKernel(uint64_t kernel, uint64_t *candidates, uint64_t *numCandidates) {
  uint64_t startingPattern = kernel;
  startingPattern += ((uint64_t)(threadIdx.x & 15)) << 10;  // set the lower row of 4 'T' bits
  startingPattern += ((uint64_t)(threadIdx.x >> 4)) << 17;  // set the upper row of 6 'T' bits
  startingPattern += ((uint64_t)(blockIdx.x & 63)) << 41;   // set the lower row of 6 'B' bits
  startingPattern += ((uint64_t)(blockIdx.x >> 6)) << 50;   // set the upper row of 4 'B' bits

  uint64_t endAt = startingPattern +
                   ((1ULL << FRAME_SEARCH_NUM_P_BITS) << 24);  // 2^16 = 65536 increments for the P bits (bits 24-39)
  uint64_t beginAt = startingPattern;

  for (uint64_t pattern = beginAt; pattern != endAt; pattern += (1ULL << 24)) {
    uint64_t g1 = pattern;
    uint16_t generations = 0;

    while (generations < FAST_SEARCH_MAX_GENERATIONS) {
      if (!step6GenerationsAndCheck(&g1, pattern, &generations, candidates, numCandidates)) {
        continue;
      }
      break;
    }
  }
}

// Cache-accelerated version of findCandidatesInKernel
__global__ void findCandidatesInKernelWithCache(uint64_t kernel, uint64_t *candidates, uint64_t *numCandidates,
                                                 const SubgridHashTable* cache) {
  uint64_t startingPattern = kernel;
  startingPattern += ((uint64_t)(threadIdx.x & 15)) << 10;  // set the lower row of 4 'T' bits
  startingPattern += ((uint64_t)(threadIdx.x >> 4)) << 17;  // set the upper row of 6 'T' bits
  startingPattern += ((uint64_t)(blockIdx.x & 63)) << 41;   // set the lower row of 6 'B' bits
  startingPattern += ((uint64_t)(blockIdx.x >> 6)) << 50;   // set the upper row of 4 'B' bits

  uint64_t endAt = startingPattern +
                   ((1ULL << FRAME_SEARCH_NUM_P_BITS) << 24);  // 2^16 = 65536 increments for the P bits (bits 24-39)
  uint64_t beginAt = startingPattern;

  for (uint64_t pattern = beginAt; pattern != endAt; pattern += (1ULL << 24)) {
    uint64_t g1 = pattern;
    uint16_t generations = 0;

    while (generations < FAST_SEARCH_MAX_GENERATIONS) {
      if (!step6GenerationsAndCheckWithCache(&g1, pattern, &generations, candidates, numCandidates, cache)) {
        continue;
      }
      break;
    }
  }
}

// CUDA execution functions

__host__ void executeKernelSearch(gol::SearchMemory &mem, ProgramArgs *cli, uint64_t frame, uint64_t frameIdx) {
  // Loop over all kernels for this frame
  for (int kernelIdx = 0; kernelIdx < (1ULL << FRAME_SEARCH_NUM_K_BITS); ++kernelIdx) {
    const uint64_t kernel = constructKernel(frame, kernelIdx);
    const double startTime = getHighResCurrentTime();

    // Phase 1: Find candidates in this kernel
    *mem.h_numCandidates() = 0;
    cudaCheckError(cudaMemcpy(mem.d_numCandidates(), mem.h_numCandidates(), sizeof(uint64_t), cudaMemcpyHostToDevice));

    // Use cache-accelerated version if cache is available
    if (mem.d_cacheTable() != nullptr) {
      findCandidatesInKernelWithCache<<<FRAME_SEARCH_GRID_SIZE, FRAME_SEARCH_THREADS_PER_BLOCK>>>(
          kernel, mem.d_candidates(), mem.d_numCandidates(), mem.d_cacheTable());
    } else {
      findCandidatesInKernel<<<FRAME_SEARCH_GRID_SIZE, FRAME_SEARCH_THREADS_PER_BLOCK>>>(
          kernel, mem.d_candidates(), mem.d_numCandidates());
    }

    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());
    cudaCheckError(cudaMemcpy(mem.h_numCandidates(), mem.d_numCandidates(), sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // Phase 2: Process candidates if found
    *mem.h_bestGenerations() = 0;
    *mem.h_bestPattern() = 0;

    if (*mem.h_numCandidates() > 0) {
      cudaCheckError(
          cudaMemcpy(mem.d_bestGenerations(), mem.h_bestGenerations(), sizeof(uint64_t), cudaMemcpyHostToDevice));
      cudaCheckError(cudaMemcpy(mem.d_bestPattern(), mem.h_bestPattern(), sizeof(uint64_t), cudaMemcpyHostToDevice));
      processCandidates<<<FRAME_SEARCH_GRID_SIZE, FRAME_SEARCH_THREADS_PER_BLOCK>>>(
          mem.d_candidates(), mem.d_numCandidates(), mem.d_bestPattern(), mem.d_bestGenerations(), cli->cycleDetection);
      cudaCheckError(cudaGetLastError());
      cudaCheckError(cudaDeviceSynchronize());

      cudaCheckError(cudaMemcpy(mem.h_bestPattern(), mem.d_bestPattern(), sizeof(uint64_t), cudaMemcpyDeviceToHost));
      cudaCheckError(
          cudaMemcpy(mem.h_bestGenerations(), mem.d_bestGenerations(), sizeof(uint64_t), cudaMemcpyDeviceToHost));
      updateBestGenerations(*mem.h_bestGenerations());
    }

    bool isFrameComplete = (kernelIdx == (1ULL << FRAME_SEARCH_NUM_K_BITS) - 1);
    reportKernelResults(mem, cli, startTime, frame, frameIdx, kernelIdx, isFrameComplete);
  }
}

__host__ void execute7x7Search(gol::SearchMemory &mem, ProgramArgs *cli, uint64_t rangeStart, uint64_t rangeEnd) {
  const double startTime = getHighResCurrentTime();

  // Phase 1: Find candidates in this range
  *mem.h_numCandidates() = 0;
  cudaCheckError(cudaMemcpy(mem.d_numCandidates(), mem.h_numCandidates(), sizeof(uint64_t), cudaMemcpyHostToDevice));

  find7x7Candidates<<<FRAME_SEARCH_GRID_SIZE, FRAME_SEARCH_THREADS_PER_BLOCK>>>(
      rangeStart, rangeEnd, mem.d_candidates(), mem.d_numCandidates(), cli->cycleDetection);

  cudaCheckError(cudaGetLastError());
  cudaCheckError(cudaDeviceSynchronize());
  cudaCheckError(cudaMemcpy(mem.h_numCandidates(), mem.d_numCandidates(), sizeof(uint64_t), cudaMemcpyDeviceToHost));

  // Phase 2: Process candidates if found (recount to find best)
  *mem.h_bestGenerations() = 0;
  *mem.h_bestPattern() = 0;

  if (*mem.h_numCandidates() > 0) {
    cudaCheckError(
        cudaMemcpy(mem.d_bestGenerations(), mem.h_bestGenerations(), sizeof(uint64_t), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(mem.d_bestPattern(), mem.h_bestPattern(), sizeof(uint64_t), cudaMemcpyHostToDevice));

    // Process 7x7 candidates
    processCandidates7x7<<<FRAME_SEARCH_GRID_SIZE, FRAME_SEARCH_THREADS_PER_BLOCK>>>(
        mem.d_candidates(), mem.d_numCandidates(), mem.d_bestPattern(), mem.d_bestGenerations(), cli->cycleDetection);

    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    cudaCheckError(cudaMemcpy(mem.h_bestPattern(), mem.d_bestPattern(), sizeof(uint64_t), cudaMemcpyDeviceToHost));
    cudaCheckError(
        cudaMemcpy(mem.h_bestGenerations(), mem.d_bestGenerations(), sizeof(uint64_t), cudaMemcpyDeviceToHost));
    updateBestGenerations(*mem.h_bestGenerations());
  }

  report7x7Results(mem, cli, startTime, rangeStart, rangeEnd);
}

__host__ void report7x7Results(gol::SearchMemory &mem, ProgramArgs *cli, double startTime,
                                uint64_t rangeStart, uint64_t rangeEnd) {
  const double searchTime = getHighResCurrentTime() - startTime;
  const uint64_t totalPatterns = rangeEnd - rangeStart;
  const uint64_t patternsPerSec = (searchTime > 0) ? (totalPatterns / searchTime) : 0;

  // Calculate progress through entire 7x7 space
  const uint64_t total7x7Patterns = (1ULL << 49);
  double progress = (double)rangeEnd / total7x7Patterns * 100.0;

  if (*mem.h_numCandidates() > 0) {
    char bestPatternBin[BINARY_STRING_BUFFER_SIZE] = {'\0'};
    asBinary(*mem.h_bestPattern(), bestPatternBin, 49);

    Logger::out() << "timestamp=" << time(NULL)
                   << ", rangeStart=" << rangeStart
                   << ", rangeEnd=" << rangeEnd
                   << ", bestGenerations=" << (int)*mem.h_bestGenerations()
                   << ", bestPattern=" << *mem.h_bestPattern()
                   << ", bestPatternBin=" << bestPatternBin
                   << ", patternsPerSec=" << formatWithCommas(patternsPerSec)
                   << ", progress=" << std::fixed << std::setprecision(2) << progress << "%\n";

    if (!cli->dontSaveResults) {
      queueGoogleSummaryData((int)*mem.h_bestGenerations(), *mem.h_bestPattern(), bestPatternBin, UINT64_MAX);

      if ((int)*mem.h_bestGenerations() >= 200) {
        queueGoogleProgress(0, 0, (int)*mem.h_bestGenerations(), *mem.h_bestPattern(), bestPatternBin);
      }
    }
  } else {
    Logger::out() << "timestamp=" << time(NULL)
                   << ", rangeStart=" << rangeStart
                   << ", rangeEnd=" << rangeEnd
                   << ", candidates=0"
                   << ", patternsPerSec=" << formatWithCommas(patternsPerSec)
                   << ", progress=" << std::fixed << std::setprecision(2) << progress << "%\n";
  }
}

__host__ void reportKernelResults(gol::SearchMemory &mem, ProgramArgs *cli, double startTime, uint64_t frame,
                                  uint64_t frameIdx, int kernelIdx, bool isFrameComplete) {
  const double kernelTime = getHighResCurrentTime() - startTime;
  const uint64_t patternsPerSec =
      (FRAME_SEARCH_GRID_SIZE * FRAME_SEARCH_THREADS_PER_BLOCK * (1ULL << FRAME_SEARCH_NUM_P_BITS)) / kernelTime;

  if (*mem.h_numCandidates() <= 0) {
    std::cerr << "timestamp=" << time(NULL) << ", frameIdx=" << frameIdx << ", kernelIdx=" << kernelIdx
              << ", error=NO_PATTERNS_FOUND\n";

    return;
  }

  char bestPatternBin[BINARY_STRING_BUFFER_SIZE] = {'\0'};
  asBinary(*mem.h_bestPattern(), bestPatternBin);

  Logger::out() << "timestamp=" << time(NULL) << ", frameIdx=" << frameIdx << ", kernelIdx=" << kernelIdx
                 << ", bestGenerations=" << (int)*mem.h_bestGenerations() << ", bestPattern=" << *mem.h_bestPattern()
                 << ", bestPatternBin=" << bestPatternBin << ", patternsPerSec=" << formatWithCommas(patternsPerSec) << "\n";

  // Always mark frame as complete locally when all kernels are done
  if (isFrameComplete) {
    setGoogleFrameCompleteInCache(frameIdx);
  }

  if (!cli->dontSaveResults) {
    // Pass frameIdx if this completes the frame (kernelIdx == 15), otherwise pass UINT64_MAX
    uint64_t completedFrameIdx = isFrameComplete ? frameIdx : UINT64_MAX;
    queueGoogleSummaryData((int)*mem.h_bestGenerations(), *mem.h_bestPattern(), bestPatternBin, completedFrameIdx);

    // Send progress data to Google if generations >= 200
    if ((int)*mem.h_bestGenerations() >= 200) {
      queueGoogleProgress(frameIdx, kernelIdx, (int)*mem.h_bestGenerations(), *mem.h_bestPattern(), bestPatternBin);
    }
  }
}

#endif
