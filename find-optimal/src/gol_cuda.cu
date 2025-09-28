// CUDA-specific implementations
#include <iostream>
#include <locale>
#include <sstream>

#include "constants.h"
#include "gol.h"
#include "logging.h"

#ifdef __NVCC__

// CUDA kernels
__global__ void processCandidates(uint64_t *candidates, uint64_t *numCandidates, uint64_t *bestPattern,
                                  uint64_t *bestGenerations, CycleDetectionAlgorithm algorithm, GolGridMode gridMode) {
  for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < *numCandidates; i += blockDim.x * gridDim.x) {
    uint64_t generations = countGenerations(candidates[i], algorithm, gridMode);
    if (generations > 0) {  // Only process if it actually ended
      // Check to see if it's higher and emit it in best(Pattern|Generations)
      uint64_t old = atomicMax((unsigned long long *)bestGenerations, (unsigned long long)generations);
      if (old < generations) {
        *bestPattern = candidates[i];
      }
    }
  }
}

__global__ void findCandidatesInKernel(uint64_t kernel, uint64_t *candidates, uint64_t *numCandidates, GolGridMode gridMode) {
  uint64_t startingPattern = kernel;
  startingPattern += ((uint64_t)(threadIdx.x & 15)) << 10;  // set the lower row of 4 'T' bits
  startingPattern += ((uint64_t)(threadIdx.x >> 4)) << 17;  // set the upper row of 6 'T' bits
  startingPattern += ((uint64_t)(blockIdx.x & 63)) << 41;   // set the lower row of 6 'B' bits
  startingPattern += ((uint64_t)(blockIdx.x >> 6)) << 50;   // set the upper row of 4 'B' bits

  uint64_t endAt = startingPattern +
                   ((1ULL << FRAME_SEARCH_NUM_P_BITS) << 23);  // 2^16 = 65536 increments for the P bits (bits 23-38)
  uint64_t beginAt = startingPattern;

  for (uint64_t pattern = beginAt; pattern < endAt; pattern += (1ULL << 23)) {
    uint64_t g1 = pattern;
    uint64_t generations = 0;

    while (generations < FAST_SEARCH_MAX_GENERATIONS) {
      if (!step6GenerationsAndCheck(&g1, pattern, &generations, candidates, numCandidates, gridMode)) {
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
    findCandidatesInKernel<<<FRAME_SEARCH_GRID_SIZE, FRAME_SEARCH_THREADS_PER_BLOCK>>>(kernel, mem.d_candidates(),
                                                                                       mem.d_numCandidates(), cli->golGridMode);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());
    cudaCheckError(cudaMemcpy(mem.h_numCandidates(), mem.d_numCandidates(), sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // Phase 2: Process candidates if found
    if (*mem.h_numCandidates() > 0) {
      *mem.h_bestGenerations() = 0;
      *mem.h_bestPattern() = 0;
      cudaCheckError(
          cudaMemcpy(mem.d_bestGenerations(), mem.h_bestGenerations(), sizeof(uint64_t), cudaMemcpyHostToDevice));
      cudaCheckError(cudaMemcpy(mem.d_bestPattern(), mem.h_bestPattern(), sizeof(uint64_t), cudaMemcpyHostToDevice));
      processCandidates<<<FRAME_SEARCH_GRID_SIZE, FRAME_SEARCH_THREADS_PER_BLOCK>>>(
          mem.d_candidates(), mem.d_numCandidates(), mem.d_bestPattern(), mem.d_bestGenerations(), cli->cycleDetection, cli->golGridMode);
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

  // Format patternsPerSec with commas using a separate stringstream
  std::ostringstream formattedRate;
  formattedRate.imbue(std::locale(""));
  formattedRate << patternsPerSec;

  Logger::out() << "timestamp=" << time(NULL) << ", frameIdx=" << frameIdx << ", kernelIdx=" << kernelIdx
                 << ", bestGenerations=" << (int)*mem.h_bestGenerations() << ", bestPattern=" << *mem.h_bestPattern()
                 << ", bestPatternBin=" << bestPatternBin << ", patternsPerSec=" << formattedRate.str() << "\n";

  // Always mark frame as complete locally when all kernels are done
  if (isFrameComplete) {
    googleSetFrameCompleteInCache(frameIdx);
  }

  if (!cli->dontSaveResults) {
    // Pass frameIdx if this completes the frame (kernelIdx == 15), otherwise pass UINT64_MAX
    uint64_t completedFrameIdx = isFrameComplete ? frameIdx : UINT64_MAX;
    googleSendSummaryDataAsync((int)*mem.h_bestGenerations(), *mem.h_bestPattern(), bestPatternBin, completedFrameIdx);

    // Send progress data to Google if generations >= 200
    if ((int)*mem.h_bestGenerations() >= 200) {
      googleSendProgressAsync(frameIdx, kernelIdx, (int)*mem.h_bestGenerations(), *mem.h_bestPattern(), bestPatternBin);
    }
  }
}

#endif
