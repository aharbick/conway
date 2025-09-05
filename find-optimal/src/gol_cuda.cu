// CUDA-specific implementations
#include "gol.h"

#include <iostream>

#ifdef __NVCC__

// CUDA kernels
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

  if (pattern >= endAt) return;

  while (generations < FAST_SEARCH_MAX_GENERATIONS) {
    if (!step6GenerationsAndCheck(&g1, pattern, &generations, candidates, numCandidates)) {
      continue;
    }
    return;
  }
}

__global__ void findCandidatesInKernel(uint64_t kernel, int chunkIdx, uint64_t chunkSize, uint64_t *candidates,
                                       uint64_t *numCandidates) {
  uint64_t startingPattern = kernel;
  startingPattern += ((uint64_t)(threadIdx.x & 15)) << 10;   // set the lower row of 4 'T' bits
  startingPattern += ((uint64_t)(threadIdx.x >> 4)) << 17;   // set the upper row of 6 'T' bits
  startingPattern += ((uint64_t)(blockIdx.x & 63)) << 41;    // set the lower row of 6 'B' bits
  startingPattern += ((uint64_t)(blockIdx.x >> 6)) << 50;    // set the upper row of 4 'B' bits

  uint64_t endAt = startingPattern + (chunkSize << 23);  // 16 bits worth of increments for the P bits (bits 23-38)
  uint64_t beginAt = startingPattern + ((uint64_t)chunkIdx * chunkSize << 23);

  for (uint64_t pattern = beginAt; pattern < endAt; pattern += (1ULL << 23)) {
    uint64_t g1 = pattern;
    uint64_t generations = 0;

    while (generations < FAST_SEARCH_MAX_GENERATIONS) {
      if (!step6GenerationsAndCheck(&g1, pattern, &generations, candidates, numCandidates)) {
        continue;
      }
      break;
    }
  }
}

// CUDA execution functions
__host__ void executeCandidateSearch(gol::SearchMemory& mem, ProgramArgs *cli, uint64_t start, uint64_t end) {
  // Phase 1: Find candidates
  *mem.h_numCandidates() = 0;
  cudaCheckError(cudaMemcpy(mem.d_numCandidates(), mem.h_numCandidates(), sizeof(uint64_t), cudaMemcpyHostToDevice));
  findCandidates<<<cli->blockSize, cli->threadsPerBlock>>>(start, end, mem.d_candidates(), mem.d_numCandidates());
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
    processCandidates<<<cli->blockSize, cli->threadsPerBlock>>>(mem.d_candidates(), mem.d_numCandidates(),
                                                                mem.d_bestPattern(), mem.d_bestGenerations());
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    cudaCheckError(cudaMemcpy(mem.h_bestPattern(), mem.d_bestPattern(), sizeof(uint64_t), cudaMemcpyDeviceToHost));
    cudaCheckError(
        cudaMemcpy(mem.h_bestGenerations(), mem.d_bestGenerations(), sizeof(uint64_t), cudaMemcpyDeviceToHost));
    bool isNewGlobalBest = updateBestGenerations(*mem.h_bestGenerations());

    // Print result for new best
    if (isNewGlobalBest) {
      char bin[BINARY_STRING_BUFFER_SIZE] = {'\0'};
      asBinary(*mem.h_bestPattern(), bin);
      std::cout << "[Thread " << cli->threadId << " - " << (uint64_t)time(NULL) 
                << "] bestGenerations=" << (int)*mem.h_bestGenerations() 
                << ", bestPattern=" << *mem.h_bestPattern() 
                << ", bestPatternBin=" << bin << "\n";
    }
  }
}

__host__ void executeKernelSearch(gol::SearchMemory& mem, ProgramArgs *cli, uint64_t frame, uint64_t frameIdx) {
  const int numChunks = FRAME_SEARCH_MAX_CHUNK_SIZE / cli->chunkSize;

  // Loop over all kernels for this frame
  for (int kernelIdx = 0; kernelIdx < FRAME_SEARCH_NUM_KERNELS; ++kernelIdx) {
    const uint64_t kernel = constructKernel(frame, kernelIdx);
    for (int chunkIdx = 0; chunkIdx < numChunks; ++chunkIdx) {
      const double startTime = getCurrentTime();

      // Phase 1: Find candidates in this kernel/chunk
      *mem.h_numCandidates() = 0;
      cudaCheckError(cudaMemcpy(mem.d_numCandidates(), mem.h_numCandidates(), sizeof(uint64_t), cudaMemcpyHostToDevice));
      findCandidatesInKernel<<<cli->blockSize, cli->threadsPerBlock>>>(kernel, chunkIdx, cli->chunkSize,
                                                                       mem.d_candidates(), mem.d_numCandidates());
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
        processCandidates<<<cli->blockSize, cli->threadsPerBlock>>>(mem.d_candidates(), mem.d_numCandidates(),
                                                                    mem.d_bestPattern(), mem.d_bestGenerations());
        cudaCheckError(cudaGetLastError());
        cudaCheckError(cudaDeviceSynchronize());

        cudaCheckError(cudaMemcpy(mem.h_bestPattern(), mem.d_bestPattern(), sizeof(uint64_t), cudaMemcpyDeviceToHost));
        cudaCheckError(
            cudaMemcpy(mem.h_bestGenerations(), mem.d_bestGenerations(), sizeof(uint64_t), cudaMemcpyDeviceToHost));
        updateBestGenerations(*mem.h_bestGenerations());
      }

      bool isFrameComplete = (kernelIdx == FRAME_SEARCH_NUM_KERNELS - 1) && (chunkIdx == numChunks - 1);
      reportChunkResults(mem, cli, startTime, frame, frameIdx, kernelIdx, chunkIdx, isFrameComplete);
    }
  }
}

__host__ void reportChunkResults(gol::SearchMemory& mem, ProgramArgs *cli, double startTime, uint64_t frame,
                                 uint64_t frameIdx, int kernelIdx, int chunkIdx, bool isFrameComplete) {
  const double chunkTime = getCurrentTime() - startTime;
  const uint64_t patternsPerSec = (FRAME_SEARCH_TOTAL_THREADS * cli->chunkSize) / chunkTime;

  if (*mem.h_numCandidates() <= 0) {
    std::cout << "[Thread " << cli->threadId << " - " << (uint64_t)time(NULL) 
              << "] WARN: NO PATTERNS FOUND frameIdx=" << frameIdx 
              << ", kernelIdx=" << kernelIdx << ", chunkIdx=" << chunkIdx << "\n";

    airtableSendProgress(isFrameComplete, frameIdx, kernelIdx, chunkIdx, patternsPerSec, 0, 0, "ERROR", false);
    return;
  }

  char bestPatternBin[BINARY_STRING_BUFFER_SIZE] = {'\0'};
  asBinary(*mem.h_bestPattern(), bestPatternBin);

  std::cout << "[Thread " << cli->threadId << " - " << (uint64_t)time(NULL) 
            << "] frameIdx=" << frameIdx << ", kernelIdx=" << kernelIdx 
            << ", chunkIdx=" << chunkIdx << ", bestGenerations=" << (int)*mem.h_bestGenerations()
            << ", bestPattern=" << *mem.h_bestPattern() << ", bestPatternBin=" << bestPatternBin 
            << ", patternsPerSec=" << patternsPerSec << "\n";

  airtableSendProgress(isFrameComplete, frameIdx, kernelIdx, chunkIdx, patternsPerSec, (int)*mem.h_bestGenerations(),
                       *mem.h_bestPattern(), bestPatternBin, false);
}

#endif
