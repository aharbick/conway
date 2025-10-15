// CUDA-specific implementations
#include <iostream>
#include <locale>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <vector>

#include <nlohmann/json.hpp>

#include "constants.h"
#include "gol.h"
#include "logging.h"
#include "subgrid_cache.h"

#ifdef __NVCC__

// CUDA kernels

// Find candidates in a range of 7x7 patterns for subgrid cache computation
// Each thread processes SUBGRID_PATTERNS_PER_THREAD patterns across 4 translations
__global__ void findSubgridCandidates(uint64_t rangeStart, uint64_t rangeEnd,
                                      SubgridCacheEntry* candidates, uint64_t* numCandidates,
                                      CycleDetectionAlgorithm algorithm, int minGenerations) {
  // Calculate which 7x7 base pattern this thread is responsible for
  uint64_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t patternsPerThread = SUBGRID_PATTERNS_PER_THREAD;
  uint64_t basePatternStart = rangeStart + (threadId * patternsPerThread);

  if (basePatternStart >= rangeEnd) return;

  uint64_t basePatternEnd = basePatternStart + patternsPerThread;
  if (basePatternEnd > rangeEnd) {
    basePatternEnd = rangeEnd;
  }

  // Process each 7x7 pattern assigned to this thread
  for (uint64_t pattern7x7 = basePatternStart; pattern7x7 < basePatternEnd; pattern7x7++) {
    // Test all 4 possible positions of 7x7 within 8x8
    // Position 0: rows 0-6, cols 0-6
    // Position 1: rows 0-6, cols 1-7
    // Position 2: rows 1-7, cols 0-6
    // Position 3: rows 1-7, cols 1-7

    for (int pos = 0; pos < 4; pos++) {
      int rowOffset = (pos >= 2) ? 1 : 0;
      int colOffset = (pos & 1) ? 1 : 0;

      uint64_t pattern8x8 = expand7x7To8x8(pattern7x7, rowOffset, colOffset);

      // Count generations for this 8x8 pattern
      int gens = countGenerations(pattern8x8, algorithm);

      // Save each 8x8 pattern that meets the threshold
      if (gens >= minGenerations) {
        uint64_t idx = atomicAdd((unsigned long long*)numCandidates, 1ULL);
        candidates[idx].pattern = pattern8x8;  // Store the 8x8 pattern
        candidates[idx].generations = gens;
      }
    }
  }
}
__global__ void processCandidates(uint64_t *candidates, uint64_t *numCandidates, uint64_t *bestPattern,
                                  uint64_t *bestGenerations, CycleDetectionAlgorithm algorithm) {
  for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < *numCandidates; i += blockDim.x * gridDim.x) {
    uint64_t generations = countGenerations(candidates[i], algorithm);
    if (generations > 0) {  // Only process if it actually ended
      // Check to see if it's higher and emit it in best(Pattern|Generations)
      uint64_t old = atomicMax((unsigned long long *)bestGenerations, (unsigned long long)generations);
      if (old < generations) {
        *bestPattern = candidates[i];
      }
    }
  }
}

__global__ void findCandidatesInKernel(uint64_t kernel, uint64_t *candidates, uint64_t *numCandidates) {
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
    uint16_t generations = 0;

    while (generations < FAST_SEARCH_MAX_GENERATIONS) {
      if (!step6GenerationsAndCheck(&g1, pattern, &generations, candidates, numCandidates)) {
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
                                                                                       mem.d_numCandidates());
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

// Subgrid cache computation (moved from subgrid_cache.cpp)
__host__ int computeSubgridCache(ProgramArgs* cli) {
  using json = nlohmann::json;

  Logger::out() << "Computing 7x7 subgrid cache...\n";
  Logger::out() << "Total 7x7 patterns to process: " << formatWithCommas(SUBGRID_TOTAL_PATTERNS) << "\n";
  Logger::out() << "Output file: " << cli->subgridCachePath << "\n";
  Logger::out() << "Minimum generations to save: " << SUBGRID_MIN_GENERATIONS << "\n\n";

  // Count existing entries if resuming
  uint64_t existingCandidates = 0;
  if (cli->subgridCacheBegin > 0) {
    std::ifstream inFile(cli->subgridCachePath);
    if (inFile.is_open()) {
      std::string line;
      while (std::getline(inFile, line)) {
        if (!line.empty()) {
          existingCandidates++;
        }
      }
      inFile.close();
      Logger::out() << "Found " << formatWithCommas(existingCandidates) << " existing candidates in cache file\n";
    }
  }

  // Open output file (append mode if resuming, otherwise truncate)
  std::ofstream outFile;
  if (cli->subgridCacheBegin > 0) {
    outFile.open(cli->subgridCachePath, std::ios::app);
  } else {
    outFile.open(cli->subgridCachePath, std::ios::trunc);
  }

  if (!outFile.is_open()) {
    std::cerr << "[ERROR] Failed to open output file: " << cli->subgridCachePath << "\n";
    return 1;
  }

  // Allocate device memory for candidates
  SubgridCacheEntry* d_candidates = nullptr;
  uint64_t* d_numCandidates = nullptr;

  cudaCheckError(cudaMalloc(&d_candidates, SUBGRID_CACHE_MAX_CANDIDATES * sizeof(SubgridCacheEntry)));
  cudaCheckError(cudaMalloc(&d_numCandidates, sizeof(uint64_t)));

  // Allocate host memory for candidates
  std::vector<SubgridCacheEntry> h_candidates(SUBGRID_CACHE_MAX_CANDIDATES);
  uint64_t h_numCandidates = 0;

  // Calculate how many patterns each kernel invocation processes
  uint64_t totalThreads = (uint64_t)SUBGRID_GRID_SIZE * SUBGRID_THREADS_PER_BLOCK;
  uint64_t patternsPerInvocation = totalThreads * SUBGRID_PATTERNS_PER_THREAD;

  uint64_t totalBatches = (SUBGRID_TOTAL_PATTERNS + patternsPerInvocation - 1) / patternsPerInvocation;

  Logger::out() << "Threads per invocation: " << formatWithCommas(totalThreads) << "\n";
  Logger::out() << "Patterns per invocation: " << formatWithCommas(patternsPerInvocation) << "\n";
  Logger::out() << "Total batches: " << formatWithCommas(totalBatches) << "\n\n";

  uint64_t totalPatternsProcessed = 0;
  uint64_t totalCandidatesSaved = existingCandidates;
  double programStartTime = getHighResCurrentTime();

  // Determine starting point (for resuming)
  uint64_t startingPattern = cli->subgridCacheBegin;
  if (startingPattern > 0) {
    Logger::out() << "Resuming from pattern: " << formatWithCommas(startingPattern) << "\n\n";
  }

  // Process in chunks
  uint64_t batchNumber = 0;
  for (uint64_t rangeStart = startingPattern; rangeStart < SUBGRID_TOTAL_PATTERNS; rangeStart += patternsPerInvocation) {
    batchNumber++;
    double batchStartTime = getHighResCurrentTime();
    uint64_t rangeEnd = rangeStart + patternsPerInvocation;
    if (rangeEnd > SUBGRID_TOTAL_PATTERNS) {
      rangeEnd = SUBGRID_TOTAL_PATTERNS;
    }

    // Reset candidate counter
    h_numCandidates = 0;
    cudaCheckError(cudaMemcpy(d_numCandidates, &h_numCandidates, sizeof(uint64_t), cudaMemcpyHostToDevice));

    // Launch kernel
    findSubgridCandidates<<<SUBGRID_GRID_SIZE, SUBGRID_THREADS_PER_BLOCK>>>(
        rangeStart, rangeEnd, d_candidates, d_numCandidates, cli->cycleDetection, SUBGRID_MIN_GENERATIONS);

    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    // Copy results back
    cudaCheckError(cudaMemcpy(&h_numCandidates, d_numCandidates, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    if (h_numCandidates > 0) {
      if (h_numCandidates > SUBGRID_CACHE_MAX_CANDIDATES) {
        std::cerr << "[ERROR] Found " << h_numCandidates << " candidates, exceeds buffer size "
                  << SUBGRID_CACHE_MAX_CANDIDATES << "\n";
        cudaFree(d_candidates);
        cudaFree(d_numCandidates);
        outFile.close();
        return 1;
      }

      cudaCheckError(cudaMemcpy(h_candidates.data(), d_candidates,
                                h_numCandidates * sizeof(SubgridCacheEntry), cudaMemcpyDeviceToHost));

      // Write entries immediately (one JSON object per line)
      for (uint64_t i = 0; i < h_numCandidates; i++) {
        json entry;
        entry["pattern"] = h_candidates[i].pattern;
        entry["generations"] = h_candidates[i].generations;
        outFile << entry.dump() << "\n";
      }
      outFile.flush();  // Ensure data is written to disk

      totalCandidatesSaved += h_numCandidates;
    }

    totalPatternsProcessed += (rangeEnd - rangeStart);

    // Performance metrics
    double batchEndTime = getHighResCurrentTime();
    double batchElapsed = batchEndTime - batchStartTime;
    double totalElapsed = batchEndTime - programStartTime;
    uint64_t patternsThisBatch = (rangeEnd - rangeStart);
    uint64_t patternsPerSec = (batchElapsed > 0) ? (uint64_t)(patternsThisBatch / batchElapsed) : 0;

    // Estimate remaining time (account for starting position)
    uint64_t totalPatternsCompleted = startingPattern + totalPatternsProcessed;
    double percentComplete = (100.0 * totalPatternsCompleted) / SUBGRID_TOTAL_PATTERNS;
    uint64_t patternsRemaining = SUBGRID_TOTAL_PATTERNS - totalPatternsCompleted;
    double estimatedSecondsRemaining = (patternsPerSec > 0) ? (double)patternsRemaining / patternsPerSec : 0;
    double estimatedDaysRemaining = estimatedSecondsRemaining / 86400.0;

    // Progress reporting (similar format to frame search)
    Logger::out() << "batch=" << batchNumber
                  << ", range=" << rangeStart << " - " << rangeEnd
                  << ", candidates=" << formatWithCommas(totalCandidatesSaved)
                  << ", patternsPerSec=" << formatWithCommas(patternsPerSec)
                  << ", progress=" << std::fixed << std::setprecision(2) << percentComplete << "%"
                  << ", estimatedDaysRemaining=" << std::setprecision(1) << estimatedDaysRemaining
                  << "\n";
  }

  // Close the file
  outFile.close();

  // Cleanup
  cudaFree(d_candidates);
  cudaFree(d_numCandidates);

  Logger::out() << "\nSubgrid cache computation complete!\n";
  Logger::out() << "Total candidates saved: " << formatWithCommas(totalCandidatesSaved) << "\n";

  return 0;
}

#endif
