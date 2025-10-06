#include "subgrid_cache.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include <nlohmann/json.hpp>

#include "cuda_utils.h"
#include "gol_memory.h"
#include "logging.h"
#include "utils.h"

using json = nlohmann::json;

int computeSubgridCache(ProgramArgs* cli) {
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
        rangeStart, rangeEnd, d_candidates, d_numCandidates, cli->cycleDetection);

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
