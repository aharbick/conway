// CUDA kernels and functions for strip-based search (reversibility optimization)
#include <iomanip>
#include <iostream>

#include "constants.h"
#include "gol.h"
#include "gol_memory.h"
#include "logging.h"

#ifdef __NVCC__

// External declaration for shared kernel
extern __global__ void processCandidates(uint64_t *candidates, uint64_t *numCandidates, uint64_t *bestPattern,
                                         uint64_t *bestGenerations, CycleDetectionAlgorithm algorithm);

// Forward declaration for progress reporting
__host__ void reportStripSearchBlockResults(ProgramArgs *cli, double intervalStartTime,
                                            uint32_t firstBlock, uint32_t lastBlock,
                                            uint64_t bestGenerations, uint64_t bestPattern,
                                            uint64_t totalTopStrips, uint64_t totalBottomStrips,
                                            uint64_t totalCombinations);

// Helper: Compute 2-generation signature for deduplication
// Strips that produce the same signature are equivalent
__device__ static inline uint32_t computeSignature(uint64_t pattern, bool isTop) {
  uint64_t gen1 = computeNextGeneration8x8(pattern);
  uint64_t gen2 = computeNextGeneration8x8(gen1);
  return isTop ? (uint32_t)(gen2 & 0xFFFFFFFFULL)   // Top 4 rows
               : (uint32_t)(gen2 >> 32);            // Bottom 4 rows
}

// CityHash-inspired hash function for 32-bit signatures
// Provides much better distribution than simple modulo
__device__ static inline uint32_t hashSignature(uint32_t sig) {
  sig *= 0x9ddfea08U;  // CityHash-style multiply
  sig ^= sig >> 16;    // Mix high bits into low bits
  return sig;
}

// Helper: Add strip to output if signature is unique (using hash table with CityHash)
__device__ static inline void addIfUnique(
    uint16_t strip,
    uint32_t signature,
    uint32_t* hashTable,
    uint16_t* uniqueStrips,
    uint32_t* numUnique
) {
  uint32_t hash = hashSignature(signature);
  uint32_t bucket = hash & STRIP_HASH_TABLE_MASK;

  // Linear probing with CityHash - much shorter probe chains than modulo
  for (int probe = 0; probe < STRIP_MAX_PROBE_LENGTH; probe++) {
    uint32_t probeBucket = (bucket + probe) & STRIP_HASH_TABLE_MASK;

    // Try to claim this bucket (0xFFFFFFFF means empty)
    uint32_t old = atomicCAS(&hashTable[probeBucket], 0xFFFFFFFF, signature);

    if (old == 0xFFFFFFFF) {
      // Claimed empty bucket - new unique signature
      uint32_t idx = atomicAdd(numUnique, 1);
      if (idx < STRIP_SEARCH_MAX_VALID_STRIPS) {
        uniqueStrips[idx] = strip;
      }
      return;
    } else if (old == signature) {
      // Already seen this signature
      return;
    }
    // else: collision, continue probing
  }
  // Safety: exceeded max probe length (should never happen with good hash)
}

// Find unique TOP strips for a given middle block
// Called first, then hash table is cleared, then findUniqueBottomStrips is called
__global__ void findUniqueTopStrips(
    uint32_t middleBlock,
    uint16_t* uniqueStrips,
    uint32_t* numUnique,
    uint32_t* hashTable
) {
  uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t totalThreads = gridDim.x * blockDim.x;

  for (uint32_t strip = threadId; strip < STRIP_SEARCH_TOTAL_STRIPS; strip += totalThreads) {
    // Construct pattern: strip in rows 0-1, middle block in rows 2-5, zeros in rows 6-7
    uint64_t pattern = ((uint64_t)strip) | ((uint64_t)middleBlock << 16);
    uint32_t signature = computeSignature(pattern, true);
    addIfUnique((uint16_t)strip, signature, hashTable, uniqueStrips, numUnique);
  }
}

// Find unique BOTTOM strips for a given middle block
// Hash table must be cleared before calling this
__global__ void findUniqueBottomStrips(
    uint32_t middleBlock,
    uint16_t* uniqueStrips,
    uint32_t* numUnique,
    uint32_t* hashTable
) {
  uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t totalThreads = gridDim.x * blockDim.x;

  for (uint32_t strip = threadId; strip < STRIP_SEARCH_TOTAL_STRIPS; strip += totalThreads) {
    // Construct pattern: zeros in rows 0-1, middle block in rows 2-5, strip in rows 6-7
    uint64_t pattern = ((uint64_t)middleBlock << 16) | ((uint64_t)strip << 48);
    uint32_t signature = computeSignature(pattern, false);
    addIfUnique((uint16_t)strip, signature, hashTable, uniqueStrips, numUnique);
  }
}

// Test all combinations of unique top and bottom strips for a given middle block
// This is the main search kernel for strip search
//
// Work division using two loops (more efficient than division/modulo):
// - Outer loop: each thread handles a subset of top strips
// - Inner loop: for each top strip, iterate through ALL bottom strips
__global__ void findCandidatesForStripBlock(
    uint32_t middleBlock,
    const uint16_t* uniqueTopStrips,
    const uint16_t* uniqueBottomStrips,
    uint32_t numUniqueTop,
    uint32_t numUniqueBottom,
    uint64_t* candidates,
    uint64_t* numCandidates
) {
  uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t totalThreads = gridDim.x * blockDim.x;

  // Outer loop: each thread handles a subset of top strips (strided access)
  for (uint32_t topIdx = threadId; topIdx < numUniqueTop; topIdx += totalThreads) {
    uint64_t topBits = (uint64_t)uniqueTopStrips[topIdx];

    // Inner loop: for each top strip, test all bottom strips
    for (uint32_t bottomIdx = 0; bottomIdx < numUniqueBottom; bottomIdx++) {
      // Construct the full 64-bit pattern
      // Layout: [top strip 16 bits][middle block 32 bits][bottom strip 16 bits]
      //         bits 0-15          bits 16-47            bits 48-63
      uint64_t pattern = topBits |
                         ((uint64_t)middleBlock << 16) |
                         ((uint64_t)uniqueBottomStrips[bottomIdx] << 48);

      // Fast evaluation loop (same as findCandidatesInKernel)
      uint64_t g1 = pattern;
      uint16_t generations = 0;

      while (generations < FAST_SEARCH_MAX_GENERATIONS) {
        generations += 6;
        uint64_t g2 = computeNextGeneration8x8(g1);
        uint64_t g3 = computeNextGeneration8x8(g2);
        uint64_t g4 = computeNextGeneration8x8(g3);
        uint64_t g5 = computeNextGeneration8x8(g4);
        uint64_t g6 = computeNextGeneration8x8(g5);
        g1 = computeNextGeneration8x8(g6);

        // Check for cycles (pattern stabilized)
        if ((g1 == g2) || (g1 == g3) || (g1 == g4)) {
          break;  // Pattern ended, not interesting
        }

        // Check if reached minimum candidate threshold
        if (generations >= MIN_CANDIDATE_GENERATIONS) {
          uint64_t idx = atomicAdd((unsigned long long*)numCandidates, 1ULL);
          if (idx < STRIP_SEARCH_MAX_CANDIDATES) {
            candidates[idx] = pattern;
          }
          break;
        }
      }
    }
  }
}

// Execute strip search for a single middle block using StripSearchMemory
// Hash table is allocated locally and reused between top/bottom strip finding
__host__ void executeStripSearchForBlock(
    uint32_t middleBlock,
    gol::StripSearchMemory& mem,
    uint32_t* d_hashTable,
    CycleDetectionAlgorithm algorithm
) {
  // Phase 1a: Find unique TOP strips
  uint32_t zero = 0;
  cudaCheckError(cudaMemcpy(mem.d_numUniqueStrips(), &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemset(d_hashTable, 0xFF, STRIP_HASH_TABLE_SIZE * sizeof(uint32_t)));

  findUniqueTopStrips<<<STRIP_SEARCH_UNIQUE_GRID_SIZE, STRIP_SEARCH_UNIQUE_THREADS_PER_BLOCK>>>(
      middleBlock, mem.d_uniqueTopStrips(), mem.d_numUniqueStrips(), d_hashTable);
  cudaCheckError(cudaGetLastError());
  cudaCheckError(cudaDeviceSynchronize());
  cudaCheckError(cudaMemcpy(mem.h_numUniqueTop(), mem.d_numUniqueStrips(), sizeof(uint32_t), cudaMemcpyDeviceToHost));

  // Phase 1b: Find unique BOTTOM strips (reuse hash table after clearing)
  cudaCheckError(cudaMemcpy(mem.d_numUniqueStrips(), &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemset(d_hashTable, 0xFF, STRIP_HASH_TABLE_SIZE * sizeof(uint32_t)));

  findUniqueBottomStrips<<<STRIP_SEARCH_UNIQUE_GRID_SIZE, STRIP_SEARCH_UNIQUE_THREADS_PER_BLOCK>>>(
      middleBlock, mem.d_uniqueBottomStrips(), mem.d_numUniqueStrips(), d_hashTable);
  cudaCheckError(cudaGetLastError());
  cudaCheckError(cudaDeviceSynchronize());
  cudaCheckError(cudaMemcpy(mem.h_numUniqueBottom(), mem.d_numUniqueStrips(), sizeof(uint32_t), cudaMemcpyDeviceToHost));

  // Phase 2: Test all combinations of unique strips
  uint64_t zero64 = 0;
  cudaCheckError(cudaMemcpy(mem.d_numCandidates(), &zero64, sizeof(uint64_t), cudaMemcpyHostToDevice));

  if (*mem.h_numUniqueTop() > 0 && *mem.h_numUniqueBottom() > 0) {
    findCandidatesForStripBlock<<<STRIP_SEARCH_COMBO_GRID_SIZE, STRIP_SEARCH_COMBO_THREADS_PER_BLOCK>>>(
        middleBlock,
        mem.d_uniqueTopStrips(),
        mem.d_uniqueBottomStrips(),
        *mem.h_numUniqueTop(),
        *mem.h_numUniqueBottom(),
        mem.d_candidates(),
        mem.d_numCandidates());
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());
  }

  cudaCheckError(cudaMemcpy(mem.h_numCandidates(), mem.d_numCandidates(), sizeof(uint64_t), cudaMemcpyDeviceToHost));

  // Phase 3: Process candidates to find best
  *mem.h_bestGenerations() = 0;
  *mem.h_bestPattern() = 0;

  if (*mem.h_numCandidates() > 0) {
    cudaCheckError(cudaMemcpy(mem.d_bestGenerations(), mem.h_bestGenerations(), sizeof(uint64_t), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(mem.d_bestPattern(), mem.h_bestPattern(), sizeof(uint64_t), cudaMemcpyHostToDevice));

    processCandidates<<<STRIP_SEARCH_COMBO_GRID_SIZE, STRIP_SEARCH_COMBO_THREADS_PER_BLOCK>>>(
        mem.d_candidates(), mem.d_numCandidates(), mem.d_bestPattern(), mem.d_bestGenerations(), algorithm);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    cudaCheckError(cudaMemcpy(mem.h_bestPattern(), mem.d_bestPattern(), sizeof(uint64_t), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(mem.h_bestGenerations(), mem.d_bestGenerations(), sizeof(uint64_t), cudaMemcpyDeviceToHost));
  }
}

// Main strip search execution function
// Iterates through all 2^32 middle blocks
__host__ void executeStripSearch(ProgramArgs* cli, uint32_t blockStart, uint32_t blockEnd) {
  Logger::out() << "Starting strip search...\n";
  Logger::out() << "Block range: " << blockStart << " to " << blockEnd << "\n";
  Logger::out() << "Total middle blocks to search: " << formatWithCommas(blockEnd - blockStart) << "\n\n";

  // Allocate memory using RAII
  gol::StripSearchMemory mem;

  // Allocate hash table used to find unique strip signatures
  uint32_t* d_hashTable = nullptr;
  cudaCheckError(cudaMalloc(&d_hashTable, STRIP_HASH_TABLE_SIZE * sizeof(uint32_t)));

  double intervalStartTime = getHighResCurrentTime();
  uint32_t intervalStartBlock = blockStart;
  uint64_t intervalBestGenerations = 0;
  uint64_t intervalBestPattern = 0;
  uint64_t intervalTotalTopStrips = 0;
  uint64_t intervalTotalBottomStrips = 0;
  uint64_t intervalTotalCombinations = 0;

  for (uint32_t middleBlock = blockStart; middleBlock < blockEnd; middleBlock++) {
    executeStripSearchForBlock(middleBlock, mem, d_hashTable, cli->cycleDetection);
    intervalTotalTopStrips += *mem.h_numUniqueTop();
    intervalTotalBottomStrips += *mem.h_numUniqueBottom();
    intervalTotalCombinations += (uint64_t)(*mem.h_numUniqueTop()) * (*mem.h_numUniqueBottom());

    // Track interval best (for reporting)
    if (*mem.h_bestGenerations() > intervalBestGenerations) {
      intervalBestGenerations = *mem.h_bestGenerations();
      intervalBestPattern = *mem.h_bestPattern();
    }

    // Track global best
    if (*mem.h_bestGenerations() > 0) {
      updateBestGenerations((int)*mem.h_bestGenerations());
    }

    // Progress reporting every STRIP_SEARCH_REPORT_INTERVAL blocks
    if ((middleBlock - blockStart + 1) % STRIP_SEARCH_REPORT_INTERVAL == 0) {
      reportStripSearchBlockResults(cli, intervalStartTime, intervalStartBlock, middleBlock,
                                    intervalBestGenerations, intervalBestPattern,
                                    intervalTotalTopStrips, intervalTotalBottomStrips,
                                    intervalTotalCombinations);
      intervalStartTime = getHighResCurrentTime();
      intervalStartBlock = middleBlock + 1;
      intervalBestGenerations = 0;
      intervalBestPattern = 0;
      intervalTotalTopStrips = 0;
      intervalTotalBottomStrips = 0;
      intervalTotalCombinations = 0;
    }
  }

  // Cleanup hash table (StripSearchMemory handles its own cleanup via RAII)
  cudaFree(d_hashTable);
}

// Report progress for strip search (follows same logging conventions as reportKernelResults)
__host__ void reportStripSearchBlockResults(ProgramArgs *cli, double intervalStartTime,
                                            uint32_t firstBlock, uint32_t lastBlock,
                                            uint64_t bestGenerations, uint64_t bestPattern,
                                            uint64_t totalTopStrips, uint64_t totalBottomStrips,
                                            uint64_t totalCombinations) {
  double elapsed = getHighResCurrentTime() - intervalStartTime;

  // Each middle block covers 2^32 patterns (all combinations of top/bottom strips)
  const uint64_t patternsPerBlock = 1ULL << 32;
  uint64_t blocksInRange = lastBlock - firstBlock + 1;  // inclusive range
  uint64_t patternsInRange = blocksInRange * patternsPerBlock;
  uint64_t patternsPerSec = (elapsed > 0) ? (uint64_t)(patternsInRange / elapsed) : 0;

  // Compute averages per block
  uint64_t avgTopStrips = totalTopStrips / blocksInRange;
  uint64_t avgBottomStrips = totalBottomStrips / blocksInRange;
  uint64_t avgCombinations = totalCombinations / blocksInRange;

  char bestPatternBin[BINARY_STRING_BUFFER_SIZE] = {'\0'};
  asBinary(bestPattern, bestPatternBin);

  Logger::out() << "timestamp=" << time(NULL)
                << ", firstBlock=" << firstBlock
                << ", lastBlock=" << lastBlock
                << ", avgTopStrips=" << formatWithCommas(avgTopStrips)
                << ", avgBottomStrips=" << formatWithCommas(avgBottomStrips)
                << ", avgCombinations=" << formatWithCommas(avgCombinations)
                << ", bestGenerations=" << bestGenerations
                << ", bestPattern=" << bestPattern
                << ", bestPatternBin=" << bestPatternBin
                << ", patternsPerSec=" << formatWithCommas(patternsPerSec)
                << "\n";

  if (!cli->dontSaveResults && bestGenerations > 0) {
    queueGoogleSummaryData((int)bestGenerations, bestPattern, bestPatternBin, UINT64_MAX);

    if (bestGenerations >= 200) {
      queueGoogleProgress(firstBlock, 0, (int)bestGenerations, bestPattern, bestPatternBin);
    }
  }
}

#endif
