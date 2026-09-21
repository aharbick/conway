// CUDA kernels and functions for strip-based search (reversibility optimization)
#include <iomanip>
#include <iostream>

#include "center4x4_utils.h"
#include "constants.h"
#include "gol.h"
#include "gol_memory.h"
#include "logging.h"
#include "oracle_progress.h"
#include "shutdown.h"
#include "subgrid_bloom.h"

#ifdef __NVCC__

// External declaration for shared kernel
extern __global__ void processCandidates(uint64_t *candidates, uint64_t *numCandidates, uint64_t *bestPattern,
                                         uint64_t *bestGenerations, CycleDetectionAlgorithm algorithm);

// Forward declaration for progress reporting
__host__ void reportStripSearchResults(ProgramArgs *cli, double intervalStartTime,
                                       uint32_t centerIdx, uint32_t middleIdx,
                                       uint64_t bestGenerations, uint64_t bestPattern,
                                       bool exactInterval, uint32_t highGenerations);

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
    uint32_t signature = computeStripSignature(pattern, true);
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
    uint32_t signature = computeStripSignature(pattern, false);
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

// Fast variant of findCandidatesForStripBlock. Produces exactly the same candidate set.
//
// Two changes over the legacy kernel:
//
// 1. Work division. The legacy kernel maps one thread per top strip, so only
//    numUniqueTop threads (2K-31K) of the 1M launched ever run and the rest of the GPU
//    sits idle. Here the grid is 2D - blockIdx.y picks the top strip, threads walk the
//    bottom strips - so parallelism scales with the number of strip *pairs*.
//
// 2. Repacking. Pattern lifetimes are very uneven (mean ~28 generations, tail past 180),
//    so a lane that finishes early would idle while the rest of its warp grinds on. Each
//    lane instead picks up its next bottom strip immediately and keeps the warp full.
__global__ void findCandidatesForStripBlockFast(
    uint32_t middleBlock,
    const uint16_t* uniqueTopStrips,
    const uint16_t* uniqueBottomStrips,
    uint32_t numUniqueTop,
    uint32_t numUniqueBottom,
    uint64_t* candidates,
    uint64_t* numCandidates
) {
  uint32_t topIdx = blockIdx.y;
  if (topIdx >= numUniqueTop) {
    return;
  }

  // Everything except the bottom strip is fixed for this block
  uint64_t base = (uint64_t)uniqueTopStrips[topIdx] | ((uint64_t)middleBlock << 16);

  uint32_t stride = gridDim.x * blockDim.x;
  uint32_t bottomIdx = blockIdx.x * blockDim.x + threadIdx.x;

  uint64_t pattern = 0;
  uint64_t g1 = 0;
  uint16_t generations = 0;
  bool active = false;

  while (true) {
    // Refill: this lane has no live pattern, so take the next bottom strip
    if (!active) {
      if (bottomIdx >= numUniqueBottom) {
        break;
      }
      pattern = base | ((uint64_t)uniqueBottomStrips[bottomIdx] << 48);
      bottomIdx += stride;
      g1 = pattern;
      generations = 0;
      active = true;
    }

    generations += 6;
    uint64_t g2 = computeNextGeneration8x8(g1);
    uint64_t g3 = computeNextGeneration8x8(g2);
    uint64_t g4 = computeNextGeneration8x8(g3);
    uint64_t g5 = computeNextGeneration8x8(g4);
    uint64_t g6 = computeNextGeneration8x8(g5);
    g1 = computeNextGeneration8x8(g6);

    if ((g1 == g2) || (g1 == g3) || (g1 == g4)) {
      active = false;  // Pattern ended, not interesting
    } else if (generations >= MIN_CANDIDATE_GENERATIONS) {
      uint64_t idx = atomicAdd((unsigned long long*)numCandidates, 1ULL);
      if (idx < STRIP_SEARCH_MAX_CANDIDATES) {
        candidates[idx] = pattern;
      }
      active = false;
    } else if (generations >= FAST_SEARCH_MAX_GENERATIONS) {
      active = false;
    }
  }
}

// Oracle variant of the phase-2 kernel: the same search, but patterns that cannot reach
// `target` generations are abandoned as soon as that is provable.
//
// Every 7x7-coverable 8x8 state with a terminating lifetime >= the cache's threshold is in
// the subgrid cache, and the cache's maximum is its longest. So once a pattern's live cells
// fit inside a 7x7 box at generation g:
//
//   g <= tier1Max  ->  total is capped at g + max, below target: discard, no lookup
//   g <= tier2Max  ->  a state absent from the cache lives < min more, so total is below
//                      target unless the filter says the state might be in it
//
// Bloom false positives only cost the pattern its early-out; it keeps simulating exactly as
// it would have, so they cannot change the result. Patterns that never fit a 7x7 box, or
// that fit one too late to decide, run to the same conclusion as the normal kernel.
//
// Unlike the plain fast kernel this advances one generation at a time and refills a lane the
// moment its pattern is abandoned. With the oracle killing ~98% of patterns by generation 8,
// waiting for a six-generation boundary to refill wasted most of every block.
__global__ void findCandidatesForStripBlockOracle(
    uint32_t middleBlock,
    const uint16_t* uniqueTopStrips,
    const uint16_t* uniqueBottomStrips,
    uint32_t numUniqueTop,
    uint32_t numUniqueBottom,
    uint64_t* candidates,
    uint64_t* numCandidates,
    const uint32_t* __restrict__ bloomFilter,
    uint16_t target,
    uint16_t tier1Max,
    uint16_t tier2Max
) {
  uint32_t topIdx = blockIdx.y;
  if (topIdx >= numUniqueTop) {
    return;
  }

  uint64_t base = (uint64_t)uniqueTopStrips[topIdx] | ((uint64_t)middleBlock << 16);
  uint32_t stride = gridDim.x * blockDim.x;
  uint32_t bottomIdx = blockIdx.x * blockDim.x + threadIdx.x;

  uint64_t pattern = 0, g = 0, h1 = 0, h2 = 0, h3 = 0;
  uint16_t generations = 0;
  bool active = false, probed = false;

  while (true) {
    if (!active) {
      if (bottomIdx >= numUniqueBottom) {
        break;
      }
      pattern = base | ((uint64_t)uniqueBottomStrips[bottomIdx] << 48);
      bottomIdx += stride;
      g = pattern;
      generations = 0;
      h1 = h2 = h3 = 0;
      active = true;
      probed = false;
    }

    g = computeNextGeneration8x8(g);
    generations++;

    // One oracle test per pattern, at the first generation it fits inside a 7x7 box
    if (!probed && generations <= tier2Max && isCoverableBy7x7OrEmpty(g)) {
      probed = true;
      if (generations <= tier1Max) {
        active = false;
        continue;
      }
      if (!subgridBloomMaybe(bloomFilter, g)) {
        active = false;
        continue;
      }
    }

    // Cycle detection against the previous three generations
    if (g == h1 || g == h2 || g == h3) {
      active = false;
      continue;
    }
    h3 = h2;
    h2 = h1;
    h1 = g;

    if (generations >= target) {
      uint64_t idx = atomicAdd((unsigned long long*)numCandidates, 1ULL);
      if (idx < STRIP_SEARCH_MAX_CANDIDATES) {
        candidates[idx] = pattern;
      }
      active = false;
    } else if (generations >= FAST_SEARCH_MAX_GENERATIONS) {
      active = false;
    }
  }
}

// Load the 7x7 Bloom filter onto the device, or fail loudly. Returns the header so the
// caller can derive the generation bounds from the artifact rather than from constants.
__host__ static bool loadSubgridBloom(const std::string& path, SubgridBloomHeader* header,
                                      uint32_t** d_filter) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) {
    Logger::out() << "[FATAL] --oracle needs the 7x7 Bloom filter, but " << path
                  << " could not be opened.\n        Build it with:\n"
                  << "          cat data/7x7subgrid-cache.json.gz.part_* | gzip -dc"
                  << " | ./build/build-subgrid-bloom " << DEFAULT_SUBGRID_BLOOM_PATH << "\n";
    return false;
  }

  if (fread(header, sizeof(*header), 1, f) != 1 ||
      memcmp(header->magic, SUBGRID_BLOOM_MAGIC, 8) != 0 ||
      header->version != SUBGRID_BLOOM_VERSION || header->bitsLog != SUBGRID_BLOOM_BITS_LOG ||
      header->blockBits != SUBGRID_BLOOM_BLOCK_BITS || header->k != SUBGRID_BLOOM_K) {
    Logger::out() << "[FATAL] " << path << " is not a filter this build understands"
                  << " - rebuild it with build-subgrid-bloom.\n";
    fclose(f);
    return false;
  }

  std::vector<uint32_t> filter(SUBGRID_BLOOM_WORDS);
  if (fread(filter.data(), sizeof(uint32_t), SUBGRID_BLOOM_WORDS, f) != SUBGRID_BLOOM_WORDS) {
    Logger::out() << "[FATAL] " << path << " is truncated\n";
    fclose(f);
    return false;
  }
  fclose(f);

  cudaCheckError(cudaMalloc(d_filter, SUBGRID_BLOOM_BYTES));
  cudaCheckError(cudaMemcpy(*d_filter, filter.data(), SUBGRID_BLOOM_BYTES, cudaMemcpyHostToDevice));
  return true;
}

// Execute strip search for a single middle block using StripSearchMemory
// Hash table is allocated locally and reused between top/bottom strip finding
// Parameters for the oracle, derived once from the filter's header
struct OracleParams {
  const uint32_t* d_filter = nullptr;  // nullptr disables the oracle
  uint16_t target = 0;
  uint16_t tier1Max = 0;
  uint16_t tier2Max = 0;
};

__host__ void executeStripSearchForBlock(
    uint32_t middleBlock,
    gol::StripSearchMemory& mem,
    uint32_t* d_hashTable,
    CycleDetectionAlgorithm algorithm,
    StripKernel stripKernel,
    const OracleParams& oracle
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
    if (oracle.d_filter != nullptr) {
      dim3 grid(STRIP_ORACLE_X_BLOCKS, *mem.h_numUniqueTop());
      findCandidatesForStripBlockOracle<<<grid, STRIP_ORACLE_THREADS_PER_BLOCK>>>(
          middleBlock,
          mem.d_uniqueTopStrips(),
          mem.d_uniqueBottomStrips(),
          *mem.h_numUniqueTop(),
          *mem.h_numUniqueBottom(),
          mem.d_candidates(),
          mem.d_numCandidates(),
          oracle.d_filter,
          oracle.target,
          oracle.tier1Max,
          oracle.tier2Max);
    } else if (stripKernel == STRIP_KERNEL_LEGACY) {
      findCandidatesForStripBlock<<<STRIP_SEARCH_COMBO_GRID_SIZE, STRIP_SEARCH_COMBO_THREADS_PER_BLOCK>>>(
          middleBlock,
          mem.d_uniqueTopStrips(),
          mem.d_uniqueBottomStrips(),
          *mem.h_numUniqueTop(),
          *mem.h_numUniqueBottom(),
          mem.d_candidates(),
          mem.d_numCandidates());
    } else {
      // One block per top strip in the y dimension (numUniqueTop <= 32768, well under the
      // 65535 gridDim.y limit), x dimension and threads walk the bottom strips.
      dim3 grid(STRIP_SEARCH_COMBO_FAST_X_BLOCKS, *mem.h_numUniqueTop());
      findCandidatesForStripBlockFast<<<grid, STRIP_SEARCH_COMBO_FAST_THREADS_PER_BLOCK>>>(
          middleBlock,
          mem.d_uniqueTopStrips(),
          mem.d_uniqueBottomStrips(),
          *mem.h_numUniqueTop(),
          *mem.h_numUniqueBottom(),
          mem.d_candidates(),
          mem.d_numCandidates());
    }
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());
  }

  cudaCheckError(cudaMemcpy(mem.h_numCandidates(), mem.d_numCandidates(), sizeof(uint64_t), cudaMemcpyDeviceToHost));

  // Warn if candidates exceeded buffer size (some were lost)
  if (*mem.h_numCandidates() > STRIP_SEARCH_MAX_CANDIDATES) {
    Logger::out() << "WARNING: " << *mem.h_numCandidates() << " candidates found, buffer only holds "
                  << STRIP_SEARCH_MAX_CANDIDATES << " - some candidates lost!\n";
  }

  // Phase 3: Process candidates to find best
  *mem.h_bestGenerations() = 0;
  *mem.h_bestPattern() = 0;

  // Cap numCandidates to buffer size to avoid out-of-bounds access
  uint64_t cappedCandidates = (*mem.h_numCandidates() > STRIP_SEARCH_MAX_CANDIDATES)
                                  ? STRIP_SEARCH_MAX_CANDIDATES
                                  : *mem.h_numCandidates();

  if (cappedCandidates > 0) {
    // Write capped count back to device so processCandidates doesn't read out-of-bounds
    cudaCheckError(cudaMemcpy(mem.d_numCandidates(), &cappedCandidates, sizeof(uint64_t), cudaMemcpyHostToDevice));
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
// Iterates through unique center 4x4 blocks (8548 total) with ear combinations (256 × 256)
// Supports partial ranges: middleStart/middleEnd apply to first/last center respectively
__host__ void executeStripSearch(ProgramArgs* cli, uint32_t centerStart, uint32_t centerEnd,
                                 uint32_t middleStart, uint32_t middleEnd) {
  // Allocate memory using RAII
  gol::StripSearchMemory mem;

  // Allocate hash table used to find unique strip signatures
  uint32_t* d_hashTable = nullptr;
  cudaCheckError(cudaMalloc(&d_hashTable, STRIP_HASH_TABLE_SIZE * sizeof(uint32_t)));

  // Set up the 7x7 oracle if asked for. The target defaults to the best known terminating
  // pattern, not one past it, so a pattern merely matching the record is still reported -
  // another example of it is worth having. Finding a longer one raises the target to that,
  // which tightens the pruning and speeds up the rest of the search.
  OracleParams oracle;
  OracleProgress oracleProgress;
  uint32_t* d_bloomFilter = nullptr;
  SubgridBloomHeader bloomHeader{};
  if (cli->useOracle) {
    SubgridBloomHeader& header = bloomHeader;
    if (!loadSubgridBloom(cli->bloomFilePath, &header, &d_bloomFilter)) {
      cudaFree(d_hashTable);
      return;
    }

    uint32_t target = cli->oracleTarget;
    if (target == 0) {
      target = (gBestGenerations > 0) ? (uint32_t)gBestGenerations : STRIP_ORACLE_FALLBACK_TARGET;
    }

    oracle.target = (uint16_t)target;
    oracle.tier1Max = (uint16_t)subgridBloomTier1Max(header, target);
    oracle.tier2Max = (uint16_t)subgridBloomTier2Max(header, target);
    oracle.d_filter = d_bloomFilter;

    Logger::out() << "7x7 oracle enabled: reporting patterns at or above " << target
                  << " generations"
                  << " (cache " << header.numKeys << " states, "
                  << header.minGenerations << ".." << header.maxGenerations << " generations)\n";
    Logger::out() << "  discarding a 7x7-coverable pattern needs no lookup up to generation "
                  << oracle.tier1Max << ", and a filter miss up to generation "
                  << oracle.tier2Max << "\n";
    Logger::out() << "  every " << STRIP_ORACLE_HISTOGRAM_SAMPLE
                  << "th middleIdx still runs the exact kernel, to keep feeding the histogram\n";

    if (!oracleProgress.load(cli->oracleProgressPath, target)) {
      cudaFree(d_hashTable);
      cudaFree(d_bloomFilter);
      return;
    }
    Logger::out() << "  completed intervals go to " << cli->oracleProgressPath
                  << ", NOT the shared completion bitmap: clearing an interval of patterns"
                  << " at or above the target is not the same as searching it exhaustively\n";

    if (oracle.tier2Max == 0) {
      Logger::out() << "[FATAL] target " << target << " is too low for the oracle to prune"
                    << " (needs > " << header.minGenerations + 1 << ")\n";
      cudaFree(d_hashTable);
      cudaFree(d_bloomFilter);
      return;
    }
  }

  double intervalStartTime = getHighResCurrentTime();
  uint64_t intervalBestGenerations = 0;
  uint64_t intervalBestPattern = 0;

  for (uint32_t centerIdx = centerStart; centerIdx < centerEnd; centerIdx++) {
    uint16_t center4x4 = get4x4CenterByIndex(centerIdx);

    // Determine middleIdx range for this center:
    // - First center: start from middleStart
    // - Last center: end at middleEnd
    // - Middle centers: full range (0 to 512)
    uint32_t thisMiddleStart = (centerIdx == centerStart) ? middleStart : 0;
    uint32_t thisMiddleEnd = (centerIdx == centerEnd - 1) ? middleEnd : STRIP_SEARCH_TOTAL_MIDDLE_IDX;

    // middleIdx 0-511, each covers 128 middle blocks
    for (uint32_t middleIdx = thisMiddleStart; middleIdx < thisMiddleEnd; middleIdx++) {
      // Skip already-completed intervals (unless --dont-save-results is set). An interval
      // searched exhaustively is also settled for the oracle - its best was recorded, and it
      // was below the target - so either bitmap is enough to skip it.
      if (!cli->dontSaveResults && isGoogleStripIntervalComplete(centerIdx, middleIdx)) {
        continue;
      }
      if (oracle.d_filter != nullptr && oracleProgress.isComplete(centerIdx, middleIdx)) {
        continue;
      }

      // A new record raises the bar to itself, so the search goes on reporting every pattern
      // that matches the new record until something beats it. Only ever upward: intervals
      // already cleared at a lower target stay valid.
      if (oracle.d_filter != nullptr && cli->oracleTarget == 0 && gBestGenerations > 0 &&
          (uint32_t)gBestGenerations > oracle.target) {
        uint32_t raised = (uint32_t)gBestGenerations;
        oracle.target = (uint16_t)raised;
        oracle.tier1Max = (uint16_t)subgridBloomTier1Max(bloomHeader, raised);
        oracle.tier2Max = (uint16_t)subgridBloomTier2Max(bloomHeader, raised);
        Logger::out() << "Oracle target raised to " << raised
                      << " (no lookup up to generation " << oracle.tier1Max
                      << ", filter miss up to " << oracle.tier2Max << ")\n";
      }

      // Process STRIP_SEARCH_MIDDLE_BLOCKS_PER_REPORT middle blocks for this middleIdx
      uint32_t blockStart = middleIdx * STRIP_SEARCH_MIDDLE_BLOCKS_PER_REPORT;
      uint32_t blockEnd = blockStart + STRIP_SEARCH_MIDDLE_BLOCKS_PER_REPORT;

      // The oracle cannot report an interval best below its target, so sample the exact
      // kernel periodically to keep the histogram populated.
      bool exactInterval = (oracle.d_filter == nullptr) ||
                           (middleIdx % STRIP_ORACLE_HISTOGRAM_SAMPLE == 0);
      OracleParams intervalOracle = exactInterval ? OracleParams{} : oracle;

      for (uint32_t blockOffset = blockStart; blockOffset < blockEnd; blockOffset++) {
        if (shutdownRequested()) {
          break;  // finish this middle block, then abandon the interval unreported
        }
        uint16_t leftEar = blockOffset / CENTER_4X4_TOTAL_EAR_VALUES;
        uint16_t rightEar = blockOffset % CENTER_4X4_TOTAL_EAR_VALUES;
        uint32_t middleBlock = reconstructMiddleBlock(center4x4, (uint8_t)leftEar, (uint8_t)rightEar);
        executeStripSearchForBlock(middleBlock, mem, d_hashTable, cli->cycleDetection,
                                   cli->stripKernel, intervalOracle);

        // Track interval best (for reporting) and update global best
        if (*mem.h_bestGenerations() > intervalBestGenerations) {
          intervalBestGenerations = *mem.h_bestGenerations();
          intervalBestPattern = *mem.h_bestPattern();
          updateBestGenerations((int)intervalBestGenerations);
        }
      }

      // An interval interrupted part way through has not been searched, so it must not be
      // reported or recorded - leaving it unmarked means the next run redoes it.
      if (shutdownRequested()) {
        Logger::out() << "Stopped before finishing centerIdx=" << centerIdx
                      << ", middleIdx=" << middleIdx << "; it will be searched again\n";
        break;
      }

      // Report at end of each middleIdx. Only an exact interval may claim exhaustive
      // completion; an oracle interval records itself in the local bitmap instead.
      //
      // The target is the best known, so it is the figure to log as the record being
      // matched. It was target - 1 while the target sat one past the best.
      reportStripSearchResults(cli, intervalStartTime, centerIdx, middleIdx,
                               intervalBestGenerations, intervalBestPattern, exactInterval,
                               (oracle.d_filter != nullptr) ? (uint32_t)oracle.target : 0);

      if (oracle.d_filter != nullptr) {
        // Exact intervals satisfy the oracle's claim too, so they mark both
        oracleProgress.markComplete(centerIdx, middleIdx, oracle.target);
        oracleProgress.save();
      }

      // Reset interval tracking
      intervalStartTime = getHighResCurrentTime();
      intervalBestGenerations = 0;
      intervalBestPattern = 0;
    }
    if (shutdownRequested()) {
      break;
    }
  }

  // Cleanup (StripSearchMemory handles its own cleanup via RAII)
  oracleProgress.save();
  cudaFree(d_hashTable);
  if (d_bloomFilter != nullptr) {
    cudaFree(d_bloomFilter);
  }
}

// Report progress for strip search (follows same logging conventions as reportKernelResults)
__host__ void reportStripSearchResults(ProgramArgs *cli, double intervalStartTime,
                                       uint32_t centerIdx, uint32_t middleIdx,
                                       uint64_t bestGenerations, uint64_t bestPattern,
                                       bool exactInterval, uint32_t highGenerations) {
  double elapsed = getHighResCurrentTime() - intervalStartTime;

  // Calculate patterns per middleIdx interval:
  // - Each middleIdx covers 128 middle blocks
  // - Patterns per middleBlock: 2^64 / 560,201,728 ≈ 32,928,752,540
  // - Patterns per middleIdx: 128 × 32,928,752,540 ≈ 4,214,880,325,120
  const uint64_t patternsPerMiddleIdx = 4214880325120ULL;
  uint64_t patternsPerSec = (elapsed > 0) ? (uint64_t)(patternsPerMiddleIdx / elapsed) : 0;

  char bestPatternBin[BINARY_STRING_BUFFER_SIZE] = {'\0'};
  asBinary(bestPattern, bestPatternBin);

  Logger& out = Logger::out();
  out << "timestamp=" << time(NULL)
      << ", centerIdx=" << centerIdx
      << ", middleIdx=" << middleIdx;

  // Report a best only when there is one. In oracle mode there usually is not, and
  // "bestGenerations=0" would be a claim rather than an absence: the interval does have a
  // best, the oracle simply never measured it, having discarded everything below its target
  // without looking. Omitting the tokens also keeps the lines that did find something
  // visible in a log where almost nothing does.
  if (bestGenerations > 0) {
    out << ", bestGenerations=" << bestGenerations
        << ", bestPattern=" << bestPattern
        << ", bestPatternBin=" << bestPatternBin;
  }

  // The bar this interval was searched against. Worth recording on every line of an oracle
  // run, including its sampled exact intervals, because the bar rises the moment a record
  // lands - so without it a line does not say what it was measured against.
  if (highGenerations > 0) {
    out << ", highGenerations=" << highGenerations;
  }

  out << ", patternsPerSec=" << formatWithCommas(patternsPerSec)
      << ", mode=" << (exactInterval ? "exact" : "oracle")
      << "\n";

  // Save to Google Sheets if appropriate
  if (!cli->dontSaveResults) {
    // Track completion for every interval, including the ones where nothing survived long
    // enough to report. Gating this on bestGenerations > 0 left those intervals forever
    // unmarked, so every run resumed at the earliest one and redid the work behind it.
    //
    // Only exact intervals may claim this, though. The shared bitmap means "searched
    // exhaustively and its best recorded", which is what a full run resumes from and what
    // feeds the histogram; an oracle interval has established only that nothing there
    // reaches the target. Marking those here would make a later exhaustive run skip them
    // and lose their distribution data for good.
    if (exactInterval) {
      queueGoogleStripCompletion(centerIdx, middleIdx);
    }

    if (bestGenerations > 0) {
      // Record summary data for histogram
      queueGoogleStripSummaryData((int)bestGenerations, bestPattern, bestPatternBin);

      // Only log detailed progress for high-generation results
      if (bestGenerations >= 204) {
        queueGoogleStripProgress(centerIdx, middleIdx, (int)bestGenerations, bestPattern, bestPatternBin);
      }
    }
  }
}

#endif
