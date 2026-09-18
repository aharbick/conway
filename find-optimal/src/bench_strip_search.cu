// Benchmark harness for strip search: times each phase on real middle blocks and
// compares phase-2 kernel variants against the production kernel, checking that every
// variant produces an identical candidate set.
//
//   ./build/bench-strip-search [centerIdx] [middleIdx] [numBlocks] [threads] [xblocks] [hist] [onlyVariant]
//
// e.g.  ./build/bench-strip-search 836 380 3 32 1 1          # all variants + lifetime histogram
//       ./build/bench-strip-search 836 380 1 64 4 0 v2       # sweep one variant's launch config
//
// Variants: v0 = legacy kernel (one thread per top strip), v1 = 2D grid with one pair per
// thread, v2 = 2D grid + lane repacking (this is --strip-kernel=fast), v3xN = v2 plus
// N-wide ILP. Measured on a 5090: v2 is 4.8x-54x faster than v0 depending on how many
// unique top strips the middle block has.
//
// NOTE: the kernels here are copies of the ones in gol_cuda_strips.cu, kept local so this
// harness links without the Google Sheets/curl dependencies. If you change a production
// kernel, mirror it here.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>

#include "center4x4_utils.h"
#include "constants.h"

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)

// ---------------------------------------------------------------- phase 1 --
__device__ static inline void addIfUnique(uint16_t strip, uint32_t signature, uint32_t* hashTable,
                                          uint16_t* uniqueStrips, uint32_t* numUnique) {
  uint32_t hash = hashSignature(signature);
  uint32_t bucket = hash & STRIP_HASH_TABLE_MASK;
  for (int probe = 0; probe < STRIP_MAX_PROBE_LENGTH; probe++) {
    uint32_t probeBucket = (bucket + probe) & STRIP_HASH_TABLE_MASK;
    uint32_t old = atomicCAS(&hashTable[probeBucket], 0xFFFFFFFF, signature);
    if (old == 0xFFFFFFFF) {
      uint32_t idx = atomicAdd(numUnique, 1);
      if (idx < STRIP_SEARCH_MAX_VALID_STRIPS) uniqueStrips[idx] = strip;
      return;
    } else if (old == signature) {
      return;
    }
  }
}

__global__ void findUniqueTopStrips(uint32_t middleBlock, uint16_t* uniqueStrips, uint32_t* numUnique,
                                    uint32_t* hashTable) {
  uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t totalThreads = gridDim.x * blockDim.x;
  for (uint32_t strip = threadId; strip < STRIP_SEARCH_TOTAL_STRIPS; strip += totalThreads) {
    uint64_t pattern = ((uint64_t)strip) | ((uint64_t)middleBlock << 16);
    addIfUnique((uint16_t)strip, computeStripSignature(pattern, true), hashTable, uniqueStrips, numUnique);
  }
}

__global__ void findUniqueBottomStrips(uint32_t middleBlock, uint16_t* uniqueStrips, uint32_t* numUnique,
                                       uint32_t* hashTable) {
  uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t totalThreads = gridDim.x * blockDim.x;
  for (uint32_t strip = threadId; strip < STRIP_SEARCH_TOTAL_STRIPS; strip += totalThreads) {
    uint64_t pattern = ((uint64_t)middleBlock << 16) | ((uint64_t)strip << 48);
    addIfUnique((uint16_t)strip, computeStripSignature(pattern, false), hashTable, uniqueStrips, numUnique);
  }
}

// ------------------------------------------------------------- 7x7 oracle --
//
// Every 7x7-coverable 8x8 state with a terminating lifetime >= 180 is in the subgrid
// cache, and the longest one there is 206 generations. So a pattern that first fits inside
// a 7x7 box at generation g can never exceed g + 206 generations in total, and a state
// absent from the cache lives fewer than 180 more. That gives two sound early-outs when
// hunting for patterns above a target T:
//
//   tier 1: coverable at g <= T - 207  ->  discard. No lookup at all.
//   tier 2: coverable at g <= T - 181  ->  discard unless the state is in the cache.
//
// Tier 2 needs a membership test, and a cache-resident blocked Bloom filter is enough: a
// false positive only means the pattern keeps simulating as it would have anyway, so the
// filter can be small and lossy without affecting correctness.
#define BLOOM_BITS_LOG 28                        // 2^28 bits = 32MB
#define BLOOM_BLOCK_BITS 512                     // one 64-byte cache line per probe
#define BLOOM_NUM_BLOCKS (1u << (BLOOM_BITS_LOG - 9))
#define BLOOM_K 6
#define SUBGRID_MAX_GENERATIONS 206              // longest terminating 7x7-coverable state

__host__ __device__ static inline uint64_t bloomHash(uint64_t key) {
  key = (key ^ (key >> 30)) * 0xBF58476D1CE4E5B9ULL;
  key = (key ^ (key >> 27)) * 0x94D049BB133111EBULL;
  return key ^ (key >> 31);
}

// Bit positions for one key, all inside a single 512-bit block
__host__ __device__ static inline void bloomSlots(uint64_t key, uint32_t& blockIdx, uint32_t bits[BLOOM_K]) {
  uint64_t h = bloomHash(key);
  blockIdx = (uint32_t)(h >> 40) & (BLOOM_NUM_BLOCKS - 1);
  uint32_t a = (uint32_t)h;
  uint32_t b = (uint32_t)(h >> 20) | 1u;  // odd, so the stride never degenerates
#pragma unroll
  for (int i = 0; i < BLOOM_K; i++) {
    bits[i] = (a + i * b) & (BLOOM_BLOCK_BITS - 1);
  }
}

static inline void bloomInsert(uint32_t* filter, uint64_t key) {
  uint32_t blk, bits[BLOOM_K];
  bloomSlots(key, blk, bits);
  uint32_t* words = filter + (size_t)blk * (BLOOM_BLOCK_BITS / 32);
  for (int i = 0; i < BLOOM_K; i++) {
    words[bits[i] >> 5] |= 1u << (bits[i] & 31);
  }
}

__device__ static inline bool bloomMaybe(const uint32_t* __restrict__ filter, uint64_t key) {
  uint32_t blk, bits[BLOOM_K];
  bloomSlots(key, blk, bits);
  const uint32_t* words = filter + (size_t)blk * (BLOOM_BLOCK_BITS / 32);
#pragma unroll
  for (int i = 0; i < BLOOM_K; i++) {
    if (!((words[bits[i] >> 5] >> (bits[i] & 31)) & 1u)) return false;
  }
  return true;
}

// ------------------------------------------------------- phase 2 variants --

// v0: current production kernel, verbatim.
__global__ void v0_baseline(uint32_t middleBlock, const uint16_t* uniqueTopStrips,
                            const uint16_t* uniqueBottomStrips, uint32_t numUniqueTop,
                            uint32_t numUniqueBottom, uint64_t* candidates, uint64_t* numCandidates) {
  uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t totalThreads = gridDim.x * blockDim.x;

  for (uint32_t topIdx = threadId; topIdx < numUniqueTop; topIdx += totalThreads) {
    uint64_t topBits = (uint64_t)uniqueTopStrips[topIdx];
    for (uint32_t bottomIdx = 0; bottomIdx < numUniqueBottom; bottomIdx++) {
      uint64_t pattern = topBits | ((uint64_t)middleBlock << 16) |
                         ((uint64_t)uniqueBottomStrips[bottomIdx] << 48);
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
        if ((g1 == g2) || (g1 == g3) || (g1 == g4)) break;
        if (generations >= MIN_CANDIDATE_GENERATIONS) {
          uint64_t idx = atomicAdd((unsigned long long*)numCandidates, 1ULL);
          if (idx < STRIP_SEARCH_MAX_CANDIDATES) candidates[idx] = pattern;
          break;
        }
      }
    }
  }
}

// v1: 2D grid, one (top,bottom) pair per thread. blockIdx.y = top strip index.
__global__ void v1_pair_per_thread(uint32_t middleBlock, const uint16_t* uniqueTopStrips,
                                   const uint16_t* uniqueBottomStrips, uint32_t numUniqueTop,
                                   uint32_t numUniqueBottom, uint64_t* candidates,
                                   uint64_t* numCandidates) {
  uint32_t topIdx = blockIdx.y;
  if (topIdx >= numUniqueTop) return;
  uint64_t base = (uint64_t)uniqueTopStrips[topIdx] | ((uint64_t)middleBlock << 16);

  for (uint32_t bottomIdx = blockIdx.x * blockDim.x + threadIdx.x; bottomIdx < numUniqueBottom;
       bottomIdx += gridDim.x * blockDim.x) {
    uint64_t pattern = base | ((uint64_t)uniqueBottomStrips[bottomIdx] << 48);
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
      if ((g1 == g2) || (g1 == g3) || (g1 == g4)) break;
      if (generations >= MIN_CANDIDATE_GENERATIONS) {
        uint64_t idx = atomicAdd((unsigned long long*)numCandidates, 1ULL);
        if (idx < STRIP_SEARCH_MAX_CANDIDATES) candidates[idx] = pattern;
        break;
      }
    }
  }
}

// v2: 2D grid + repacking. A lane that finishes its pattern immediately picks up the
// next one instead of idling while the rest of the warp finishes (divergence fix).
__global__ void v2_repack(uint32_t middleBlock, const uint16_t* uniqueTopStrips,
                          const uint16_t* uniqueBottomStrips, uint32_t numUniqueTop,
                          uint32_t numUniqueBottom, uint64_t* candidates, uint64_t* numCandidates) {
  uint32_t topIdx = blockIdx.y;
  if (topIdx >= numUniqueTop) return;
  uint64_t base = (uint64_t)uniqueTopStrips[topIdx] | ((uint64_t)middleBlock << 16);

  uint32_t stride = gridDim.x * blockDim.x;
  uint32_t bottomIdx = blockIdx.x * blockDim.x + threadIdx.x;

  uint64_t pattern = 0, g1 = 0;
  uint16_t generations = 0;
  bool active = false;

  while (true) {
    if (!active) {
      if (bottomIdx >= numUniqueBottom) break;
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
      active = false;
    } else if (generations >= MIN_CANDIDATE_GENERATIONS) {
      uint64_t idx = atomicAdd((unsigned long long*)numCandidates, 1ULL);
      if (idx < STRIP_SEARCH_MAX_CANDIDATES) candidates[idx] = pattern;
      active = false;
    } else if (generations >= FAST_SEARCH_MAX_GENERATIONS) {
      active = false;
    }
  }
}

// v3: repacking + 2-wide ILP (two independent patterns per thread interleaved).
template <int W>
__global__ void v3_repack_ilp(uint32_t middleBlock, const uint16_t* uniqueTopStrips,
                              const uint16_t* uniqueBottomStrips, uint32_t numUniqueTop,
                              uint32_t numUniqueBottom, uint64_t* candidates, uint64_t* numCandidates) {
  uint32_t topIdx = blockIdx.y;
  if (topIdx >= numUniqueTop) return;
  uint64_t base = (uint64_t)uniqueTopStrips[topIdx] | ((uint64_t)middleBlock << 16);

  uint32_t stride = gridDim.x * blockDim.x;
  uint32_t next = blockIdx.x * blockDim.x + threadIdx.x;

  uint64_t pattern[W], g1[W];
  uint16_t gens[W];
  bool active[W];
#pragma unroll
  for (int w = 0; w < W; w++) { active[w] = false; pattern[w] = 0; g1[w] = 0; gens[w] = 0; }

  int live = 0;
  while (true) {
    live = 0;
#pragma unroll
    for (int w = 0; w < W; w++) {
      if (!active[w] && next < numUniqueBottom) {
        pattern[w] = base | ((uint64_t)uniqueBottomStrips[next] << 48);
        next += stride;
        g1[w] = pattern[w];
        gens[w] = 0;
        active[w] = true;
      }
      if (active[w]) live++;
    }
    if (live == 0) break;

    uint64_t g2[W], g3[W], g4[W];
#pragma unroll
    for (int w = 0; w < W; w++) {
      g2[w] = computeNextGeneration8x8(g1[w]);
    }
#pragma unroll
    for (int w = 0; w < W; w++) g3[w] = computeNextGeneration8x8(g2[w]);
#pragma unroll
    for (int w = 0; w < W; w++) g4[w] = computeNextGeneration8x8(g3[w]);
#pragma unroll
    for (int w = 0; w < W; w++) g1[w] = computeNextGeneration8x8(g4[w]);
#pragma unroll
    for (int w = 0; w < W; w++) g1[w] = computeNextGeneration8x8(g1[w]);
#pragma unroll
    for (int w = 0; w < W; w++) g1[w] = computeNextGeneration8x8(g1[w]);

#pragma unroll
    for (int w = 0; w < W; w++) {
      if (!active[w]) continue;
      gens[w] += 6;
      if ((g1[w] == g2[w]) || (g1[w] == g3[w]) || (g1[w] == g4[w])) {
        active[w] = false;
      } else if (gens[w] >= MIN_CANDIDATE_GENERATIONS) {
        uint64_t idx = atomicAdd((unsigned long long*)numCandidates, 1ULL);
        if (idx < STRIP_SEARCH_MAX_CANDIDATES) candidates[idx] = pattern[w];
        active[w] = false;
      } else if (gens[w] >= FAST_SEARCH_MAX_GENERATIONS) {
        active[w] = false;
      }
    }
  }
}

// v4/v5: v2 plus the 7x7 oracle. tier1Max/tier2Max are the highest generation at which
// each early-out is sound for the target; pass tier2Max = 0 (and filter = nullptr) to
// measure tier 1 alone.
__global__ void v5_oracle(uint32_t middleBlock, const uint16_t* uniqueTopStrips,
                          const uint16_t* uniqueBottomStrips, uint32_t numUniqueTop,
                          uint32_t numUniqueBottom, uint64_t* candidates, uint64_t* numCandidates,
                          const uint32_t* __restrict__ filter, uint16_t target,
                          uint16_t tier1Max, uint16_t tier2Max) {
  uint32_t topIdx = blockIdx.y;
  if (topIdx >= numUniqueTop) return;
  uint64_t base = (uint64_t)uniqueTopStrips[topIdx] | ((uint64_t)middleBlock << 16);

  uint32_t stride = gridDim.x * blockDim.x;
  uint32_t bottomIdx = blockIdx.x * blockDim.x + threadIdx.x;

  uint64_t pattern = 0, g = 0;
  uint16_t gens = 0;
  bool active = false, probed = false;

  while (true) {
    if (!active) {
      if (bottomIdx >= numUniqueBottom) break;
      pattern = base | ((uint64_t)uniqueBottomStrips[bottomIdx] << 48);
      bottomIdx += stride;
      g = pattern;
      gens = 0;
      active = true;
      probed = false;
    }

    uint64_t prev[3];
#pragma unroll
    for (int step = 0; step < 6; step++) {
      g = computeNextGeneration8x8(g);
      gens++;
      if (step < 3) prev[step] = g;

      // One oracle test per pattern, at the first generation it fits a 7x7 box
      if (!probed && gens <= tier2Max && isCoverableBy7x7(g)) {
        probed = true;
        if (gens <= tier1Max) {
          active = false;  // cannot exceed gens + 206, so cannot reach the target
          break;
        }
        if (filter != nullptr && !bloomMaybe(filter, g)) {
          active = false;  // lives < 180 more, so cannot reach the target
          break;
        }
      }
    }
    if (!active) continue;

    // Same cycle test as the production kernel: g is 6 generations on from its start
    if ((g == prev[0]) || (g == prev[1]) || (g == prev[2])) {
      active = false;
    } else if (gens >= target) {
      uint64_t idx = atomicAdd((unsigned long long*)numCandidates, 1ULL);
      if (idx < STRIP_SEARCH_MAX_CANDIDATES) candidates[idx] = pattern;
      active = false;
    } else if (gens >= FAST_SEARCH_MAX_GENERATIONS) {
      active = false;
    }
  }
}

// v6: the oracle, with the loop restructured for it.
//
// v5 kept v2's shape - six generations, then decide - which was fine when patterns ran ~28
// generations but wastes most of a block once the oracle kills them at generation ~6: a
// lane that discards at generation 2 still idles through four more before it can refill.
// Here the loop advances one generation at a time and refills the instant a pattern is
// discarded, with a rolling 3-generation history for the same cycle test. The coverability
// test is also reordered to fail on its first term, which is the common case early on.
__device__ static inline bool coverable7x7Fast(uint64_t g) {
  // Rows first: all live cells must miss row 7 or miss row 0. Dense early patterns fail
  // here immediately, which is most of the calls.
  bool rows = ((g & 0xFF00000000000000ULL) == 0) || ((g & 0x00000000000000FFULL) == 0);
  if (!rows) return false;
  return ((g & 0x8080808080808080ULL) == 0) || ((g & 0x0101010101010101ULL) == 0);
}

__global__ void v6_oracle_fused(uint32_t middleBlock, const uint16_t* uniqueTopStrips,
                                const uint16_t* uniqueBottomStrips, uint32_t numUniqueTop,
                                uint32_t numUniqueBottom, uint64_t* candidates,
                                uint64_t* numCandidates, const uint32_t* __restrict__ filter,
                                uint16_t target, uint16_t tier1Max, uint16_t tier2Max) {
  uint32_t topIdx = blockIdx.y;
  if (topIdx >= numUniqueTop) return;
  uint64_t base = (uint64_t)uniqueTopStrips[topIdx] | ((uint64_t)middleBlock << 16);

  uint32_t stride = gridDim.x * blockDim.x;
  uint32_t bottomIdx = blockIdx.x * blockDim.x + threadIdx.x;

  uint64_t pattern = 0, g = 0, h1 = 0, h2 = 0, h3 = 0;
  uint16_t gens = 0;
  bool active = false, probed = false;

  while (true) {
    if (!active) {
      if (bottomIdx >= numUniqueBottom) break;
      pattern = base | ((uint64_t)uniqueBottomStrips[bottomIdx] << 48);
      bottomIdx += stride;
      g = pattern;
      gens = 0;
      h1 = h2 = h3 = 0;
      active = true;
      probed = false;
    }

    g = computeNextGeneration8x8(g);
    gens++;

    // Oracle: one test per pattern, at the first generation it fits a 7x7 box
    if (!probed && gens <= tier2Max && coverable7x7Fast(g)) {
      probed = true;
      if (gens <= tier1Max) {
        active = false;
        continue;
      }
      if (filter != nullptr && !bloomMaybe(filter, g)) {
        active = false;
        continue;
      }
    }

    // Cycle test against the last three generations, equivalent to the production check
    if (g == h1 || g == h2 || g == h3) {
      active = false;
      continue;
    }
    h3 = h2;
    h2 = h1;
    h1 = g;

    if (gens >= target) {
      uint64_t idx = atomicAdd((unsigned long long*)numCandidates, 1ULL);
      if (idx < STRIP_SEARCH_MAX_CANDIDATES) candidates[idx] = pattern;
      active = false;
    } else if (gens >= FAST_SEARCH_MAX_GENERATIONS) {
      active = false;
    }
  }
}

// v7: v6, with the cycle test deferred past the oracle window.
//
// While the oracle window is open, the cycle test is nearly dead weight: a pattern that
// settles into a still life or a blinker fits inside a 7x7 box, so tier 1 discards it
// anyway. The only patterns it would catch early are ones that both cycle before
// generation tier2Max and never fit a 7x7 box, which is a fraction of the 1.8% that are
// never coverable - a rounding error against six fewer instructions on every generation of
// the hot path. The history is still maintained so the test is exact once it turns on.
template <int W>
__global__ void v7_oracle_lean(uint32_t middleBlock, const uint16_t* uniqueTopStrips,
                               const uint16_t* uniqueBottomStrips, uint32_t numUniqueTop,
                               uint32_t numUniqueBottom, uint64_t* candidates,
                               uint64_t* numCandidates, const uint32_t* __restrict__ filter,
                               uint16_t target, uint16_t tier1Max, uint16_t tier2Max) {
  uint32_t topIdx = blockIdx.y;
  if (topIdx >= numUniqueTop) return;
  uint64_t base = (uint64_t)uniqueTopStrips[topIdx] | ((uint64_t)middleBlock << 16);

  uint32_t stride = gridDim.x * blockDim.x;
  uint32_t next = blockIdx.x * blockDim.x + threadIdx.x;

  uint64_t pattern[W], g[W], h1[W], h2[W], h3[W];
  uint16_t gens[W];
  bool active[W], probed[W];
#pragma unroll
  for (int w = 0; w < W; w++) {
    active[w] = false;
    probed[w] = false;
    pattern[w] = g[w] = h1[w] = h2[w] = h3[w] = 0;
    gens[w] = 0;
  }

  while (true) {
    int live = 0;
#pragma unroll
    for (int w = 0; w < W; w++) {
      if (!active[w] && next < numUniqueBottom) {
        pattern[w] = base | ((uint64_t)uniqueBottomStrips[next] << 48);
        next += stride;
        g[w] = pattern[w];
        gens[w] = 0;
        h1[w] = h2[w] = h3[w] = 0;
        active[w] = true;
        probed[w] = false;
      }
      if (active[w]) live++;
    }
    if (live == 0) break;

#pragma unroll
    for (int w = 0; w < W; w++) {
      if (!active[w]) continue;

      g[w] = computeNextGeneration8x8(g[w]);
      gens[w]++;

      if (!probed[w] && gens[w] <= tier2Max && coverable7x7Fast(g[w])) {
        probed[w] = true;
        if (gens[w] <= tier1Max) {
          active[w] = false;
          continue;
        }
        if (filter != nullptr && !bloomMaybe(filter, g[w])) {
          active[w] = false;
          continue;
        }
      }

      // Only compare once the oracle can no longer decide it
      if (gens[w] > tier2Max && (g[w] == h1[w] || g[w] == h2[w] || g[w] == h3[w])) {
        active[w] = false;
        continue;
      }
      h3[w] = h2[w];
      h2[w] = h1[w];
      h1[w] = g[w];

      if (gens[w] >= target) {
        uint64_t idx = atomicAdd((unsigned long long*)numCandidates, 1ULL);
        if (idx < STRIP_SEARCH_MAX_CANDIDATES) candidates[idx] = pattern[w];
        active[w] = false;
      } else if (gens[w] >= FAST_SEARCH_MAX_GENERATIONS) {
        active[w] = false;
      }
    }
  }
}

// Exhaustive check of the oracle over every pair of a middle block: compute each pattern's
// true lifetime, and confirm the oracle never discards one that reaches the target.
__global__ void k_validateOracle(uint32_t middleBlock, const uint16_t* uniqueTopStrips,
                                 const uint16_t* uniqueBottomStrips, uint32_t numUniqueTop,
                                 uint32_t numUniqueBottom, const uint32_t* __restrict__ filter,
                                 uint16_t target, uint16_t tier1Max, uint16_t tier2Max,
                                 unsigned long long* stats) {
  uint32_t topIdx = blockIdx.y;
  if (topIdx >= numUniqueTop) return;
  uint64_t base = (uint64_t)uniqueTopStrips[topIdx] | ((uint64_t)middleBlock << 16);

  unsigned long long t1 = 0, t2 = 0, kept = 0, atTarget = 0, violations = 0;

  for (uint32_t bottomIdx = blockIdx.x * blockDim.x + threadIdx.x; bottomIdx < numUniqueBottom;
       bottomIdx += gridDim.x * blockDim.x) {
    uint64_t pattern = base | ((uint64_t)uniqueBottomStrips[bottomIdx] << 48);

    // What the oracle would decide
    int decision = 0;  // 0 = kept, 1 = tier 1 discard, 2 = tier 2 discard
    uint64_t g = pattern;
    for (uint16_t gen = 1; gen <= tier2Max; gen++) {
      g = computeNextGeneration8x8(g);
      if (isCoverableBy7x7(g)) {
        if (gen <= tier1Max) decision = 1;
        else if (filter != nullptr && !bloomMaybe(filter, g)) decision = 2;
        break;
      }
    }

    // The truth
    int exact = countGenerations(pattern, computeNextGeneration8x8, CYCLE_DETECTION_FLOYD);
    if (exact >= target) {
      atTarget++;
      if (decision != 0) violations++;
    }
    if (decision == 1) t1++;
    else if (decision == 2) t2++;
    else kept++;
  }

  atomicAdd(&stats[0], t1);
  atomicAdd(&stats[1], t2);
  atomicAdd(&stats[2], kept);
  atomicAdd(&stats[3], atTarget);
  atomicAdd(&stats[4], violations);
}

// Phase 3, as it exists in production: re-run each candidate from generation 0.
__global__ void processCandidates(uint64_t* candidates, uint64_t* numCandidates, uint64_t* bestPattern,
                                  uint64_t* bestGenerations, CycleDetectionAlgorithm algorithm) {
  for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < *numCandidates;
       i += blockDim.x * gridDim.x) {
    uint64_t generations = countGenerations(candidates[i], computeNextGeneration8x8, algorithm);
    if (generations > 0) {
      uint64_t old = atomicMax((unsigned long long*)bestGenerations, (unsigned long long)generations);
      if (old < generations) *bestPattern = candidates[i];
    }
  }
}

// Pure generation throughput: no early exit, no divergence, W independent patterns per
// thread for instruction-level parallelism. This is the ceiling the hardware can sustain
// for this generation function, to compare against what the real kernel achieves.
template <int W>
__global__ void k_throughput(uint64_t seed, int iters, uint64_t* sink) {
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t s[W];
#pragma unroll
  for (int w = 0; w < W; w++) {
    // splitmix64 so the patterns are unrelated and reasonably dense
    uint64_t z = seed + (tid * W + w) * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    s[w] = z ^ (z >> 31);
  }

  for (int i = 0; i < iters; i++) {
#pragma unroll
    for (int w = 0; w < W; w++) {
      s[w] = computeNextGeneration8x8(s[w]);
    }
  }

  uint64_t acc = 0;
#pragma unroll
  for (int w = 0; w < W; w++) acc ^= s[w];
  if (acc == 0x123456789ABCDEFULL) sink[tid] = acc;  // keep it, never taken
}

// Instrumentation: where the generations are spent relative to the point where a pattern's
// live cells first fit inside a 7x7 box. Everything after that point is what a perfect 7x7
// lookup table could replace, so this bounds what the subgrid cache can save.
__global__ void k_coverage(uint32_t middleBlock, const uint16_t* uniqueTopStrips,
                           const uint16_t* uniqueBottomStrips, uint32_t numUniqueTop,
                           uint32_t numUniqueBottom, unsigned long long* stats) {
  uint32_t topIdx = blockIdx.y;
  if (topIdx >= numUniqueTop) return;
  uint64_t base = (uint64_t)uniqueTopStrips[topIdx] | ((uint64_t)middleBlock << 16);

  unsigned long long total = 0, afterCover = 0, everCover = 0, pairs = 0, coverGenSum = 0;
  unsigned long long cov6 = 0, cov12 = 0, cov20 = 0, cov34 = 0;

  for (uint32_t bottomIdx = blockIdx.x * blockDim.x + threadIdx.x; bottomIdx < numUniqueBottom;
       bottomIdx += gridDim.x * blockDim.x) {
    uint64_t g1 = base | ((uint64_t)uniqueBottomStrips[bottomIdx] << 48);
    uint16_t generations = 0;
    int firstCover = -1;

    while (generations < FAST_SEARCH_MAX_GENERATIONS) {
      uint64_t g2 = computeNextGeneration8x8(g1);
      uint64_t g3 = computeNextGeneration8x8(g2);
      uint64_t g4 = computeNextGeneration8x8(g3);
      uint64_t g5 = computeNextGeneration8x8(g4);
      uint64_t g6 = computeNextGeneration8x8(g5);
      uint64_t g7 = computeNextGeneration8x8(g6);

      if (firstCover < 0) {
        if (isCoverableBy7x7(g2)) firstCover = generations + 1;
        else if (isCoverableBy7x7(g3)) firstCover = generations + 2;
        else if (isCoverableBy7x7(g4)) firstCover = generations + 3;
        else if (isCoverableBy7x7(g5)) firstCover = generations + 4;
        else if (isCoverableBy7x7(g6)) firstCover = generations + 5;
        else if (isCoverableBy7x7(g7)) firstCover = generations + 6;
      }

      generations += 6;
      g1 = g7;
      if ((g1 == g2) || (g1 == g3) || (g1 == g4)) break;
      if (generations >= MIN_CANDIDATE_GENERATIONS) break;
    }

    pairs++;
    total += generations;
    if (firstCover >= 0) {
      everCover++;
      coverGenSum += firstCover;
      if (generations > firstCover) afterCover += generations - firstCover;
      if (firstCover <= 6) cov6++;
      if (firstCover <= 12) cov12++;
      if (firstCover <= 20) cov20++;
      if (firstCover <= 34) cov34++;
    }
  }

  atomicAdd(&stats[0], pairs);
  atomicAdd(&stats[1], total);
  atomicAdd(&stats[2], afterCover);
  atomicAdd(&stats[3], everCover);
  atomicAdd(&stats[4], coverGenSum);
  atomicAdd(&stats[5], cov6);
  atomicAdd(&stats[6], cov12);
  atomicAdd(&stats[7], cov20);
  atomicAdd(&stats[8], cov34);
}

// Instrumentation: how deep the 7x7 subgrid cache would have to go. Samples the 2^49 7x7
// patterns uniformly, expands each to all 4 positions in the 8x8 (exactly like the real
// cache build), and histograms the resulting 8x8 lifetimes. The count of states above a
// threshold sets both the table size and, via `firstCover + L <= 180`, how early a cache
// miss can be turned into a sound discard.
__global__ void k_cacheDepth(uint64_t sampleBase, uint64_t samples, unsigned long long* hist) {
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

  for (uint64_t i = tid; i < samples; i += stride) {
    // splitmix64 the index into a uniform 49-bit 7x7 pattern
    uint64_t z = (sampleBase + i) * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    uint64_t pattern7x7 = (z ^ (z >> 31)) & ((1ULL << 49) - 1);

    for (int pos = 0; pos < 4; pos++) {
      uint64_t pattern8x8 = expand7x7To8x8(pattern7x7, (pos >= 2) ? 1 : 0, (pos & 1) ? 1 : 0);
      int gens = countGenerations(pattern8x8, computeNextGeneration8x8, CYCLE_DETECTION_FLOYD);
      if (gens >= 100) {
        int bucket = gens / 10;  // 10..29 covers 100..299
        if (bucket > 29) bucket = 29;
        atomicAdd(&hist[bucket], 1ULL);
      }
    }
  }
}

// Instrumentation: histogram of how many generations each pattern survives.
__global__ void k_lifetimes(uint32_t middleBlock, const uint16_t* uniqueTopStrips,
                            const uint16_t* uniqueBottomStrips, uint32_t numUniqueTop,
                            uint32_t numUniqueBottom, unsigned long long* hist,
                            unsigned long long* totalSteps) {
  uint32_t topIdx = blockIdx.y;
  if (topIdx >= numUniqueTop) return;
  uint64_t base = (uint64_t)uniqueTopStrips[topIdx] | ((uint64_t)middleBlock << 16);
  unsigned long long mySteps = 0;

  for (uint32_t bottomIdx = blockIdx.x * blockDim.x + threadIdx.x; bottomIdx < numUniqueBottom;
       bottomIdx += gridDim.x * blockDim.x) {
    uint64_t g1 = base | ((uint64_t)uniqueBottomStrips[bottomIdx] << 48);
    uint16_t generations = 0;
    while (generations < FAST_SEARCH_MAX_GENERATIONS) {
      generations += 6;
      uint64_t g2 = computeNextGeneration8x8(g1);
      uint64_t g3 = computeNextGeneration8x8(g2);
      uint64_t g4 = computeNextGeneration8x8(g3);
      uint64_t g5 = computeNextGeneration8x8(g4);
      uint64_t g6 = computeNextGeneration8x8(g5);
      g1 = computeNextGeneration8x8(g6);
      if ((g1 == g2) || (g1 == g3) || (g1 == g4)) break;
      if (generations >= MIN_CANDIDATE_GENERATIONS) break;
    }
    mySteps += generations;
    atomicAdd(&hist[generations / 6], 1ULL);
  }
  atomicAdd(totalSteps, mySteps);
}

// ------------------------------------------------------------------ host ---
struct Buffers {
  uint16_t *d_top, *d_bottom;
  uint32_t *d_numUnique, *d_hash;
  uint64_t *d_candidates, *d_numCandidates;
};

// 7x7 oracle state, set up in main()
static uint32_t* g_dFilter = nullptr;   // nullptr if the key file was not found
static uint16_t g_target = 215;         // beat the current record of 214
static uint16_t g_tier1Max = 0;         // target - 207
static uint16_t g_tier2Max = 0;         // target - 181

// Build the blocked Bloom filter from the subgrid cache keys.
static bool loadOracle(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    printf("oracle: %s not found, skipping tier-2 variants\n", path);
    return false;
  }
  fseek(f, 0, SEEK_END);
  long bytes = ftell(f);
  fseek(f, 0, SEEK_SET);
  size_t n = bytes / sizeof(uint64_t);

  const size_t words = (1ull << BLOOM_BITS_LOG) / 32;
  std::vector<uint32_t> filter(words, 0u);
  std::vector<uint64_t> keys(1 << 20);
  size_t read = 0;
  while (read < n) {
    size_t chunk = keys.size() < (n - read) ? keys.size() : (n - read);
    if (fread(keys.data(), sizeof(uint64_t), chunk, f) != chunk) break;
    for (size_t i = 0; i < chunk; i++) bloomInsert(filter.data(), keys[i]);
    read += chunk;
  }
  fclose(f);

  size_t set = 0;
  for (uint32_t w : filter) set += __builtin_popcount(w);
  double fill = (double)set / (words * 32.0);
  printf("oracle: %zu keys -> %.0f MB filter, %.1f%% of bits set, est. false positive %.3f%%\n",
         read, (words * 4.0) / 1e6, 100.0 * fill, 100.0 * pow(fill, BLOOM_K));

  CHECK(cudaMalloc(&g_dFilter, words * sizeof(uint32_t)));
  CHECK(cudaMemcpy(g_dFilter, filter.data(), words * sizeof(uint32_t), cudaMemcpyHostToDevice));
  return true;
}

static void phase1(Buffers& b, uint32_t middleBlock, uint32_t& nTop, uint32_t& nBottom) {
  uint32_t zero = 0;
  CHECK(cudaMemcpy(b.d_numUnique, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));
  CHECK(cudaMemset(b.d_hash, 0xFF, STRIP_HASH_TABLE_SIZE * sizeof(uint32_t)));
  findUniqueTopStrips<<<STRIP_SEARCH_UNIQUE_GRID_SIZE, STRIP_SEARCH_UNIQUE_THREADS_PER_BLOCK>>>(
      middleBlock, b.d_top, b.d_numUnique, b.d_hash);
  CHECK(cudaMemcpy(&nTop, b.d_numUnique, sizeof(uint32_t), cudaMemcpyDeviceToHost));

  CHECK(cudaMemcpy(b.d_numUnique, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));
  CHECK(cudaMemset(b.d_hash, 0xFF, STRIP_HASH_TABLE_SIZE * sizeof(uint32_t)));
  findUniqueBottomStrips<<<STRIP_SEARCH_UNIQUE_GRID_SIZE, STRIP_SEARCH_UNIQUE_THREADS_PER_BLOCK>>>(
      middleBlock, b.d_bottom, b.d_numUnique, b.d_hash);
  CHECK(cudaMemcpy(&nBottom, b.d_numUnique, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  CHECK(cudaDeviceSynchronize());
}

static std::vector<uint64_t> runVariant(const std::string& name, Buffers& b, uint32_t middleBlock,
                                        uint32_t nTop, uint32_t nBottom, float& ms, int threads,
                                        int xblocks) {
  uint64_t zero64 = 0;
  CHECK(cudaMemcpy(b.d_numCandidates, &zero64, sizeof(uint64_t), cudaMemcpyHostToDevice));
  cudaEvent_t t0, t1;
  CHECK(cudaEventCreate(&t0));
  CHECK(cudaEventCreate(&t1));
  CHECK(cudaEventRecord(t0));

  dim3 grid2d((unsigned)xblocks, nTop);
  if (name == "v0") {
    v0_baseline<<<STRIP_SEARCH_COMBO_GRID_SIZE, STRIP_SEARCH_COMBO_THREADS_PER_BLOCK>>>(
        middleBlock, b.d_top, b.d_bottom, nTop, nBottom, b.d_candidates, b.d_numCandidates);
  } else if (name == "v1") {
    dim3 g((nBottom + threads - 1) / threads, nTop);
    v1_pair_per_thread<<<g, threads>>>(middleBlock, b.d_top, b.d_bottom, nTop, nBottom,
                                       b.d_candidates, b.d_numCandidates);
  } else if (name == "v2") {
    v2_repack<<<grid2d, threads>>>(middleBlock, b.d_top, b.d_bottom, nTop, nBottom, b.d_candidates,
                                   b.d_numCandidates);
  } else if (name == "v3x2") {
    v3_repack_ilp<2><<<grid2d, threads>>>(middleBlock, b.d_top, b.d_bottom, nTop, nBottom,
                                          b.d_candidates, b.d_numCandidates);
  } else if (name == "v3x4") {
    v3_repack_ilp<4><<<grid2d, threads>>>(middleBlock, b.d_top, b.d_bottom, nTop, nBottom,
                                          b.d_candidates, b.d_numCandidates);
  } else if (name == "v4") {
    // tier 1 only: no filter, no memory traffic
    v5_oracle<<<grid2d, threads>>>(middleBlock, b.d_top, b.d_bottom, nTop, nBottom,
                                   b.d_candidates, b.d_numCandidates, nullptr, g_target,
                                   g_tier1Max, g_tier1Max);
  } else if (name == "v5") {
    v5_oracle<<<grid2d, threads>>>(middleBlock, b.d_top, b.d_bottom, nTop, nBottom,
                                   b.d_candidates, b.d_numCandidates, g_dFilter, g_target,
                                   g_tier1Max, g_tier2Max);
  } else if (name == "v6") {
    v6_oracle_fused<<<grid2d, threads>>>(middleBlock, b.d_top, b.d_bottom, nTop, nBottom,
                                         b.d_candidates, b.d_numCandidates, g_dFilter, g_target,
                                         g_tier1Max, g_tier2Max);
  } else if (name == "v7") {
    v7_oracle_lean<1><<<grid2d, threads>>>(middleBlock, b.d_top, b.d_bottom, nTop, nBottom,
                                           b.d_candidates, b.d_numCandidates, g_dFilter, g_target,
                                           g_tier1Max, g_tier2Max);
  } else if (name == "v7x2") {
    v7_oracle_lean<2><<<grid2d, threads>>>(middleBlock, b.d_top, b.d_bottom, nTop, nBottom,
                                           b.d_candidates, b.d_numCandidates, g_dFilter, g_target,
                                           g_tier1Max, g_tier2Max);
  } else if (name == "v7x4") {
    v7_oracle_lean<4><<<grid2d, threads>>>(middleBlock, b.d_top, b.d_bottom, nTop, nBottom,
                                           b.d_candidates, b.d_numCandidates, g_dFilter, g_target,
                                           g_tier1Max, g_tier2Max);
  } else if (name == "v6t1") {
    v6_oracle_fused<<<grid2d, threads>>>(middleBlock, b.d_top, b.d_bottom, nTop, nBottom,
                                         b.d_candidates, b.d_numCandidates, nullptr, g_target,
                                         g_tier1Max, g_tier1Max);
  }
  CHECK(cudaEventRecord(t1));
  CHECK(cudaEventSynchronize(t1));
  CHECK(cudaGetLastError());
  CHECK(cudaEventElapsedTime(&ms, t0, t1));
  CHECK(cudaEventDestroy(t0));
  CHECK(cudaEventDestroy(t1));

  uint64_t n = 0;
  CHECK(cudaMemcpy(&n, b.d_numCandidates, sizeof(uint64_t), cudaMemcpyDeviceToHost));
  std::vector<uint64_t> out(n);
  if (n) CHECK(cudaMemcpy(out.data(), b.d_candidates, n * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  std::sort(out.begin(), out.end());
  return out;
}

int main(int argc, char** argv) {
  uint32_t centerIdx = (argc > 1) ? atoi(argv[1]) : 836;
  uint32_t middleIdx = (argc > 2) ? atoi(argv[2]) : 380;
  int nBlocks = (argc > 3) ? atoi(argv[3]) : 4;     // middle blocks to sample
  int threads = (argc > 4) ? atoi(argv[4]) : 256;   // threads/block for new variants
  int xblocks = (argc > 5) ? atoi(argv[5]) : 8;     // x-dim blocks for repacking variants
  bool doHist = (argc > 6) ? atoi(argv[6]) != 0 : false;
  std::string only = (argc > 7) ? argv[7] : "";      // run just one variant, e.g. "v2"

  if (argc > 8) g_target = (uint16_t)atoi(argv[8]);
  g_tier1Max = (g_target > 207) ? (uint16_t)(g_target - 207) : 0;
  g_tier2Max = (g_target > 181) ? (uint16_t)(g_target - 181) : 0;
  printf("oracle target=%u generations -> tier1 discards coverage at gen<=%u, "
         "tier2 at gen<=%u\n", g_target, g_tier1Max, g_tier2Max);
  loadOracle(getenv("BENCH_ORACLE_KEYS") ? getenv("BENCH_ORACLE_KEYS") : "/tmp/subgrid-keys.bin");

  initializeUnique4x4Centers();
  uint16_t center4x4 = get4x4CenterByIndex(centerIdx);

  Buffers b{};
  CHECK(cudaMalloc(&b.d_top, STRIP_SEARCH_MAX_VALID_STRIPS * sizeof(uint16_t)));
  CHECK(cudaMalloc(&b.d_bottom, STRIP_SEARCH_MAX_VALID_STRIPS * sizeof(uint16_t)));
  CHECK(cudaMalloc(&b.d_numUnique, sizeof(uint32_t)));
  CHECK(cudaMalloc(&b.d_hash, STRIP_HASH_TABLE_SIZE * sizeof(uint32_t)));
  CHECK(cudaMalloc(&b.d_candidates, (1ULL << 22) * sizeof(uint64_t)));  // 4M is plenty per block
  CHECK(cudaMalloc(&b.d_numCandidates, sizeof(uint64_t)));

  const char* variants[] = {"v0", "v1", "v2", "v3x2", "v3x4", "v4", "v5", "v6t1", "v6",
                            "v7", "v7x2", "v7x4"};
  double totals[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  double phase1Total = 0, phase3Total = 0;

  printf("centerIdx=%u middleIdx=%u blocks=%d threads=%d xblocks=%d\n", centerIdx, middleIdx,
         nBlocks, threads, xblocks);

  uint32_t blockStart = middleIdx * STRIP_SEARCH_MIDDLE_BLOCKS_PER_REPORT;
  for (int i = 0; i < nBlocks; i++) {
    uint32_t blockOffset = blockStart + i;
    uint16_t leftEar = blockOffset / CENTER_4X4_TOTAL_EAR_VALUES;
    uint16_t rightEar = blockOffset % CENTER_4X4_TOTAL_EAR_VALUES;
    uint32_t middleBlock = reconstructMiddleBlock(center4x4, (uint8_t)leftEar, (uint8_t)rightEar);

    uint32_t nTop = 0, nBottom = 0;
    cudaEvent_t p0, p1;
    CHECK(cudaEventCreate(&p0));
    CHECK(cudaEventCreate(&p1));
    CHECK(cudaEventRecord(p0));
    phase1(b, middleBlock, nTop, nBottom);
    CHECK(cudaEventRecord(p1));
    CHECK(cudaEventSynchronize(p1));
    float p1ms = 0;
    CHECK(cudaEventElapsedTime(&p1ms, p0, p1));
    phase1Total += p1ms;

    printf("\nmiddleBlock=0x%08X  uniqueTop=%u uniqueBottom=%u pairs=%.1fM  phase1=%.2fms\n",
           middleBlock, nTop, nBottom, (double)nTop * nBottom / 1e6, p1ms);

    std::vector<uint64_t> ref;
    for (int v = 0; v < 12; v++) {
      if (!only.empty() && only != variants[v]) continue;
      if (v >= 5 && g_tier1Max == 0) continue;               // target too low for tier 1
      if (v != 5 && v != 7 && v >= 5 && g_dFilter == nullptr) continue;
      float ms = 0;
      std::vector<uint64_t> got = runVariant(variants[v], b, middleBlock, nTop, nBottom, ms, threads, xblocks);
      totals[v] += ms;
      bool isOracle = (v >= 5);
      if (ref.empty() && !isOracle) ref = got;
      const char* ok = isOracle ? "(target-filtered)" : (got == ref ? "ok" : "MISMATCH");
      printf("  %-5s %8.2f ms  candidates=%zu  %s\n", variants[v], ms, got.size(), ok);
    }

    if (doHist && g_tier2Max > 0) {
      // Exhaustive: every pair's true lifetime vs the oracle's decision
      unsigned long long* d_v;
      CHECK(cudaMalloc(&d_v, 8 * sizeof(unsigned long long)));
      CHECK(cudaMemset(d_v, 0, 8 * sizeof(unsigned long long)));
      dim3 gv((nBottom + 255) / 256, nTop);
      cudaEvent_t s0, s1;
      CHECK(cudaEventCreate(&s0));
      CHECK(cudaEventCreate(&s1));
      CHECK(cudaEventRecord(s0));
      k_validateOracle<<<gv, 256>>>(middleBlock, b.d_top, b.d_bottom, nTop, nBottom, g_dFilter,
                                    g_target, g_tier1Max, g_tier2Max, d_v);
      CHECK(cudaEventRecord(s1));
      CHECK(cudaEventSynchronize(s1));
      CHECK(cudaGetLastError());
      float vms = 0;
      CHECK(cudaEventElapsedTime(&vms, s0, s1));
      unsigned long long v[8];
      CHECK(cudaMemcpy(v, d_v, sizeof(v), cudaMemcpyDeviceToHost));
      uint64_t pairs = (uint64_t)nTop * nBottom;
      printf("  oracle check (%.0f ms, exact lifetime for all %.1fM pairs):\n", vms, pairs / 1e6);
      printf("    tier1 discards %.1f%%   tier2 discards %.1f%%   kept %.1f%%\n",
             100.0 * v[0] / pairs, 100.0 * v[1] / pairs, 100.0 * v[2] / pairs);
      printf("    patterns reaching >=%u generations: %llu, of which wrongly discarded: %llu %s\n",
             g_target, v[3], v[4], v[4] == 0 ? "<- sound" : "<- UNSOUND");
      CHECK(cudaFree(d_v));
    }

    {  // phase 3 cost, using the candidate set left in the buffer by the last variant
      uint64_t *d_best, *d_bestGen;
      CHECK(cudaMalloc(&d_best, sizeof(uint64_t)));
      CHECK(cudaMalloc(&d_bestGen, sizeof(uint64_t)));
      CHECK(cudaMemset(d_best, 0, sizeof(uint64_t)));
      CHECK(cudaMemset(d_bestGen, 0, sizeof(uint64_t)));
      cudaEvent_t c0, c1;
      CHECK(cudaEventCreate(&c0)); CHECK(cudaEventCreate(&c1));
      CHECK(cudaEventRecord(c0));
      processCandidates<<<STRIP_SEARCH_COMBO_GRID_SIZE, STRIP_SEARCH_COMBO_THREADS_PER_BLOCK>>>(
          b.d_candidates, b.d_numCandidates, d_best, d_bestGen, CYCLE_DETECTION_FLOYD);
      CHECK(cudaEventRecord(c1));
      CHECK(cudaEventSynchronize(c1));
      float cms = 0; CHECK(cudaEventElapsedTime(&cms, c0, c1));
      uint64_t bg = 0; CHECK(cudaMemcpy(&bg, d_bestGen, sizeof(uint64_t), cudaMemcpyDeviceToHost));
      printf("  phase3 %8.2f ms  bestGenerations=%llu\n", cms, (unsigned long long)bg);
      phase3Total += cms;
      CHECK(cudaFree(d_best)); CHECK(cudaFree(d_bestGen));
    }

    if (doHist) {
      unsigned long long* d_stats;
      CHECK(cudaMalloc(&d_stats, 16 * sizeof(unsigned long long)));
      CHECK(cudaMemset(d_stats, 0, 16 * sizeof(unsigned long long)));
      dim3 gc((nBottom + 255) / 256, nTop);
      k_coverage<<<gc, 256>>>(middleBlock, b.d_top, b.d_bottom, nTop, nBottom, d_stats);
      CHECK(cudaDeviceSynchronize());
      CHECK(cudaGetLastError());
      unsigned long long st[16];
      CHECK(cudaMemcpy(st, d_stats, sizeof(st), cudaMemcpyDeviceToHost));
      printf("  7x7 coverage: %.1f%% of pairs ever fit a 7x7 box (mean at gen %.1f); "
             "%.1f%% of all generation-steps happen after that point\n",
             100.0 * st[3] / st[0], (double)st[4] / (st[3] ? st[3] : 1),
             100.0 * st[2] / st[1]);
      printf("    first coverable at gen <=6: %.1f%%  <=12: %.1f%%  <=20: %.1f%%  <=34: %.1f%%  never: %.1f%%\n",
             100.0 * st[5] / st[0], 100.0 * st[6] / st[0], 100.0 * st[7] / st[0],
             100.0 * st[8] / st[0], 100.0 * (st[0] - st[3]) / st[0]);
      CHECK(cudaFree(d_stats));
    }

    if (doHist) {
      unsigned long long *d_hist, *d_steps;
      CHECK(cudaMalloc(&d_hist, 64 * sizeof(unsigned long long)));
      CHECK(cudaMalloc(&d_steps, sizeof(unsigned long long)));
      CHECK(cudaMemset(d_hist, 0, 64 * sizeof(unsigned long long)));
      CHECK(cudaMemset(d_steps, 0, sizeof(unsigned long long)));
      dim3 g((nBottom + 255) / 256, nTop);
      k_lifetimes<<<g, 256>>>(middleBlock, b.d_top, b.d_bottom, nTop, nBottom, d_hist, d_steps);
      CHECK(cudaDeviceSynchronize());
      unsigned long long hist[64], steps;
      CHECK(cudaMemcpy(hist, d_hist, sizeof(hist), cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(&steps, d_steps, sizeof(steps), cudaMemcpyDeviceToHost));
      uint64_t pairs = (uint64_t)nTop * nBottom;
      printf("  lifetime: mean=%.1f gens; buckets(gens:pct):", (double)steps / pairs);
      for (int k = 0; k < 64; k++)
        if (hist[k] * 1000ull / pairs > 0) printf(" %d:%.1f%%", k * 6, 100.0 * hist[k] / pairs);
      printf("\n");
      CHECK(cudaFree(d_hist));
      CHECK(cudaFree(d_steps));
    }
  }

  // ---- pure generation throughput ceiling ----
  {
    uint64_t* d_sink = nullptr;
    CHECK(cudaMalloc(&d_sink, 256 * 1024 * sizeof(uint64_t)));
    const int blocks = 8192, tpb = 256, iters = 2000;
    printf("\n=== pure generation throughput (no early exit, %d x %d threads, %d iters) ===\n",
           blocks, tpb, iters);
    double best = 0;
    for (int variant = 0; variant < 4; variant++) {
      cudaEvent_t a, b;
      CHECK(cudaEventCreate(&a));
      CHECK(cudaEventCreate(&b));
      CHECK(cudaEventRecord(a));
      int w = 0;
      switch (variant) {
        case 0: w = 1; k_throughput<1><<<blocks, tpb>>>(1, iters, d_sink); break;
        case 1: w = 2; k_throughput<2><<<blocks, tpb>>>(1, iters, d_sink); break;
        case 2: w = 4; k_throughput<4><<<blocks, tpb>>>(1, iters, d_sink); break;
        case 3: w = 8; k_throughput<8><<<blocks, tpb>>>(1, iters, d_sink); break;
      }
      CHECK(cudaEventRecord(b));
      CHECK(cudaEventSynchronize(b));
      CHECK(cudaGetLastError());
      float ms = 0;
      CHECK(cudaEventElapsedTime(&ms, a, b));
      double gensPerSec = (double)blocks * tpb * w * iters / (ms / 1000.0);
      if (gensPerSec > best) best = gensPerSec;
      printf("  ILP=%d  %7.1f ms  %.2f G pattern-generations/sec\n", w, ms, gensPerSec / 1e9);
      CHECK(cudaEventDestroy(a));
      CHECK(cudaEventDestroy(b));
    }
    printf("  ceiling: %.2f G pattern-generations/sec\n", best / 1e9);
    CHECK(cudaFree(d_sink));
  }

  // ---- how big a 7x7 cache would have to be at various thresholds ----
  {
    const uint64_t samples = 1ULL << 30;  // 1.07e9 7x7 patterns x 4 positions
    unsigned long long* d_hist;
    CHECK(cudaMalloc(&d_hist, 32 * sizeof(unsigned long long)));
    CHECK(cudaMemset(d_hist, 0, 32 * sizeof(unsigned long long)));
    cudaEvent_t a, bb;
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&bb));
    CHECK(cudaEventRecord(a));
    k_cacheDepth<<<8192, 256>>>(0x1234, samples, d_hist);
    CHECK(cudaEventRecord(bb));
    CHECK(cudaEventSynchronize(bb));
    CHECK(cudaGetLastError());
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, a, bb));
    unsigned long long hist[32];
    CHECK(cudaMemcpy(hist, d_hist, sizeof(hist), cudaMemcpyDeviceToHost));

    const double evaluated = (double)samples * 4.0;
    const double population = 4.0 * 562949953421312.0;  // 4 positions x 2^49
    printf("\n=== 7x7 cache depth (sampled %.2e of %.2e states, %.1f ms) ===\n",
           evaluated, population, ms);
    printf("  threshold   sampled   estimated entries   table @16B/entry\n");
    for (int t = 18; t >= 13; t--) {  // thresholds 180 down to 130
      unsigned long long atLeast = 0;
      for (int k = t; k < 30; k++) atLeast += hist[k];
      double est = population * (double)atLeast / evaluated;
      printf("  >=%3d gens %9llu   %15.3e   %8.2f GB\n", t * 10, atLeast, est, est * 16 / 1e9);
    }
    CHECK(cudaFree(d_hist));
  }

  printf("\n=== totals over %d middle blocks (phase1 %.1f ms, phase3 %.1f ms) ===\n", nBlocks, phase1Total, phase3Total);
  for (int v = 0; v < 12; v++)
    if (totals[v] > 0)
      printf("  %-5s %9.1f ms   speedup vs v0: %.2fx\n", variants[v], totals[v], totals[0] / totals[v]);
  return 0;
}
