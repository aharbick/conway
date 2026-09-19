#ifndef _SUBGRID_BLOOM_H_
#define _SUBGRID_BLOOM_H_

#include <cstdint>

#include "cuda_utils.h"

// Blocked Bloom filter over the 7x7 subgrid cache keys.
//
// The cache holds every 7x7-coverable 8x8 pattern whose terminating lifetime is at least
// SUBGRID_MIN_GENERATIONS, keyed by the 8x8 pattern. Two facts about it drive the strip
// search's early-outs, for a target of T generations:
//
//   * No 7x7-coverable state lives longer than the cache's maximum (206 generations), since
//     the cache is exhaustive above its threshold. A pattern that first fits inside a 7x7
//     box at generation g is therefore capped at g + max, so it can be discarded outright
//     when g + max < T. No lookup needed.
//
//   * A state absent from the cache lives fewer than min more generations, so a pattern
//     covered at generation g can be discarded when g + min < T unless the state is in the
//     cache.
//
// Only the second test needs a lookup, and only membership - never the stored count. A
// blocked Bloom filter is therefore enough, and its false positives are harmless: the
// pattern simply keeps simulating as it would have without the filter. Every probe touches
// exactly one 64-byte line, and at 32MB the filter stays close enough to cache that the
// lookups do not become the bottleneck.
//
// The thresholds live in the filter file's header rather than in code, so the artifact and
// the bounds derived from it can never drift apart.

#define SUBGRID_BLOOM_MAGIC "GOLBLOOM"
#define SUBGRID_BLOOM_VERSION 1u
#define SUBGRID_BLOOM_BITS_LOG 28u                                  // 2^28 bits = 32MB
#define SUBGRID_BLOOM_BLOCK_BITS 512u                               // one 64-byte line
#define SUBGRID_BLOOM_K 6u
#define SUBGRID_BLOOM_WORDS ((1ull << SUBGRID_BLOOM_BITS_LOG) / 32)
#define SUBGRID_BLOOM_NUM_BLOCKS (1u << (SUBGRID_BLOOM_BITS_LOG - 9))
#define SUBGRID_BLOOM_BYTES (SUBGRID_BLOOM_WORDS * sizeof(uint32_t))

struct SubgridBloomHeader {
  char magic[8];             // SUBGRID_BLOOM_MAGIC, not null terminated
  uint32_t version;
  uint32_t bitsLog;
  uint32_t blockBits;
  uint32_t k;
  uint64_t numKeys;          // keys inserted
  uint32_t minGenerations;   // the cache's threshold: absent => lives fewer than this
  uint32_t maxGenerations;   // the cache's maximum: no coverable state exceeds this
  uint64_t bitsSet;          // for a fill/false-positive sanity check on load
  uint64_t reserved;
};

__host__ __device__ static inline uint64_t subgridBloomHash(uint64_t key) {
  key = (key ^ (key >> 30)) * 0xBF58476D1CE4E5B9ULL;
  key = (key ^ (key >> 27)) * 0x94D049BB133111EBULL;
  return key ^ (key >> 31);
}

// The K bit positions for a key, all inside one 512-bit block
__host__ __device__ static inline uint32_t subgridBloomSlots(uint64_t key, uint32_t bits[SUBGRID_BLOOM_K]) {
  uint64_t h = subgridBloomHash(key);
  uint32_t a = (uint32_t)h;
  uint32_t b = (uint32_t)(h >> 20) | 1u;  // odd, so the stride cannot degenerate
  for (uint32_t i = 0; i < SUBGRID_BLOOM_K; i++) {
    bits[i] = (a + i * b) & (SUBGRID_BLOOM_BLOCK_BITS - 1);
  }
  return (uint32_t)(h >> 40) & (SUBGRID_BLOOM_NUM_BLOCKS - 1);
}

__host__ __device__ static inline void subgridBloomInsert(uint32_t* filter, uint64_t key) {
  uint32_t bits[SUBGRID_BLOOM_K];
  uint32_t blk = subgridBloomSlots(key, bits);
  uint32_t* words = filter + (size_t)blk * (SUBGRID_BLOOM_BLOCK_BITS / 32);
  for (uint32_t i = 0; i < SUBGRID_BLOOM_K; i++) {
    words[bits[i] >> 5] |= 1u << (bits[i] & 31);
  }
}

// False positives are possible, false negatives are not
__host__ __device__ static inline bool subgridBloomMaybe(const uint32_t* filter, uint64_t key) {
  uint32_t bits[SUBGRID_BLOOM_K];
  uint32_t blk = subgridBloomSlots(key, bits);
  const uint32_t* words = filter + (size_t)blk * (SUBGRID_BLOOM_BLOCK_BITS / 32);
  for (uint32_t i = 0; i < SUBGRID_BLOOM_K; i++) {
    if (!((words[bits[i] >> 5] >> (bits[i] & 31)) & 1u)) return false;
  }
  return true;
}

// Highest generation at which a coverable state can be discarded with no lookup, and the
// highest at which a filter miss is enough. Zero means the test cannot apply at this target.
static inline uint32_t subgridBloomTier1Max(const SubgridBloomHeader& h, uint32_t target) {
  return (target > h.maxGenerations + 1) ? (target - h.maxGenerations - 1) : 0u;
}

static inline uint32_t subgridBloomTier2Max(const SubgridBloomHeader& h, uint32_t target) {
  return (target > h.minGenerations + 1) ? (target - h.minGenerations - 1) : 0u;
}

#endif
