#ifndef _SUBGRID_BLOOM_H_
#define _SUBGRID_BLOOM_H_

#include <cstdint>

#include "cuda_utils.h"

// Blocked Bloom filters over the 7x7 subgrid cache, tiered by lifetime.
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
//   * Otherwise the pattern needs a state that lives at least T - g more generations, so a
//     state absent from the cache means it can be discarded.
//
// That second test is where tiering pays. A pattern covered at generation 9 hunting for 215
// needs a state living 206 more generations, and there are sixteen of those - asking a
// filter built over all 19.7 million states at least 180 is asking a far broader question
// than necessary, and a far bigger filter. So the artifact carries several filters, each
// over the states at or above one threshold, and a lookup uses the most selective one whose
// threshold still proves what it needs:
//
//   needs >= 206   ->  the >=204 filter, 368 keys, a few KB
//   needs >= 200   ->  the >=200 filter, 26K keys, 64KB
//   needs >= 196   ->  the >=196 filter, 116K keys, 256KB
//   needs >= 181   ->  the >=180 filter, 19.7M keys, 32MB
//
// The first few sit in cache, and they are the ones almost every probe uses, because most
// patterns fit a 7x7 box within a dozen generations.
//
// Only membership is ever needed, never the stored count, so Bloom filters suffice and
// their false positives are harmless: the pattern simply keeps simulating as it would have
// without the filter. Every probe touches exactly one 64-byte line.
//
// The thresholds live in the file header rather than in code, so the artifact and the
// bounds derived from it cannot drift apart.

#define SUBGRID_BLOOM_MAGIC "GOLBLOOM"
#define SUBGRID_BLOOM_VERSION 2u
#define SUBGRID_BLOOM_BLOCK_BITS 512u          // one 64-byte line per probe
#define SUBGRID_BLOOM_WORDS_PER_BLOCK (SUBGRID_BLOOM_BLOCK_BITS / 32)
#define SUBGRID_BLOOM_K 6u
#define SUBGRID_BLOOM_MAX_TIERS 8u
#define SUBGRID_BLOOM_MAX_GEN 128u             // generations the tier table covers
#define SUBGRID_BLOOM_NO_TIER 0xFFu

// Bits per key. The small tiers can afford to be generous - they are only kilobytes - while
// the widest tier is the one that has to stay a sane size.
#define SUBGRID_BLOOM_BITS_PER_KEY 16u
#define SUBGRID_BLOOM_WIDE_BITS_PER_KEY 13u
#define SUBGRID_BLOOM_MIN_BLOCKS 8u

struct SubgridBloomTier {
  uint32_t threshold;    // this filter holds every cache state with lifetime >= threshold
  uint32_t blocks;       // power of two
  uint64_t wordOffset;   // into the filter words following the header
  uint64_t keys;
};

struct SubgridBloomHeader {
  char magic[8];             // SUBGRID_BLOOM_MAGIC, not null terminated
  uint32_t version;
  uint32_t blockBits;
  uint32_t k;
  uint32_t numTiers;
  uint64_t numKeys;          // keys in the widest tier
  uint32_t minGenerations;   // the cache's threshold: absent => lives fewer than this
  uint32_t maxGenerations;   // the cache's maximum: no coverable state exceeds this
  uint64_t totalWords;       // filter words following the header
  uint64_t reserved;
  SubgridBloomTier tiers[SUBGRID_BLOOM_MAX_TIERS];  // ascending by threshold
};

__host__ __device__ static inline uint64_t subgridBloomHash(uint64_t key) {
  key = (key ^ (key >> 30)) * 0xBF58476D1CE4E5B9ULL;
  key = (key ^ (key >> 27)) * 0x94D049BB133111EBULL;
  return key ^ (key >> 31);
}

// The K bit positions for a key, all inside one 512-bit block of the given filter
__host__ __device__ static inline uint32_t subgridBloomSlots(uint64_t key, uint32_t blockMask,
                                                             uint32_t bits[SUBGRID_BLOOM_K]) {
  uint64_t h = subgridBloomHash(key);
  uint32_t a = (uint32_t)h;
  uint32_t b = (uint32_t)(h >> 20) | 1u;  // odd, so the stride cannot degenerate
  for (uint32_t i = 0; i < SUBGRID_BLOOM_K; i++) {
    bits[i] = (a + i * b) & (SUBGRID_BLOOM_BLOCK_BITS - 1);
  }
  return (uint32_t)(h >> 40) & blockMask;
}

__host__ __device__ static inline void subgridBloomInsert(uint32_t* filter, uint32_t blockMask,
                                                          uint64_t key) {
  uint32_t bits[SUBGRID_BLOOM_K];
  uint32_t blk = subgridBloomSlots(key, blockMask, bits);
  uint32_t* words = filter + (size_t)blk * SUBGRID_BLOOM_WORDS_PER_BLOCK;
  for (uint32_t i = 0; i < SUBGRID_BLOOM_K; i++) {
    words[bits[i] >> 5] |= 1u << (bits[i] & 31);
  }
}

// False positives are possible, false negatives are not
__host__ __device__ static inline bool subgridBloomMaybe(const uint32_t* filter, uint32_t blockMask,
                                                         uint64_t key) {
  uint32_t bits[SUBGRID_BLOOM_K];
  uint32_t blk = subgridBloomSlots(key, blockMask, bits);
  const uint32_t* words = filter + (size_t)blk * SUBGRID_BLOOM_WORDS_PER_BLOCK;
  for (uint32_t i = 0; i < SUBGRID_BLOOM_K; i++) {
    if (!((words[bits[i] >> 5] >> (bits[i] & 31)) & 1u)) return false;
  }
  return true;
}

// Everything the kernel needs to run the oracle, resolved once per target on the host.
//
// Tiers are held as plain arrays indexed at compile time, never by a per-lane value: an
// index that varies across a warp turns a parameter-bank read into a serialized one, and
// this sits on the hot path.
struct OracleTierTable {
  const uint32_t* base;                            // all tiers, concatenated, on the device
  uint32_t threshold[SUBGRID_BLOOM_MAX_TIERS];     // ascending
  uint32_t wordOffset[SUBGRID_BLOOM_MAX_TIERS];
  uint32_t blockMask[SUBGRID_BLOOM_MAX_TIERS];
  uint8_t tierForGen[SUBGRID_BLOOM_MAX_GEN];       // resolved per generation on the host
  uint32_t numTiers;
  uint16_t target;
  uint16_t tier1Max;                               // discard with no lookup at or below this
  uint16_t tier2Max;                               // highest generation any tier can decide
};

// Can a pattern covered at `generation` still reach the target?
//
// False means provably not, so it can be abandoned. True means either a filter said "maybe"
// or no tier could decide - both mean keep simulating, so a false positive costs nothing but
// work the search would have done anyway.
//
// The tier per generation is resolved on the host, because the choice depends only on the
// target. Selecting it here instead, by predication across every tier, measured slower: the
// extra work lands on every probe, while the lookup is a parameter-bank read.
__host__ __device__ static inline bool subgridBloomMightReach(const OracleTierTable& tiers,
                                                              uint64_t state, uint32_t generation) {
  uint8_t tier = tiers.tierForGen[generation];
  if (tier == SUBGRID_BLOOM_NO_TIER) return true;
  return subgridBloomMaybe(tiers.base + tiers.wordOffset[tier], tiers.blockMask[tier], state);
}

// Highest generation at which a coverable state can be discarded with no lookup: beyond
// g + maxGenerations the target is simply unreachable. Zero means the test cannot apply.
static inline uint32_t subgridBloomTier1Max(const SubgridBloomHeader& h, uint32_t target) {
  return (target > h.maxGenerations + 1) ? (target - h.maxGenerations - 1) : 0u;
}

// Highest generation at which any filter can decide: below the cache's own threshold
// absence proves nothing, because such states were never enumerated.
static inline uint32_t subgridBloomTier2Max(const SubgridBloomHeader& h, uint32_t target) {
  return (target > h.minGenerations + 1) ? (target - h.minGenerations - 1) : 0u;
}

// Resolve the filter layout and the two generation bounds for a target.
static inline void subgridBloomBuildTierTable(const SubgridBloomHeader& h, uint32_t target,
                                              OracleTierTable* table) {
  table->target = (uint16_t)target;
  table->tier1Max = (uint16_t)subgridBloomTier1Max(h, target);
  table->tier2Max = (uint16_t)subgridBloomTier2Max(h, target);
  table->numTiers = (h.numTiers < SUBGRID_BLOOM_MAX_TIERS) ? h.numTiers : SUBGRID_BLOOM_MAX_TIERS;

  for (uint32_t i = 0; i < SUBGRID_BLOOM_MAX_TIERS; i++) {
    bool present = i < table->numTiers;
    table->threshold[i] = present ? h.tiers[i].threshold : 0xFFFFFFFFu;  // never selected
    table->wordOffset[i] = present ? (uint32_t)h.tiers[i].wordOffset : 0u;
    table->blockMask[i] = present ? (h.tiers[i].blocks - 1u) : 0u;
  }

  // A pattern covered at generation g needs a state living target - g more generations, so
  // any tier at or below that proves it falls short; the highest such one is the smallest
  // filter. Ascending thresholds mean the last match wins.
  for (uint32_t g = 0; g < SUBGRID_BLOOM_MAX_GEN; g++) {
    table->tierForGen[g] = SUBGRID_BLOOM_NO_TIER;
    if (g == 0 || g > table->tier2Max) continue;
    uint32_t needed = target - g;
    for (uint32_t i = 0; i < table->numTiers; i++) {
      if (table->threshold[i] <= needed) table->tierForGen[g] = (uint8_t)i;
    }
  }
}

// Which tier a pattern covered at `generation` would probe, or -1 for none. Mirrors the
// selection in subgridBloomMightReach; for tests and diagnostics.
static inline int subgridBloomTierForGeneration(const OracleTierTable& tiers, uint32_t generation) {
  if (generation >= SUBGRID_BLOOM_MAX_GEN) return -1;
  uint8_t tier = tiers.tierForGen[generation];
  return (tier == SUBGRID_BLOOM_NO_TIER) ? -1 : (int)tier;
}

#endif
