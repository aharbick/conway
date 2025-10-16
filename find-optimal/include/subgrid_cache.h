#ifndef _SUBGRID_CACHE_H_
#define _SUBGRID_CACHE_H_

#include <cstdint>
#include <string>
#include <unordered_map>

#include "cli_parser.h"
#include "constants.h"
#include "cuda_utils.h"

// Hash table constants for 7x7 subgrid cache
#define CACHE_TABLE_SIZE_BITS 26  // 2^26 = 67M slots (~33% load factor for 19.6M entries)
#define CACHE_TABLE_SIZE (1ULL << CACHE_TABLE_SIZE_BITS)
#define CACHE_EMPTY_VALUE 0  // 0 means no entry (since valid patterns are non-zero)
#define CACHE_MAX_PROBE_LENGTH 256  // Safety limit (max observed was 170 in testing)

// CityHash * >>33 - optimized 3-operation hash function
__host__ __device__ static inline uint64_t hash_cityhash_33(uint64_t key) {
  key *= 0x9ddfea08eb382d69ULL;  // CityHash constant
  key ^= key >> 33;                // Mix bits
  return key;
}

// GPU hash table structure for 7x7 subgrid cache
// Uses linear probing with CityHash * >>33
struct SubgridHashTable {
  uint64_t* keys;     // Pattern values (0 = empty slot)
  uint16_t* values;   // Generation counts

  __host__ __device__ inline uint16_t get(uint64_t pattern) const {
    if (pattern == 0) return 0;  // Empty patterns don't exist in cache

    uint64_t hash = hash_cityhash_33(pattern);
    uint64_t index = hash & (CACHE_TABLE_SIZE - 1);  // Modulo by table size (power of 2)

    // Linear probing with safety limit
    for (int probe = 0; probe < CACHE_MAX_PROBE_LENGTH; probe++) {
      uint64_t key = keys[index];
      if (key == CACHE_EMPTY_VALUE) {
        return 0;  // Not found
      }
      if (key == pattern) {
        return values[index];  // Found!
      }
      index = (index + 1) & (CACHE_TABLE_SIZE - 1);  // Next slot (wrap around)
    }

    // Safety: if we've probed too many times, treat as not found
    return 0;
  }
};

// Entry in the subgrid cache file
struct SubgridCacheEntry {
  uint64_t pattern;      // The 8x8 pattern (includes position information)
  uint16_t generations;  // Number of generations (180-65535)
};

// Singleton class for managing the subgrid cache
class SubgridCache {
 private:
  std::unordered_map<uint64_t, uint16_t> cache_;
  bool loaded_ = false;
  mutable SubgridHashTable* gpuHashTable_ = nullptr;  // GPU hash table struct (on GPU)
  mutable uint64_t* gpuKeys_ = nullptr;               // Keys array (on GPU)
  mutable uint16_t* gpuValues_ = nullptr;             // Values array (on GPU)

  SubgridCache() = default;

  ~SubgridCache();

 public:
  // Singleton instance getter
  static SubgridCache& getInstance() {
    static SubgridCache instance;
    return instance;
  }

  // Delete copy/move constructors
  SubgridCache(const SubgridCache&) = delete;
  SubgridCache& operator=(const SubgridCache&) = delete;

  // Load cache from file
  void load(const std::string& filePath);

  // Check if cache is loaded
  bool isLoaded() const { return loaded_; }

  // Look up a pattern in the cache
  // Returns 0 if not found, otherwise returns generation count
  uint16_t lookup(uint64_t pattern) const {
    auto it = cache_.find(pattern);
    return (it != cache_.end()) ? it->second : 0;
  }

  // Get cache size
  size_t size() const { return cache_.size(); }

  // Build and return GPU hash table from loaded cache
  // Returns nullptr if cache is not loaded
  // Result is memoized - subsequent calls return the same pointer
  // Hash table is owned by this singleton and freed in destructor
  SubgridHashTable* getGpuHashTable() const;
};

// Compute the 7x7 subgrid cache and save to disk
int computeSubgridCache(ProgramArgs* cli);

// Forward declaration of CUDA kernel (defined in gol_cuda.cu)
#ifdef __NVCC__
__global__ void findSubgridCandidates(uint64_t rangeStart, uint64_t rangeEnd,
                                      SubgridCacheEntry* candidates, uint64_t* numCandidates,
                                      CycleDetectionAlgorithm algorithm, int minGenerations);
#endif

#endif
