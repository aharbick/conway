#ifndef _SUBGRID_CACHE_H_
#define _SUBGRID_CACHE_H_

#include <cstdint>
#include <string>
#include <unordered_map>

#include "cli_parser.h"
#include "constants.h"

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

  SubgridCache() = default;

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
