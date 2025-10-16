#include "subgrid_cache.h"

#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>

#include "logging.h"
#include "utils.h"

using json = nlohmann::json;

SubgridCache::~SubgridCache() {
#ifdef __NVCC__
  if (gpuKeys_) {
    cudaFree(gpuKeys_);
  }
  if (gpuValues_) {
    cudaFree(gpuValues_);
  }
  if (gpuHashTable_) {
    cudaFree(gpuHashTable_);
  }
#endif
}

void SubgridCache::load(const std::string& filePath) {
  if (loaded_) {
    return;
  }

  std::ifstream inFile(filePath);
  if (!inFile.is_open()) {
    std::cerr << "[ERROR] Failed to open subgrid cache file: " << filePath << "\n";
    return;
  }

  std::string line;
  while (std::getline(inFile, line)) {
    if (line.empty()) continue;

    try {
      json entry = json::parse(line);
      uint64_t pattern = entry["pattern"];
      uint16_t generations = entry["generations"];
      cache_[pattern] = generations;
    } catch (const std::exception& e) {
      // Skip malformed lines silently
    }
  }

  inFile.close();
  loaded_ = true;

  Logger::out() << "Using subgrid cache with " << formatWithCommas(cache_.size()) << " entries.\n";
}

SubgridHashTable* SubgridCache::getGpuHashTable() const {
  // Return memoized table if already built
  if (gpuHashTable_ != nullptr) {
    return gpuHashTable_;
  }

  if (!loaded_ || cache_.empty()) {
    return nullptr;
  }

#ifdef __NVCC__
  // Allocate host-side hash table
  uint64_t* h_keys = new uint64_t[CACHE_TABLE_SIZE]();  // Zero-initialized
  uint16_t* h_values = new uint16_t[CACHE_TABLE_SIZE]();

  // Build hash table using same logic as GPU lookup
  for (const auto& entry : cache_) {
    uint64_t pattern = entry.first;
    uint16_t generations = entry.second;

    uint64_t hash = hash_cityhash_33(pattern);
    uint64_t index = hash & (CACHE_TABLE_SIZE - 1);

    // Linear probing to find empty slot
    for (int probe = 0; probe < CACHE_MAX_PROBE_LENGTH; probe++) {
      if (h_keys[index] == CACHE_EMPTY_VALUE) {
        h_keys[index] = pattern;
        h_values[index] = generations;
        break;
      }
      index = (index + 1) & (CACHE_TABLE_SIZE - 1);
    }
  }

  // Allocate GPU memory for arrays
  uint64_t* d_keys = nullptr;
  uint16_t* d_values = nullptr;
  cudaCheckError(cudaMalloc(&d_keys, CACHE_TABLE_SIZE * sizeof(uint64_t)));
  cudaCheckError(cudaMalloc(&d_values, CACHE_TABLE_SIZE * sizeof(uint16_t)));

  // Copy arrays to GPU
  cudaCheckError(cudaMemcpy(d_keys, h_keys, CACHE_TABLE_SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(d_values, h_values, CACHE_TABLE_SIZE * sizeof(uint16_t), cudaMemcpyHostToDevice));

  // Clean up host memory
  delete[] h_keys;
  delete[] h_values;

  // Save pointers for cleanup
  gpuKeys_ = d_keys;
  gpuValues_ = d_values;

  // Allocate the SubgridHashTable struct itself on GPU
  SubgridHashTable* d_table = nullptr;
  cudaCheckError(cudaMalloc(&d_table, sizeof(SubgridHashTable)));

  // Create struct on host with GPU pointers, then copy to GPU
  SubgridHashTable h_table{d_keys, d_values};
  cudaCheckError(cudaMemcpy(d_table, &h_table, sizeof(SubgridHashTable), cudaMemcpyHostToDevice));

  gpuHashTable_ = d_table;

  Logger::out() << "Built GPU hash table with " << formatWithCommas(cache_.size())
                << " entries in " << formatWithCommas(CACHE_TABLE_SIZE) << " slots\n";

  return gpuHashTable_;
#else
  return nullptr;  // Non-CUDA builds
#endif
}
