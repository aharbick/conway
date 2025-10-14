#include "subgrid_cache.h"

#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>

#include "logging.h"
#include "utils.h"

using json = nlohmann::json;

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
