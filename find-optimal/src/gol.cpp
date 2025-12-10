#include "gol.h"

#include <algorithm>
#include <iomanip>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "logging.h"

// Global variables updated across threads
std::mutex gMutex;
int gBestGenerations = 0;

__host__ bool updateBestGenerations(int generations) {
  std::lock_guard<std::mutex> lock(gMutex);

  bool isNewGlobalBest = (gBestGenerations < generations);
  if (isNewGlobalBest) {
    gBestGenerations = generations;
  }

  return isNewGlobalBest;
}

__host__ void *search(void *args) {
  ProgramArgs *cli = static_cast<ProgramArgs *>(args);
  gol::SearchMemory mem(FRAME_SEARCH_MAX_CANDIDATES);

  if (cli->gridSize == GRID_SIZE_7X7) {
    // 7x7 exhaustive search: iterate through specified pattern range
    const uint64_t rangeStart = cli->grid7x7StartPattern;
    const uint64_t rangeEnd = cli->grid7x7EndPattern;
    const uint64_t totalPatterns = rangeEnd - rangeStart;

    Logger::out() << "Searching " << formatWithCommas(totalPatterns) << " patterns in 7x7 grid space "
                   << "(range " << formatWithCommas(rangeStart) << " to " << formatWithCommas(rangeEnd - 1) << ")\n";

    // Worker partitioning: divide the pattern space among workers
    uint64_t workerStart = rangeStart;
    uint64_t workerEnd = rangeEnd;
    uint64_t workerPatterns = totalPatterns;

    if (cli->totalWorkers > 1) {
      uint64_t patternsPerWorker = totalPatterns / cli->totalWorkers;
      uint64_t remainder = totalPatterns % cli->totalWorkers;

      workerStart = rangeStart + (cli->workerNum - 1) * patternsPerWorker;
      workerEnd = workerStart + patternsPerWorker;

      if (cli->workerNum == cli->totalWorkers) {
        workerEnd += remainder;
      }

      workerPatterns = workerEnd - workerStart;
      Logger::out() << "Worker " << cli->workerNum << "/" << cli->totalWorkers
                     << " processing patterns " << formatWithCommas(workerStart) << " to " << formatWithCommas(workerEnd - 1)
                     << " (" << formatWithCommas(workerPatterns) << " total)\n";
    }

    // Process patterns in batches
    const uint64_t batchSize = 1ULL << 32;  // 4 billion patterns per batch

    for (uint64_t batchStart = workerStart; batchStart < workerEnd; batchStart += batchSize) {
      uint64_t batchEnd = batchStart + batchSize;
      if (batchEnd > workerEnd) {
        batchEnd = workerEnd;
      }

      execute7x7Search(mem, cli, batchStart, batchEnd);
    }

    Logger::out() << "7x7 search complete. Processed " << formatWithCommas(workerEnd - workerStart) << " patterns\n";

  } else {
    // 8x8 frame-based search: use symmetry reduction
    const std::string searchRangeMessage = getSearchDescription(cli);
    Logger::out() << "Searching " << searchRangeMessage << "\n";

    // Build worker-specific frame list
    std::vector<uint64_t> workerFrames;

    if (cli->frameMode == FRAME_MODE_RANDOM || cli->frameMode == FRAME_MODE_SEQUENTIAL) {
      // Generate all incomplete frames for this worker
      for (uint64_t frameIdx = 0; frameIdx < FRAME_SEARCH_TOTAL_MINIMAL_FRAMES; ++frameIdx) {
        if ((frameIdx % cli->totalWorkers) == (cli->workerNum - 1)) {
          if (!getGoogleFrameCompleteFromCache(frameIdx)) {
            workerFrames.push_back(frameIdx);
          }
        }
      }

      // Shuffle for random mode
      if (cli->frameMode == FRAME_MODE_RANDOM && !workerFrames.empty()) {
        std::mt19937_64 rng(static_cast<uint64_t>(time(nullptr)) + cli->workerNum);
        std::shuffle(workerFrames.begin(), workerFrames.end(), rng);
      }
    } else if (cli->frameMode == FRAME_MODE_INDEX) {
      workerFrames.push_back(cli->frameModeIndex);
    }

    Logger::out() << "Processing " << workerFrames.size() << " frames for this worker\n";

    // Process all frames
    for (uint64_t frameIdx : workerFrames) {
      uint64_t frame = getFrameByIndex(frameIdx);
      if (frame != 0) {
        executeKernelSearch(mem, cli, frame, frameIdx);
      }
    }
  }

#ifdef __NVCC__
  cudaDeviceSynchronize();
#endif

  return NULL;
}

__host__ std::string getSearchDescription(ProgramArgs *cli) {
  std::ostringstream oss;

  if (cli->frameMode == FRAME_MODE_RANDOM) {
    oss << "RANDOMLY among incomplete frames";
  } else if (cli->frameMode == FRAME_MODE_SEQUENTIAL) {
    oss << "SEQUENTIALLY through incomplete frames";
  } else if (cli->frameMode == FRAME_MODE_INDEX) {
    oss << "SINGLE frame at index " << cli->frameModeIndex;
  } else {
    oss << "ERROR: Invalid frame mode";
  }

  if (cli->totalWorkers > 1) {
    oss << " [worker " << cli->workerNum << "/" << cli->totalWorkers << "]";
  }

  return oss.str();
}

__host__ void compareCycleDetectionAlgorithms(ProgramArgs *cli, uint64_t frameIdx) {
  Logger::out() << "Comparing cycle detection algorithms on frameIdx: " << frameIdx << "\n";

  uint64_t frame = getFrameByIndex(frameIdx);
  if (frame == 0) {
    std::cerr << "[ERROR] Invalid frame index: " << frameIdx << "\n";
    return;
  }

  Logger::out() << "Frame value: " << frame << "\n\n";

  gol::SearchMemory mem(FRAME_SEARCH_MAX_CANDIDATES);

  // Save original settings
  CycleDetectionAlgorithm originalAlgorithm = cli->cycleDetection;
  bool originalDontSave = cli->dontSaveResults;

  // Enable don't save results for testing
  cli->dontSaveResults = true;

  // Test Floyd's algorithm
  cli->cycleDetection = CYCLE_DETECTION_FLOYD;
  Logger::out() << "=== Testing Floyd's cycle detection ===\n";
  double startTime = getHighResCurrentTime();
  executeKernelSearch(mem, cli, frame, frameIdx);
  double floydTime = getHighResCurrentTime() - startTime;
  Logger::out() << "Floyd's algorithm completed in " << std::fixed << std::setprecision(3) << floydTime
                 << " seconds\n\n";

  // Test Nivasch's algorithm
  cli->cycleDetection = CYCLE_DETECTION_NIVASCH;
  Logger::out() << "=== Testing Nivasch's cycle detection ===\n";
  startTime = getHighResCurrentTime();
  executeKernelSearch(mem, cli, frame, frameIdx);
  double nivaschTime = getHighResCurrentTime() - startTime;
  Logger::out() << "Nivasch's algorithm completed in " << std::fixed << std::setprecision(3) << nivaschTime
                 << " seconds\n\n";

  // Restore original settings
  cli->cycleDetection = originalAlgorithm;
  cli->dontSaveResults = originalDontSave;

  // Summary
  Logger::out() << "=== Performance Comparison ===\n";
  Logger::out() << "Floyd's algorithm:   " << std::fixed << std::setprecision(3) << floydTime << " seconds\n";
  Logger::out() << "Nivasch's algorithm: " << std::fixed << std::setprecision(3) << nivaschTime << " seconds\n";

  if (floydTime < nivaschTime) {
    double timeDiff = nivaschTime - floydTime;
    double percentDiff = (timeDiff / nivaschTime) * 100.0;
    Logger::out() << "Floyd's was faster by " << std::fixed << std::setprecision(1) << percentDiff << "% ("
                   << std::setprecision(3) << timeDiff << " seconds)\n";
  } else if (nivaschTime < floydTime) {
    double timeDiff = floydTime - nivaschTime;
    double percentDiff = (timeDiff / floydTime) * 100.0;
    Logger::out() << "Nivasch's was faster by " << std::fixed << std::setprecision(1) << percentDiff << "% ("
                   << std::setprecision(3) << timeDiff << " seconds)\n";
  } else {
    Logger::out() << "Both algorithms took the same time\n";
  }
}