#include "gol.h"

#include <algorithm>
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

  const std::string searchRangeMessage = getSearchDescription(cli);
  Logger::out() << "Searching " << searchRangeMessage << "\n";

  gol::SearchMemory mem(FRAME_SEARCH_MAX_CANDIDATES);

  // Build worker-specific frame list
  std::vector<uint64_t> workerFrames;

  if (cli->frameMode == FRAME_MODE_RANDOM || cli->frameMode == FRAME_MODE_SEQUENTIAL) {
    // Generate all incomplete frames for this worker
    for (uint64_t frameIdx = 0; frameIdx < FRAME_SEARCH_TOTAL_MINIMAL_FRAMES; ++frameIdx) {
      // Check if this frame belongs to this worker using modulo partitioning
      if ((frameIdx % cli->totalWorkers) == (cli->workerNum - 1)) {
        // Check if frame is incomplete
        if (!getGoogleFrameCompleteFromCache(frameIdx)) {
          workerFrames.push_back(frameIdx);
        }
      }
    }

    // Shuffle for random mode, keep in order for sequential mode
    if (cli->frameMode == FRAME_MODE_RANDOM && !workerFrames.empty()) {
      std::mt19937_64 rng(static_cast<uint64_t>(time(nullptr)) + cli->workerNum);
      std::shuffle(workerFrames.begin(), workerFrames.end(), rng);
    }
  } else if (cli->frameMode == FRAME_MODE_INDEX) {
    // Use the pre-parsed frame index
    workerFrames.push_back(cli->frameModeIndex);
  }

  // Report how many frames will be processed
  Logger::out() << "Processing " << workerFrames.size() << " frames for this worker\n";

  // Process all frames in the worker's list
  for (uint64_t frameIdx : workerFrames) {
    // Get the actual frame value for this index
    uint64_t frame = getFrameByIndex(frameIdx);
    if (frame != 0) {
      executeKernelSearch(mem, cli, frame, frameIdx);
    }
  }

#ifdef __NVCC__
  // Ensure all CUDA operations complete before returning
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

  // Add worker information
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