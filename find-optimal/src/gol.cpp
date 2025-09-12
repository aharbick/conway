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
  Logging::out() << "Searching " << searchRangeMessage << "\n";

  gol::SearchMemory mem(FRAME_SEARCH_MAX_CANDIDATES);

  // Build worker-specific frame list
  std::vector<uint64_t> workerFrames;

  if (cli->frameMode == "missing") {
    // Get incomplete frames from the API
    std::vector<uint64_t> incompleteFrames = googleGetIncompleteFrames();

    // Filter for this worker using modulo partitioning
    for (uint64_t frameIdx : incompleteFrames) {
      if ((frameIdx % cli->totalWorkers) == (cli->workerNum - 1)) {
        workerFrames.push_back(frameIdx);
      }
    }
  } else if (cli->frameMode == "random") {
    // Generate all incomplete frames for this worker
    for (uint64_t frameIdx = 0; frameIdx < FRAME_SEARCH_TOTAL_MINIMAL_FRAMES; ++frameIdx) {
      // Check if this frame belongs to this worker using modulo partitioning
      if ((frameIdx % cli->totalWorkers) == (cli->workerNum - 1)) {
        // Check if frame is incomplete
        if (!googleIsFrameCompleteFromCache(frameIdx)) {
          workerFrames.push_back(frameIdx);
        }
      }
    }

    // Shuffle for random mode
    if (!workerFrames.empty()) {
      std::mt19937_64 rng(static_cast<uint64_t>(time(nullptr)) + cli->workerNum);
      std::shuffle(workerFrames.begin(), workerFrames.end(), rng);
    }
  }

  // Process all frames in the worker's list
  for (uint64_t frameIdx : workerFrames) {
    // Get the actual frame value for this index
    uint64_t frame = getFrameByIndex(frameIdx);
    if (frame != 0) {
      executeKernelSearch(mem, cli, frame, frameIdx);
    }
  }

  return NULL;
}


__host__ std::string getSearchDescription(ProgramArgs *cli) {
  std::ostringstream oss;

  if (cli->frameMode == "missing") {
    oss << "SEQUENTIALLY through frames missing kernels";
  } else if (cli->frameMode == "random") {
    oss << "RANDOMLY among incomplete frames";
  } else {
    oss << "ERROR: Invalid frame mode";
  }

  // Add worker information
  if (cli->totalWorkers > 1) {
    oss << " [worker " << cli->workerNum << "/" << cli->totalWorkers << "]";
  }

  return oss.str();
}