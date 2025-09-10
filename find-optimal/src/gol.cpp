#include "gol.h"

#include <mutex>
#include <random>
#include <sstream>
#include <string>

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

  printThreadStatus(cli->threadId, "Running with CUDA enabled");

  if (cli->random) {
    printThreadStatus(cli->threadId, "searching RANDOMLY %llu candidates", cli->randomSamples);
    searchRandom(cli);
  } else {
    const std::string searchRangeMessage = getSearchDescription(cli);
    printThreadStatus(cli->threadId, "searching %s", searchRangeMessage.c_str());
    searchAll(cli);
  }

  return NULL;
}

__host__ void searchRandom(ProgramArgs *cli) {
  std::mt19937_64 rng(static_cast<uint64_t>(time(nullptr)));

  gol::SearchMemory mem(RANDOM_SEARCH_MAX_CANDIDATES);

  // We're randomly searching..  I didn't get cuRAND to work so we randomize our batches. Each
  // call to findCandidates is sequential but we look at random locations across all possible.
  const uint64_t chunkSize = RANDOM_SEARCH_CHUNK_SIZE;
  const uint64_t iterations = cli->randomSamples / chunkSize;
  for (uint64_t i = 0; i < iterations; ++i) {
    const uint64_t start = rng();
    const uint64_t end = start + chunkSize;
    executeCandidateSearch(mem, cli, start, end);
  }
}

__host__ void searchAll(ProgramArgs *cli) {
  gol::SearchMemory mem(FRAME_SEARCH_MAX_CANDIDATES);

  if (cli->randomFrameMode) {
    // Random frame processing
    std::mt19937_64 rng(static_cast<uint64_t>(time(nullptr)) + cli->threadId);
    std::uniform_int_distribution<uint64_t> dist(0, cli->frameEndIdx - 1);

    while (true) {
      // Generate random frame index
      uint64_t randomFrameIdx = dist(rng);

      // Check if this frame is already complete
      if (googleIsFrameCompleteFromCache(randomFrameIdx)) {
        continue;  // Skip completed frames
      }

      // Get the actual frame value for this index
      uint64_t frame = getFrameByIndex(randomFrameIdx);
      if (frame != 0) {
        printThreadStatus(cli->threadId, "Processing random frame %llu", randomFrameIdx);
        executeKernelSearch(mem, cli, frame, randomFrameIdx);
      }
    }
  } else {
    // Sequential frame processing
    for (uint64_t i = 0, currentFrameIdx = 0; i < FRAME_SEARCH_MAX_FRAMES && currentFrameIdx < cli->frameEndIdx; ++i) {
      const uint64_t frame = spreadBitsToFrame(i);
      if (isMinimalFrame(frame)) {
        if (currentFrameIdx >= cli->frameBeginIdx) {
          // Check if this frame is already complete (for sequential processing too)
          if (googleIsFrameCompleteFromCache(currentFrameIdx)) {
            printThreadStatus(cli->threadId, "Skipping completed frame %llu", currentFrameIdx);
          } else {
            executeKernelSearch(mem, cli, frame, currentFrameIdx);
          }
        }
        ++currentFrameIdx;
      }
    }
  }
}


__host__ std::string getSearchDescription(ProgramArgs *cli) {
  if (cli->randomFrameMode) {
    return "RANDOMLY among all incomplete frames";
  } else if (cli->frameBeginIdx > 0 || cli->frameEndIdx > 0) {
    std::ostringstream oss;
    oss << "ALL in frames (" << cli->frameBeginIdx << " - " << cli->frameEndIdx << ")";
    return oss.str();
  } else if (cli->beginAt > 0 || cli->endAt > 0) {
    std::ostringstream oss;
    oss << "ALL in range (" << cli->beginAt << " - " << cli->endAt << ")";
    return oss.str();
  } else {
    return "ALL";
  }
}