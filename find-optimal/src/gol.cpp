#include "gol.h"

#include "mt.h"

// Global variables updated across threads
pthread_mutex_t gMutex = PTHREAD_MUTEX_INITIALIZER;
int gBestGenerations = 0;

__host__ bool updateBestGenerations(int generations) {
  pthread_mutex_lock(&gMutex);

  bool isNewGlobalBest = (gBestGenerations < generations);
  if (isNewGlobalBest) {
    gBestGenerations = generations;
  }

  pthread_mutex_unlock(&gMutex);
  return isNewGlobalBest;
}

__host__ void *search(void *args) {
  ProgramArgs *cli = (ProgramArgs *)args;

  printThreadStatus(cli->threadId, "Running with CUDA enabled");

  if (cli->random) {
    printThreadStatus(cli->threadId, "searching RANDOMLY %llu candidates", cli->randomSamples);
    searchRandom(cli);
  } else {
    char searchRangeMessage[MESSAGE_BUFFER_SIZE] = {'\0'};
    getSearchDescription(cli, searchRangeMessage, sizeof(searchRangeMessage));
    printThreadStatus(cli->threadId, "searching %s", searchRangeMessage);
    searchAll(cli);
  }

  return NULL;
}

__host__ void searchRandom(ProgramArgs *cli) {
  init_genrand64((uint64_t)time(NULL));

  SearchMemory *mem = allocateSearchMemory(RANDOM_SEARCH_MAX_CANDIDATES);

  // We're randomly searching..  I didn't get cuRAND to work so we randomize our batches. Each
  // call to findCandidates is sequential but we look at random locations across all possible.
  uint64_t chunkSize = RANDOM_SEARCH_CHUNK_SIZE;
  uint64_t iterations = cli->randomSamples / chunkSize;
  for (uint64_t i = 0; i < iterations; i++) {
    uint64_t start = genrand64_int64();
    uint64_t end = start + chunkSize;
    executeCandidateSearch(mem, cli, start, end);
  }

  // Add cleanup
  freeSearchMemory(mem);
}

__host__ void searchAll(ProgramArgs *cli) {
  SearchMemory *mem = allocateSearchMemory(FRAME_SEARCH_MAX_CANDIDATES);

  // Iterate through frame range or all possible 24-bit numbers and use spreadBitsToFrame to cover all 64-bit "frames"
  for (uint64_t i = 0, currentFrameIdx = 0; i < FRAME_SEARCH_MAX_FRAMES && currentFrameIdx < cli->frameEndIdx; i++) {
    uint64_t frame = spreadBitsToFrame(i);
    if (isMinimalFrame(frame)) {
      if (currentFrameIdx >= cli->frameBeginIdx) {
        executeKernelSearch(mem, cli, frame, currentFrameIdx);
      }
      currentFrameIdx++;
    }
  }

  // Cleanup
  freeSearchMemory(mem);
}


__host__ const char *getSearchDescription(ProgramArgs *cli, char *buffer, size_t bufferSize) {
  if (cli->frameBeginIdx > 0 || cli->frameEndIdx > 0) {
    snprintf(buffer, bufferSize, "ALL in frames (%lu - %lu)", cli->frameBeginIdx, cli->frameEndIdx);
  } else if (cli->beginAt > 0 || cli->endAt > 0) {
    snprintf(buffer, bufferSize, "ALL in range (%lu - %lu)", cli->beginAt, cli->endAt);
  } else {
    strcpy(buffer, "ALL");
  }
  return buffer;
}