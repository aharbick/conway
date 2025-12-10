#include "gol_search.h"

#include <chrono>
#include <iostream>
#include <thread>

#include "gol.h"
#include "google_client.h"
#include "logging.h"

int executeMainSearch(ProgramArgs* cli) {
  // Only query Google Sheets if we're saving results
  if (!cli->dontSaveResults) {
    // Initialize global best generations from Google Sheets database
    int dbBestGenerations = getGoogleBestResult();
    if (dbBestGenerations > 0) {
      gBestGenerations = dbBestGenerations;
      Logger::out() << "Best generations so far: " << gBestGenerations << "\n";
    }

    // Initialize frame completion cache for 8x8 search (only useful for random/sequential modes, not index mode)
    // 7x7 doesn't use frame cache since it's exhaustive search
    if (cli->gridSize == GRID_SIZE_8X8 && cli->frameMode != FRAME_MODE_INDEX && loadGoogleFrameCache()) {
      uint64_t completedFrames = getGoogleFrameCacheCompletedCount();
      Logger::out() << "Frame completion cache initialized with " << completedFrames << " completed frames\n";
    }
  }

  search(cli);

  // Wait for any async upload threads to complete (only if we're saving results)
  if (!cli->dontSaveResults) {
    std::this_thread::sleep_for(std::chrono::seconds(10));
  }

  return 0;
}