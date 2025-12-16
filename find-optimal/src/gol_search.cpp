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
    // Determine search type based on CLI parameters
    bool isStripSearch = (cli->stripMode != STRIP_MODE_NONE);
    GoogleSearchType searchType = isStripSearch ? GOOGLE_SEARCH_STRIP : GOOGLE_SEARCH_FRAME;

    // Initialize global best generations from Google Sheets database
    int dbBestGenerations = getGoogleBestResult(searchType);
    if (dbBestGenerations > 0) {
      gBestGenerations = dbBestGenerations;
      Logger::out() << "Best generations so far: " << gBestGenerations << "\n";
    }

    // Initialize completion cache based on search type
    if (isStripSearch) {
      // Strip search uses strip completion cache
      if (loadGoogleStripCache()) {
        uint64_t completedIntervals = getGoogleStripCacheCompletedCount();
        Logger::out() << "Strip completion cache initialized with " << completedIntervals << " completed middle block intervals\n";

        // For range mode searches, advance to the first incomplete interval
        if (cli->stripMode == STRIP_MODE_RANGE || cli->stripMode == STRIP_MODE_NONE) {
          uint32_t resumeCenterIdx, resumeMiddleIdx;
          if (findFirstIncompleteStripInterval(cli->centerIdxStart, cli->centerIdxEnd,
                                               cli->middleIdxStart, cli->middleIdxEnd,
                                               resumeCenterIdx, resumeMiddleIdx)) {
            cli->centerIdxStart = resumeCenterIdx;
            cli->middleIdxStart = resumeMiddleIdx;
          } else {
            Logger::out() << "All intervals in range are already complete!\n";
            return 0;
          }
        }
      }
    } else if (cli->gridSize == GRID_SIZE_8X8 && cli->frameMode != FRAME_MODE_INDEX) {
      // Frame search uses frame completion cache (only useful for random/sequential modes, not index mode)
      // 7x7 doesn't use frame cache since it's exhaustive search
      if (loadGoogleFrameCache()) {
        uint64_t completedFrames = getGoogleFrameCacheCompletedCount();
        Logger::out() << "Frame completion cache initialized with " << completedFrames << " completed frames\n";
      }
    }
  }

  search(cli);

  // Wait for any async upload threads to complete (only if we're saving results)
  if (!cli->dontSaveResults) {
    std::this_thread::sleep_for(std::chrono::seconds(10));
  }

  return 0;
}