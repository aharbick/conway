#include "gol_search.h"

#include <chrono>
#include <iostream>
#include <thread>

#include "gol.h"
#include "google_client.h"
#include "logging.h"

int executeMainSearch(ProgramArgs* cli) {
  // Initialize global best generations from Google Sheets database
  int dbBestGenerations = googleGetBestResult();
  if (dbBestGenerations > 0) {
    gBestGenerations = dbBestGenerations;
    Logger::out() << "Best generations so far: " << gBestGenerations << "\n";
  }

  // Initialize frame completion cache
  if (googleLoadFrameCache()) {
    uint64_t completedFrames = googleGetFrameCacheCompletedCount();
    Logger::out() << "Frame completion cache initialized with " << completedFrames << " completed frames\n";
  }

  search(cli);

  // Wait for any async upload threads to complete
  std::this_thread::sleep_for(std::chrono::seconds(10));

  return 0;
}