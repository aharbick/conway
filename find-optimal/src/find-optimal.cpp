#include <locale.h>
#include <stdio.h>

#include <iostream>
#include <memory>

#include "api_test_handlers.h"
#include "cli_parser.h"
#include "constants.h"
#include "gol_search.h"
#include "google_client.h"
#include "google_request_queue.h"
#include "logging.h"
#include "simulation_handlers.h"
#include "subgrid_cache.h"

static void printCudaDeviceInfo(ProgramArgs* cli) {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  Logger::out() << "CUDA devices available: " << deviceCount << "\n";
  Logger::out() << "Using 1 GPU with blockSize=" << FRAME_SEARCH_GRID_SIZE
                 << ", threadsPerBlock=" << FRAME_SEARCH_THREADS_PER_BLOCK << "\n";
}

int main(int argc, char** argv) {
  // Set locale for number formatting with thousands separators
  setlocale(LC_NUMERIC, "");

  // Change stdout to not buffered
  setvbuf(stdout, NULL, _IONBF, 0);

  // Process the arguments using CLI parser first
  auto* cli = parseCommandLineArgs(argc, argv);
  if (!cli) {
    std::cerr << "[ERROR] Failed to parse command line arguments\n";
    return 1;
  }

  // Initialize Google client
  initGoogleClient();

  // Initialize logging system
  Logger::initialize(cli);

  // Print CUDA device info early
  printCudaDeviceInfo(cli);

  // Handle Google Sheets configuration
  if (cli->dontSaveResults) {
    Logger::out() << "Not saving results to Google Sheets (--dont-save-results specified)\n";
  } else {
    // Check Google configuration and warn if not configured
    GoogleConfig googleConfig;
    if (getGoogleConfig(&googleConfig) != GOOGLE_SUCCESS) {
      Logger::out() << "Google Sheets API is not configured. Processing all frames and not saving results.\n";
      cli->dontSaveResults = true;  // Prevent sending attempts
    } else {
      // Initialize and start the request queue
      if (!initGoogleRequestQueue(cli->queueDirectory)) {
        std::cerr << "[ERROR] Failed to initialize request queue\n";
        cleanupProgramArgs(cli);
        cleanupGoogleClient();
        return 1;
      }
    }
  }


  // Load subgrid cache if specified (but not when computing it)
  if (!cli->subgridCachePath.empty() && !cli->computeSubgridCache) {
    SubgridCache::getInstance().load(cli->subgridCachePath);
  }

  // Determine which action to take and execute it
  int result;
  if (cli->computeSubgridCache) {
    result = computeSubgridCache(cli);
  } else if (!cli->testApi.empty()) {
    if (cli->testApi == "framecache") {
      result = handleTestFrameCache(cli);
    } else if (cli->testApi == "progress") {
      handleTestProgressApi(cli);
      result = 0;  // API tests always return success
    } else {  // testApi == "summary"
      handleTestSummaryApi(cli);
      result = 0;  // API tests always return success
    }
  } else if (cli->compareCycleAlgorithms) {
    result = handleCompareCycleAlgorithms(cli);
  } else if (cli->simulateMode) {
    result = handleSimulationMode(cli);
  } else {
    // Default: execute main search
    result = executeMainSearch(cli);
  }

  // Common cleanup for all execution paths
  stopGoogleRequestQueue();
  cleanupProgramArgs(cli);
  cleanupGoogleClient();

  return result;
}
