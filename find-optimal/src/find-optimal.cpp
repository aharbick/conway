#include <locale.h>
#include <stdio.h>

#include <iostream>
#include <memory>

#include "api_test_handlers.h"
#include "center4x4_utils.h"
#include "cli_parser.h"
#include "constants.h"
#include "gol_search.h"
#include "google_client.h"
#include "google_request_queue.h"
#include "logging.h"
#include "queue_handlers.h"
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

  // Initialize center 4x4 lookup table for strip search symmetry reduction
  initializeUnique4x4Centers();

  // Initialize logging system
  Logger::initialize(cli);

  // Print CUDA device info early
  printCudaDeviceInfo(cli);

  // Print topology information
#ifdef TOPOLOGY_TORUS
  Logger::out() << "Topology: Torus (wrapping boundaries)\n";
#else
  Logger::out() << "Topology: Box/Plane (non-wrapping boundaries)\n";
#endif

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
  if (cli->drainRequestQueue) {
    result = handleDrainRequestQueue(cli);
  } else if (cli->computeSubgridCache) {
    result = computeSubgridCache(cli);
  } else if (cli->testApi != TEST_API_NONE) {
    if (cli->testApi == TEST_API_FRAMECACHE) {
      result = handleTestFrameCache(cli);
    } else if (cli->testApi == TEST_API_PROGRESS) {
      handleTestProgressApi(cli);
      result = 0;  // API tests always return success
    } else if (cli->testApi == TEST_API_SUMMARY) {
      handleTestSummaryApi(cli);
      result = 0;  // API tests always return success
    } else if (cli->testApi == TEST_API_STRIP_PROGRESS) {
      handleTestStripProgressApi(cli);
      result = 0;  // API tests always return success
    } else if (cli->testApi == TEST_API_STRIP_SUMMARY) {
      handleTestStripSummaryApi(cli);
      result = 0;  // API tests always return success
    } else if (cli->testApi == TEST_API_STRIP_CACHE) {
      result = handleTestStripCache(cli);
    } else if (cli->testApi == TEST_API_STRIP_COMPLETION) {
      handleTestStripCompletionApi(cli);
      result = 0;
    } else {
      std::cerr << "[ERROR] Unknown test API type\n";
      result = 1;
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
