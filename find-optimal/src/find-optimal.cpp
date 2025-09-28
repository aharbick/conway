#include <locale.h>
#include <stdio.h>

#include <iostream>
#include <memory>

#include "api_test_handlers.h"
#include "cli_parser.h"
#include "constants.h"
#include "gol_search.h"
#include "google_client.h"
#include "logging.h"
#include "simulation_handlers.h"

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
  googleInit();

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
    if (googleGetConfig(&googleConfig) != GOOGLE_SUCCESS) {
      Logger::out() << "Google Sheets API is not configured. Processing all frames and not saving results.\n";
      cli->dontSaveResults = true;  // Prevent sending attempts
    }
  }


  // Determine which action to take and execute it
  int result;
  if (cli->testFrameCache) {
    result = handleTestFrameCache(cli);
  } else if (cli->testProgressApi) {
    handleTestProgressApi(cli);
    result = 0;  // API tests always return success
  } else if (cli->testSummaryApi) {
    handleTestSummaryApi(cli);
    result = 0;  // API tests always return success
  } else if (cli->compareCycleAlgorithms) {
    result = handleCompareCycleAlgorithms(cli);
  } else if (cli->simulateMode) {
    result = handleSimulationMode(cli);
  } else {
    // Default: execute main search
    result = executeMainSearch(cli);
  }

  // Common cleanup for all execution paths
  cleanupProgramArgs(cli);
  googleCleanup();

  return result;
}
