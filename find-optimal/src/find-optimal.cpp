#include <locale.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "cli_parser.h"
#include "constants.h"
#include "gol.h"
#include "logging.h"

static void printCudaDeviceInfo(ProgramArgs* cli) {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  Logging::out() << "CUDA devices available: " << deviceCount << "\n";
  Logging::out() << "Using 1 GPU with blockSize=" << FRAME_SEARCH_GRID_SIZE
                 << ", threadsPerBlock=" << FRAME_SEARCH_THREADS_PER_BLOCK << "\n";
}

int main(int argc, char** argv) {
  // Set locale for number formatting with thousands separators
  setlocale(LC_NUMERIC, "");

  // Change stdout to not buffered
  setvbuf(stdout, NULL, _IONBF, 0);

  // Initialize Google client
  googleInit();

  // Check Google configuration and warn once if not configured
  GoogleConfig googleConfig;
  if (googleGetConfig(&googleConfig) != GOOGLE_SUCCESS) {
    std::cerr << "[ERROR] Progress will not be saved to Google Sheets\n";
  }

  // Process the arguments using CLI parser
  auto* cli = parseCommandLineArgs(argc, argv);
  if (!cli) {
    std::cerr << "[ERROR] Failed to parse command line arguments\n";
    googleCleanup();
    return 1;
  }

  // Initialize logging system
  Logging::LogManager::initialize(cli);


  // Handle test-frame-cache flag
  if (cli->testFrameCache) {
    std::cout << "Testing Frame Completion Cache...\n";

    std::cout << "Loading frame cache from Google Sheets...\n";
    if (!googleLoadFrameCache()) {
      std::cerr << "[ERROR] Failed to load frame cache from Google Sheets API\n";
      cleanupProgramArgs(cli);
      googleCleanup();
      return 1;
    }

    std::cout << "Cache loaded successfully. Scanning all " << FRAME_CACHE_TOTAL_FRAMES << " frames...\n";

    uint64_t completedFrames = 0;
    uint64_t totalFrames = FRAME_CACHE_TOTAL_FRAMES;

    // Sample frames for progress reporting
    for (uint64_t frameIdx = 0; frameIdx < totalFrames; frameIdx++) {
      if (googleIsFrameCompleteFromCache(frameIdx)) {
        completedFrames++;
      }

      // Progress report every 100K frames
      if (frameIdx > 0 && frameIdx % 100000 == 0) {
        double progress = (double)frameIdx / totalFrames * 100.0;
        std::cout << "Progress: " << frameIdx << "/" << totalFrames << " (" << std::fixed << std::setprecision(1)
                  << progress << "%), " << "completed so far: " << completedFrames << "\n";
      }
    }

    std::cout << "\n=== Frame Cache Test Results ===\n";
    std::cout << "Total frames: " << totalFrames << "\n";
    std::cout << "Completed frames: " << completedFrames << "\n";
    std::cout << "Completion rate: " << std::fixed << std::setprecision(2)
              << (double)completedFrames / totalFrames * 100.0 << "%\n";
    std::cout << "Cache size: " << FRAME_CACHE_BITMAP_BYTES << " bytes\n";

    cleanupProgramArgs(cli);
    googleCleanup();
    return 0;
  }

  // Handle test-progress-api flag
  if (cli->testProgressApi) {
    std::cout << "Testing Google Sheets Progress API with fake data...\n";

    // Generate realistic but varying test data based on current time
    time_t now = time(NULL);
    srand(now);  // Seed random number generator with current time

    // Generate realistic progress data
    uint64_t testFrameId = 1000000 + (rand() % 1000000);  // Frame IDs in realistic range
    int testKernelId = rand() % 16;                       // Kernel IDs 0-15

    // Generate realistic result data
    int testGenerations = 180 + (rand() % 200);                // Generations 180-379 (realistic range)
    uint64_t testPattern = ((uint64_t)rand() << 32) | rand();  // Random 64-bit pattern

    // Generate a realistic 64-bit binary pattern string
    char testPatternBin[65];
    for (int i = 0; i < 64; i++) {
      testPatternBin[i] = ((testPattern >> (63 - i)) & 1) ? '1' : '0';
    }
    testPatternBin[64] = '\0';

    // Test unified progress upload with best result data
    std::cout << "Testing progress upload (frameIdx=" << testFrameId << ", kernelIdx=" << testKernelId
              << ", generations=" << testGenerations << ", pattern=" << std::hex << testPattern << std::dec << ")...\n";
    bool success = googleSendProgress(testFrameId, testKernelId, testGenerations, testPattern, testPatternBin);
    std::cout << "Progress upload: " << (success ? "SUCCESS" : "FAILED") << "\n";

    // Test querying best result
    int bestResult = googleGetBestResult();
    std::cout << "Best result query: " << bestResult << "\n";

    // Cleanup and exit
    cleanupProgramArgs(cli);
    googleCleanup();
    // Always return success - Google Sheets errors shouldn't fail the program
    return 0;
  }

  // Handle test-summary-api flag
  if (cli->testSummaryApi) {
    std::cout << "Testing Google Sheets Summary API with test data...\n";

    // Use easily identifiable test data (bestGenerations=6 is well below normal range)
    time_t now = time(NULL);
    srand(now);  // Seed random number generator with current time

    // Generate test pattern data
    uint64_t testPattern = ((uint64_t)rand() << 32) | rand();  // Random 64-bit pattern

    // Generate a 64-bit binary pattern string
    char testPatternBin[65];
    for (int i = 0; i < 64; i++) {
      testPatternBin[i] = ((testPattern >> (63 - i)) & 1) ? '1' : '0';
    }
    testPatternBin[64] = '\0';

    // Test initial summary data upload with bestGenerations=6
    std::cout << "Testing summary data upload (generations=6, pattern=" << testPattern << ")...\n";
    bool success = googleSendSummaryData(6, testPattern, testPatternBin);
    std::cout << "Summary data upload: " << (success ? "SUCCESS" : "FAILED") << "\n";

    // Test duplicate entry (should increment count for bestGenerations=6)
    uint64_t testPattern2 = ((uint64_t)rand() << 32) | rand();
    for (int i = 0; i < 64; i++) {
      testPatternBin[i] = ((testPattern2 >> (63 - i)) & 1) ? '1' : '0';
    }

    std::cout << "Testing DUPLICATE summary data upload (generations=6, pattern=" << testPattern2 << ")...\n";
    success = googleSendSummaryData(6, testPattern2, testPatternBin);
    std::cout << "Duplicate summary data upload: " << (success ? "SUCCESS" : "FAILED") << "\n";

    // Cleanup and exit
    cleanupProgramArgs(cli);
    googleCleanup();
    return 0;
  }

  // Handle compare-cycle-algorithms flag
  if (cli->compareCycleAlgorithms) {
    compareCycleDetectionAlgorithms(cli, cli->compareFrameIdx);
    cleanupProgramArgs(cli);
    googleCleanup();
    return 0;
  }

#ifdef __NVCC__
  printCudaDeviceInfo(cli);
#endif

  // Initialize global best generations from Google Sheets database
  int dbBestGenerations = googleGetBestResult();
  if (dbBestGenerations > 0) {
    gBestGenerations = dbBestGenerations;
    Logging::out() << "Best generations so far: " << gBestGenerations << "\n";
  }

  // Initialize frame completion cache
  if (googleLoadFrameCache()) {
    uint64_t completedFrames = googleGetFrameCacheCompletedCount();
    Logging::out() << "Frame completion cache initialized with " << completedFrames << " completed frames\n";
  }

  search(cli);
  cleanupProgramArgs(cli);

  // Wait for any async upload threads to complete
  std::this_thread::sleep_for(std::chrono::seconds(10));

  // Cleanup Google client
  googleCleanup();

  return 0;
}
