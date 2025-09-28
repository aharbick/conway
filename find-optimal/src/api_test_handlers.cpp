#include "api_test_handlers.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>

#include "google_client.h"

int handleTestFrameCache(ProgramArgs* cli) {
  std::cout << "Testing Frame Completion Cache...\n";

  std::cout << "Loading frame cache from Google Sheets...\n";
  if (!googleLoadFrameCache()) {
    std::cerr << "[ERROR] Failed to load frame cache from Google Sheets API\n";
    return 1;
  }

  std::cout << "Cache loaded successfully. Scanning all " << FRAME_CACHE_TOTAL_FRAMES << " frames...\n";

  uint64_t completedFrames = 0;
  uint64_t totalFrames = FRAME_CACHE_TOTAL_FRAMES;

  // Sample frames for progress reporting
  for (uint64_t frameIdx = 0; frameIdx < totalFrames; frameIdx++) {
    if (googleGetFrameCompleteFromCache(frameIdx)) {
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

  return 0;
}

int handleTestProgressApi(ProgramArgs* cli) {
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

  return 0;
}

int handleTestSummaryApi(ProgramArgs* cli) {
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

  return 0;
}