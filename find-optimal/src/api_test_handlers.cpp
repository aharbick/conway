#include "api_test_handlers.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>

#include "google_client.h"

int handleTestFrameCache(ProgramArgs* cli) {
  std::cout << "Testing Frame Completion Cache...\n";

  std::cout << "Loading frame cache from Google Sheets...\n";
  if (!loadGoogleFrameCache()) {
    std::cerr << "[ERROR] Failed to load frame cache from Google Sheets API\n";
    return 1;
  }

  std::cout << "Cache loaded successfully. Scanning all " << FRAME_CACHE_TOTAL_FRAMES << " frames...\n";

  uint64_t completedFrames = 0;
  uint64_t totalFrames = FRAME_CACHE_TOTAL_FRAMES;

  // Sample frames for progress reporting
  for (uint64_t frameIdx = 0; frameIdx < totalFrames; frameIdx++) {
    if (getGoogleFrameCompleteFromCache(frameIdx)) {
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
  bool success = sendGoogleProgress(testFrameId, testKernelId, testGenerations, testPattern, testPatternBin);
  std::cout << "Progress upload: " << (success ? "SUCCESS" : "FAILED") << "\n";

  // Test querying best result
  int bestResult = getGoogleBestResult();
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
  bool success = sendGoogleSummaryData(6, testPattern, testPatternBin);
  std::cout << "Summary data upload: " << (success ? "SUCCESS" : "FAILED") << "\n";

  // Test duplicate entry (should increment count for bestGenerations=6)
  uint64_t testPattern2 = ((uint64_t)rand() << 32) | rand();
  for (int i = 0; i < 64; i++) {
    testPatternBin[i] = ((testPattern2 >> (63 - i)) & 1) ? '1' : '0';
  }

  std::cout << "Testing DUPLICATE summary data upload (generations=6, pattern=" << testPattern2 << ")...\n";
  success = sendGoogleSummaryData(6, testPattern2, testPatternBin);
  std::cout << "Duplicate summary data upload: " << (success ? "SUCCESS" : "FAILED") << "\n";

  return 0;
}

// ============================================================================
// STRIP SEARCH API TEST HANDLERS
// ============================================================================

int handleTestStripProgressApi(ProgramArgs* cli) {
  std::cout << "Testing Google Sheets Strip Progress API with fake data...\n";

  // Generate realistic but varying test data based on current time
  time_t now = time(NULL);
  srand(now);

  // Generate realistic strip progress data
  uint32_t testCenterIdx = rand() % STRIP_CACHE_TOTAL_CENTERS;  // 0-8547
  uint32_t testMiddleIdx = rand() % STRIP_CACHE_MIDDLE_IDX_COUNT;  // 0-511

  // Generate realistic result data
  int testGenerations = 180 + (rand() % 200);  // Generations 180-379
  uint64_t testPattern = ((uint64_t)rand() << 32) | rand();

  // Generate a 64-bit binary pattern string
  char testPatternBin[65];
  for (int i = 0; i < 64; i++) {
    testPatternBin[i] = ((testPattern >> (63 - i)) & 1) ? '1' : '0';
  }
  testPatternBin[64] = '\0';

  // Test strip progress upload
  std::cout << "Testing strip progress upload (centerIdx=" << testCenterIdx << ", middleIdx=" << testMiddleIdx
            << ", generations=" << testGenerations << ", pattern=" << std::hex << testPattern << std::dec << ")...\n";
  bool success = sendGoogleStripProgress(testCenterIdx, testMiddleIdx, testGenerations, testPattern, testPatternBin);
  std::cout << "Strip progress upload: " << (success ? "SUCCESS" : "FAILED") << "\n";

  return 0;
}

int handleTestStripSummaryApi(ProgramArgs* cli) {
  std::cout << "Testing Google Sheets Strip Summary API with test data...\n";

  // Use easily identifiable test data (bestGenerations=7 to distinguish from frame test)
  time_t now = time(NULL);
  srand(now);

  // Generate test pattern data
  uint64_t testPattern = ((uint64_t)rand() << 32) | rand();

  // Generate a 64-bit binary pattern string
  char testPatternBin[65];
  for (int i = 0; i < 64; i++) {
    testPatternBin[i] = ((testPattern >> (63 - i)) & 1) ? '1' : '0';
  }
  testPatternBin[64] = '\0';

  // Test strip summary data upload with bestGenerations=7
  std::cout << "Testing strip summary data upload (generations=7, pattern=" << testPattern << ")...\n";
  bool success = sendGoogleStripSummaryData(7, testPattern, testPatternBin);
  std::cout << "Strip summary data upload: " << (success ? "SUCCESS" : "FAILED") << "\n";

  // Test duplicate entry (should increment count for bestGenerations=7)
  uint64_t testPattern2 = ((uint64_t)rand() << 32) | rand();
  for (int i = 0; i < 64; i++) {
    testPatternBin[i] = ((testPattern2 >> (63 - i)) & 1) ? '1' : '0';
  }

  std::cout << "Testing DUPLICATE strip summary data upload (generations=7, pattern=" << testPattern2 << ")...\n";
  success = sendGoogleStripSummaryData(7, testPattern2, testPatternBin);
  std::cout << "Duplicate strip summary data upload: " << (success ? "SUCCESS" : "FAILED") << "\n";

  return 0;
}

int handleTestStripCache(ProgramArgs* cli) {
  std::cout << "Testing Strip Completion Cache...\n";

  std::cout << "Loading strip cache from Google Sheets...\n";
  if (!loadGoogleStripCache()) {
    std::cerr << "[ERROR] Failed to load strip cache from Google Sheets API\n";
    return 1;
  }

  uint64_t totalIntervals = STRIP_CACHE_TOTAL_INTERVALS;
  uint64_t completedIntervals = getGoogleStripCacheCompletedCount();

  std::cout << "\n=== Strip Cache Test Results ===\n";
  std::cout << "Total centers: " << STRIP_CACHE_TOTAL_CENTERS << "\n";
  std::cout << "Middle indices per center: " << STRIP_CACHE_MIDDLE_IDX_COUNT << "\n";
  std::cout << "Total intervals: " << totalIntervals << "\n";
  std::cout << "Completed intervals: " << completedIntervals << "\n";
  std::cout << "Completion rate: " << std::fixed << std::setprecision(2)
            << (double)completedIntervals / totalIntervals * 100.0 << "%\n";
  std::cout << "Cache size: " << STRIP_CACHE_BITMAP_BYTES << " bytes\n";

  // Show completion status for first few intervals
  std::cout << "\nFirst 20 intervals completion status:\n";
  for (uint32_t i = 0; i < 20; i++) {
    uint32_t centerIdx = i / STRIP_CACHE_MIDDLE_IDX_COUNT;
    uint32_t middleIdx = i % STRIP_CACHE_MIDDLE_IDX_COUNT;
    bool complete = isGoogleStripIntervalComplete(centerIdx, middleIdx);
    std::cout << "  " << centerIdx << ":" << middleIdx << " = " << (complete ? "complete" : "incomplete") << "\n";
  }

  return 0;
}

int handleTestStripCompletionApi(ProgramArgs* cli) {
  std::cout << "Testing Google Sheets Strip Completion API...\n";

  // Use centerIdx=0, middleIdx=0 for easy verification and reversal
  uint32_t testCenterIdx = 0;
  uint32_t testMiddleIdx = 0;

  // Test strip completion
  std::cout << "Testing strip completion (centerIdx=" << testCenterIdx << ", middleIdx=" << testMiddleIdx << ")...\n";
  bool success = sendGoogleStripCompletion(testCenterIdx, testMiddleIdx);
  std::cout << "Strip completion: " << (success ? "SUCCESS" : "FAILED") << "\n";

  return 0;
}