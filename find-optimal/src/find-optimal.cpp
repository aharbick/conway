#include <locale.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "cli_parser.h"
#include "gol.h"
#include "logging.h"

// Thread context class with RAII
class ThreadContext {
 public:
  std::vector<pthread_t> threads;
  std::vector<std::unique_ptr<ProgramArgs>> threadArgs;
  int numThreads;

  ThreadContext(int count) : numThreads(count) {
    threads.resize(count);
    threadArgs.reserve(count);
  }
};

#ifdef __NVCC__
static void printCudaDeviceInfo(ProgramArgs* cli) {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  Logging::out() << "CUDA devices available: " << deviceCount << "\n";
  Logging::out() << "Using " << cli->gpusToUse << " GPU(s) with blockSize=" << cli->blockSize
                 << ", threadsPerBlock=" << cli->threadsPerBlock << "\n";
}
#endif

static std::unique_ptr<ProgramArgs> createThreadArgs(ProgramArgs* cli, int threadId, uint64_t patternsPerThread) {
  auto targs = std::make_unique<ProgramArgs>(*cli);
  targs->threadId = threadId;

  // Each thread has the same endAt but different beginAt offsets
  targs->beginAt = cli->beginAt + (threadId * patternsPerThread);
  if (threadId == cli->cpuThreads - 1) {
    // Last thread gets any remaining patterns
    targs->endAt = cli->endAt;
  } else {
    targs->endAt = targs->beginAt + patternsPerThread;
  }

  return targs;
}

static std::unique_ptr<ThreadContext> createAndStartThreads(ProgramArgs* cli) {
  uint64_t patternsPerThread = ((cli->endAt > 0) ? cli->endAt - cli->beginAt : ULONG_MAX) / cli->cpuThreads;

  auto context = std::make_unique<ThreadContext>(cli->cpuThreads);

  // Start threads
  for (int t = 0; t < cli->cpuThreads; ++t) {
    context->threadArgs.emplace_back(createThreadArgs(cli, t, patternsPerThread));
    pthread_create(&context->threads[t], NULL, search, context->threadArgs[t].get());
  }

  return context;
}


static void joinAndCleanupThreads(std::unique_ptr<ThreadContext>& context) {
  for (int t = 0; t < context->numThreads; ++t) {
    pthread_join(context->threads[t], NULL);
    printThreadStatus(t, "COMPLETE");
  }
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

    // Debug: Check first few frames manually
    std::cout << "Debug: Testing first 10 frames manually:\n";
    for (uint64_t i = 0; i < 10; i++) {
      bool isComplete = googleIsFrameCompleteFromCache(i);
      std::cout << "  Frame " << i << ": " << (isComplete ? "complete" : "incomplete") << "\n";
    }

    // Debug: Check some frames that you know should be complete
    std::cout << "Debug: Testing some potentially complete frames:\n";
    uint64_t testFrames[] = {0, 1, 100, 1000, 10000, 100000};
    for (uint64_t frame : testFrames) {
      bool isComplete = googleIsFrameCompleteFromCache(frame);
      std::cout << "  Frame " << frame << ": " << (isComplete ? "complete" : "incomplete") << "\n";
    }

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
                  << progress << "%), "
                  << "completed so far: " << completedFrames << "\n";
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

  // Handle test-google-api flag
  if (cli->testGoogleApi) {
    std::cout << "Testing Google Sheets API with fake data...\n";

    // Generate realistic but varying test data based on current time
    time_t now = time(NULL);
    srand(now);  // Seed random number generator with current time

    // Generate realistic progress data
    uint64_t testFrameId = 1000000 + (rand() % 1000000);  // Frame IDs in realistic range
    int testKernelId = rand() % 16;                       // Kernel IDs 0-15
    int testChunkId = rand() % 100;                       // Chunk IDs 0-99
    double testRate = 500000.0 + (rand() % 1000000);      // Realistic patterns/sec rate

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
              << ", chunkIdx=" << testChunkId << ", rate=" << testRate << ", generations=" << testGenerations
              << ", pattern=" << std::hex << testPattern << std::dec << ")...\n";
    std::string progressResponse =
        googleSendProgressWithResponse(false, testFrameId, testKernelId, testChunkId, (uint64_t)testRate,
                                       testGenerations, testPattern, testPatternBin, true);
    std::cout << "Progress upload response: " << progressResponse << "\n";

    // Test querying best result
    int bestResult = googleGetBestResult();
    std::cout << "Best result query: " << bestResult << "\n";

    // Test querying best complete frame
    std::cout << "Testing best complete frame query...\n";
    uint64_t bestFrame = googleGetBestCompleteFrame();
    if (bestFrame == ULLONG_MAX) {
      std::cout << "Best complete frame query: no completed frames found\n";
    } else {
      std::cout << "Best complete frame query result: " << bestFrame << "\n";
    }

    // Cleanup and exit
    cleanupProgramArgs(cli);
    googleCleanup();
    // Always return success - Google Sheets errors shouldn't fail the program
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

  // Print resume message if frame-based search is being used
  if (cli->frameBeginIdx > 0 && cli->frameEndIdx > 0) {
    Logging::out() << "Resuming from frame: " << cli->frameBeginIdx << "\n";
  } else if (cli->frameBeginIdx == 0 && cli->frameEndIdx > 0) {
    Logging::out() << "Starting from frame 0\n";
  }

  // Create and start threads, then wait for completion
  auto context = createAndStartThreads(cli);
  joinAndCleanupThreads(context);
  cleanupProgramArgs(cli);

  // Cleanup Google client
  googleCleanup();

  return 0;
}
