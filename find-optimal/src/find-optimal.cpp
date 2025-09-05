#include <locale.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include <iostream>
#include <memory>
#include <vector>

#include "cli_parser.h"
#include "gol.h"

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
  std::cout << "CUDA devices available: " << deviceCount << "\n";
  std::cout << "Using " << cli->gpusToUse << " GPU(s) with blockSize=" << cli->blockSize
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

  // Initialize Airtable client
  airtableInit();

  // Process the arguments using CLI parser
  auto* cli = parseCommandLineArgs(argc, argv);
  if (!cli) {
    std::cerr << "[ERROR] Failed to parse command line arguments\n";
    airtableCleanup();
    return 1;
  }

  // Handle test-airtable flag
  if (cli->testAirtable) {
    std::cout << "Testing Airtable API with fake data...\n";

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
    bool sendProgressResult = airtableSendProgress(false, testFrameId, testKernelId, testChunkId, (uint64_t)testRate,
                                                   testGenerations, testPattern, testPatternBin, true);
    std::cout << "Progress upload " << (sendProgressResult ? "succeeded" : "failed") << "\n";

    // Test querying best result
    int bestResult = airtableGetBestResult();
    std::cout << "Best result query: " << bestResult << "\n";

    // Test querying best complete frame
    std::cout << "Testing best complete frame query...\n";
    uint64_t bestFrame = airtableGetBestCompleteFrame();
    if (bestFrame == ULLONG_MAX) {
      std::cout << "Best complete frame query: no completed frames found\n";
    } else {
      std::cout << "Best complete frame query result: " << bestFrame << "\n";
    }

    // Cleanup and exit
    cleanupProgramArgs(cli);
    airtableCleanup();
    return (sendProgressResult && bestResult >= 0) ? 0 : 1;
  }

#ifdef __NVCC__
  printCudaDeviceInfo(cli);
#endif

  // Initialize global best generations from Airtable database
  int dbBestGenerations = airtableGetBestResult();
  if (dbBestGenerations > 0) {
    gBestGenerations = dbBestGenerations;
    std::cout << "Best generations so far: " << gBestGenerations << "\n";
  }

  // Print resume message if frame-based search is being used
  if (cli->frameBeginIdx > 0 && cli->frameEndIdx > 0) {
    std::cout << "Resuming from frame: " << cli->frameBeginIdx << "\n";
  } else if (cli->frameBeginIdx == 0 && cli->frameEndIdx > 0) {
    std::cout << "Starting from frame 0\n";
  }

  // Create and start threads, then wait for completion
  auto context = createAndStartThreads(cli);
  joinAndCleanupThreads(context);
  cleanupProgramArgs(cli);

  // Cleanup Airtable client
  airtableCleanup();

  return 0;
}
