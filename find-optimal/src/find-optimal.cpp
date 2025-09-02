#include <locale.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include "cli_parser.h"
#include "gol.h"

// Thread context structure
typedef struct {
  pthread_t* threads;
  ProgramArgs** threadArgs;
  int numThreads;
} ThreadContext;

#ifdef __NVCC__
static void printCudaDeviceInfo(ProgramArgs* cli) {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  printf("CUDA devices available: %d\n", deviceCount);
  printf("Using %d GPU(s) with blockSize=%d, threadsPerBlock=%d\n", cli->gpusToUse, cli->blockSize,
         cli->threadsPerBlock);
}
#endif

static ProgramArgs* createThreadArgs(ProgramArgs* cli, int threadId, uint64_t patternsPerThread) {
  ProgramArgs* targs = (ProgramArgs*)malloc(sizeof(ProgramArgs));
  memcpy(targs, cli, sizeof(ProgramArgs));
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

static ThreadContext* createAndStartThreads(ProgramArgs* cli) {
  uint64_t patternsPerThread = ((cli->endAt > 0) ? cli->endAt - cli->beginAt : ULONG_MAX) / cli->cpuThreads;

  ThreadContext* context = (ThreadContext*)malloc(sizeof(ThreadContext));
  context->threads = (pthread_t*)malloc(sizeof(pthread_t) * cli->cpuThreads);
  context->threadArgs = (ProgramArgs**)malloc(sizeof(ProgramArgs*) * cli->cpuThreads);
  context->numThreads = cli->cpuThreads;

  // Start threads
  for (int t = 0; t < cli->cpuThreads; t++) {
    context->threadArgs[t] = createThreadArgs(cli, t, patternsPerThread);
    pthread_create(&context->threads[t], NULL, search, context->threadArgs[t]);
  }

  return context;
}


static void joinAndCleanupThreads(ThreadContext* context) {
  for (int t = 0; t < context->numThreads; t++) {
    pthread_join(context->threads[t], NULL);
    printThreadStatus(t, "COMPLETE");
    free(context->threadArgs[t]);
  }
  free(context->threads);
  free(context->threadArgs);
  free(context);
}

int main(int argc, char** argv) {
  // Set locale for number formatting with thousands separators
  setlocale(LC_NUMERIC, "");

  // Change stdout to not buffered
  setvbuf(stdout, NULL, _IONBF, 0);

  // Initialize Airtable client
  airtableInit();

  // Process the arguments using CLI parser
  ProgramArgs* cli = parseCommandLineArgs(argc, argv);
  if (!cli) {
    printf("[ERROR] Failed to parse command line arguments\n");
    airtableCleanup();
    return 1;
  }

  // Handle test-airtable flag
  if (cli->testAirtable) {
    printf("Testing Airtable API with fake data...\n");

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
    printf(
        "Testing progress upload (frameIdx=%llu, kernelIdx=%d, chunkIdx=%d, rate=%.0f, generations=%d, "
        "pattern=%llX)...\n",
        testFrameId, testKernelId, testChunkId, testRate, testGenerations, testPattern);
    bool sendProgressResult = airtableSendProgress(false, testFrameId, testKernelId, testChunkId, (uint64_t)testRate,
                                                   testGenerations, testPattern, testPatternBin, true);
    printf("Progress upload %s\n", sendProgressResult ? "succeeded" : "failed");

    // Test querying best result
    int bestResult = airtableGetBestResult();
    printf("Best result query: %d\n", bestResult);

    // Test querying best complete frame
    printf("Testing best complete frame query...\n");
    uint64_t bestFrame = airtableGetBestCompleteFrame();
    if (bestFrame == ULLONG_MAX) {
      printf("Best complete frame query: no completed frames found\n");
    } else {
      printf("Best complete frame query result: %llu\n", bestFrame);
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
    printf("Best generations so far: %d\n", gBestGenerations);
  }

  // Print resume message if frame-based search is being used
  if (cli->frameBeginIdx > 0 && cli->frameEndIdx > 0) {
    printf("Resuming from frame: %lu\n", cli->frameBeginIdx);
  } else if (cli->frameBeginIdx == 0 && cli->frameEndIdx > 0) {
    printf("Starting from frame 0\n");
  }

  // Create and start threads, then wait for completion
  ThreadContext* context = createAndStartThreads(cli);
  joinAndCleanupThreads(context);
  cleanupProgramArgs(cli);

  // Cleanup Airtable client
  airtableCleanup();

  return 0;
}
