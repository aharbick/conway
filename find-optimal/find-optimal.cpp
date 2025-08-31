#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <pthread.h>
#include <limits.h>
#include <unistd.h>
#include <string.h>
#include <argp.h>
#include <sys/time.h>
#include <locale.h>
#include <errno.h>

#include "gol.h"

#define STRINGIFY_CONSTANT(x) #x

const char *prog = "find-optimal v0.1";
const char *prog_bug_email = "aharbick@aharbick.com";
static char prog_doc[] = "Search for terminal and stable states in an 8x8 bounded Conway's Game of Life grid";
static char prog_args_doc[] = "";
static struct argp_option argp_options[] = {
  { "cudaconfig", 'c', "config", 0, "CUDA kernel params numgpus:blocksize:threadsperblock (e.g. 1:1024:1024)"},
  { "threads", 't', "num", 0, "Number of CPU threads (if you use more than one GPU you should use matching threads)."},
#ifndef __NVCC__
  { "range", 'r', "BEGIN[:END]", 0, "Range to search (e.g., 1: or 1:1012415). Default end is ULONG_MAX."},
#endif
  { "frame-range", 'f', "BEGIN[:END]", 0, "Frame range to search (e.g., 1: or 1:12515). Default end is " STRINGIFY_CONSTANT(FRAME_SEARCH_TOTAL_FRAMES) "."},
  { "chunk-size", 'k', "size", 0, "Chunk size for pattern processing (default: 32768)."},
  { "verbose", 'v', NULL, 0, "Enable verbose output."},
  { "random", 'R', NULL, 0, "Use random patterns."},
  { "randomsamples", 's', "num", 0, "How many random samples to run. Default is 1 billion."},
  { 0 }
};

static bool validatePositiveInteger(long value, const char* name, const char* input) {
  if (value <= 0) {
    printf("[ERROR] invalid %s '%s', must be greater than 0\n", name, input);
    return false;
  }
  return true;
}

static bool validateIntegerString(const char* str, const char* name) {
  if (!str || *str == '\0') {
    printf("[ERROR] %s cannot be empty\n", name);
    return false;
  }

  char* end;
  errno = 0;
  long value = strtol(str, &end, 10);

  if (errno == ERANGE || value == LONG_MAX || value == LONG_MIN) {
    printf("[ERROR] %s '%s' is out of range\n", name, str);
    return false;
  }

  if (*end != '\0') {
    printf("[ERROR] %s '%s' contains invalid characters\n", name, str);
    return false;
  }

  return true;
}

static bool validateUnsignedLongString(const char* str, const char* name) {
  if (!str || *str == '\0') {
    printf("[ERROR] %s cannot be empty\n", name);
    return false;
  }

  char* end;
  errno = 0;
  unsigned long long value = strtoull(str, &end, 10);

  if (errno == ERANGE || value == ULLONG_MAX) {
    printf("[ERROR] %s '%s' is out of range\n", name, str);
    return false;
  }

  if (*end != '\0') {
    printf("[ERROR] %s '%s' contains invalid characters\n", name, str);
    return false;
  }

  return true;
}

static bool validateRange(ulong64 begin, ulong64 end, const char* beginStr, const char* endStr) {
  if (end <= begin) {
    printf("[WARN] invalid range '%s:%s', end must be greater than begin\n", beginStr, endStr);
    return false;
  }
  return true;
}

static bool parseCudaConfig(const char *arg, prog_args *a) {
  if (!arg || *arg == '\0') {
    printf("[ERROR] CUDA config cannot be empty\n");
    return false;
  }

  char *argCopy = strdup(arg);  // Make a copy to avoid modifying original
  if (!argCopy) {
    printf("[ERROR] Memory allocation failed\n");
    return false;
  }

  char *saveptr;
  char *gpuStr = strtok_r(argCopy, ":", &saveptr);

  if (!gpuStr) {
    printf("[ERROR] Invalid CUDA config format '%s', expected numgpus:blocksize:threadsperblock\n", arg);
    free(argCopy);
    return false;
  }

  if (!validateIntegerString(gpuStr, "gpusToUse")) {
    free(argCopy);
    return false;
  }

  a->gpusToUse = strtol(gpuStr, NULL, 10);
  if (!validatePositiveInteger(a->gpusToUse, "gpusToUse", gpuStr)) {
    free(argCopy);
    return false;
  }

  char *blockStr = strtok_r(NULL, ":", &saveptr);
  if (blockStr) {
    if (!validateIntegerString(blockStr, "blockSize")) {
      free(argCopy);
      return false;
    }
    a->blockSize = strtol(blockStr, NULL, 10);
    if (!validatePositiveInteger(a->blockSize, "blockSize", blockStr)) {
      free(argCopy);
      return false;
    }
  } else {
    printf("[WARN] blockSize not specified in '%s', using default (%d)\n", arg, DEFAULT_CUDA_GRID_SIZE);
    a->blockSize = DEFAULT_CUDA_GRID_SIZE;
  }

  char *threadsStr = strtok_r(NULL, ":", &saveptr);
  if (threadsStr) {
    if (!validateIntegerString(threadsStr, "threadsPerBlock")) {
      free(argCopy);
      return false;
    }
    a->threadsPerBlock = strtol(threadsStr, NULL, 10);
    if (!validatePositiveInteger(a->threadsPerBlock, "threadsPerBlock", threadsStr)) {
      free(argCopy);
      return false;
    }
  } else {
    printf("[WARN] threadsPerBlock not specified in '%s', using default (%d)\n", arg, DEFAULT_CUDA_THREADS_PER_BLOCK);
    a->threadsPerBlock = DEFAULT_CUDA_THREADS_PER_BLOCK;
  }

  free(argCopy);
  return true;
}

static void initializeDefaultArgs(prog_args* cli) {
  cli->cpuThreads = 1;
  cli->gpusToUse = 1;
  cli->blockSize = DEFAULT_CUDA_GRID_SIZE;
  cli->threadsPerBlock = DEFAULT_CUDA_THREADS_PER_BLOCK;
  cli->beginAt = 0;
  cli->endAt = 0;
  cli->frameBeginAt = 0;
  cli->frameEndAt = 0;
  cli->random = false;
  cli->verbose = false;
  cli->randomSamples = ULONG_MAX;
  cli->chunkSize = FRAME_SEARCH_DEFAULT_CHUNK_SIZE;
}

#ifdef __NVCC__
static void printCudaDeviceInfo(prog_args* cli) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Running on GPU: %s\n", prop.name);
  printf("Compute capability: %d.%d\n", prop.major, prop.minor);
  printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
  printf("Max thread dimensions: [%d, %d, %d]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
  printf("Using %d blocks with %d threads per block\n", cli->blockSize, cli->threadsPerBlock);
}
#endif

static prog_args* createThreadArgs(prog_args* cli, int threadId, ulong64 patternsPerThread) {
  prog_args* targs = (prog_args*)malloc(sizeof(prog_args));
  memcpy(targs, cli, sizeof(prog_args));
  targs->threadId = threadId;
#ifndef __NVCC__
  targs->beginAt = (cli->beginAt > 0) ? cli->beginAt : threadId * patternsPerThread + 1;
  targs->endAt = targs->beginAt + patternsPerThread - 1;
#endif
  return targs;
}

typedef struct {
  pthread_t* threads;
  prog_args** threadArgs;
  int numThreads;
} thread_context_t;

static thread_context_t* createAndStartThreads(prog_args* cli) {
  ulong64 patternsPerThread = ((cli->endAt > 0) ? cli->endAt - cli->beginAt : ULONG_MAX) / cli->cpuThreads;

  thread_context_t* context = (thread_context_t*)malloc(sizeof(thread_context_t));
  context->threads = (pthread_t*)malloc(sizeof(pthread_t) * cli->cpuThreads);
  context->threadArgs = (prog_args**)malloc(sizeof(prog_args*) * cli->cpuThreads);
  context->numThreads = cli->cpuThreads;

  for (int t = 0; t < cli->cpuThreads; t++) {
    context->threadArgs[t] = createThreadArgs(cli, t, patternsPerThread);
    pthread_create(&context->threads[t], NULL, search, (void*)context->threadArgs[t]);
  }

  return context;
}

static void printThreadCompletion(int threadId, const char* status) {
  printf("\n[Thread %d - %llu] %s\n", threadId, (ulong64)time(NULL), status);
}

static void joinAndCleanupThreads(thread_context_t* context) {
  for (int t = 0; t < context->numThreads; t++) {
    pthread_join(context->threads[t], NULL);
    printThreadCompletion(t, "COMPLETE");
    free(context->threadArgs[t]);
  }
  free(context->threads);
  free(context->threadArgs);
  free(context);
}

static void cleanupProgArgs(prog_args* cli) {
  free(cli);
}

static bool parseRange(char *arg, ulong64 *begin, ulong64 *end, ulong64 defaultEnd) {
  if (!arg || *arg == '\0') {
    printf("[ERROR] range cannot be empty\n");
    return false;
  }

  char *argCopy = strdup(arg);  // Make a copy to avoid modifying original
  if (!argCopy) {
    printf("[ERROR] Memory allocation failed\n");
    return false;
  }

  char *colon = strchr(argCopy, ':');
  if (colon == NULL) {
    printf("[ERROR] invalid range format '%s', expected BEGIN: or BEGIN:END\n", arg);
    free(argCopy);
    return false;
  }

  *colon = '\0';

  // Validate begin value
  if (!validateUnsignedLongString(argCopy, "range begin")) {
    free(argCopy);
    return false;
  }
  *begin = strtoull(argCopy, NULL, 10);

  // Handle end value
  if (*(colon + 1) == '\0') {
    *end = defaultEnd;
  } else {
    if (!validateUnsignedLongString(colon + 1, "range end")) {
      free(argCopy);
      return false;
    }
    *end = strtoull(colon + 1, NULL, 10);
    if (!validateRange(*begin, *end, argCopy, colon + 1)) {
      free(argCopy);
      return false;
    }
  }

  free(argCopy);
  return true;
}

static error_t parse_argp_options(int key, char *arg, struct argp_state *state) {
  prog_args *a = (prog_args *)state->input;

  switch(key) {
  case 'c':
    if (!parseCudaConfig(arg, a)) {
      return ARGP_ERR_UNKNOWN;
    }
    break;

  case 't':
    if (!validateIntegerString(arg, "threads")) {
      return ARGP_ERR_UNKNOWN;
    }
    a->cpuThreads = strtol(arg, NULL, 10);
    if (!validatePositiveInteger(a->cpuThreads, "threads", arg)) {
      return ARGP_ERR_UNKNOWN;
    }
    break;

#ifndef __NVCC__
  case 'r':
    if (!parseRange(arg, &a->beginAt, &a->endAt, ULONG_MAX)) {
      return ARGP_ERR_UNKNOWN;
    }
    break;
#endif

  case 'f':
    if (!parseRange(arg, &a->frameBeginAt, &a->frameEndAt, FRAME_SEARCH_TOTAL_FRAMES)) {
      return ARGP_ERR_UNKNOWN;
    }
    break;

  case 'v':
    a->verbose = true;
    break;

  case 'R':
    a->random = true;
    break;

  case 's':
    if (!validateUnsignedLongString(arg, "randomsamples")) {
      return ARGP_ERR_UNKNOWN;
    }
    a->randomSamples = strtoull(arg, NULL, 10);
    if (a->randomSamples == 0) {
      printf("[ERROR] randomsamples must be greater than 0\n");
      return ARGP_ERR_UNKNOWN;
    }
    break;

  case 'k':
    if (!validateUnsignedLongString(arg, "chunk-size")) {
      return ARGP_ERR_UNKNOWN;
    }
    a->chunkSize = strtoull(arg, NULL, 10);
    if (a->chunkSize == 0 || a->chunkSize > FRAME_SEARCH_MAX_CHUNK_SIZE) {
      printf("[ERROR] invalid chunk-size '%s', must be between 1 and %d\n", arg, FRAME_SEARCH_MAX_CHUNK_SIZE);
      return ARGP_ERR_UNKNOWN;
    }
    break;

  default:
    return ARGP_ERR_UNKNOWN;
  }
  return 0;
}


struct argp argp = {argp_options, parse_argp_options, prog_args_doc, prog_doc, 0, 0};

int main(int argc, char **argv) {
  // Set locale for number formatting with thousands separators
  setlocale(LC_NUMERIC, "");

  // Change stdout to not buffered
  setvbuf(stdout, NULL, _IONBF, 0);

  // Process the arguments
  prog_args *cli = (prog_args *) malloc(sizeof(prog_args));
  initializeDefaultArgs(cli);
  argp_parse(&argp, argc, argv, 0, 0, cli);

#ifdef __NVCC__
  printCudaDeviceInfo(cli);
#endif

  // Create and start threads, then wait for completion
  thread_context_t *context = createAndStartThreads(cli);
  joinAndCleanupThreads(context);
  cleanupProgArgs(cli);

  return 0;
}

