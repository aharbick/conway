#include "cli_parser.h"

#include <argp.h>
#include <errno.h>
#include <limits.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "airtable_client.h"
#include "constants.h"

// Program information
const char* prog = "find-optimal v0.1";
const char* prog_bug_email = "aharbick@aharbick.com";
static char prog_doc[] = "Search for terminal and stable states in an 8x8 bounded Conway's Game of Life grid";
static char ProgramArgs_doc[] = "";

// Command line options
static struct argp_option argp_options[] = {
    {"cudaconfig", 'c', "config", 0, "CUDA kernel params numgpus:blocksize:threadsperblock (e.g. 1:1024:1024)"},
    {"threads", 't', "num", 0, "Number of CPU threads (if you use more than one GPU you should use matching threads)."},
#ifndef __NVCC__
    {"range", 'r', "BEGIN[:END]", 0, "Range to search (e.g., 1: or 1:1012415). Default end is ULONG_MAX."},
#endif
    {"frame-index-range", 'f', "BEGIN[:END]", 0,
     "Frame index range to search from 0 to 2102800 or 'resume' (e.g. 0: or 1:1234 or resume)"},
    {"chunk-size", 'k', "size", 0, "Chunk size for pattern processing (default: 65536)."},
    {"random", 'R', "samples", 0, "Random search mode with specified number of samples."},
    {"verbose", 'v', 0, 0, "Verbose output."},
    {"test-airtable", 'T', 0, 0, "Test Airtable API functionality and exit."},
    {0}};

// Validation functions
static bool validatePositiveInteger(long value, const char* name, const char* str) {
  if (value <= 0) {
    printf("[ERROR] %s '%s' must be positive\n", name, str);
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

static bool validateRange(uint64_t begin, uint64_t end, const char* beginStr, const char* endStr) {
  if (end <= begin) {
    printf("[WARN] invalid range '%s:%s', end must be greater than begin\n", beginStr, endStr);
    return false;
  }
  return true;
}

static bool parseCudaConfig(const char* arg, ProgramArgs* a) {
  if (!arg || *arg == '\0') {
    printf("[ERROR] CUDA config cannot be empty\n");
    return false;
  }

  char* argCopy = strdup(arg);
  if (!argCopy) {
    printf("[ERROR] Memory allocation failed\n");
    return false;
  }

  char* saveptr;
  char* gpuStr = strtok_r(argCopy, ":", &saveptr);

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

  char* blockStr = strtok_r(NULL, ":", &saveptr);
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
  }

  // Handle threadsPerBlock if present, or skip to use default
  if (blockStr) {
    char* threadsStr = strtok_r(NULL, ":", &saveptr);
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
    }
  }

  free(argCopy);
  return true;
}

static bool parseRangeArg(const char* arg, uint64_t* begin, uint64_t* end) {
  if (!arg || *arg == '\0') {
    printf("[ERROR] Range argument cannot be empty\n");
    return false;
  }

  // Handle special 'resume' case
  if (strcmp(arg, "resume") == 0) {
    uint64_t resumeFrame = airtableGetBestCompleteFrame();
    *begin = (resumeFrame == ULLONG_MAX) ? 0 : resumeFrame + 1;
    *end = FRAME_SEARCH_TOTAL_MINIMAL_FRAMES;
    return true;
  }

  char* argCopy = strdup(arg);
  if (!argCopy) {
    printf("[ERROR] Memory allocation failed\n");
    return false;
  }

  char* colon = strchr(argCopy, ':');
  if (colon == NULL) {
    printf("[ERROR] invalid range format '%s', expected BEGIN: or BEGIN:END or 'resume'\n", arg);
    free(argCopy);
    return false;
  }

  *colon = '\0';  // Split the string

  // Validate begin value
  if (!validateUnsignedLongString(argCopy, "range begin")) {
    free(argCopy);
    return false;
  }

  *begin = strtoull(argCopy, NULL, 10);

  // Check if end is specified
  if (*(colon + 1) == '\0') {
    *end = 0;  // No end specified
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

static error_t parseArgpOptions(int key, char* arg, struct argp_state* state) {
  ProgramArgs* a = (ProgramArgs*)state->input;

  switch (key) {
  case 'c':
    if (!parseCudaConfig(arg, a)) {
      argp_failure(state, 1, 0, "Invalid CUDA configuration");
    }
    break;
  case 't':
    if (!validateIntegerString(arg, "threads")) {
      argp_failure(state, 1, 0, "Invalid thread count");
    }
    a->cpuThreads = strtol(arg, NULL, 10);
    if (!validatePositiveInteger(a->cpuThreads, "threads", arg)) {
      argp_failure(state, 1, 0, "Thread count must be positive");
    }
    break;
  case 'r':
#ifndef __NVCC__
    if (!parseRangeArg(arg, &a->beginAt, &a->endAt)) {
      argp_failure(state, 1, 0, "Invalid range specification");
    }
#endif
    break;
  case 'f':
    if (!parseRangeArg(arg, &a->frameBeginIdx, &a->frameEndIdx)) {
      argp_failure(state, 1, 0, "Invalid frame index range");
    }
    break;
  case 'k':
    if (!validateIntegerString(arg, "chunk-size")) {
      argp_failure(state, 1, 0, "Invalid chunk size");
    }
    a->chunkSize = strtol(arg, NULL, 10);
    if (!validatePositiveInteger(a->chunkSize, "chunk-size", arg)) {
      argp_failure(state, 1, 0, "Chunk size must be positive");
    }
    break;
  case 'R':
    a->random = true;
    if (!validateIntegerString(arg, "random-samples")) {
      argp_failure(state, 1, 0, "Invalid random sample count");
    }
    a->randomSamples = strtoull(arg, NULL, 10);
    if (a->randomSamples == 0) {
      argp_failure(state, 1, 0, "Random samples must be positive");
    }
    break;
  case 'v':
    a->verbose = true;
    break;
  case 'T':
    a->testAirtable = true;
    break;
  default:
    return ARGP_ERR_UNKNOWN;
  }

  return 0;
}

struct argp argp = {argp_options, parseArgpOptions, ProgramArgs_doc, prog_doc, 0, 0};

void initializeDefaultArgs(ProgramArgs* args) {
  args->threadId = 0;
  args->cpuThreads = 1;
  args->gpusToUse = 1;
  args->blockSize = 1024;
  args->threadsPerBlock = 1024;
  args->random = false;
  args->verbose = false;
  args->testAirtable = false;
  args->resumeFromDatabase = false;
  args->randomSamples = 0;
  args->beginAt = 1;
  args->endAt = 0;
  args->frameBeginIdx = 0;
  args->frameEndIdx = 0;
  args->chunkSize = FRAME_SEARCH_DEFAULT_CHUNK_SIZE;
}

ProgramArgs* parseCommandLineArgs(int argc, char** argv) {
  ProgramArgs* args = (ProgramArgs*)malloc(sizeof(ProgramArgs));
  if (!args) {
    printf("[ERROR] Memory allocation failed\n");
    return nullptr;
  }

  initializeDefaultArgs(args);

  if (argp_parse(&argp, argc, argv, 0, 0, args) != 0) {
    free(args);
    return nullptr;
  }

  return args;
}

void cleanupProgramArgs(ProgramArgs* args) {
  if (args) {
    free(args);
  }
}