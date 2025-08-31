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

#include "gol.h"

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
  { "frame-range", 'f', "BEGIN[:END]", 0, "Frame range to search (e.g., 1: or 1:12515). Default end is 2102800."},
  { "chunk-size", 'k', "size", 0, "Chunk size for pattern processing (default: 32768)."},
  { "verbose", 'v', NULL, 0, "Enable verbose output."},
  { "random", 'R', NULL, 0, "Use random patterns."},
  { "randomsamples", 's', "num", 0, "How many random samples to run. Default is 1 billion."},
  { 0 }
};

static error_t parse_argp_options(int key, char *arg, struct argp_state *state) {
  prog_args *a = (prog_args *)state->input;
  char *end;
  switch(key) {
  case 'c':
    a->gpusToUse = strtol(arg, &end, 10);
    if (a->gpusToUse <= 0) {
      printf("[WARN] invalid gpusToUse '%s', must be greater than 0\n", arg);
      return ARGP_ERR_UNKNOWN;
    }
    if (*end == ':') {
      a->blockSize = strtol(end+1, &end, 10);
      if (a->blockSize <= 0) {
        printf("[WARN] invalid blockSize '%s', must be greater than 0\n", end+1);
        return ARGP_ERR_UNKNOWN;
      }
    }
    else {
      printf("[WARN] invalid cudaconfig '%s', using default (1024) for blockSize\n", arg);
      a->blockSize = 1024;
    }
    if (*end == ':') {
      a->threadsPerBlock = strtol(end+1, NULL, 10);
    }
    else {
      printf("[WARN] invalid cudaconfig '%s', using default (1024) for threadsPerBlock\n", arg);
      a->threadsPerBlock = 1024;
    }
    break;
  case 't':
    a->cpuThreads = strtol(arg, NULL, 10);
    break;
#ifndef __NVCC__
  case 'r': {
    char *colon = strchr(arg, ':');
    if (colon == NULL) {
      printf("[WARN] invalid range format '%s', expected BEGIN: or BEGIN:END\n", arg);
      return ARGP_ERR_UNKNOWN;
    }
    *colon = '\0';
    a->beginAt = strtoull(arg, NULL, 10);
    if (*(colon + 1) == '\0') {
      // End not specified, use default
      a->endAt = ULONG_MAX;
    } else {
      a->endAt = strtoull(colon + 1, NULL, 10);
      if (a->endAt <= a->beginAt) {
        printf("[WARN] invalid range '%s:%s', end must be greater than begin\n", arg, colon + 1);
        return ARGP_ERR_UNKNOWN;
      }
    }
    break;
  }
#endif
  case 'f': {
    char *colon = strchr(arg, ':');
    if (colon == NULL) {
      printf("[WARN] invalid frame-range format '%s', expected BEGIN: or BEGIN:END\n", arg);
      return ARGP_ERR_UNKNOWN;
    }
    *colon = '\0';
    a->frameBeginAt = strtoull(arg, NULL, 10);
    if (*(colon + 1) == '\0') {
      // End not specified, use default
      a->frameEndAt = 2102800;
    } else {
      a->frameEndAt = strtoull(colon + 1, NULL, 10);
      if (a->frameEndAt <= a->frameBeginAt) {
        printf("[WARN] invalid frame-range '%s:%s', end must be greater than begin\n", arg, colon + 1);
        return ARGP_ERR_UNKNOWN;
      }
    }
    break;
  }
  case 'v':
    a->verbose = true;
    break;
  case 'R':
    a->random = true;
    break;
  case 's': {
    a->randomSamples = strtoull(arg, NULL, 10);
    break;
  }
  case 'k': {
    a->chunkSize = strtoull(arg, NULL, 10);
    if (a->chunkSize == 0 || a->chunkSize > 65536) {
      printf("[WARN] invalid chunk-size '%s', must be between 1 and 65536\n", arg);
      return ARGP_ERR_UNKNOWN;
    }
    break;
  }
  default: return ARGP_ERR_UNKNOWN;
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
  cli->cpuThreads = 1;
  cli->gpusToUse = 1;
  cli->blockSize = 1024;
  cli->threadsPerBlock = 1024;
  cli->beginAt = 0;
  cli->endAt = 0;
  cli->frameBeginAt = 0;
  cli->frameEndAt = 0;
  cli->random = false;
  cli->verbose = false;
  cli->randomSamples = ULONG_MAX;
  cli->chunkSize = 32768;
  argp_parse(&argp, argc, argv, 0, 0, cli);

#ifdef __NVCC__
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Running on GPU: %s\n", prop.name);
  printf("Compute capability: %d.%d\n", prop.major, prop.minor);
  printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
  printf("Max thread dimensions: [%d, %d, %d]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
  printf("Using %d blocks with %d threads per block\n", cli->blockSize, cli->threadsPerBlock);
#endif

  // Allocate an array of threads
  ulong64 patternsPerThread = ((cli->endAt > 0) ? cli->endAt - cli->beginAt : ULONG_MAX) / cli->cpuThreads;
  pthread_t *threads = (pthread_t *) malloc(sizeof(pthread_t) * cli->cpuThreads);

  for (int t = 0; t < cli->cpuThreads; t++) {
    // Spin up a thread per gpu
    prog_args *targs = (prog_args *) malloc(sizeof(prog_args));
    memcpy(targs, cli, sizeof(prog_args));
    targs->threadId = t;
#ifndef __NVCC__
    targs->beginAt = (cli->beginAt > 0) ? cli->beginAt : t * patternsPerThread + 1;
    targs->endAt = targs->beginAt + patternsPerThread -1;
#endif
    pthread_create(&threads[t], NULL, search, (void*) targs);
  }

  for (int t = 0; t < cli->cpuThreads; t++) {
    pthread_join(threads[t], NULL);
    printf("\n[Thread %d - %llu] COMPLETE\n", t, (ulong64)time(NULL));
  }
}

