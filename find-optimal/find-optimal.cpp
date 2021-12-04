#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <pthread.h>
#include <limits.h>
#include <unistd.h>
#include <string.h>
#include <argp.h>

#include "cli.h"
#include "utils.h"

#ifndef HAS_CUDA
#include "gol.h"
#else
#include "gol_cuda.h"
#endif

const char *prog = "find-optimal v0.1";
const char *prog_bug_email = "aharbick@aharbick.com";
static char prog_doc[] = "Search for terminal and stable states in an 8x8 bounded Conway's Game of Life grid";
static char prog_args_doc[] = "";
static struct argp_option argp_options[] = {
  { "cudaconfig", 'c', "config", 0, "CUDA kernel params numgpus:blocksize:threadsperblock (e.g. 1:4096:256)"},
  { "threads", 't', "num", 0, "Number of CPU threads (if you use more than one GPU you should use matching threads)."},
  { "beginat", 'b', "num", 0, "Explicit beginAt."},
  { "endat", 'e', "num", 0, "Explicit endAt."},
  { "cachingfilter", 'f', "y,x", 0, "Used for caching (e.g. 5x7)" },
  { "random", 'r', "ignorerange", OPTION_ARG_OPTIONAL, "Use random patterns. Default in [beginAt-endAt]. -r1 [1-ULONG_MAX]."},
  { 0 }
};

static error_t parse_argp_options(int key, char *arg, struct argp_state *state) {
  prog_args *a = (prog_args *)state->input;
  char *end;
  switch(key) {
  case 'c':
    a->gpusToUse = strtol(arg, &end, 10);
    if (*end == ':') {
      a->blockSize = strtol(end+1, &end, 10);
    }
    else {
      printf("[WARN] invalid cudaconfig '%s', using default (4096) for blockSize\n", arg);
      a->blockSize = 4096;
    }
    if (*end == ':') {
      a->threadsPerBlock = strtol(end+1, NULL, 10);
    }
    else {
      printf("[WARN] invalid cudaconfig '%s', using default (256) for threadsPerBlock\n", arg);
      a->threadsPerBlock = 256;
    }
    break;
  case 't':
    a->cpuThreads = strtol(arg, NULL, 10);
    break;
  case 'b':
    a->beginAt = strtoull(arg, NULL, 10);
    break;
  case 'e':
    a->endAt = strtoull(arg, NULL, 10);
    break;
  case 'f':
    a->filter.y = strtol(arg, &end, 10);
    if (*end == ',') {
      a->filter.x = strtol(end+1, &end, 10);
    }
    else {
      printf("[WARN] invalid cachefilter '%s', using default (5x7) for filter\n", arg);
      a->filter.x = 7;
      a->filter.y = 5;
    }
    break;
  case 'r':
    a->random = true;
    if (arg) {
      a->unrestrictedRandom = true;
    }
    break;
  default: return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

struct argp argp = {argp_options, parse_argp_options, prog_args_doc, prog_doc, 0, 0};

int main(int argc, char **argv) {
  // Change stdout to not buffered
  setvbuf(stdout, NULL, _IONBF, 0);

  // Process the arguments
  prog_args *cli = (prog_args *) malloc(sizeof(prog_args));
  cli->cpuThreads = 1;
  cli->gpusToUse = 1;
  cli->blockSize = 4096;
  cli->threadsPerBlock = 256;
  cli->filter.x = 4;
  cli->filter.y = 4;
  cli->beginAt = 0;
  cli->endAt = 0;
  cli->random = false;
  cli->unrestrictedRandom = false;
  argp_parse(&argp, argc, argv, 0, 0, cli);

  // Allocate an array of threads
  ulong64 patternsPerThread = ((cli->endAt > 0) ? cli->endAt - cli->beginAt : ULONG_MAX) / cli->cpuThreads;
  pthread_t *threads = (pthread_t *) malloc(sizeof(pthread_t) * cli->cpuThreads);

  for (int t = 0; t < cli->cpuThreads; t++) {
    // Spin up a thread per gpu
    prog_args *targs = (prog_args *) malloc(sizeof(prog_args));
    memcpy(targs, cli, sizeof(prog_args));
    targs->threadId = t;
    targs->beginAt = (cli->beginAt > 0) ? cli->beginAt : t * patternsPerThread + 1;
    targs->endAt = targs->beginAt + patternsPerThread -1;
    pthread_create(&threads[t], NULL, search, (void*) targs);
  }

  for (int t = 0; t < cli->cpuThreads; t++) {
    pthread_join(threads[t], NULL);
    printf("[Thread %d] COMPLETE\n", t);
  }
}
