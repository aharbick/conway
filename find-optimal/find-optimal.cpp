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
  { "beginat", 'b', "num", 0, "Explicit beginAt."},
  { "endat", 'e', "num", 0, "Explicit endAt."},
  { "random", 'r', NULL, 0, "Use random patterns."},
  { "randomsamples", 's', "num", 0, "How many random samples to run. Default is 1 billion."},
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
  case 'b':
    a->beginAt = strtoull(arg, NULL, 10);
    break;
  case 'e':
    a->endAt = strtoull(arg, NULL, 10);
    break;
  case 'r':
    a->random = true;
    break;
  case 's': {
    a->randomSamples = strtoull(arg, NULL, 10);
    break;
  }
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
  cli->blockSize = 1024;
  cli->threadsPerBlock = 1024;
  cli->beginAt = 0;
  cli->endAt = 0;
  cli->random = false;
  cli->randomSamples = 1ULL << 30; // 1 billion
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

