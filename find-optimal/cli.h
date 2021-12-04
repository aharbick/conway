#ifndef _CLI_H_
#define _CLI_H_

#include "utils.h"

typedef struct prog_args {
  int threadId;
  int cpuThreads;
  int gpusToUse;
  int blockSize;
  int threadsPerBlock;
  bool random;
  bool unrestrictedRandom;
  ulong64 beginAt;
  ulong64 endAt;
} prog_args;

#endif
