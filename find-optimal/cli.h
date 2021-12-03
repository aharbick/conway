#ifndef _CLI_H_
#define _CLI_H_

#include "utils.h"

struct caching_filter {
  int x;
  int y;
};

typedef struct prog_args {
  int threadId;
  int cpuThreads;
  int gpusToUse;
  int blockSize;
  int threadsPerBlock;
  struct caching_filter filter;
  bool random;
  ulong64 beginAt;
  ulong64 endAt;
} prog_args;

#endif
