#include "utils.h"

// Time utilities for performance measurement
__host__ double getCurrentTime() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec / 1000000000.0;
}

// Thread status and logging utilities
__host__ void printThreadStatus(int threadId, const char *format, ...) {
  va_list args;
  va_start(args, format);
  printf("[Thread %d - %lu] ", threadId, (uint64_t)time(NULL));
  vprintf(format, args);
  printf("\n");
  va_end(args);
}