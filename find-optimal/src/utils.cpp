#include "utils.h"

#include <iostream>
#include <sstream>

#include "logging.h"

// Time utilities for performance measurement
__host__ double getHighResCurrentTime() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec / 1000000000.0;
}