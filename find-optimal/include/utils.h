#ifndef _UTILS_H_
#define _UTILS_H_

#include <stdarg.h>

#include <cstdint>
#include <cstdio>
#include <ctime>
#include <sstream>
#include <string>

#include "cuda_utils.h"

// Time utilities for performance measurement
__host__ double getHighResCurrentTime();

// Number formatting utilities
inline std::string formatWithCommas(uint64_t value) {
  std::ostringstream oss;
  oss.imbue(std::locale(""));
  oss << value;
  return oss.str();
}

#endif