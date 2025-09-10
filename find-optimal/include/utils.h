#ifndef _UTILS_H_
#define _UTILS_H_

#include <stdarg.h>

#include <cstdint>
#include <cstdio>
#include <ctime>

#include "cuda_utils.h"

// Time utilities for performance measurement
__host__ double getHighResCurrentTime();

#endif