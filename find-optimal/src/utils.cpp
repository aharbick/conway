#include "utils.h"

#include <iostream>
#include <sstream>

#include "logging.h"
#include "platform_compat.h"

// Time utilities for performance measurement
__host__ double getHighResCurrentTime() {
  return getPlatformHighResTime();
}