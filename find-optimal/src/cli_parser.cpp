#include "cli_parser.h"

#include <argp.h>
#include <errno.h>
#include <limits.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#include "constants.h"
#include "google_client.h"

// Program information
const char* prog = "find-optimal v0.1";
const char* prog_bug_email = "aharbick@aharbick.com";
static char prog_doc[] = "Search for terminal and stable states in an 8x8 bounded Conway's Game of Life grid";
static char ProgramArgs_doc[] = "";

// Command line options - organized by group for display order
// Group 1: Search modes (primary operations)
// Group 2: Search options (cache, algorithm settings)
// Group 3: Output/logging
// Group 4: Worker/queue configuration
// Group 5: Debug/testing utilities
static struct argp_option argp_options[] = {
    // Group 1: Search modes
    {0, 0, 0, 0, "Search modes:", 1},
    {"frame-mode", 'f', "MODE", 0, "Frame search mode: 'random', 'sequential', or 'index:XXXXX' (single frame by index)", 1},
    {"strip-mode", 'S', "MODE", OPTION_ARG_OPTIONAL, "Strip search mode: 'index:C[:M[-M2]]' (single center with optional middle range), 'range:C1[:M1]-C2[:M2]' (range). C=0-8547, M=0-511. Omit MODE for full search.", 1},
    {"7x7-mode", '7', "RANGE", OPTION_ARG_OPTIONAL, "Search 7x7 grid: 'start:end', ':end' (from 1), 'start:' (to 2^49), or omit for full search", 1},

    // Group 2: Search options
    {0, 0, 0, 0, "Search options:", 2},
    {"cycle-detection", 'D', "ALGORITHM", 0, "Cycle detection algorithm: 'floyd' or 'nivasch' (default: 'floyd')", 2},
    {"subgrid-cache-file", 'C', "FILE", 0, "Load 7x7 subgrid cache from FILE to use for early termination optimization.", 2},
    {"compute-subgrid-cache", 'c', "PATH", 0, "Compute 7x7 subgrid cache for all 2^49 patterns and save to disk at PATH.", 2},
    {"subgrid-cache-begin", 'b', "NUMBER", 0, "Starting pattern index for subgrid cache computation (for resuming).", 2},

    // Group 3: Output/logging
    {0, 0, 0, 0, "Output options:", 3},
    {"log-file", 'l', "PATH", 0, "Path to log file for progress output.", 3},
    {"dont-save-results", 'r', 0, 0, "Don't save results to Google Sheets (for testing/benchmarking).", 3},

    // Group 4: Worker/queue
    {0, 0, 0, 0, "Distributed processing:", 4},
    {"worker", 'w', "N:M", 0, "Worker configuration N:M where N is worker number (1-based) and M is total workers (default: 1:1).", 4},
    {"queue-directory", 'q', "PATH", 0, "Directory for persistent request queue (default: ./request-queue).", 4},
    {"drain-request-queue", 'Q', 0, 0, "Process all pending requests in queue and exit.", 4},

    // Group 5: Debug/testing
    {0, 0, 0, 0, "Debug and testing:", 5},
    {"simulate", 's', "TYPE", 0, "Simulate mode: 'pattern' (8x8 evolution), 'pattern:7x7' (7x7 evolution), or 'symmetry' (transformations).", 5},
    {"compare-cycle-algorithms", 'A', "FRAME_IDX", 0, "Compare Floyd's vs Nivasch's cycle detection on given frame index and exit.", 5},
    {"test-api", 'T', "TYPE", 0, "Test API functionality and exit: 'progress', 'summary', 'framecache', 'strip-progress', 'strip-summary', 'strip-cache', or 'strip-completion'.", 5},

    {0}};


static bool parseFrameMode(const char* arg, ProgramArgs* args) {
  std::string str(arg);

  if (str == "random") {
    args->frameMode = FRAME_MODE_RANDOM;
    return true;
  } else if (str == "sequential") {
    args->frameMode = FRAME_MODE_SEQUENTIAL;
    return true;
  }

  // Check for index:XXXXX format
  if (str.substr(0, 6) == "index:") {
    std::string frameIdxStr = str.substr(6);
    if (frameIdxStr.empty()) {
      std::cerr << "[ERROR] Frame mode 'index:' requires a frame index\n";
      return false;
    }

    // Validate that the frame index is a valid number
    try {
      uint64_t frameIdx = std::stoull(frameIdxStr);
      if (frameIdx >= FRAME_SEARCH_TOTAL_MINIMAL_FRAMES) {
        std::cerr << "[ERROR] Frame index " << frameIdx << " exceeds total frames ("
                  << FRAME_SEARCH_TOTAL_MINIMAL_FRAMES << ")\n";
        return false;
      }
      args->frameMode = FRAME_MODE_INDEX;
      args->frameModeIndex = frameIdx;
      return true;
    } catch (const std::exception&) {
      std::cerr << "[ERROR] Invalid frame index '" << frameIdxStr << "', expected a valid integer\n";
      return false;
    }
  }

  std::cerr << "[ERROR] Invalid frame mode '" << arg << "', expected 'random', 'sequential', or 'index:XXXXX'\n";
  return false;
}

static bool parseCycleDetection(const char* arg, ProgramArgs* args) {
  std::string str(arg);

  if (str == "floyd") {
    args->cycleDetection = CYCLE_DETECTION_FLOYD;
    return true;
  } else if (str == "nivasch") {
    args->cycleDetection = CYCLE_DETECTION_NIVASCH;
    return true;
  }

  std::cerr << "[ERROR] Invalid cycle detection algorithm '" << arg << "', expected 'floyd' or 'nivasch'\n";
  return false;
}

static bool parseStripMode(const char* arg, ProgramArgs* args) {
  // Center 4x4 symmetry reduction: 8548 unique centers (from Burnside's lemma)
  const uint32_t maxCenter = CENTER_4X4_TOTAL_UNIQUE;
  const uint32_t maxMiddle = STRIP_SEARCH_TOTAL_MIDDLE_IDX;  // 512

  // Handle no argument or empty string: default to full range
  if (!arg || *arg == '\0') {
    args->stripMode = STRIP_MODE_RANGE;
    args->centerIdxStart = 0;
    args->centerIdxEnd = maxCenter;
    args->middleIdxStart = 0;
    args->middleIdxEnd = maxMiddle;
    return true;
  }

  std::string str(arg);

  // Check for index:C[:M[-M2]] format (single center, optionally with middle index or range)
  if (str.substr(0, 6) == "index:") {
    std::string indexStr = str.substr(6);
    if (indexStr.empty()) {
      std::cerr << "[ERROR] Strip mode 'index:' requires a center index (0-" << maxCenter - 1 << ")\n";
      return false;
    }

    try {
      size_t colonPos = indexStr.find(':');
      uint64_t centerIdx;
      uint64_t middleIdxStart = 0;
      uint64_t middleIdxEnd = maxMiddle;
      bool hasMiddle = false;

      if (colonPos == std::string::npos) {
        // Just center: index:C
        centerIdx = std::stoull(indexStr);
      } else {
        // Center and middle: index:C:M or index:C:M1-M2
        centerIdx = std::stoull(indexStr.substr(0, colonPos));
        std::string middleStr = indexStr.substr(colonPos + 1);
        hasMiddle = true;

        size_t dashPos = middleStr.find('-');
        if (dashPos == std::string::npos) {
          // Single middle: index:C:M
          middleIdxStart = std::stoull(middleStr);
          middleIdxEnd = middleIdxStart + 1;
        } else {
          // Middle range: index:C:M1-M2
          middleIdxStart = std::stoull(middleStr.substr(0, dashPos));
          middleIdxEnd = std::stoull(middleStr.substr(dashPos + 1)) + 1;  // +1 for exclusive end
        }
      }

      if (centerIdx >= maxCenter) {
        std::cerr << "[ERROR] Center index " << centerIdx << " exceeds maximum (" << maxCenter - 1 << ")\n";
        return false;
      }
      if (hasMiddle && middleIdxStart >= maxMiddle) {
        std::cerr << "[ERROR] Middle index " << middleIdxStart << " exceeds maximum (" << maxMiddle - 1 << ")\n";
        return false;
      }
      if (hasMiddle && middleIdxEnd > maxMiddle) {
        std::cerr << "[ERROR] Middle end index " << (middleIdxEnd - 1) << " exceeds maximum (" << maxMiddle - 1 << ")\n";
        return false;
      }
      if (hasMiddle && middleIdxStart >= middleIdxEnd) {
        std::cerr << "[ERROR] Middle start must be less than or equal to middle end\n";
        return false;
      }

      args->stripMode = STRIP_MODE_INDEX;
      args->centerIdxStart = (uint32_t)centerIdx;
      args->centerIdxEnd = (uint32_t)centerIdx + 1;
      args->middleIdxStart = (uint32_t)middleIdxStart;
      args->middleIdxEnd = (uint32_t)middleIdxEnd;
      return true;
    } catch (const std::exception&) {
      std::cerr << "[ERROR] Invalid index format '" << indexStr << "', expected C, C:M, or C:M1-M2\n";
      return false;
    }
  }

  // Check for range:C1[:M1]-C2[:M2] format
  if (str.substr(0, 6) == "range:") {
    std::string rangeStr = str.substr(6);
    if (rangeStr.empty()) {
      std::cerr << "[ERROR] Strip mode 'range:' requires at least a start value\n";
      return false;
    }

    // Find the dash separator between start and end
    size_t dashPos = rangeStr.find('-');

    try {
      uint64_t startCenter, endCenter;
      uint64_t startMiddle = 0, endMiddle = maxMiddle;

      if (dashPos == std::string::npos) {
        // Only start provided: range:C1[:M1] (end defaults to max)
        std::string startStr = rangeStr;
        size_t colonPos = startStr.find(':');

        if (colonPos == std::string::npos) {
          startCenter = std::stoull(startStr);
        } else {
          startCenter = std::stoull(startStr.substr(0, colonPos));
          startMiddle = std::stoull(startStr.substr(colonPos + 1));
        }
        endCenter = maxCenter;
        endMiddle = maxMiddle;
      } else {
        // Both start and end: range:C1[:M1]-C2[:M2]
        std::string startStr = rangeStr.substr(0, dashPos);
        std::string endStr = rangeStr.substr(dashPos + 1);

        // Parse start (C1[:M1])
        size_t startColonPos = startStr.find(':');
        if (startColonPos == std::string::npos) {
          startCenter = std::stoull(startStr);
        } else {
          startCenter = std::stoull(startStr.substr(0, startColonPos));
          startMiddle = std::stoull(startStr.substr(startColonPos + 1));
        }

        // Parse end (C2[:M2])
        if (endStr.empty()) {
          endCenter = maxCenter;
          endMiddle = maxMiddle;
        } else {
          size_t endColonPos = endStr.find(':');
          if (endColonPos == std::string::npos) {
            endCenter = std::stoull(endStr);
            endMiddle = maxMiddle;
          } else {
            endCenter = std::stoull(endStr.substr(0, endColonPos));
            endMiddle = std::stoull(endStr.substr(endColonPos + 1));
          }
        }
      }

      // Validate bounds
      if (startCenter >= maxCenter) {
        std::cerr << "[ERROR] Start center " << startCenter << " exceeds maximum (" << maxCenter - 1 << ")\n";
        return false;
      }
      if (endCenter > maxCenter) {
        std::cerr << "[ERROR] End center " << endCenter << " exceeds maximum (" << maxCenter << ")\n";
        return false;
      }
      if (startMiddle >= maxMiddle) {
        std::cerr << "[ERROR] Start middle " << startMiddle << " exceeds maximum (" << maxMiddle - 1 << ")\n";
        return false;
      }
      if (endMiddle > maxMiddle) {
        std::cerr << "[ERROR] End middle " << endMiddle << " exceeds maximum (" << maxMiddle << ")\n";
        return false;
      }

      // Validate ordering (lexicographic on center:middle pairs)
      if (startCenter > endCenter || (startCenter == endCenter && startMiddle >= endMiddle)) {
        std::cerr << "[ERROR] Start position must be before end position\n";
        return false;
      }

      args->stripMode = STRIP_MODE_RANGE;
      args->centerIdxStart = (uint32_t)startCenter;
      args->centerIdxEnd = (uint32_t)endCenter;
      args->middleIdxStart = (uint32_t)startMiddle;
      args->middleIdxEnd = (uint32_t)endMiddle;
      return true;
    } catch (const std::exception&) {
      std::cerr << "[ERROR] Invalid range format '" << rangeStr << "', expected C1[:M1]-C2[:M2]\n";
      return false;
    }
  }

  std::cerr << "[ERROR] Invalid strip mode '" << arg << "', expected 'index:C[:M]' or 'range:C1[:M1]-C2[:M2]'\n";
  return false;
}


static bool parseSimulateType(const char* arg, ProgramArgs* args) {
  std::string str(arg);

  if (str == "pattern") {
    args->simulateType = SIMULATE_PATTERN;
    args->gridSize = GRID_SIZE_8X8;
    return true;
  } else if (str == "pattern:7x7") {
    args->simulateType = SIMULATE_PATTERN;
    args->gridSize = GRID_SIZE_7X7;
    return true;
  } else if (str == "symmetry") {
    args->simulateType = SIMULATE_SYMMETRY;
    return true;
  }

  std::cerr << "[ERROR] Invalid simulate type '" << arg << "', expected 'pattern', 'pattern:7x7', or 'symmetry'\n";
  return false;
}

static bool parseWorkerConfig(const char* arg, ProgramArgs* a) {
  std::string str(arg);
  size_t colonPos = str.find(':');

  if (colonPos == std::string::npos) {
    std::cerr << "[ERROR] Invalid worker format '" << arg << "', expected N:M\n";
    return false;
  }

  try {
    a->workerNum = std::stoi(str.substr(0, colonPos));
    a->totalWorkers = std::stoi(str.substr(colonPos + 1));

    if (a->workerNum <= 0 || a->totalWorkers <= 0) {
      std::cerr << "[ERROR] Worker numbers must be positive\n";
      return false;
    }

    if (a->workerNum > a->totalWorkers) {
      std::cerr << "[ERROR] Worker number cannot exceed total workers\n";
      return false;
    }

    return true;
  } catch (const std::exception&) {
    std::cerr << "[ERROR] Invalid worker format '" << arg << "', expected N:M\n";
    return false;
  }
}

static bool parseTestApi(const char* arg, ProgramArgs* args) {
  std::string str(arg);

  if (str == "progress") {
    args->testApi = TEST_API_PROGRESS;
    return true;
  } else if (str == "summary") {
    args->testApi = TEST_API_SUMMARY;
    return true;
  } else if (str == "framecache") {
    args->testApi = TEST_API_FRAMECACHE;
    return true;
  } else if (str == "strip-progress") {
    args->testApi = TEST_API_STRIP_PROGRESS;
    return true;
  } else if (str == "strip-summary") {
    args->testApi = TEST_API_STRIP_SUMMARY;
    return true;
  } else if (str == "strip-cache") {
    args->testApi = TEST_API_STRIP_CACHE;
    return true;
  } else if (str == "strip-completion") {
    args->testApi = TEST_API_STRIP_COMPLETION;
    return true;
  }

  std::cerr << "[ERROR] Invalid test API type '" << arg << "', expected 'progress', 'summary', 'framecache', 'strip-progress', 'strip-summary', 'strip-cache', or 'strip-completion'\n";
  return false;
}

static bool parse7x7Range(const char* arg, ProgramArgs* args) {
  const uint64_t max7x7 = 1ULL << 49;

  if (!arg || *arg == '\0') {
    // No argument provided: use full range
    args->grid7x7StartPattern = 0;
    args->grid7x7EndPattern = max7x7;
    args->gridSize = GRID_SIZE_7X7;
    return true;
  }

  std::string str(arg);
  size_t colonPos = str.find(':');

  if (colonPos == std::string::npos) {
    std::cerr << "[ERROR] Invalid 7x7 range format '" << arg << "', expected 'start:end', ':end', or 'start:'\n";
    return false;
  }

  try {
    std::string startStr = str.substr(0, colonPos);
    std::string endStr = str.substr(colonPos + 1);

    // Parse start value (default to 0 if empty)
    if (startStr.empty()) {
      args->grid7x7StartPattern = 0;
    } else {
      args->grid7x7StartPattern = std::stoull(startStr);
      if (args->grid7x7StartPattern > max7x7) {
        std::cerr << "[ERROR] Start pattern " << args->grid7x7StartPattern << " exceeds maximum 7x7 pattern (" << max7x7 << ")\n";
        return false;
      }
    }

    // Parse end value (default to max if empty)
    if (endStr.empty()) {
      args->grid7x7EndPattern = max7x7;
    } else {
      args->grid7x7EndPattern = std::stoull(endStr);
      if (args->grid7x7EndPattern > max7x7) {
        std::cerr << "[ERROR] End pattern " << args->grid7x7EndPattern << " exceeds maximum 7x7 pattern (" << max7x7 << ")\n";
        return false;
      }
    }

    // Validate range
    if (args->grid7x7StartPattern >= args->grid7x7EndPattern) {
      std::cerr << "[ERROR] Start pattern must be less than end pattern\n";
      return false;
    }

    args->gridSize = GRID_SIZE_7X7;
    return true;
  } catch (const std::exception&) {
    std::cerr << "[ERROR] Invalid 7x7 range format '" << arg << "', expected numeric values\n";
    return false;
  }
}

static error_t parseArgpOptions(int key, char* arg, struct argp_state* state) {
  ProgramArgs* a = (ProgramArgs*)state->input;

  switch (key) {
  case 'f':
    if (!parseFrameMode(arg, a)) {
      argp_failure(state, 1, 0, "Invalid frame mode");
    }
    break;
  case 'S':
    if (!parseStripMode(arg, a)) {
      argp_failure(state, 1, 0, "Invalid strip mode");
    }
    break;
  case '7':
    if (!parse7x7Range(arg, a)) {
      argp_failure(state, 1, 0, "Invalid 7x7 range specification");
    }
    break;
  case 'D':
    if (!parseCycleDetection(arg, a)) {
      argp_failure(state, 1, 0, "Invalid cycle detection algorithm");
    }
    break;
  case 's':
    if (!parseSimulateType(arg, a)) {
      argp_failure(state, 1, 0, "Invalid simulate type");
    }
    a->simulateMode = true;
    break;
  case 'A':
    try {
      a->compareFrameIdx = std::stoull(arg);
      a->compareCycleAlgorithms = true;
    } catch (const std::exception&) {
      argp_failure(state, 1, 0, "Invalid frame index for comparison");
    }
    break;
  case 'c':
    if (!arg || *arg == '\0') {
      argp_failure(state, 1, 0, "Subgrid cache path cannot be empty");
    }
    a->subgridCachePath = arg;
    a->computeSubgridCache = true;
    break;
  case 'b':
    try {
      a->subgridCacheBegin = std::stoull(arg);
    } catch (const std::exception&) {
      argp_failure(state, 1, 0, "Invalid subgrid cache begin value");
    }
    break;
  case 'C':
    if (!arg || *arg == '\0') {
      argp_failure(state, 1, 0, "Subgrid cache file path cannot be empty");
    }
    a->subgridCachePath = arg;
    break;
  case 'r':
    a->dontSaveResults = true;
    break;
  case 'v':
    break;
  case 'T':
    if (!parseTestApi(arg, a)) {
      argp_failure(state, 1, 0, "Invalid test API type");
    }
    break;
  case 'l':
    if (!arg || *arg == '\0') {
      argp_failure(state, 1, 0, "Log file path cannot be empty");
    }
    a->logFilePath = std::string(arg);
    break;
  case 'w':
    if (!parseWorkerConfig(arg, a)) {
      argp_failure(state, 1, 0, "Invalid worker configuration");
    }
    break;
  case 'q':
    if (!arg || *arg == '\0') {
      argp_failure(state, 1, 0, "Queue directory path cannot be empty");
    }
    a->queueDirectory = std::string(arg);
    break;
  case 'Q':
    a->drainRequestQueue = true;
    break;
  default:
    return ARGP_ERR_UNKNOWN;
  }

  return 0;
}

struct argp argp = {argp_options, parseArgpOptions, ProgramArgs_doc, prog_doc, 0, 0};

void initializeDefaultArgs(ProgramArgs* args) {
  args->resumeFromDatabase = false;
  args->compareCycleAlgorithms = false;
  args->dontSaveResults = false;
  args->simulateMode = false;
  args->computeSubgridCache = false;
  args->drainRequestQueue = false;
  args->testApi = TEST_API_NONE;
  args->simulateType = SIMULATE_NONE;
  args->frameMode = FRAME_MODE_NONE;
  args->stripMode = STRIP_MODE_NONE;
  args->gridSize = GRID_SIZE_8X8;
  args->frameModeIndex = 0;
  args->grid7x7StartPattern = 0;
  args->grid7x7EndPattern = (1ULL << 49) - 1;
  args->centerIdxStart = 0;
  args->centerIdxEnd = CENTER_4X4_TOTAL_UNIQUE;  // 8548 unique center patterns
  args->middleIdxStart = 0;
  args->middleIdxEnd = STRIP_SEARCH_TOTAL_MIDDLE_IDX;  // 512 middle indices per center
  args->logFilePath = "";
  args->queueDirectory = "./request-queue";
  args->subgridCachePath = "";
  args->cycleDetection = CYCLE_DETECTION_FLOYD;
  args->compareFrameIdx = 0;
  args->subgridCacheBegin = 0;
  args->workerNum = 1;
  args->totalWorkers = 1;
}

ProgramArgs* parseCommandLineArgs(int argc, char** argv) {
  auto args = std::make_unique<ProgramArgs>();
  if (!args) {
    std::cerr << "[ERROR] Memory allocation failed\n";
    return nullptr;
  }

  initializeDefaultArgs(args.get());

  if (argp_parse(&argp, argc, argv, 0, 0, args.get()) != 0) {
    return nullptr;
  }

  return args.release();
}

void cleanupProgramArgs(ProgramArgs* args) {
  delete args;
}