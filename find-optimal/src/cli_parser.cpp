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

// Command line options
static struct argp_option argp_options[] = {
    {"frame-mode", 'f', "MODE", 0, "Frame search mode: 'random' or 'sequential'"},
    {"cycle-detection", 'D', "ALGORITHM", 0, "Cycle detection algorithm: 'floyd' or 'nivasch' (default: 'floyd')"},
    {"simulate", 's', "TYPE", 0, "Simulate mode: 'pattern' (evolution) or 'symmetry' (transformations)."},
    {"compare-cycle-algorithms", 'A', "FRAME_IDX", 0,
     "Compare Floyd's vs Nivasch's cycle detection on given frame index and exit."},
    {"compute-subgrid-cache", 'c', "PATH", 0, "Compute 7x7 subgrid cache for all 2^49 patterns and save to disk at PATH."},
    {"subgrid-cache-begin", 'b', "NUMBER", 0, "Starting pattern index for subgrid cache computation (for resuming)."},
    {"dont-save-results", 'r', 0, 0, "Don't save results to Google Sheets (for testing/benchmarking)."},
    {"test-api", 'T', "TYPE", 0, "Test API functionality and exit: 'progress', 'summary', or 'framecache'."},
    {"log-file", 'l', "PATH", 0, "Path to log file for progress output."},
    {"worker", 'w', "N:M", 0,
     "Worker configuration N:M where N is worker number (1-based) and M is total workers (default: 1:1)."},
    {"queue-directory", 'q', "PATH", 0, "Directory for persistent request queue (default: ./request-queue)."},
    {0}};


static bool parseFrameMode(const char* arg, ProgramArgs* args) {
  std::string str(arg);

  if (str == "random" || str == "sequential") {
    args->frameMode = str;
    return true;
  }

  std::cerr << "[ERROR] Invalid frame mode '" << arg << "', expected 'random' or 'sequential'\n";
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


static bool parseSimulateType(const char* arg, ProgramArgs* args) {
  std::string str(arg);

  if (str == "pattern") {
    args->simulateType = str;
    return true;
  } else if (str == "symmetry") {
    args->simulateType = str;
    return true;
  }

  std::cerr << "[ERROR] Invalid simulate type '" << arg << "', expected 'pattern' or 'symmetry'\n";
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

  if (str == "progress" || str == "summary" || str == "framecache") {
    args->testApi = str;
    return true;
  }

  std::cerr << "[ERROR] Invalid test API type '" << arg << "', expected 'progress', 'summary', or 'framecache'\n";
  return false;
}

static error_t parseArgpOptions(int key, char* arg, struct argp_state* state) {
  ProgramArgs* a = (ProgramArgs*)state->input;

  switch (key) {
  case 'f':
    if (!parseFrameMode(arg, a)) {
      argp_failure(state, 1, 0, "Invalid frame mode");
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
  args->testApi = "";
  args->simulateType = "";
  args->frameMode = "";
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