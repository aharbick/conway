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
    {"frame-mode", 'f', "MODE", 0, "Frame search mode: 'random' or 'missing'"},
    {"verbose", 'v', 0, 0, "Verbose output."},
    {"test-google-api", 'T', 0, 0, "Test Google Sheets API functionality and exit."},
    {"test-frame-cache", 'C', 0, 0, "Test frame completion cache functionality and exit."},
    {"test-missing-frames", 'M', 0, 0, "Test missing frames API and display incomplete frames."},
    {"log-file", 'l', "PATH", 0, "Path to log file for progress output."},
    {"worker", 'w', "N:M", 0,
     "Worker configuration N:M where N is worker number (1-based) and M is total workers (default: 1:1)."},
    {0}};



static bool parseFrameMode(const char* arg, ProgramArgs* args) {
  std::string str(arg);

  if (str == "random" || str == "missing") {
    args->frameMode = str;
    return true;
  }

  std::cerr << "[ERROR] Invalid frame mode '" << arg << "', expected 'random' or 'missing'\n";
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

static error_t parseArgpOptions(int key, char* arg, struct argp_state* state) {
  ProgramArgs* a = (ProgramArgs*)state->input;

  switch (key) {
  case 'f':
    if (!parseFrameMode(arg, a)) {
      argp_failure(state, 1, 0, "Invalid frame mode");
    }
    break;
  case 'v':
    a->verbose = true;
    break;
  case 'T':
    a->testGoogleApi = true;
    break;
  case 'C':
    a->testFrameCache = true;
    break;
  case 'M':
    a->testMissingFrames = true;
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
  default:
    return ARGP_ERR_UNKNOWN;
  }

  return 0;
}

struct argp argp = {argp_options, parseArgpOptions, ProgramArgs_doc, prog_doc, 0, 0};

void initializeDefaultArgs(ProgramArgs* args) {
  args->verbose = false;
  args->testGoogleApi = false;
  args->testFrameCache = false;
  args->testMissingFrames = false;
  args->resumeFromDatabase = false;
  args->frameMode = "";
  args->logFilePath = "";
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