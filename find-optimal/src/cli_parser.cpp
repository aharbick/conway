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
    {"cudaconfig", 'c', "config", 0, "CUDA kernel params numgpus:blocksize:threadsperblock (e.g. 1:1024:1024)"},
    {"threads", 't', "num", 0, "Number of CPU threads (if you use more than one GPU you should use matching threads)."},
    {"frame-index-range", 'f', "BEGIN[:END]", 0,
     "Frame index range to search from 0 to 2102800 or 'resume' (e.g. 0: or 1:1234 or resume)"},
    {"chunk-size", 'k', "size", 0, "Chunk size for pattern processing (default: 65536)."},
    {"verbose", 'v', 0, 0, "Verbose output."},
    {"test-google-api", 'T', 0, 0, "Test Google Sheets API functionality and exit."},
    {"test-frame-cache", 'C', 0, 0, "Test frame completion cache functionality and exit."},
    {"log-file", 'l', "PATH", 0, "Path to log file for progress output."},
    {"worker", 'w', "N:M", 0, "Worker configuration N:M where N is worker number (1-based) and M is total workers (default: 1:1)."},
    {0}};


static bool parseCudaConfig(const char* arg, ProgramArgs* a) {
  std::string str(arg);
  std::vector<std::string> parts;
  
  // Split by colon
  size_t pos = 0;
  while (pos < str.length()) {
    size_t colonPos = str.find(':', pos);
    if (colonPos == std::string::npos) {
      parts.push_back(str.substr(pos));
      break;
    }
    parts.push_back(str.substr(pos, colonPos - pos));
    pos = colonPos + 1;
  }
  
  if (parts.empty()) {
    std::cerr << "[ERROR] Invalid CUDA config format '" << arg << "', expected numgpus:blocksize:threadsperblock\n";
    return false;
  }
  
  try {
    // Parse gpusToUse (required)
    a->gpusToUse = std::stoi(parts[0]);
    if (a->gpusToUse <= 0) {
      std::cerr << "[ERROR] gpusToUse must be positive\n";
      return false;
    }
    
    // Parse blockSize (optional)
    if (parts.size() > 1) {
      a->blockSize = std::stoi(parts[1]);
      if (a->blockSize <= 0) {
        std::cerr << "[ERROR] blockSize must be positive\n";
        return false;
      }
    }
    
    // Parse threadsPerBlock (optional)
    if (parts.size() > 2) {
      a->threadsPerBlock = std::stoi(parts[2]);
      if (a->threadsPerBlock <= 0) {
        std::cerr << "[ERROR] threadsPerBlock must be positive\n";
        return false;
      }
    }
    
    return true;
  } catch (const std::exception&) {
    std::cerr << "[ERROR] Invalid CUDA config format '" << arg << "', expected numgpus:blocksize:threadsperblock\n";
    return false;
  }
}

static bool parseRangeArg(const char* arg, uint64_t* begin, uint64_t* end, ProgramArgs* args = nullptr) {
  std::string str(arg);
  
  // Handle special cases
  if (str == "resume") {
    uint64_t resumeFrame = googleGetBestCompleteFrame();
    *begin = (resumeFrame == ULLONG_MAX) ? 0 : resumeFrame + 1;
    *end = FRAME_SEARCH_TOTAL_MINIMAL_FRAMES;
    return true;
  }

  if (str == "random") {
    *begin = 0;
    *end = FRAME_SEARCH_TOTAL_MINIMAL_FRAMES;
    if (args) {
      args->randomFrameMode = true;
    }
    return true;
  }

  // Parse BEGIN:END format
  size_t colonPos = str.find(':');
  if (colonPos == std::string::npos) {
    std::cerr << "[ERROR] Invalid range format '" << arg << "', expected BEGIN: or BEGIN:END or 'resume'\n";
    return false;
  }

  try {
    // Parse begin value
    *begin = std::stoull(str.substr(0, colonPos));
    
    // Parse end value (if specified)
    if (colonPos + 1 >= str.length()) {
      *end = 0;  // No end specified
    } else {
      *end = std::stoull(str.substr(colonPos + 1));
      if (*end <= *begin) {
        std::cerr << "[ERROR] Invalid range, end must be greater than begin\n";
        return false;
      }
    }
    
    return true;
  } catch (const std::exception&) {
    std::cerr << "[ERROR] Invalid range format '" << arg << "', expected BEGIN: or BEGIN:END or 'resume'\n";
    return false;
  }
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
  case 'c':
    if (!parseCudaConfig(arg, a)) {
      argp_failure(state, 1, 0, "Invalid CUDA configuration");
    }
    break;
  case 't':
    try {
      a->cpuThreads = std::stoi(arg);
      if (a->cpuThreads <= 0) {
        argp_failure(state, 1, 0, "Thread count must be positive");
      }
    } catch (const std::exception&) {
      argp_failure(state, 1, 0, "Invalid thread count");
    }
    break;
  case 'f':
    if (!parseRangeArg(arg, &a->frameBeginIdx, &a->frameEndIdx, a)) {
      argp_failure(state, 1, 0, "Invalid frame index range");
    }
    break;
  case 'k':
    try {
      a->chunkSize = std::stoull(arg);
      if (a->chunkSize <= 0) {
        argp_failure(state, 1, 0, "Chunk size must be positive");
      }
    } catch (const std::exception&) {
      argp_failure(state, 1, 0, "Invalid chunk size");
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
  args->threadId = 0;
  args->cpuThreads = 1;
  args->gpusToUse = 1;
  args->blockSize = 1024;
  args->threadsPerBlock = 1024;
  args->verbose = false;
  args->testGoogleApi = false;
  args->testFrameCache = false;
  args->resumeFromDatabase = false;
  args->randomFrameMode = false;
  args->frameBeginIdx = 0;
  args->frameEndIdx = 0;
  args->chunkSize = FRAME_SEARCH_DEFAULT_CHUNK_SIZE;
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