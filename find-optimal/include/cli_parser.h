#ifndef _CLI_PARSER_H_
#define _CLI_PARSER_H_

#include <cstdint>
#include <string>

typedef struct ProgramArgs {
  int gpusToUse;
  int blockSize;
  int threadsPerBlock;
  bool verbose;
  bool testGoogleApi;
  bool testFrameCache;
  bool testMissingFrames;
  bool resumeFromDatabase;
  bool randomFrameMode;
  bool missingFrameMode;
  uint64_t frameBeginIdx;
  uint64_t frameEndIdx;
  uint64_t chunkSize;
  std::string logFilePath;
  int workerNum;
  int totalWorkers;
} ProgramArgs;

// Initialize ProgramArgs with default values
void initializeDefaultArgs(ProgramArgs* args);

// Parse command line arguments using argp
ProgramArgs* parseCommandLineArgs(int argc, char** argv);

// Clean up ProgramArgs allocated by parseCommandLineArgs
void cleanupProgramArgs(ProgramArgs* args);

#endif