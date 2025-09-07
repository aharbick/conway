#ifndef _CLI_PARSER_H_
#define _CLI_PARSER_H_

#include <cstdint>

typedef struct ProgramArgs {
  int threadId;
  int cpuThreads;
  int gpusToUse;
  int blockSize;
  int threadsPerBlock;
  bool random;
  bool verbose;
  bool testGoogleApi;
  bool resumeFromDatabase;
  bool randomFrameMode;
  uint64_t randomSamples;
  uint64_t beginAt;
  uint64_t endAt;
  uint64_t frameBeginIdx;
  uint64_t frameEndIdx;
  uint64_t chunkSize;
} ProgramArgs;

// Initialize ProgramArgs with default values
void initializeDefaultArgs(ProgramArgs* args);

// Parse command line arguments using argp
ProgramArgs* parseCommandLineArgs(int argc, char** argv);

// Clean up ProgramArgs allocated by parseCommandLineArgs
void cleanupProgramArgs(ProgramArgs* args);

#endif