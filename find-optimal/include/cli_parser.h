#ifndef _CLI_PARSER_H_
#define _CLI_PARSER_H_

#include <cstdint>
#include <string>

typedef struct ProgramArgs {
  bool verbose;
  bool testGoogleApi;
  bool testFrameCache;
  bool testMissingFrames;
  bool resumeFromDatabase;
  std::string frameMode;
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