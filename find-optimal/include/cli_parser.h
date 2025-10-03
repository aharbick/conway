#ifndef _CLI_PARSER_H_
#define _CLI_PARSER_H_

#include <cstdint>
#include <string>

typedef enum { CYCLE_DETECTION_FLOYD, CYCLE_DETECTION_NIVASCH } CycleDetectionAlgorithm;


typedef struct ProgramArgs {
  bool testProgressApi;
  bool testSummaryApi;
  bool testFrameCache;
  bool resumeFromDatabase;
  bool compareCycleAlgorithms;
  bool dontSaveResults;
  bool simulateMode;
  std::string simulateType;
  std::string frameMode;
  std::string logFilePath;
  std::string queueDirectory;
  CycleDetectionAlgorithm cycleDetection;
  uint64_t compareFrameIdx;
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