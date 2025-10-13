#ifndef _CLI_PARSER_H_
#define _CLI_PARSER_H_

#include <cstdint>
#include <string>

typedef enum { CYCLE_DETECTION_FLOYD, CYCLE_DETECTION_NIVASCH } CycleDetectionAlgorithm;


typedef struct ProgramArgs {
  bool resumeFromDatabase;
  bool compareCycleAlgorithms;
  bool dontSaveResults;
  bool simulateMode;
  bool computeSubgridCache;
  std::string testApi;
  std::string simulateType;
  std::string frameMode;
  std::string logFilePath;
  std::string queueDirectory;
  std::string subgridCachePath;
  CycleDetectionAlgorithm cycleDetection;
  uint64_t compareFrameIdx;
  uint64_t subgridCacheBegin;
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