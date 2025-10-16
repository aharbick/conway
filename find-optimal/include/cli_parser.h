#ifndef _CLI_PARSER_H_
#define _CLI_PARSER_H_

#include <cstdint>
#include <string>

enum CycleDetectionAlgorithm { CYCLE_DETECTION_FLOYD, CYCLE_DETECTION_NIVASCH };

enum FrameMode {
  FRAME_MODE_RANDOM,
  FRAME_MODE_SEQUENTIAL,
  FRAME_MODE_INDEX,
  FRAME_MODE_NONE
};

enum TestApiType {
  TEST_API_PROGRESS,
  TEST_API_SUMMARY,
  TEST_API_FRAMECACHE,
  TEST_API_NONE
};

enum SimulateType {
  SIMULATE_PATTERN,
  SIMULATE_SYMMETRY,
  SIMULATE_NONE
};

typedef struct ProgramArgs {
  bool resumeFromDatabase;
  bool compareCycleAlgorithms;
  bool dontSaveResults;
  bool simulateMode;
  bool computeSubgridCache;
  bool drainRequestQueue;
  TestApiType testApi;
  SimulateType simulateType;
  FrameMode frameMode;
  uint64_t frameModeIndex;  // Only used when frameMode == FRAME_MODE_INDEX
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