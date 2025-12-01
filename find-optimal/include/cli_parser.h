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

enum GridSize {
  GRID_SIZE_8X8,
  GRID_SIZE_7X7
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
  GridSize gridSize;
  uint64_t frameModeIndex;  // Only used when frameMode == FRAME_MODE_INDEX
  uint64_t grid7x7StartPattern;  // Starting pattern for 7x7 search (default: 0)
  uint64_t grid7x7EndPattern;    // Ending pattern for 7x7 search (default: 2^49 - 1)
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