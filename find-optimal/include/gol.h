#ifndef _GOL_H_
#define _GOL_H_

// Standard library includes
#include <assert.h>
#include <limits.h>
#include <locale.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Platform compatibility
#include "platform_compat.h"

#include <cstdint>
#include <mutex>

// Project modules (organized logically)
#include "cli_parser.h"
#include "constants.h"
#include "cuda_utils.h"
#include "display_utils.h"
#include "symmetry_utils.h"
#include "gol_core.h"
#include "gol_memory.h"
#include "google_client.h"
#include "google_request_queue.h"
#include "utils.h"

// Global variables updated across threads
extern std::mutex gMutex;
extern int gBestGenerations;

// Global state management
__host__ bool updateBestGenerations(int generations);

// Search algorithm functions
__host__ void *search(void *args);

// Search execution functions (using RAII memory management)
__host__ void executeKernelSearch(gol::SearchMemory &mem, ProgramArgs *cli, uint64_t frame, uint64_t frameIdx);
__host__ void execute7x7Search(gol::SearchMemory &mem, ProgramArgs *cli, uint64_t rangeStart, uint64_t rangeEnd);
__host__ void executeStripSearch(ProgramArgs* cli, uint32_t centerStart, uint32_t centerEnd);
__host__ void reportKernelResults(gol::SearchMemory &mem, ProgramArgs *cli, double startTime, uint64_t frame,
                                  uint64_t frameIdx, int kernelIdx, bool isFrameComplete);
__host__ void report7x7Results(gol::SearchMemory &mem, ProgramArgs *cli, double startTime,
                                uint64_t rangeStart, uint64_t rangeEnd);

// Helper functions
__host__ std::string getSearchDescription(ProgramArgs *cli);
__host__ void compareCycleDetectionAlgorithms(ProgramArgs *cli, uint64_t frameIdx);

#endif