#ifndef _GOL_H_
#define _GOL_H_

// Standard library includes
#include <assert.h>
#include <limits.h>
#include <locale.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <cstdint>
#include <mutex>

// Project modules (organized logically)
#include "cli_parser.h"
#include "constants.h"
#include "cuda_utils.h"
#include "display_utils.h"
#include "frame_utils.h"
#include "gol_core.h"
#include "gol_memory.h"
#include "google_client.h"
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
__host__ void reportKernelResults(gol::SearchMemory &mem, ProgramArgs *cli, double startTime, uint64_t frame,
                                  uint64_t frameIdx, int kernelIdx, bool isFrameComplete);

// Helper functions
__host__ std::string getSearchDescription(ProgramArgs *cli);
__host__ void compareCycleDetectionAlgorithms(ProgramArgs *cli, uint64_t frameIdx);

#endif