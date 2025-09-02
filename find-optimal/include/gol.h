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

// Project modules (organized logically)
#include "airtable_client.h"
#include "cli_parser.h"
#include "constants.h"
#include "cuda_utils.h"
#include "display_utils.h"
#include "frame_utils.h"
#include "gol_core.h"
#include "search_memory.h"
#include "utils.h"

// Global variables updated across threads
extern pthread_mutex_t gMutex;
extern int gBestGenerations;

// Global state management
__host__ bool updateBestGenerations(int generations);

// Search algorithm functions
__host__ void searchRandom(ProgramArgs *cli);
__host__ void searchAll(ProgramArgs *cli);
__host__ void *search(void *args);

// Search execution functions
__host__ void executeCandidateSearch(SearchMemory *mem, ProgramArgs *cli, uint64_t start, uint64_t end);
__host__ void executeKernelSearch(SearchMemory *mem, ProgramArgs *cli, uint64_t frame, uint64_t frameIdx);
__host__ void reportChunkResults(SearchMemory *mem, ProgramArgs *cli, double startTime, uint64_t frame,
                                 uint64_t frameIdx, int kernelIdx, int chunkIdx, bool isFrameComplete);

// Helper functions
__host__ const char *getSearchDescription(ProgramArgs *cli, char *buffer, size_t bufferSize);

#endif