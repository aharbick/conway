#ifndef _SEARCH_MEMORY_H_
#define _SEARCH_MEMORY_H_

#include <cstdint>
#include <cstdlib>

#include "cuda_utils.h"

// Memory structure for search operations
typedef struct {
  uint64_t *d_candidates, *d_numCandidates, *d_bestPattern, *d_bestGenerations;
  uint64_t *h_candidates, *h_numCandidates, *h_bestPattern, *h_bestGenerations;
} SearchMemory;

// Memory management functions
__host__ SearchMemory *allocateSearchMemory(size_t candidateSize);
__host__ void freeSearchMemory(SearchMemory *mem);

#endif