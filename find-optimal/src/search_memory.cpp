#include "search_memory.h"

__host__ SearchMemory *allocateSearchMemory(size_t candidateSize) {
  SearchMemory *mem = static_cast<SearchMemory *>(malloc(sizeof(SearchMemory)));

  // Allocate device memory
  cudaCheckError(cudaMalloc((void **)&mem->d_candidates, sizeof(uint64_t) * candidateSize));
  cudaCheckError(cudaMalloc((void **)&mem->d_numCandidates, sizeof(uint64_t)));
  cudaCheckError(cudaMalloc((void **)&mem->d_bestPattern, sizeof(uint64_t)));
  cudaCheckError(cudaMalloc((void **)&mem->d_bestGenerations, sizeof(uint64_t)));

  // Allocate host memory
  mem->h_candidates = static_cast<uint64_t *>(calloc(candidateSize, sizeof(uint64_t)));
  mem->h_numCandidates = static_cast<uint64_t *>(malloc(sizeof(uint64_t)));
  mem->h_bestPattern = static_cast<uint64_t *>(malloc(sizeof(uint64_t)));
  mem->h_bestGenerations = static_cast<uint64_t *>(malloc(sizeof(uint64_t)));

  return mem;
}

__host__ void freeSearchMemory(SearchMemory *mem) {
  if (!mem)
    return;

  // Free device memory
  cudaCheckError(cudaFree(mem->d_candidates));
  cudaCheckError(cudaFree(mem->d_numCandidates));
  cudaCheckError(cudaFree(mem->d_bestPattern));
  cudaCheckError(cudaFree(mem->d_bestGenerations));

  // Free host memory
  free(mem->h_candidates);
  free(mem->h_numCandidates);
  free(mem->h_bestPattern);
  free(mem->h_bestGenerations);

  free(mem);
}