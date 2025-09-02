#ifndef _CUDA_UTILS_H_
#define _CUDA_UTILS_H_

#include <cstdio>
#include <cstdlib>

#ifdef __NVCC__
#include <cuda.h>

// CUDA error checking macro
#define cudaCheckError(ans) \
  { cudaAssert((ans), __FILE__, __LINE__); }

inline void cudaAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}

#else
// Define empty CUDA decorators for non-CUDA compilation
#define __device__
#define __host__
#define __global__
#endif

#endif