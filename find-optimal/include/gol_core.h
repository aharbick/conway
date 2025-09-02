#ifndef _GOL_CORE_H_
#define _GOL_CORE_H_

#include "types.h"

// CUDA decorators - defined here or by NVCC
#ifndef __NVCC__
#define __device__
#define __host__
#endif

// Conway's Game of Life bit manipulation masks
#define GOL_HORIZONTAL_SHIFT_MASK 0x7f7f7f7f7f7f7f7f
#define GOL_VERTICAL_SHIFT_MASK 0xFEFEFEFEFEFEFEFE

// Generation counting constants
#define MIN_CANDIDATE_GENERATIONS 180
#define FAST_SEARCH_MAX_GENERATIONS 300

// Core GOL computation functions - the mathematical heart of the algorithm

__host__ __device__ static inline void add2(ulong64 a, ulong64 b, ulong64 &s0, ulong64 &s1) {
  s0 = a ^ b ;
  s1 = a & b ;
}

__host__ __device__ static inline void add3(ulong64 a, ulong64 b, ulong64 c, ulong64 &s0, ulong64 &s1) {
  ulong64 t0, t1, t2 ;
  add2(a, b, t0, t1) ;
  add2(t0, c, s0, t2) ;
  s1 = t1 ^ t2 ;
}

// Tom Rokicki's optimized 19-operation Conway's Game of Life computation
__host__ __device__ static inline ulong64 computeNextGeneration(ulong64 a) {
  ulong64 s0, sh2, a0, a1, sll, slh ;
  add2((a & GOL_HORIZONTAL_SHIFT_MASK)<<1, (a & GOL_VERTICAL_SHIFT_MASK)>>1, s0, sh2) ;
  add2(s0, a, a0, a1) ;
  a1 |= sh2 ;
  add3(a0>>8, a0<<8, s0, sll, slh) ;
  ulong64 y = a1 >> 8 ;
  ulong64 x = a1 << 8 ;
  return (x^y^sh2^slh)&((x|y)^(sh2|slh))&(sll|a) ;
}

__host__ __device__ static inline int adjustGenerationsForDeadout(int generations, ulong64 g2, ulong64 g3, ulong64 g4, ulong64 g5, ulong64 g6) {
  if (g2 == 0) return generations - 5;
  if (g3 == 0) return generations - 4;
  if (g4 == 0) return generations - 3;
  if (g5 == 0) return generations - 2;
  if (g6 == 0) return generations - 1;
  return generations;
}

// Core generation counting logic - determines how long patterns run
__host__ __device__ static inline int countGenerations(ulong64 pattern) {
  bool ended = false;
  int generations = 0;

  //  Run for up to 300 generations just checking if we cycle within 6
  //  generations or die out.
  ulong64 g1, g2, g3, g4, g5, g6;
  g1 = pattern;
  do {
    generations+=6;
    g2 = computeNextGeneration(g1);
    g3 = computeNextGeneration(g2);
    g4 = computeNextGeneration(g3);
    g5 = computeNextGeneration(g4);
    g6 = computeNextGeneration(g5);
    g1 = computeNextGeneration(g6);

    if (g1 == 0) {
      ended = true; // died out
      generations = adjustGenerationsForDeadout(generations, g2, g3, g4, g5, g6);
      break;
    }

    if ((g1 == g2) || (g1 == g3) || (g1 == g4)) {
      // periodic
      break;
    }

  }
  while (generations < FAST_SEARCH_MAX_GENERATIONS);

  // Fall back to Floyd's cycle detection algorithm if we haven't
  // we didn't exit the previous loop because of die out or cycle.
  if (!ended && generations >= FAST_SEARCH_MAX_GENERATIONS) {
    ulong64 slow = g1;
    ulong64 fast = computeNextGeneration(slow);
    do {
      generations++;
      ulong64 nextSlow = computeNextGeneration(slow);

      if (slow == nextSlow) {
        ended = true; // If we didn't change then we ended
        break;
      }
      slow = nextSlow;
      fast = computeNextGeneration(computeNextGeneration(fast));
    }
    while (slow != fast);
    ended = slow == 0; // If we died out then we ended
  }

  return ended ? generations : 0;
}

// CUDA kernel construction function
__host__ __device__ static inline ulong64 constructKernel(ulong64 frame, int kernelIndex) {
  ulong64 kernel = frame;
  kernel += ((ulong64)(kernelIndex & 3)) << 3;      // lower pair of K bits
  kernel += ((ulong64)(kernelIndex >> 2)) << 59;    // upper pair of K bits
  return kernel;
}

// Atomic increment helper - works in both device and host contexts
__host__ __device__ static inline ulong64 getNextCandidateIndex(ulong64* numCandidates) {
#ifdef __CUDA_ARCH__
  return atomicAdd(numCandidates, 1);
#else
  return (*numCandidates)++;
#endif
}

// Core 6-generation stepping and cycle detection logic
// Used by both CPU tests and CUDA kernels
__host__ __device__ static inline bool step6GenerationsAndCheck(ulong64* g1, ulong64 pattern, ulong64* generations,
                                        ulong64* candidates, ulong64* numCandidates) {
  *generations += 6;
  ulong64 g2 = computeNextGeneration(*g1);
  ulong64 g3 = computeNextGeneration(g2);
  ulong64 g4 = computeNextGeneration(g3);
  ulong64 g5 = computeNextGeneration(g4);
  ulong64 g6 = computeNextGeneration(g5);
  *g1 = computeNextGeneration(g6);

  // Check for cycles
  if ((*g1 == g2) || (*g1 == g3) || (*g1 == g4)) {
    *generations = 0;
    return true; // Pattern ended/cyclical, advance to next
  }

  // Check if reached minimum candidate generations
  if (*generations >= MIN_CANDIDATE_GENERATIONS) {
    ulong64 idx = getNextCandidateIndex(numCandidates);
    candidates[idx] = pattern;
    *generations = 0;
    return true; // Candidate found, advance to next
  }

  return false; // Continue with current pattern
}

#endif