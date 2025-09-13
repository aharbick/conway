#ifndef _GOL_CORE_H_
#define _GOL_CORE_H_

#include <cstdint>
#include <string>

#include "cli_parser.h"
#include "constants.h"

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

__host__ __device__ static inline void add2(uint64_t a, uint64_t b, uint64_t& s0, uint64_t& s1) {
  s0 = a ^ b;
  s1 = a & b;
}

__host__ __device__ static inline void add3(uint64_t a, uint64_t b, uint64_t c, uint64_t& s0, uint64_t& s1) {
  uint64_t t0, t1, t2;
  add2(a, b, t0, t1);
  add2(t0, c, s0, t2);
  s1 = t1 ^ t2;
}

// Tom Rokicki's optimized 19-operation Conway's Game of Life computation
__host__ __device__ static inline uint64_t computeNextGeneration(uint64_t a) {
  uint64_t s0, sh2, a0, a1, sll, slh;
  add2((a & GOL_HORIZONTAL_SHIFT_MASK) << 1, (a & GOL_VERTICAL_SHIFT_MASK) >> 1, s0, sh2);
  add2(s0, a, a0, a1);
  a1 |= sh2;
  add3(a0 >> 8, a0 << 8, s0, sll, slh);
  uint64_t y = a1 >> 8;
  uint64_t x = a1 << 8;
  return (x ^ y ^ sh2 ^ slh) & ((x | y) ^ (sh2 | slh)) & (sll | a);
}

__host__ __device__ static inline int adjustGenerationsForDeadout(int generations, uint64_t g2, uint64_t g3,
                                                                  uint64_t g4, uint64_t g5, uint64_t g6) {
  if (g2 == 0)
    return generations - 5;
  if (g3 == 0)
    return generations - 4;
  if (g4 == 0)
    return generations - 3;
  if (g5 == 0)
    return generations - 2;
  if (g6 == 0)
    return generations - 1;
  return generations;
}

// Floyd's cycle detection algorithm
__host__ __device__ static inline int floydCycleDetection(uint64_t startState, int startGenerations) {
  bool ended = false;
  int generations = startGenerations;

  uint64_t slow = startState;
  uint64_t fast = computeNextGeneration(slow);

  do {
    generations++;
    uint64_t nextSlow = computeNextGeneration(slow);

    if (slow == nextSlow) {
      ended = true;  // If we didn't change then we ended
      break;
    }
    slow = nextSlow;
    fast = computeNextGeneration(computeNextGeneration(fast));
  } while (slow != fast);

  ended = slow == 0;  // If we died out then we ended
  return ended ? generations : 0;
}

// Nivasch's cycle detection algorithm - stack-based approach
struct NivaschStackEntry {
  uint64_t value;
  int time;
};

__host__ __device__ static inline int nivaschCycleDetection(uint64_t startState, int startGenerations) {
  bool ended = false;

  // Multi-stack Nivasch algorithm
  NivaschStackEntry stacks[NIVASCH_NUM_STACKS][NIVASCH_MAX_STACK_SIZE];
  int stackSizes[NIVASCH_NUM_STACKS] = {0};

  uint64_t current = startState;
  int time = startGenerations;

  while (time < 10000) {  // Reasonable upper bound
    // Determine which stack to use based on hash of current value
    int stackIdx = (current % NIVASCH_NUM_STACKS);
    NivaschStackEntry* stack = stacks[stackIdx];
    int& stackSize = stackSizes[stackIdx];

    // Pop entries from stack where value > current
    while (stackSize > 0 && stack[stackSize - 1].value > current) {
      stackSize--;
    }

    // Check if current value matches top of stack
    if (stackSize > 0 && stack[stackSize - 1].value == current) {
      // Found cycle! Return total generations until cycle detected
      return time;
    }

    // Push current value onto stack if there's room
    if (stackSize < NIVASCH_MAX_STACK_SIZE) {
      stack[stackSize].value = current;
      stack[stackSize].time = time;
      stackSize++;
    }

    // Advance to next generation
    uint64_t next = computeNextGeneration(current);
    if (next == 0) {
      ended = true;
      break;
    }
    if (next == current) {
      // Found stable state
      break;
    }

    current = next;
    time++;
  }

  ended = (current == 0);
  return ended ? time : 0;
}

// Core generation counting logic - determines how long patterns run
__host__ __device__ static inline int countGenerations(uint64_t pattern,
                                                       CycleDetectionAlgorithm algorithm = CYCLE_DETECTION_FLOYD) {
  bool ended = false;
  int generations = 0;

  //  Run for up to 300 generations just checking if we cycle within 6
  //  generations or die out.
  uint64_t g1, g2, g3, g4, g5, g6;
  g1 = pattern;
  do {
    generations += 6;
    g2 = computeNextGeneration(g1);
    g3 = computeNextGeneration(g2);
    g4 = computeNextGeneration(g3);
    g5 = computeNextGeneration(g4);
    g6 = computeNextGeneration(g5);
    g1 = computeNextGeneration(g6);

    if (g1 == 0) {
      ended = true;  // died out
      generations = adjustGenerationsForDeadout(generations, g2, g3, g4, g5, g6);
      break;
    }

    if ((g1 == g2) || (g1 == g3) || (g1 == g4)) {
      // periodic
      break;
    }

  } while (generations < FAST_SEARCH_MAX_GENERATIONS);

  // Fall back to chosen cycle detection algorithm if we haven't
  // exited the previous loop because of die out or cycle.
  if (!ended && generations >= FAST_SEARCH_MAX_GENERATIONS) {
    if (algorithm == CYCLE_DETECTION_NIVASCH) {
      return nivaschCycleDetection(g1, generations);
    } else {
      return floydCycleDetection(g1, generations);
    }
  }

  return ended ? generations : 0;
}

// CUDA kernel construction function
__host__ __device__ static inline uint64_t constructKernel(uint64_t frame, int kernelIndex) {
  uint64_t kernel = frame;
  kernel += ((uint64_t)(kernelIndex & 3)) << 3;    // lower pair of K bits
  kernel += ((uint64_t)(kernelIndex >> 2)) << 59;  // upper pair of K bits
  return kernel;
}

// Atomic increment helper - works in both device and host contexts
__host__ __device__ static inline uint64_t getNextCandidateIndex(uint64_t* numCandidates) {
#ifdef __CUDA_ARCH__
  return atomicAdd((unsigned long long*)numCandidates, 1ULL);
#else
  return (*numCandidates)++;
#endif
}

// Core 6-generation stepping and cycle detection logic
// Used by both CPU tests and CUDA kernels
__host__ __device__ static inline bool step6GenerationsAndCheck(uint64_t* g1, uint64_t pattern, uint64_t* generations,
                                                                uint64_t* candidates, uint64_t* numCandidates) {
  *generations += 6;
  uint64_t g2 = computeNextGeneration(*g1);
  uint64_t g3 = computeNextGeneration(g2);
  uint64_t g4 = computeNextGeneration(g3);
  uint64_t g5 = computeNextGeneration(g4);
  uint64_t g6 = computeNextGeneration(g5);
  *g1 = computeNextGeneration(g6);

  // Check for cycles
  if ((*g1 == g2) || (*g1 == g3) || (*g1 == g4)) {
    *generations = 0;
    return true;  // Pattern ended/cyclical, advance to next
  }

  // Check if reached minimum candidate generations
  if (*generations >= MIN_CANDIDATE_GENERATIONS) {
    uint64_t idx = getNextCandidateIndex(numCandidates);
    candidates[idx] = pattern;
    *generations = 0;
    return true;  // Candidate found, advance to next
  }

  return false;  // Continue with current pattern
}

#endif