#ifndef _GOL_CORE_H_
#define _GOL_CORE_H_

#include <cstdint>
#include <string>

#include "cli_parser.h"
#include "constants.h"
#include "subgrid_cache.h"

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
    time++;
    uint64_t next = computeNextGeneration(current);

    // Check for stable state (no change)
    if (current == next) {
      ended = true;  // If we didn't change then we ended
      break;
    }

    // Check for death (pattern becomes empty)
    if (next == 0) {
      ended = true;
      current = next;
      break;
    }

    // Determine which stack to use based on hash of current value
    int stackIdx = (current % NIVASCH_NUM_STACKS);
    NivaschStackEntry* stack = stacks[stackIdx];
    int& stackSize = stackSizes[stackIdx];

    // Pop entries from stack where value > current
    while (stackSize > 0 && stack[stackSize - 1].value > current) {
      stackSize--;
    }

    // Check if current value matches top of stack (cycle detected)
    if (stackSize > 0 && stack[stackSize - 1].value == current) {
      // Found cycle! Check if we died out
      ended = (current == 0);
      break;
    }

    // Push current value onto stack if there's room
    if (stackSize < NIVASCH_MAX_STACK_SIZE) {
      stack[stackSize].value = current;
      stack[stackSize].time = time;
      stackSize++;
    }

    current = next;
  }

  // Only return generation count if pattern actually ended (died out or stabilized)
  return ended ? time : 0;
}

// Core generation counting logic - determines how long patterns run
__host__ __device__ static inline int countGenerations(uint64_t pattern,
                                                       CycleDetectionAlgorithm algorithm = CYCLE_DETECTION_FLOYD) {
  // Empty pattern (all zeros) has 0 generations
  if (pattern == 0) {
    return 0;
  }

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

// Expand a compact 7x7 pattern (7 bits/row) to 8x8 grid format (8 bits/row) at given position
// pattern7x7: compact format with 7 consecutive bits per row (49 bits total)
// rowOffset: 0 for rows 0-6, 1 for rows 1-7
// colOffset: 0 for cols 0-6, 1 for cols 1-7
__host__ __device__ static inline uint64_t expand7x7To8x8(uint64_t pattern7x7, int rowOffset, int colOffset) {
  uint64_t result = 0;

  // Extract each row from compact format and place in 8x8 grid
  for (int row = 0; row < 7; row++) {
    // Extract 7 bits for this row from compact pattern
    uint64_t rowBits = (pattern7x7 >> (row * 7)) & 0x7F;

    // Place in 8x8 grid at the appropriate row and column offset
    result |= (rowBits << colOffset) << ((row + rowOffset) * 8);
  }

  return result;
}

// Check if all live cells in a pattern can be covered by a 7x7 subgrid
// Returns true if coverable, false otherwise
__host__ __device__ static inline bool isCoverableBy7x7(uint64_t pattern) {
  if (pattern == 0) return false;

  // Test 4 possible 7x7 positions within the 8x8 grid:
  // Position 1: rows 0-6, cols 0-6
  if ((pattern & ~0x7F7F7F7F7F7F7FULL) == 0) return true;
  // Position 2: rows 0-6, cols 1-7
  if ((pattern & ~0xFEFEFEFEFEFEFEULL) == 0) return true;
  // Position 3: rows 1-7, cols 0-6
  if ((pattern & ~0x7F7F7F7F7F7F7F00ULL) == 0) return true;
  // Position 4: rows 1-7, cols 1-7
  if ((pattern & ~0xFEFEFEFEFEFEFE00ULL) == 0) return true;

  return false;
}

// Core 6-generation stepping and cycle detection logic
// Used by both CPU tests and CUDA kernels
__host__ __device__ static inline bool step6GenerationsAndCheck(uint64_t* g1, uint64_t pattern, uint16_t* generations,
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

// Version with 7x7 subgrid coverage detection
// Splits candidates into covered vs uncovered arrays
__host__ __device__ static inline bool step6GenerationsAndCheckWithCoverage(
    uint64_t* g1, uint64_t pattern, uint16_t* generations,
    uint64_t* uncoveredCandidates, uint64_t* numUncoveredCandidates,
    CoveredCandidate* coveredCandidates, uint64_t* numCoveredCandidates) {

  // Check if current state has 7x7 coverage
  if (isCoverableBy7x7(*g1)) {
    uint64_t idx = getNextCandidateIndex(numCoveredCandidates);
    coveredCandidates[idx] = {pattern, *g1, *generations};
    *generations = 0;
    return true;
  }

  // Compute next 6 generations, checking for coverage after each step
  uint64_t g2 = computeNextGeneration(*g1);
  if (isCoverableBy7x7(g2)) {
    uint64_t idx = getNextCandidateIndex(numCoveredCandidates);
    coveredCandidates[idx] = {pattern, g2, (uint16_t)(*generations + 1)};
    *generations = 0;
    return true;
  }

  uint64_t g3 = computeNextGeneration(g2);
  if (isCoverableBy7x7(g3)) {
    uint64_t idx = getNextCandidateIndex(numCoveredCandidates);
    coveredCandidates[idx] = {pattern, g3, (uint16_t)(*generations + 2)};
    *generations = 0;
    return true;
  }

  uint64_t g4 = computeNextGeneration(g3);
  if (isCoverableBy7x7(g4)) {
    uint64_t idx = getNextCandidateIndex(numCoveredCandidates);
    coveredCandidates[idx] = {pattern, g4, (uint16_t)(*generations + 3)};
    *generations = 0;
    return true;
  }

  uint64_t g5 = computeNextGeneration(g4);
  if (isCoverableBy7x7(g5)) {
    uint64_t idx = getNextCandidateIndex(numCoveredCandidates);
    coveredCandidates[idx] = {pattern, g5, (uint16_t)(*generations + 4)};
    *generations = 0;
    return true;
  }

  uint64_t g6 = computeNextGeneration(g5);
  if (isCoverableBy7x7(g6)) {
    uint64_t idx = getNextCandidateIndex(numCoveredCandidates);
    coveredCandidates[idx] = {pattern, g6, (uint16_t)(*generations + 5)};
    *generations = 0;
    return true;
  }

  *g1 = computeNextGeneration(g6);
  if (isCoverableBy7x7(*g1)) {
    uint64_t idx = getNextCandidateIndex(numCoveredCandidates);
    coveredCandidates[idx] = {pattern, *g1, (uint16_t)(*generations + 6)};
    *generations = 0;
    return true;
  }

  *generations += 6;

  // Check for cycles
  if ((*g1 == g2) || (*g1 == g3) || (*g1 == g4)) {
    *generations = 0;
    return true;  // Pattern ended/cyclical, advance to next
  }

  // Check if reached minimum candidate generations (not covered)
  if (*generations >= MIN_CANDIDATE_GENERATIONS) {
    uint64_t idx = getNextCandidateIndex(numUncoveredCandidates);
    uncoveredCandidates[idx] = pattern;
    *generations = 0;
    return true;  // Candidate found, advance to next
  }

  return false;  // Continue with current pattern
}

#endif