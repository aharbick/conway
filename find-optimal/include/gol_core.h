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

// Conway's Game of Life bit manipulation masks - 8x8 grid
#define GOL_8X8_HORIZONTAL_SHIFT_MASK 0x7F7F7F7F7F7F7F7FULL  // Bits 0-6 in each byte
#define GOL_8X8_VERTICAL_SHIFT_MASK   0xFEFEFEFEFEFEFEFEULL  // Bits 1-7 in each byte
#define GOL_8X8_LEFT_EDGE_MASK        0x8080808080808080ULL  // Bit 7 in each byte (left edge)
#define GOL_8X8_RIGHT_EDGE_MASK       0x0101010101010101ULL  // Bit 0 in each byte (right edge)

// 7x7 grid masks (byte-aligned format: 7 bits per byte, 7 rows)
#define GOL_7X7_HORIZONTAL_SHIFT_MASK 0x3F3F3F3F3F3F3F3FULL  // Bits 0-5 in each byte
#define GOL_7X7_VERTICAL_SHIFT_MASK   0x7E7E7E7E7E7E7E7EULL  // Bits 1-6 in each byte
#define GOL_7X7_LEFT_EDGE_MASK        0x4040404040404040ULL  // Bit 6 in each byte (left edge)
#define GOL_7X7_RIGHT_EDGE_MASK       0x0101010101010101ULL  // Bit 0 in each byte (right edge)

// 7x7 subgrid position masks for coverage testing on 8x8 grid
#define GOL_7X7_POSITION_0_MASK       0x7F7F7F7F7F7F7FULL    // Rows 0-6, cols 0-6
#define GOL_7X7_POSITION_1_MASK       0xFEFEFEFEFEFEFEULL    // Rows 0-6, cols 1-7
#define GOL_7X7_POSITION_2_MASK       0x7F7F7F7F7F7F7F00ULL  // Rows 1-7, cols 0-6
#define GOL_7X7_POSITION_3_MASK       0xFEFEFEFEFEFEFE00ULL  // Rows 1-7, cols 1-7

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

// ============================================================================
// Grid Shift Operations - 8x8 and 7x7
// ============================================================================
// Tom Rokicki's optimized 19-operation Conway's Game of Life computation
// Topology selection: define TOPOLOGY_TORUS at compile time for wrapping boundaries
// Default (no flag): box/plane topology with non-wrapping boundaries

// Helper: Unpack compact 7x7 to byte-aligned format for easier bit manipulation
// Converts: row at (7*n) bit offset → row at (8*n) bit offset (byte-aligned)
__host__ __device__ static inline uint64_t unpack7x7(uint64_t compact) {
  uint64_t result = 0;
  result |= ((compact >>  0) & 0x7F) <<  0;  // Row 0: bits 0-6 → bits 0-6
  result |= ((compact >>  7) & 0x7F) <<  8;  // Row 1: bits 7-13 → bits 8-14
  result |= ((compact >> 14) & 0x7F) << 16;  // Row 2: bits 14-20 → bits 16-22
  result |= ((compact >> 21) & 0x7F) << 24;  // Row 3: bits 21-27 → bits 24-30
  result |= ((compact >> 28) & 0x7F) << 32;  // Row 4: bits 28-34 → bits 32-38
  result |= ((compact >> 35) & 0x7F) << 40;  // Row 5: bits 35-41 → bits 40-46
  result |= ((compact >> 42) & 0x7F) << 48;  // Row 6: bits 42-48 → bits 48-54
  return result;
}

// Helper: Pack byte-aligned format back to compact 7x7
__host__ __device__ static inline uint64_t pack7x7(uint64_t unpacked) {
  uint64_t result = 0;
  result |= ((unpacked >>  0) & 0x7F) <<  0;  // Row 0: bits 0-6 → bits 0-6
  result |= ((unpacked >>  8) & 0x7F) <<  7;  // Row 1: bits 8-14 → bits 7-13
  result |= ((unpacked >> 16) & 0x7F) << 14;  // Row 2: bits 16-22 → bits 14-20
  result |= ((unpacked >> 24) & 0x7F) << 21;  // Row 3: bits 24-30 → bits 21-27
  result |= ((unpacked >> 32) & 0x7F) << 28;  // Row 4: bits 32-38 → bits 28-34
  result |= ((unpacked >> 40) & 0x7F) << 35;  // Row 5: bits 40-46 → bits 35-41
  result |= ((unpacked >> 48) & 0x7F) << 42;  // Row 6: bits 48-54 → bits 42-48
  return result;
}

#ifdef TOPOLOGY_TORUS

// Circular shift helpers for torus topology (wrapping boundaries) - 8x8 grid
__host__ __device__ static inline uint64_t shiftLeft8x8(uint64_t a) {
  uint64_t shifted = (a & GOL_8X8_HORIZONTAL_SHIFT_MASK) << 1;
  uint64_t wrapped = (a & GOL_8X8_LEFT_EDGE_MASK) >> 7;
  return shifted | wrapped;
}

__host__ __device__ static inline uint64_t shiftRight8x8(uint64_t a) {
  uint64_t shifted = (a & GOL_8X8_VERTICAL_SHIFT_MASK) >> 1;
  uint64_t wrapped = (a & GOL_8X8_RIGHT_EDGE_MASK) << 7;
  return shifted | wrapped;
}

__host__ __device__ static inline uint64_t shiftUp8x8(uint64_t a) {
  uint64_t shifted = a << 8;
  uint64_t wrapped = (a >> 56) & 0xFFULL;
  return shifted | wrapped;
}

__host__ __device__ static inline uint64_t shiftDown8x8(uint64_t a) {
  uint64_t shifted = a >> 8;
  uint64_t wrapped = (a & 0xFFULL) << 56;
  return shifted | wrapped;
}

// Circular shift helpers for torus topology (wrapping boundaries) - 7x7 grid
__host__ __device__ static inline uint64_t shiftLeft7x7(uint64_t a) {
  uint64_t shifted = (a & GOL_7X7_HORIZONTAL_SHIFT_MASK) << 1;
  uint64_t wrapped = (a & GOL_7X7_LEFT_EDGE_MASK) >> 6;
  return shifted | wrapped;
}

__host__ __device__ static inline uint64_t shiftRight7x7(uint64_t a) {
  uint64_t shifted = (a & GOL_7X7_VERTICAL_SHIFT_MASK) >> 1;
  uint64_t wrapped = (a & GOL_7X7_RIGHT_EDGE_MASK) << 6;
  return shifted | wrapped;
}

__host__ __device__ static inline uint64_t shiftUp7x7(uint64_t a) {
  uint64_t shifted = a << 8;
  uint64_t wrapped = (a >> 48) & 0x7FULL;
  return shifted | wrapped;
}

__host__ __device__ static inline uint64_t shiftDown7x7(uint64_t a) {
  uint64_t shifted = a >> 8;
  uint64_t wrapped = (a & 0x7FULL) << 48;
  return shifted | wrapped;
}

#else

// Box/plane shift helpers for non-wrapping boundaries - 8x8 grid
__host__ __device__ static inline uint64_t shiftLeft8x8(uint64_t a) {
  return (a & GOL_8X8_HORIZONTAL_SHIFT_MASK) << 1;
}

__host__ __device__ static inline uint64_t shiftRight8x8(uint64_t a) {
  return (a & GOL_8X8_VERTICAL_SHIFT_MASK) >> 1;
}

__host__ __device__ static inline uint64_t shiftUp8x8(uint64_t a) {
  return a << 8;
}

__host__ __device__ static inline uint64_t shiftDown8x8(uint64_t a) {
  return a >> 8;
}

// Box/plane shift helpers for non-wrapping boundaries - 7x7 grid
__host__ __device__ static inline uint64_t shiftLeft7x7(uint64_t a) {
  return (a & GOL_7X7_HORIZONTAL_SHIFT_MASK) << 1;
}

__host__ __device__ static inline uint64_t shiftRight7x7(uint64_t a) {
  return (a & GOL_7X7_VERTICAL_SHIFT_MASK) >> 1;
}

__host__ __device__ static inline uint64_t shiftUp7x7(uint64_t a) {
  return a << 8;
}

__host__ __device__ static inline uint64_t shiftDown7x7(uint64_t a) {
  return a >> 8;
}

#endif

// Generic template for computing next generation using Rokicki's algorithm
// Works with any set of shift functions (8x8 or 7x7, torus or plane)
template<typename ShiftLeft, typename ShiftRight, typename ShiftUp, typename ShiftDown>
__host__ __device__ static inline uint64_t computeNextGeneration(
    uint64_t a,
    ShiftLeft shiftLeft,
    ShiftRight shiftRight,
    ShiftUp shiftUp,
    ShiftDown shiftDown) {
  uint64_t s0, sh2, a0, a1, sll, slh;
  add2(shiftLeft(a), shiftRight(a), s0, sh2);
  add2(s0, a, a0, a1);
  a1 |= sh2;
  add3(shiftDown(a0), shiftUp(a0), s0, sll, slh);
  uint64_t y = shiftDown(a1);
  uint64_t x = shiftUp(a1);
  return (x ^ y ^ sh2 ^ slh) & ((x | y) ^ (sh2 | slh)) & (sll | a);
}

// 8x8 next generation computation
__host__ __device__ static inline uint64_t computeNextGeneration8x8(uint64_t a) {
  return computeNextGeneration(a, shiftLeft8x8, shiftRight8x8, shiftUp8x8, shiftDown8x8);
}

// 7x7 unpacked format valid bits mask (bits 0-6 of each byte, 7 rows only - byte 8 should be 0)
#define GOL_7X7_UNPACKED_MASK 0x007F7F7F7F7F7F7FULL

// 7x7 next generation computation using Rokicki's algorithm adapted for 7x7
// Works for both torus and plane topologies (determined by shift functions above)
// Input and output are in unpacked format (byte-aligned: 7 bits per byte, 7 rows)
__host__ __device__ static inline uint64_t computeNextGeneration7x7(uint64_t unpacked7x7) {
  uint64_t result = computeNextGeneration(unpacked7x7, shiftLeft7x7, shiftRight7x7, shiftUp7x7, shiftDown7x7);
  return result & GOL_7X7_UNPACKED_MASK;  // Mask to ensure only valid 7x7 bits are set
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

// Floyd's cycle detection algorithm - generic version using template
template<typename ComputeNextGenerationFunc>
__host__ __device__ static inline int floydCycleDetection(uint64_t startState, int startGenerations,
                                                          ComputeNextGenerationFunc computeNextGeneration) {
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

// Nivasch's cycle detection algorithm - stack-based approach with template
struct NivaschStackEntry {
  uint64_t value;
  int time;
};

template<typename ComputeNextGenerationFunc>
__host__ __device__ static inline int nivaschCycleDetection(uint64_t startState, int startGenerations,
                                                            ComputeNextGenerationFunc computeNextGeneration) {
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

// Core generation counting logic - template version that works with any computeNextGeneration function
template<typename ComputeNextGenerationFunc>
__host__ __device__ static inline int countGenerations(uint64_t pattern,
                                                       ComputeNextGenerationFunc computeNextGeneration,
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
      return nivaschCycleDetection(g1, generations, computeNextGeneration);
    } else {
      return floydCycleDetection(g1, generations, computeNextGeneration);
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

// ============================================================================
// 8x8 Grid Helper - Expand 7x7 pattern to specific position on 8x8 grid
// ============================================================================
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

  // Test 4 possible 7x7 positions within the 8x8 grid
  if ((pattern & ~GOL_7X7_POSITION_0_MASK) == 0) return true;
  if ((pattern & ~GOL_7X7_POSITION_1_MASK) == 0) return true;
  if ((pattern & ~GOL_7X7_POSITION_2_MASK) == 0) return true;
  if ((pattern & ~GOL_7X7_POSITION_3_MASK) == 0) return true;

  return false;
}

// Core 6-generation stepping and cycle detection logic
// Used by both CPU tests and CUDA kernels
__host__ __device__ static inline bool step6GenerationsAndCheck(uint64_t* g1, uint64_t pattern, uint16_t* generations,
                                                                uint64_t* candidates, uint64_t* numCandidates) {
  *generations += 6;
  uint64_t g2 = computeNextGeneration8x8(*g1);
  uint64_t g3 = computeNextGeneration8x8(g2);
  uint64_t g4 = computeNextGeneration8x8(g3);
  uint64_t g5 = computeNextGeneration8x8(g4);
  uint64_t g6 = computeNextGeneration8x8(g5);
  *g1 = computeNextGeneration8x8(g6);

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

// Macro for early termination based on 7x7 cache lookup
// Checks if generation is 7x7-coverable and if so, looks up in cache
// Cache contains patterns with ≥180 generations
// Logic: cache hit OR (cache miss + *generations > 20) → save as candidate
//        cache miss + *generations ≤ 20 → dead end (max 20+179=199)
#define RETURN_EARLY_IF_7X7_COVERAGE(gen, generations, pattern, candidates, numCandidates, cache) \
  do { \
    if (isCoverableBy7x7(gen)) { \
      uint16_t cachedGens = cache->get(gen); \
      if (cachedGens > 0 || *generations > 20) { \
        uint64_t idx = getNextCandidateIndex(numCandidates); \
        candidates[idx] = pattern; \
        *generations = 0; \
        return true; \
      } \
      *generations = 0; \
      return true; \
    } \
  } while (0)

// Cache-accelerated 6-generation stepping with early candidate detection
// Checks each generation against the 7x7 subgrid cache to detect long-running patterns early
// This version checks after computing each generation to terminate as early as possible
__host__ __device__ static inline bool step6GenerationsAndCheckWithCache(
    uint64_t* g1, uint64_t pattern, uint16_t* generations,
    uint64_t* candidates, uint64_t* numCandidates,
    const SubgridHashTable* cache) {

  // TODO in the non-cache step6 function we do:
  //   *generations += 6;
  //
  // But for some reason here...  If we do that instead of individual (*generations)++ it crashes with
  //    CUDA Error: an illegal memory access was encountered /home/aharbick/conway/find-optimal/src/gol_cuda.cu 149

  // Check g1 first (carried over from previous iteration)
  (*generations)++;
  RETURN_EARLY_IF_7X7_COVERAGE(*g1, generations, pattern, candidates, numCandidates, cache);

  // Compute g2 and check cache immediately
  uint64_t g2 = computeNextGeneration8x8(*g1);
  (*generations)++;
  RETURN_EARLY_IF_7X7_COVERAGE(g2, generations, pattern, candidates, numCandidates, cache);

  // Compute g3 and check cache immediately
  uint64_t g3 = computeNextGeneration8x8(g2);
  (*generations)++;
  RETURN_EARLY_IF_7X7_COVERAGE(g3, generations, pattern, candidates, numCandidates, cache);

  // Compute g4 and check cache immediately
  uint64_t g4 = computeNextGeneration8x8(g3);
  (*generations)++;
  RETURN_EARLY_IF_7X7_COVERAGE(g4, generations, pattern, candidates, numCandidates, cache);

  // Compute g5 and check cache immediately
  uint64_t g5 = computeNextGeneration8x8(g4);
  (*generations)++;
  RETURN_EARLY_IF_7X7_COVERAGE(g5, generations, pattern, candidates, numCandidates, cache);

  // Compute g6 and check cache immediately
  uint64_t g6 = computeNextGeneration8x8(g5);
  (*generations)++;
  RETURN_EARLY_IF_7X7_COVERAGE(g6, generations, pattern, candidates, numCandidates, cache);

  // Compute next g1... cache check and generation happens at the beginning
  *g1 = computeNextGeneration8x8(g6);

  // Check for cycles
  if ((*g1 == g2) || (*g1 == g3) || (*g1 == g4)) {
    *generations = 0;
    return true;
  }

  // Check if reached minimum candidate generations
  if (*generations >= MIN_CANDIDATE_GENERATIONS) {
    uint64_t idx = getNextCandidateIndex(numCandidates);
    candidates[idx] = pattern;
    *generations = 0;
    return true;
  }

  return false;
}

#endif