#ifndef _CONSTANTS_H_
#define _CONSTANTS_H_

// Frame search constants
#define FRAME_SEARCH_GRID_SIZE 1024
#define FRAME_SEARCH_THREADS_PER_BLOCK 1024
#define FRAME_SEARCH_NUM_K_BITS 4
#define FRAME_SEARCH_NUM_P_BITS 16
#define FRAME_SEARCH_MAX_CANDIDATES (1 << 30)
#define FRAME_SEARCH_MAX_FRAMES (1 << 24)
#define FRAME_SEARCH_KERNEL_PATTERN_INCREMENT 0x1000000ULL
#define FRAME_SEARCH_TOTAL_MINIMAL_FRAMES 2102800

// Subgrid cache constants
#define SUBGRID_TOTAL_PATTERNS (1ULL << 49)  // 7x7 grid = 2^49 patterns
#define SUBGRID_MIN_GENERATIONS 180           // Only save patterns with >= 180 generations
#define SUBGRID_GRID_SIZE 1024
#define SUBGRID_THREADS_PER_BLOCK 1024
#define SUBGRID_PATTERNS_PER_THREAD (1 << 16)  // Each thread processes 2^16 base 7x7 patterns (×4 translations)
#define SUBGRID_CACHE_MAX_CANDIDATES (1 << 20)  // 1M candidates per batch (~14MB)


// Common buffer sizes
#define BINARY_STRING_BUFFER_SIZE 65
#define MESSAGE_BUFFER_SIZE 64

// Nivasch algorithm parameters
#define NIVASCH_MAX_STACK_SIZE 50
#define NIVASCH_NUM_STACKS 10

#endif
