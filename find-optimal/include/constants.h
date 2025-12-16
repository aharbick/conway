#ifndef _CONSTANTS_H_
#define _CONSTANTS_H_

// Frame search constants
#define FRAME_SEARCH_GRID_SIZE 1024
#define FRAME_SEARCH_THREADS_PER_BLOCK 1024
#define FRAME_SEARCH_NUM_K_BITS 4
#define FRAME_SEARCH_NUM_P_BITS 16
#define FRAME_SEARCH_MAX_CANDIDATES (1ULL << 31)  // 2B candidates = 16GB
#define FRAME_SEARCH_MAX_FRAMES (1 << 24)
#define FRAME_SEARCH_KERNEL_PATTERN_INCREMENT 0x1000000ULL
#define FRAME_SEARCH_TOTAL_MINIMAL_FRAMES 2102800

// Strip search constants (reversibility-based optimization)
// Middle block is rows 2-5 (32 bits), strips are rows 0-1 and 6-7 (16 bits each)
//
// Center 4x4 symmetry reduction:
// The middle 8x4 block is split into left ear (2 cols), center (4 cols), right ear (2 cols).
// By applying D4 symmetry to the center 4x4, we reduce the search space from 2^32 to
// 8548 × 256 × 256 = ~560M combinations (7.7x reduction).
//
// From Burnside's lemma: (2^16 + 2×2^8 + 2×2^10 + 2×2^4 + 2^8) / 8 = 8548
// Reference: https://oeis.org/A054247
#define CENTER_4X4_TOTAL_UNIQUE 8548
#define CENTER_4X4_TOTAL_EAR_VALUES 256              // 2^8 possible values per ear (left or right)
#define CENTER_4X4_EARS_PER_CENTER 65536             // 256 × 256 ear combinations per center
#define MIDDLE_BLOCK_REPORT_INTERVAL 100             // Report progress every N middle blocks
//
// Phase 1: Find unique strips (small kernel, 65K strips to test)
#define STRIP_SEARCH_UNIQUE_GRID_SIZE 256
#define STRIP_SEARCH_UNIQUE_THREADS_PER_BLOCK 256
#define STRIP_SEARCH_TOTAL_STRIPS 65536              // 2^16 possible strips
#define STRIP_SEARCH_MAX_VALID_STRIPS 32768          // Max unique strips per block (measured up to ~26K)
//
// StripHashTable: CityHash-based hash table for signature deduplication
// Uses power-of-2 size with bitwise AND for fast indexing
#define STRIP_HASH_TABLE_SIZE_BITS 17                // 2^17 = 128K slots (~20% load factor for 26K entries)
#define STRIP_HASH_TABLE_SIZE (1U << STRIP_HASH_TABLE_SIZE_BITS)
#define STRIP_HASH_TABLE_MASK (STRIP_HASH_TABLE_SIZE - 1)
#define STRIP_MAX_PROBE_LENGTH 512                   // Safety limit (should need < 100 with good hash)
//
// Phase 2: Test strip combinations (large kernel, ~289M combinations)
#define STRIP_SEARCH_COMBO_GRID_SIZE 1024
#define STRIP_SEARCH_COMBO_THREADS_PER_BLOCK 1024
#define STRIP_SEARCH_MAX_CANDIDATES (1ULL << 31)     // 2B candidates per middle block (matches frame search)

// Subgrid cache constants
#define SUBGRID_TOTAL_PATTERNS (1ULL << 49)  // 7x7 grid = 2^49 patterns
#ifdef TOPOLOGY_TORUS
#define SUBGRID_MIN_GENERATIONS 260           // Torus topology has more long-running patterns.  This setting results in a comparably sized cache.
#else
#define SUBGRID_MIN_GENERATIONS 180           // Box/plane topology (empirically results in about 1GB cache)
#endif
#define SUBGRID_GRID_SIZE 1024
#define SUBGRID_THREADS_PER_BLOCK 1024
#define SUBGRID_PATTERNS_PER_THREAD (1 << 16)  // Each thread processes 2^16 base 7x7 patterns (×4 translations)
#define SUBGRID_CACHE_MAX_CANDIDATES (1 << 26)  // 64M candidates per batch (~896MB buffer for generation)


// Common buffer sizes
#define BINARY_STRING_BUFFER_SIZE 65
#define MESSAGE_BUFFER_SIZE 64

// Nivasch algorithm parameters
#define NIVASCH_MAX_STACK_SIZE 50
#define NIVASCH_NUM_STACKS 10

#endif
