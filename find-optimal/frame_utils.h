#ifndef _FRAME_UTILS_H_
#define _FRAME_UTILS_H_

#include <stdbool.h>
#include "types.h"

// Frame bit positions (F) in the 8x8 grid:
//   FFFooFFF
//   FFooooFF 
//   FooooooF
//   oooooooo
//   oooooooo
//   FooooooF
//   FFooooFF
//   FFFooFFF

// Rotates a pattern 90 degrees clockwise
__host__ __device__ inline ulong64 rotate90(ulong64 pattern) {
    ulong64 result = 0;
    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            if (pattern & (1ULL << (row * 8 + col))) {
                // In rotated pattern, row becomes col, col becomes (7-row)
                result |= 1ULL << ((7-col) * 8 + row);
            }
        }
    }
    return result;
}

// Reflects a pattern horizontally
__host__ __device__ inline ulong64 reflectHorizontal(ulong64 pattern) {
    ulong64 result = 0;
    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            if (pattern & (1ULL << (row * 8 + col))) {
                result |= 1ULL << (row * 8 + (7-col));
            }
        }
    }
    return result;
}

// Extract frame bits from a pattern
__host__ __device__ inline ulong64 extractFrame(ulong64 pattern) {
    // Frame mask - 1s for frame positions, 0s elsewhere
    static const ulong64 FRAME_MASK = 
        0xE7ULL << 56 |  // FFFooFFF
        0xC3ULL << 48 |  // FFooooFF
        0x81ULL << 40 |  // FooooooF
                         // oooooooo
                         // oooooooo
        0x81ULL << 16 |  // FooooooF
        0xC3ULL << 8  |  // FFooooFF
        0xE7ULL;         // FFFooFFF

    return pattern & FRAME_MASK;
}

// Spreads 24 bits into frame positions
__host__ __device__ inline ulong64 spreadBitsToFrame(ulong64 bits) {
    ulong64 result = 0;
    int bitPos = 0;
    
    // Row offsets and patterns
    static const int offsets[] = {56, 48, 40, 16, 8, 0};  // Row bit offsets
    static const ulong64 patterns[] = {
        0xE7,  // FFFooFFF - 11100111
        0xC3,  // FFooooFF - 11000011
        0x81,  // FooooooF - 10000001
        0x81,  // FooooooF - 10000001
        0xC3,  // FFooooFF - 11000011
        0xE7   // FFFooFFF - 11100111
    };
    
    // Process each row
    for (int row = 0; row < 6; row++) {
        ulong64 pattern = patterns[row];
        int offset = offsets[row];
        
        // Process each bit in the pattern
        for (int pos = 7; pos >= 0; pos--) {
            if (pattern & (1ULL << pos)) {
                result |= ((bits >> bitPos++) & 1ULL) << (offset + pos);
            }
        }
    }
    
    return result;
}

// Returns true if this frame is the lexicographically minimal version
// among all its rotations and reflections
__host__ __device__ inline bool isMinimalFrame(ulong64 frame) {
    ulong64 min = frame;
    
    // Check all rotations
    ulong64 rotated = frame;
    for (int i = 0; i < 3; i++) {
        rotated = rotate90(rotated);
        if (rotated < min) return false;
    }
    
    // Check horizontal reflection and its rotations
    ulong64 reflected = reflectHorizontal(frame);
    if (reflected < min) return false;
    
    rotated = reflected;
    for (int i = 0; i < 3; i++) {
        rotated = rotate90(rotated);
        if (rotated < min) return false;
    }
    
    return true;
}

#endif 