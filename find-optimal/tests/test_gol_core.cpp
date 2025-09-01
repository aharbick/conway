#include <gtest/gtest.h>

// Define CUDA decorators as empty for CPU compilation  
#ifndef __NVCC__
#define __host__
#define __device__
#define __global__
#endif

#include "../gol_core.h"

class GOLComputationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Any setup needed before each test
    }

    // Helper function to create a pattern from a 2D array
    ulong64 createPattern(const char grid[8][8]) {
        ulong64 pattern = 0;
        for (int row = 0; row < 8; row++) {
            for (int col = 0; col < 8; col++) {
                if (grid[row][col] == '1') {
                    pattern |= 1ULL << (row * 8 + col);
                }
            }
        }
        return pattern;
    }

    // Helper to verify a pattern matches expected grid
    void verifyPattern(ulong64 pattern, const char expected[8][8]) {
        for (int row = 0; row < 8; row++) {
            for (int col = 0; col < 8; col++) {
                bool bitSet = (pattern & (1ULL << (row * 8 + col))) != 0;
                bool expectedSet = (expected[row][col] == '1');
                EXPECT_EQ(bitSet, expectedSet) << "Mismatch at (" << row << "," << col << ")";
            }
        }
    }
};

TEST_F(GOLComputationTest, ComputeNextGenerationEmpty) {
    // Empty grid should stay empty
    ulong64 empty = 0;
    ulong64 next = computeNextGeneration(empty);
    EXPECT_EQ(next, 0);
}

TEST_F(GOLComputationTest, ComputeNextGenerationSingleCell) {
    // Single cell should die (underpopulation)
    ulong64 single = 1ULL << (4 * 8 + 4); // Center cell
    ulong64 next = computeNextGeneration(single);
    EXPECT_EQ(next, 0);
}

TEST_F(GOLComputationTest, ComputeNextGenerationBlock) {
    // 2x2 block should be stable (still life)
    const char block[8][8] = {
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','1','1','0','0','0'},
        {'0','0','0','1','1','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'}
    };
    
    ulong64 pattern = createPattern(block);
    ulong64 next = computeNextGeneration(pattern);
    
    // Should remain unchanged
    EXPECT_EQ(next, pattern);
}

TEST_F(GOLComputationTest, ComputeNextGenerationBlinker) {
    // Vertical blinker should become horizontal
    const char verticalBlinker[8][8] = {
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','1','0','0','0','0'},
        {'0','0','0','1','0','0','0','0'},
        {'0','0','0','1','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'}
    };
    
    const char horizontalBlinker[8][8] = {
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','1','1','1','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'}
    };
    
    ulong64 vertical = createPattern(verticalBlinker);
    ulong64 next = computeNextGeneration(vertical);
    
    verifyPattern(next, horizontalBlinker);
}

TEST_F(GOLComputationTest, ComputeNextGenerationGlider) {
    // Classic glider pattern - should move down-right
    const char glider[8][8] = {
        {'0','0','0','0','0','0','0','0'},
        {'0','0','1','0','0','0','0','0'},
        {'0','0','0','1','0','0','0','0'},
        {'0','1','1','1','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'}
    };
    
    const char gliderNext[8][8] = {
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','1','0','1','0','0','0','0'},
        {'0','0','1','1','0','0','0','0'},
        {'0','0','1','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'}
    };
    
    ulong64 pattern = createPattern(glider);
    ulong64 next = computeNextGeneration(pattern);
    
    verifyPattern(next, gliderNext);
}

TEST_F(GOLComputationTest, ComputeNextGenerationBeehive) {
    // Beehive is a stable pattern (still life)
    const char beehive[8][8] = {
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','1','1','0','0','0'},
        {'0','0','1','0','0','1','0','0'},
        {'0','0','0','1','1','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'}
    };
    
    ulong64 pattern = createPattern(beehive);
    ulong64 next = computeNextGeneration(pattern);
    
    // Should remain unchanged
    EXPECT_EQ(next, pattern);
}

TEST_F(GOLComputationTest, ComputeNextGenerationBoundaryConditions) {
    // Test that cells at edges are treated as having dead neighbors
    // Single cell at corner should die
    ulong64 corner = 1ULL << (0 * 8 + 0); // Top-left corner
    ulong64 next = computeNextGeneration(corner);
    EXPECT_EQ(next, 0);
    
    // Single cell at edge should die
    ulong64 edge = 1ULL << (0 * 8 + 4); // Top edge
    next = computeNextGeneration(edge);
    EXPECT_EQ(next, 0);
}

TEST_F(GOLComputationTest, ComputeNextGenerationOvercrowding) {
    // Test overcrowding rule with a simpler case
    // Single cell with too many neighbors should die
    const char overcrowded[8][8] = {
        {'0','0','0','0','0','0','0','0'},
        {'0','1','1','1','0','0','0','0'},
        {'0','1','1','1','0','0','0','0'},
        {'0','1','1','1','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'}
    };
    
    const char overcrowdedNext[8][8] = {
        {'0','0','1','0','0','0','0','0'},
        {'0','1','0','1','0','0','0','0'},
        {'1','0','0','0','1','0','0','0'},
        {'0','1','0','1','0','0','0','0'},
        {'0','0','1','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'}
    };
    ulong64 pattern = createPattern(overcrowded);
    ulong64 next = computeNextGeneration(pattern);
    
    verifyPattern(next, overcrowdedNext);
}

TEST_F(GOLComputationTest, ComputeNextGenerationBirth) {
    // Test birth rule - empty cell with exactly 3 neighbors should become alive
    const char birthSetup[8][8] = {
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','1','1','0','0','0','0'},
        {'0','0','0','1','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'}
    };
    
    const char afterBirth[8][8] = {
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','1','1','0','0','0','0'},
        {'0','0','1','1','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'},
        {'0','0','0','0','0','0','0','0'}
    };
    
    ulong64 pattern = createPattern(birthSetup);
    ulong64 next = computeNextGeneration(pattern);
    
    verifyPattern(next, afterBirth);
}
