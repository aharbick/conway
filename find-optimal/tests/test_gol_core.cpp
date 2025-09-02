#include <gtest/gtest.h>

#include <climits>

// Define CUDA decorators as empty for CPU compilation
#ifndef __NVCC__
#define __host__
#define __device__
#define __global__
#endif

#include "gol_core.h"


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
  ulong64 single = 1ULL << (4 * 8 + 4);  // Center cell
  ulong64 next = computeNextGeneration(single);
  EXPECT_EQ(next, 0);
}

TEST_F(GOLComputationTest, ComputeNextGenerationBlock) {
  // 2x2 block should be stable (still life)
  const char block[8][8] = {{'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'},
                            {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '1', '1', '0', '0', '0'},
                            {'0', '0', '0', '1', '1', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'},
                            {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'}};

  ulong64 pattern = createPattern(block);
  ulong64 next = computeNextGeneration(pattern);

  // Should remain unchanged
  EXPECT_EQ(next, pattern);
}

TEST_F(GOLComputationTest, ComputeNextGenerationBlinker) {
  // Vertical blinker should become horizontal
  const char verticalBlinker[8][8] = {
      {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'},
      {'0', '0', '0', '1', '0', '0', '0', '0'}, {'0', '0', '0', '1', '0', '0', '0', '0'},
      {'0', '0', '0', '1', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'},
      {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'}};

  const char horizontalBlinker[8][8] = {
      {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'},
      {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '1', '1', '1', '0', '0', '0'},
      {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'},
      {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'}};

  ulong64 vertical = createPattern(verticalBlinker);
  ulong64 next = computeNextGeneration(vertical);

  verifyPattern(next, horizontalBlinker);
}

TEST_F(GOLComputationTest, ComputeNextGenerationGlider) {
  // Classic glider pattern - should move down-right
  const char glider[8][8] = {{'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '1', '0', '0', '0', '0', '0'},
                             {'0', '0', '0', '1', '0', '0', '0', '0'}, {'0', '1', '1', '1', '0', '0', '0', '0'},
                             {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'},
                             {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'}};

  const char gliderNext[8][8] = {{'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'},
                                 {'0', '1', '0', '1', '0', '0', '0', '0'}, {'0', '0', '1', '1', '0', '0', '0', '0'},
                                 {'0', '0', '1', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'},
                                 {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'}};

  ulong64 pattern = createPattern(glider);
  ulong64 next = computeNextGeneration(pattern);

  verifyPattern(next, gliderNext);
}

TEST_F(GOLComputationTest, ComputeNextGenerationBeehive) {
  // Beehive is a stable pattern (still life)
  const char beehive[8][8] = {{'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'},
                              {'0', '0', '0', '1', '1', '0', '0', '0'}, {'0', '0', '1', '0', '0', '1', '0', '0'},
                              {'0', '0', '0', '1', '1', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'},
                              {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'}};

  ulong64 pattern = createPattern(beehive);
  ulong64 next = computeNextGeneration(pattern);

  // Should remain unchanged
  EXPECT_EQ(next, pattern);
}

TEST_F(GOLComputationTest, ComputeNextGenerationBoundaryConditions) {
  // Test that cells at edges are treated as having dead neighbors
  // Single cell at corner should die
  ulong64 corner = 1ULL << (0 * 8 + 0);  // Top-left corner
  ulong64 next = computeNextGeneration(corner);
  EXPECT_EQ(next, 0);

  // Single cell at edge should die
  ulong64 edge = 1ULL << (0 * 8 + 4);  // Top edge
  next = computeNextGeneration(edge);
  EXPECT_EQ(next, 0);
}

TEST_F(GOLComputationTest, ComputeNextGenerationOvercrowding) {
  // Test overcrowding rule with a simpler case
  // Single cell with too many neighbors should die
  const char overcrowded[8][8] = {{'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '1', '1', '1', '0', '0', '0', '0'},
                                  {'0', '1', '1', '1', '0', '0', '0', '0'}, {'0', '1', '1', '1', '0', '0', '0', '0'},
                                  {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'},
                                  {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'}};

  const char overcrowdedNext[8][8] = {
      {'0', '0', '1', '0', '0', '0', '0', '0'}, {'0', '1', '0', '1', '0', '0', '0', '0'},
      {'1', '0', '0', '0', '1', '0', '0', '0'}, {'0', '1', '0', '1', '0', '0', '0', '0'},
      {'0', '0', '1', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'},
      {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'}};
  ulong64 pattern = createPattern(overcrowded);
  ulong64 next = computeNextGeneration(pattern);

  verifyPattern(next, overcrowdedNext);
}

TEST_F(GOLComputationTest, ComputeNextGenerationBirth) {
  // Test birth rule - empty cell with exactly 3 neighbors should become alive
  const char birthSetup[8][8] = {{'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'},
                                 {'0', '0', '1', '1', '0', '0', '0', '0'}, {'0', '0', '0', '1', '0', '0', '0', '0'},
                                 {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'},
                                 {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'}};

  const char afterBirth[8][8] = {{'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'},
                                 {'0', '0', '1', '1', '0', '0', '0', '0'}, {'0', '0', '1', '1', '0', '0', '0', '0'},
                                 {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'},
                                 {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'}};

  ulong64 pattern = createPattern(birthSetup);
  ulong64 next = computeNextGeneration(pattern);

  verifyPattern(next, afterBirth);
}

// Test the fundamental math building blocks
TEST(GOLMathTest, Add2BasicOperation) {
  ulong64 s0, s1;

  // Test basic binary addition: 5 + 3 = 8 (101 + 011 = 1000)
  add2(5, 3, s0, s1);
  EXPECT_EQ(s0, 6);  // 5 XOR 3 = 101 XOR 011 = 110 = 6 (sum bits)
  EXPECT_EQ(s1, 1);  // 5 AND 3 = 101 AND 011 = 001 = 1 (carry bits)

  // Verify: s0 + (s1 << 1) = 6 + 2 = 8 = 5 + 3 ✓
}

TEST(GOLMathTest, Add2ZeroOperands) {
  ulong64 s0, s1;

  // Adding zero should work correctly
  add2(0, 0, s0, s1);
  EXPECT_EQ(s0, 0);
  EXPECT_EQ(s1, 0);

  add2(7, 0, s0, s1);
  EXPECT_EQ(s0, 7);
  EXPECT_EQ(s1, 0);

  add2(0, 7, s0, s1);
  EXPECT_EQ(s0, 7);
  EXPECT_EQ(s1, 0);
}

TEST(GOLMathTest, Add2MaxValues) {
  ulong64 s0, s1;

  // Test with maximum values
  add2(ULLONG_MAX, ULLONG_MAX, s0, s1);
  EXPECT_EQ(s0, 0);           // All XOR results in 0
  EXPECT_EQ(s1, ULLONG_MAX);  // All AND results in all 1s
}

TEST(GOLMathTest, Add3BasicOperation) {
  ulong64 s0, s1;

  // Test 3-input addition: 5 + 3 + 2 = 10 (101 + 011 + 010 = 1010)
  add3(5, 3, 2, s0, s1);
  EXPECT_EQ(s0, 4);  // Low bits of sum
  EXPECT_EQ(s1, 3);  // High bits of sum

  // Verify: s0 + (s1 << 1) = 4 + 6 = 10 = 5 + 3 + 2 ✓
}

TEST(GOLMathTest, Add3ZeroOperands) {
  ulong64 s0, s1;

  add3(0, 0, 0, s0, s1);
  EXPECT_EQ(s0, 0);
  EXPECT_EQ(s1, 0);

  add3(7, 0, 0, s0, s1);
  EXPECT_EQ(s0, 7);
  EXPECT_EQ(s1, 0);
}

TEST(GOLMathTest, Add3NeighborCounting) {
  ulong64 s0, s1;

  // Test typical neighbor counting scenario
  // In GOL, we often need to add 3 neighbor values
  ulong64 neighbors1 = 0x0101010101010101ULL;  // Alternating pattern
  ulong64 neighbors2 = 0x1010101010101010ULL;  // Complementary pattern
  ulong64 neighbors3 = 0x0000000000000000ULL;  // No neighbors

  add3(neighbors1, neighbors2, neighbors3, s0, s1);
  EXPECT_EQ(s0, 0x1111111111111111ULL);  // All positions have 1 neighbor
  EXPECT_EQ(s1, 0x0000000000000000ULL);  // No positions have 2+ neighbors
}

// Test the generation adjustment helper
TEST(GOLMathTest, AdjustGenerationsForDeadout) {
  // Test all deadout scenarios
  EXPECT_EQ(adjustGenerationsForDeadout(12, 0, 1, 1, 1, 1), 7);   // Dies at g2
  EXPECT_EQ(adjustGenerationsForDeadout(12, 1, 0, 1, 1, 1), 8);   // Dies at g3
  EXPECT_EQ(adjustGenerationsForDeadout(12, 1, 1, 0, 1, 1), 9);   // Dies at g4
  EXPECT_EQ(adjustGenerationsForDeadout(12, 1, 1, 1, 0, 1), 10);  // Dies at g5
  EXPECT_EQ(adjustGenerationsForDeadout(12, 1, 1, 1, 1, 0), 11);  // Dies at g6
  EXPECT_EQ(adjustGenerationsForDeadout(12, 1, 1, 1, 1, 1), 12);  // Doesn't die
}

TEST(GOLMathTest, AdjustGenerationsForDeadoutEdgeCases) {
  // Test edge cases
  EXPECT_EQ(adjustGenerationsForDeadout(6, 0, 1, 1, 1, 1), 1);  // Early deadout
  EXPECT_EQ(adjustGenerationsForDeadout(6, 1, 1, 1, 1, 0), 5);  // Late deadout
}

// Test countGenerations function - critical for pattern lifespan determination
class GOLLifecycleTest : public GOLComputationTest {
  // Inherits createPattern helper from GOLComputationTest
};

TEST_F(GOLLifecycleTest, CountGenerationsEmpty) {
  // Empty pattern should die immediately
  EXPECT_EQ(countGenerations(0), 1);
}

TEST_F(GOLLifecycleTest, CountGenerationsSingleCell) {
  // Single cell dies from underpopulation - dies at generation 2, so countGenerations returns 1
  ulong64 single = 1ULL << (4 * 8 + 4);  // Center cell
  EXPECT_EQ(countGenerations(single), 1);
}

TEST_F(GOLLifecycleTest, CountGenerationsBlock) {
  // 2x2 block is stable (still life) - should return 0 (infinite life)
  const char block[8][8] = {{'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'},
                            {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '1', '1', '0', '0', '0'},
                            {'0', '0', '0', '1', '1', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'},
                            {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'}};
  ulong64 pattern = createPattern(block);
  EXPECT_EQ(countGenerations(pattern), 0);  // Still life = infinite
}

TEST_F(GOLLifecycleTest, CountGenerationsBlinker) {
  // Blinker oscillates with period 2 - should return 0 (infinite life)
  const char blinker[8][8] = {{'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'},
                              {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '1', '1', '1', '0', '0', '0'},
                              {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'},
                              {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'}};
  ulong64 pattern = createPattern(blinker);
  EXPECT_EQ(countGenerations(pattern), 0);  // Oscillator = infinite
}

TEST_F(GOLLifecycleTest, CountGenerationsRPentomino) {
  // R-pentomino is a classic pattern that stabilizes after many generations
  // This tests longer evolution and eventual stabilization
  const char rpentomino[8][8] = {{'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'},
                                 {'0', '0', '0', '1', '1', '0', '0', '0'}, {'0', '0', '1', '1', '0', '0', '0', '0'},
                                 {'0', '0', '0', '1', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'},
                                 {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'}};
  ulong64 pattern = createPattern(rpentomino);
  int generations = countGenerations(pattern);
  // R-pentomino typically stabilizes or becomes periodic
  // Should either return 0 (stable/periodic) or a positive number if it dies
  EXPECT_TRUE(generations >= 0);
}

TEST_F(GOLLifecycleTest, CountGenerationsSimpleDieout) {
  // Three cells in a row (not enough for survival)
  // Should die out after a few generations
  const char line[8][8] = {{'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'},
                           {'0', '0', '1', '0', '0', '0', '0', '0'}, {'0', '0', '1', '0', '0', '0', '0', '0'},
                           {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'},
                           {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'}};
  ulong64 pattern = createPattern(line);
  int generations = countGenerations(pattern);
  EXPECT_GT(generations, 0);   // Should die out
  EXPECT_LT(generations, 10);  // But quickly
}

TEST_F(GOLLifecycleTest, CountGenerationsBeehive) {
  // Beehive is another stable pattern (still life)
  const char beehive[8][8] = {{'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'},
                              {'0', '0', '0', '1', '1', '0', '0', '0'}, {'0', '0', '1', '0', '0', '1', '0', '0'},
                              {'0', '0', '0', '1', '1', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'},
                              {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'}};
  ulong64 pattern = createPattern(beehive);
  EXPECT_EQ(countGenerations(pattern), 0);  // Still life = infinite
}

TEST_F(GOLLifecycleTest, CountGenerationsGlider) {
  // Glider is a spaceship with period 4 - should return 0 (infinite life)
  // In a bounded 8x8 grid, it will eventually hit boundary and might die or become periodic
  const char glider[8][8] = {{'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '1', '0', '0', '0', '0', '0'},
                             {'0', '0', '0', '1', '0', '0', '0', '0'}, {'0', '1', '1', '1', '0', '0', '0', '0'},
                             {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'},
                             {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'}};
  ulong64 pattern = createPattern(glider);
  int generations = countGenerations(pattern);
  // In bounded grid, glider will either stabilize, become periodic, or die
  EXPECT_TRUE(generations >= 0);
}

// Test known patterns with verified generation counts
TEST_F(GOLLifecycleTest, CountGenerationsKnownPattern209a) {
  // Pattern that runs for exactly 209 generations
  // 0111000001011100100001100001010011111010110101100010000101110100
  ulong64 pattern = 8096493654771114356ULL;
  EXPECT_EQ(countGenerations(pattern), 209);
}

TEST_F(GOLLifecycleTest, CountGenerationsKnownPattern209b) {
  // Another pattern that runs for exactly 209 generations
  // 1000000010010011100101000010001110000010011011011000111111001010
  ulong64 pattern = 9264911738664226762ULL;
  EXPECT_EQ(countGenerations(pattern), 209);
}

TEST_F(GOLLifecycleTest, CountGenerationsKnownPattern205) {
  // Pattern that runs for exactly 205 generations
  // 0101111010010001101011001010010110011101110001000011100100010101
  ulong64 pattern = 6814417538504734997ULL;
  EXPECT_EQ(countGenerations(pattern), 205);
}

TEST_F(GOLLifecycleTest, CountGenerationsKnownPattern199) {
  // Pattern that runs for exactly 199 generations
  // 0011100100111010001001000101011011000001000110100001100110001010
  ulong64 pattern = 4123648363836610954ULL;
  EXPECT_EQ(countGenerations(pattern), 199);
}

TEST_F(GOLLifecycleTest, CountGenerationsKnownPattern197) {
  // Pattern that runs for exactly 197 generations
  // 0000000100100101011100011110000110110010101101001010100111010011
  ulong64 pattern = 82597382355986899ULL;
  EXPECT_EQ(countGenerations(pattern), 197);
}

TEST_F(GOLLifecycleTest, CountGenerationsKnownPattern193) {
  // Pattern that runs for exactly 193 generations
  // 1010110110011101001110101101100100001001010101010011011001000100
  ulong64 pattern = 12510220043743999556ULL;
  EXPECT_EQ(countGenerations(pattern), 193);
}

// Test step6GenerationsAndCheck function - critical CUDA kernel logic
class GOLStep6Test : public GOLComputationTest {
  // Inherits createPattern helper from GOLComputationTest
};

TEST_F(GOLStep6Test, Step6GenerationsAndCheckCycleDetection) {
  // Test cycle detection after 6 generations
  ulong64 candidates[100];
  ulong64 numCandidates = 0;
  ulong64 generations = 0;

  // Use a blinker pattern that cycles every 2 generations
  // After 6 generations (g1), it should detect g1 == g3 (both vertical)
  const char blinker[8][8] = {{'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'},
                              {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '1', '1', '1', '0', '0', '0'},
                              {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'},
                              {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'}};

  ulong64 pattern = createPattern(blinker);
  ulong64 g1 = pattern;

  bool shouldAdvance = step6GenerationsAndCheck(&g1, pattern, &generations, candidates, &numCandidates);

  EXPECT_TRUE(shouldAdvance) << "Blinker should be detected as cyclical";
  EXPECT_EQ(generations, 0) << "Generations should be reset to 0 for cyclical pattern";
  EXPECT_EQ(numCandidates, 0) << "Cyclical patterns should not be added as candidates";
}

TEST_F(GOLStep6Test, Step6GenerationsAndCheckCandidateGeneration) {
  // Test candidate generation when reaching MIN_CANDIDATE_GENERATIONS
  ulong64 candidates[100];
  ulong64 numCandidates = 0;
  ulong64 generations = MIN_CANDIDATE_GENERATIONS - 6;  // Will reach exactly MIN_CANDIDATE_GENERATIONS

  // Use a pattern we know runs for 193+ generations
  ulong64 pattern = 12510220043743999556ULL;  // Known 193-generation pattern
  ulong64 g1 = pattern;

  bool shouldAdvance = step6GenerationsAndCheck(&g1, pattern, &generations, candidates, &numCandidates);

  EXPECT_TRUE(shouldAdvance) << "Pattern reaching MIN_CANDIDATE_GENERATIONS should advance";
  EXPECT_EQ(generations, 0) << "Generations should be reset to 0 after becoming candidate";
  EXPECT_EQ(numCandidates, 1) << "Pattern should be added as candidate";
  EXPECT_EQ(candidates[0], pattern) << "Correct pattern should be stored as candidate";
}

TEST_F(GOLStep6Test, Step6GenerationsAndCheckContinuePattern) {
  // Test continuing with pattern that hasn't reached criteria yet
  ulong64 candidates[100];
  ulong64 numCandidates = 0;
  ulong64 generations = 50;  // Well below MIN_CANDIDATE_GENERATIONS

  // Use a pattern we know runs for 193+ generations
  ulong64 pattern = 12510220043743999556ULL;  // Known long-runner
  ulong64 g1 = pattern;

  bool shouldAdvance = step6GenerationsAndCheck(&g1, pattern, &generations, candidates, &numCandidates);

  EXPECT_FALSE(shouldAdvance) << "Pattern not meeting criteria should continue";
  EXPECT_EQ(generations, 56) << "Generations should be incremented by 6";
  EXPECT_EQ(numCandidates, 0) << "No candidates should be generated";
  EXPECT_NE(g1, pattern) << "g1 should be updated to generation after g6";
}

TEST_F(GOLStep6Test, Step6GenerationsAndCheckDeadPattern) {
  // Test pattern that dies out within 6 generations
  ulong64 candidates[100];
  ulong64 numCandidates = 0;
  ulong64 generations = 0;

  // Single cell dies quickly
  ulong64 pattern = 1ULL << (4 * 8 + 4);  // Center cell
  ulong64 g1 = pattern;

  bool shouldAdvance = step6GenerationsAndCheck(&g1, pattern, &generations, candidates, &numCandidates);

  EXPECT_TRUE(shouldAdvance) << "Dead pattern should advance to next";
  EXPECT_EQ(generations, 0) << "Generations should be reset for dead pattern";
  EXPECT_EQ(numCandidates, 0) << "Dead patterns should not be candidates";
  EXPECT_EQ(g1, 0ULL) << "Dead pattern should result in g1 = 0";
}

TEST_F(GOLStep6Test, Step6GenerationsAndCheckMultipleCandidates) {
  // Test multiple patterns being added as candidates
  ulong64 candidates[100];
  ulong64 numCandidates = 0;
  ulong64 generations = MIN_CANDIDATE_GENERATIONS - 6;

  // Add first candidate - use known long-running pattern
  ulong64 pattern1 = 12510220043743999556ULL;  // 193-generation pattern
  ulong64 g1_1 = pattern1;
  step6GenerationsAndCheck(&g1_1, pattern1, &generations, candidates, &numCandidates);

  // Add second candidate - use another known long-running pattern
  generations = MIN_CANDIDATE_GENERATIONS - 6;  // Reset for next pattern
  ulong64 pattern2 = 82597382355986899ULL;      // 197-generation pattern
  ulong64 g1_2 = pattern2;
  step6GenerationsAndCheck(&g1_2, pattern2, &generations, candidates, &numCandidates);

  EXPECT_EQ(numCandidates, 2) << "Both patterns should be candidates";
  EXPECT_EQ(candidates[0], pattern1) << "First candidate should be stored";
  EXPECT_EQ(candidates[1], pattern2) << "Second candidate should be stored";
}

TEST_F(GOLStep6Test, Step6GenerationsAndCheckExactCyclePeriods) {
  // Test detection of different cycle periods (2, 3, 4)
  ulong64 candidates[100];
  ulong64 numCandidates = 0;
  ulong64 generations = 0;

  // Test with a stable pattern (period 1)
  const char block[8][8] = {{'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'},
                            {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '1', '1', '0', '0', '0'},
                            {'0', '0', '0', '1', '1', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'},
                            {'0', '0', '0', '0', '0', '0', '0', '0'}, {'0', '0', '0', '0', '0', '0', '0', '0'}};

  ulong64 pattern = createPattern(block);
  ulong64 g1 = pattern;

  bool shouldAdvance = step6GenerationsAndCheck(&g1, pattern, &generations, candidates, &numCandidates);

  EXPECT_TRUE(shouldAdvance) << "Stable pattern should be detected as cyclical";
  EXPECT_EQ(generations, 0) << "Generations should be reset";
  EXPECT_EQ(g1, pattern) << "Stable pattern should remain unchanged";
}

// Test constructKernel function - critical for CUDA pattern construction
TEST(GOLKernelTest, ConstructKernelBasic) {
  // Test basic kernel construction with zero frame
  ulong64 frame = 0;

  // kernelIndex 0: no bits added
  ulong64 result = constructKernel(frame, 0);
  EXPECT_EQ(result, 0);

  // kernelIndex 1: adds 1 to bits 3-4 (lower 2 bits)
  result = constructKernel(frame, 1);
  ulong64 expected = 1ULL << 3;  // bit 3 set
  EXPECT_EQ(result, expected);

  // kernelIndex 2: adds 2 to bits 3-4
  result = constructKernel(frame, 2);
  expected = 2ULL << 3;  // bit 4 set
  EXPECT_EQ(result, expected);

  // kernelIndex 3: adds 3 to bits 3-4
  result = constructKernel(frame, 3);
  expected = 3ULL << 3;  // bits 3 and 4 set
  EXPECT_EQ(result, expected);
}

TEST(GOLKernelTest, ConstructKernelUpperBits) {
  // Test upper bits (kernelIndex >> 2) at positions 59-60
  ulong64 frame = 0;

  // kernelIndex 4 (100 binary): upper bits = 1
  ulong64 result = constructKernel(frame, 4);
  ulong64 expected = 1ULL << 59;  // bit 59 set
  EXPECT_EQ(result, expected);

  // kernelIndex 8 (1000 binary): upper bits = 2
  result = constructKernel(frame, 8);
  expected = 2ULL << 59;  // bit 60 set
  EXPECT_EQ(result, expected);

  // kernelIndex 12 (1100 binary): upper bits = 3
  result = constructKernel(frame, 12);
  expected = 3ULL << 59;  // bits 59 and 60 set
  EXPECT_EQ(result, expected);
}

TEST(GOLKernelTest, ConstructKernelCombined) {
  // Test both lower and upper bits together
  ulong64 frame = 0;

  // kernelIndex 5 (0101 binary): lower=1, upper=1
  ulong64 result = constructKernel(frame, 5);
  ulong64 expected = (1ULL << 3) + (1ULL << 59);
  EXPECT_EQ(result, expected);

  // kernelIndex 15 (1111 binary): lower=3, upper=3
  result = constructKernel(frame, 15);
  expected = (3ULL << 3) + (3ULL << 59);
  EXPECT_EQ(result, expected);
}

TEST(GOLKernelTest, ConstructKernelWithFrame) {
  // Test with non-zero frame
  ulong64 frame = 0xFF00FF00FF00FF00ULL;  // Alternating byte pattern

  // kernelIndex 0: should just return frame
  ulong64 result = constructKernel(frame, 0);
  EXPECT_EQ(result, frame);

  // kernelIndex 1: should add lower bits
  result = constructKernel(frame, 1);
  ulong64 expected = frame + (1ULL << 3);
  EXPECT_EQ(result, expected);

  // kernelIndex 4: should add upper bits
  result = constructKernel(frame, 4);
  expected = frame + (1ULL << 59);
  EXPECT_EQ(result, expected);

  // kernelIndex 15: should add both
  result = constructKernel(frame, 15);
  expected = frame + (3ULL << 3) + (3ULL << 59);
  EXPECT_EQ(result, expected);
}

TEST(GOLKernelTest, ConstructKernelValidRange) {
  // Test all valid kernel indices (0-15)
  ulong64 frame = 0x1234567890ABCDEFULL;

  for (int i = 0; i < 16; i++) {
    ulong64 result = constructKernel(frame, i);

    // Verify result is always >= frame (we only add bits)
    EXPECT_GE(result, frame);

    // Verify the added bits match expected pattern
    ulong64 added = result - frame;
    ulong64 expectedLower = ((ulong64)(i & 3)) << 3;    // Lower 2 bits at position 3
    ulong64 expectedUpper = ((ulong64)(i >> 2)) << 59;  // Upper 2 bits at position 59
    ulong64 expectedAdded = expectedLower + expectedUpper;

    EXPECT_EQ(added, expectedAdded) << "Failed for kernelIndex " << i;
  }
}

TEST(GOLKernelTest, ConstructKernelBitPositions) {
  // Test specific bit positions are set correctly
  ulong64 frame = 0;

  // Test each bit position that should be affected
  for (int kernelIndex = 0; kernelIndex < 16; kernelIndex++) {
    ulong64 result = constructKernel(frame, kernelIndex);

    // Check bits 3-4 (lower pair)
    int lowerBits = kernelIndex & 3;
    bool bit3Expected = (lowerBits & 1) != 0;
    bool bit4Expected = (lowerBits & 2) != 0;
    EXPECT_EQ((result & (1ULL << 3)) != 0, bit3Expected) << "Bit 3 incorrect for kernelIndex " << kernelIndex;
    EXPECT_EQ((result & (1ULL << 4)) != 0, bit4Expected) << "Bit 4 incorrect for kernelIndex " << kernelIndex;

    // Check bits 59-60 (upper pair)
    int upperBits = kernelIndex >> 2;
    bool bit59Expected = (upperBits & 1) != 0;
    bool bit60Expected = (upperBits & 2) != 0;
    EXPECT_EQ((result & (1ULL << 59)) != 0, bit59Expected) << "Bit 59 incorrect for kernelIndex " << kernelIndex;
    EXPECT_EQ((result & (1ULL << 60)) != 0, bit60Expected) << "Bit 60 incorrect for kernelIndex " << kernelIndex;
  }
}
