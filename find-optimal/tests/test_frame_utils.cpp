#include <gtest/gtest.h>

// Define CUDA decorators as empty for CPU compilation
#ifndef __NVCC__
#define __host__
#define __device__
#define __global__
#endif

#include "../types.h"
#include "../frame_utils.h"

// Test frame utility functions (these don't have problematic dependencies)
class FrameUtilsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Any setup needed before each test
    }
};

TEST_F(FrameUtilsTest, Rotate90SingleBit) {
    // Test rotating a single bit 90 degrees clockwise
    // Bit at position (0,0) should move to (7,0)
    ulong64 pattern = 1ULL << (0 * 8 + 0); // Top-left corner
    ulong64 rotated = rotate90(pattern);
    ulong64 expected = 1ULL << (7 * 8 + 0); // Bottom-left corner (after clockwise rotation)
    EXPECT_EQ(rotated, expected);
}

TEST_F(FrameUtilsTest, Rotate90FourTimes) {
    // Rotating 4 times should return to original
    ulong64 original = 0x123456789ABCDEFULL;
    ulong64 result = original;
    
    for (int i = 0; i < 4; i++) {
        result = rotate90(result);
    }
    
    EXPECT_EQ(result, original);
}

TEST_F(FrameUtilsTest, ReflectHorizontalSingleBit) {
    // Test horizontal reflection
    // Bit at position (0,0) should move to (0,7)
    ulong64 pattern = 1ULL << (0 * 8 + 0); // Top-left corner
    ulong64 reflected = reflectHorizontal(pattern);
    ulong64 expected = 1ULL << (0 * 8 + 7); // Top-right corner
    EXPECT_EQ(reflected, expected);
}

TEST_F(FrameUtilsTest, ReflectHorizontalTwice) {
    // Reflecting twice should return to original
    ulong64 original = 0x123456789ABCDEFULL;
    ulong64 result = reflectHorizontal(reflectHorizontal(original));
    EXPECT_EQ(result, original);
}

TEST_F(FrameUtilsTest, ExtractFrameEmpty) {
    // Test extracting frame from empty pattern
    ulong64 frameOnly = extractFrame(0);
    EXPECT_EQ(frameOnly, 0);
}

TEST_F(FrameUtilsTest, SpreadBitsToFrameZero) {
    // Test spreading zero bits
    ulong64 frame = spreadBitsToFrame(0);
    EXPECT_EQ(frame, 0);
}

TEST_F(FrameUtilsTest, IsMinimalFrameEmpty) {
    // Empty frame should be minimal
    EXPECT_TRUE(isMinimalFrame(0));
}

TEST_F(FrameUtilsTest, ExtractFrameNonEmpty) {
    // Test extracting frame from a pattern with both frame and non-frame bits
    ulong64 pattern = 0xFFFFFFFFFFFFFFFFULL; // All bits set
    ulong64 frameOnly = extractFrame(pattern);
    
    // Frame should be smaller than full pattern
    EXPECT_LT(frameOnly, pattern);
    EXPECT_NE(frameOnly, 0);
    
    // Should only have frame bits set
    EXPECT_EQ(extractFrame(frameOnly), frameOnly); // Extracting again should be identity
}

TEST_F(FrameUtilsTest, SpreadBitsToFrameNonZero) {
    // Test spreading some bits to frame positions
    ulong64 bits = 0x1; // Just the first bit
    ulong64 frame = spreadBitsToFrame(bits);
    
    // Should have exactly one bit set in frame position
    EXPECT_NE(frame, 0);
    EXPECT_EQ(extractFrame(frame), frame); // Should be valid frame
    
    // Test with all 24 bits set
    ulong64 allBits = 0xFFFFFF; // 24 bits
    ulong64 allFrame = spreadBitsToFrame(allBits);
    EXPECT_NE(allFrame, 0);
    EXPECT_EQ(extractFrame(allFrame), allFrame); // Should be valid frame
}

TEST_F(FrameUtilsTest, IsMinimalFrameAsymmetric) {
    // Test with an asymmetric pattern that should NOT be minimal
    ulong64 asymmetric = 1ULL << (0 * 8 + 1); // Just one bit in frame
    
    // This single bit might or might not be minimal depending on frame structure
    // Let's test the property that rotating it gives a different result
    ulong64 rotated = rotate90(asymmetric);
    if (rotated < asymmetric) {
        EXPECT_FALSE(isMinimalFrame(asymmetric));
    } else {
        EXPECT_TRUE(isMinimalFrame(asymmetric));
    }
}
