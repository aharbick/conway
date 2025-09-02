#include <gtest/gtest.h>

// Define CUDA decorators as empty for CPU compilation
#ifndef __NVCC__
#define __host__
#define __device__
#define __global__
#endif

#include <string.h>

#include <climits>

#include "display_utils.h"

class DisplayUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Any setup needed before each test
  }
};

TEST_F(DisplayUtilsTest, AsBinaryZero) {
  char buffer[65];
  asBinary(0, buffer);

  // Should be all zeros
  EXPECT_STREQ(buffer, "0000000000000000000000000000000000000000000000000000000000000000");
  EXPECT_EQ(strlen(buffer), 64);
}

TEST_F(DisplayUtilsTest, AsBinaryOne) {
  char buffer[65];
  asBinary(1, buffer);

  // Should be 63 zeros followed by a 1
  EXPECT_STREQ(buffer, "0000000000000000000000000000000000000000000000000000000000000001");
  EXPECT_EQ(strlen(buffer), 64);
}

TEST_F(DisplayUtilsTest, AsBinaryMaxValue) {
  char buffer[65];
  asBinary(ULLONG_MAX, buffer);  // 2^64 - 1

  // Should be all ones
  EXPECT_STREQ(buffer, "1111111111111111111111111111111111111111111111111111111111111111");
  EXPECT_EQ(strlen(buffer), 64);
}

TEST_F(DisplayUtilsTest, AsBinaryPowerOfTwo) {
  char buffer[65];
  asBinary(256, buffer);  // 2^8 = 100000000 (binary)

  // Should be 55 zeros, then a 1, then 8 zeros
  EXPECT_STREQ(buffer, "0000000000000000000000000000000000000000000000000000000100000000");
  EXPECT_EQ(strlen(buffer), 64);
}

TEST_F(DisplayUtilsTest, AsBinaryProblematicAirtableValue) {
  // Test the specific value that was problematic with Airtable precision
  char buffer[65];
  uint64_t problematic_value = 1736161183231225856ULL;
  asBinary(problematic_value, buffer);

  // Verify the string is exactly 64 characters and null terminated
  EXPECT_EQ(strlen(buffer), 64);

  // Test against the known correct pattern
  EXPECT_STREQ(buffer, "0001100000011000000101010110010111000111010100011011000000000000");

  // Verify we can reconstruct the original value from the binary string
  uint64_t reconstructed = 0;
  for (int i = 0; i < 64; i++) {
    if (buffer[i] == '1') {
      reconstructed |= (1ULL << (63 - i));
    }
  }
  EXPECT_EQ(reconstructed, problematic_value);
}

TEST_F(DisplayUtilsTest, AsBinaryLargeNumber) {
  char buffer[65];
  uint64_t large_number = 9223372036854775808ULL;  // 2^63
  asBinary(large_number, buffer);

  // Should start with 1 followed by 63 zeros (MSB set)
  EXPECT_EQ(buffer[0], '1');
  for (int i = 1; i < 64; i++) {
    EXPECT_EQ(buffer[i], '0');
  }
  EXPECT_EQ(strlen(buffer), 64);
}

TEST_F(DisplayUtilsTest, AsBinaryPattern) {
  char buffer[65];
  // Test a specific pattern: alternating bits 0x5555555555555555
  uint64_t pattern = 0x5555555555555555ULL;
  asBinary(pattern, buffer);

  // Should alternate 01010101... for all 64 bits
  for (int i = 0; i < 64; i++) {
    char expected = (i % 2 == 0) ? '0' : '1';
    EXPECT_EQ(buffer[i], expected);
  }
  EXPECT_EQ(strlen(buffer), 64);
}

TEST_F(DisplayUtilsTest, AsBinaryBufferSafety) {
  char buffer[65];
  memset(buffer, 'X', sizeof(buffer));  // Fill with X's

  asBinary(42, buffer);

  // Should have proper null termination
  EXPECT_EQ(buffer[64], '\0');
  EXPECT_EQ(strlen(buffer), 64);

  // No buffer overflow - the 65th position should be null
  EXPECT_NE(buffer[64], 'X');
}