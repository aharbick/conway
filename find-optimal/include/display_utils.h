#ifndef _DISPLAY_UTILS_H_
#define _DISPLAY_UTILS_H_

#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <iomanip>

// Convert a number to its binary string representation
// numBits: number of bits to output (e.g., 64 for 8x8, 49 for 7x7)
// Note: Buffer must be at least (numBits + 1) bytes to hold result + null terminator
constexpr int BINARY_STRING_SIZE = 65;
constexpr int BINARY_STRING_MAX_BITS = 64;

static inline void asBinary(uint64_t number, char *buf, int numBits = 64) {
  // Validate arguments to prevent buffer overflow
  if (numBits < 0 || numBits > BINARY_STRING_MAX_BITS) {
    // Invalid numBits - write error marker and return
    buf[0] = '?';
    buf[1] = '\0';
    return;
  }

  for (int i = 0; i < numBits; ++i) {
    buf[i] = (number >> (numBits - 1 - i)) & 1 ? '1' : '0';
  }
  buf[numBits] = '\0';  // Ensure null termination
}


// Print a pattern with configurable grid size (compact format: gridSize consecutive bits per row)
// gridSize: 7 for 7x7 grid, 8 for 8x8 grid (default)
static inline void printPattern(uint64_t number, int gridSize = 8) {
  for (int row = 0; row < gridSize; row++) {
    for (int col = 0; col < gridSize; col++) {
      int bitPos = row * gridSize + col;
      char bit = (number & (1ULL << bitPos)) ? '1' : '.';
      std::cout << bit;
      if (col < gridSize - 1) std::cout << " ";
    }
    std::cout << "\n";
  }
}

// Get pattern lines as strings for side-by-side display
static inline std::array<std::string, 8> getPatternLines(uint64_t number) {
  std::array<std::string, 8> lines;
  for (int row = 0; row < 8; row++) {
    std::string line;
    for (int col = 0; col < 8; col++) {
      int bitPos = row * 8 + col;
      char bit = (number & (1ULL << bitPos)) ? '1' : '.';
      line += bit;
      if (col < 7) line += " ";
    }
    lines[row] = line;
  }
  return lines;
}

// Print four patterns side by side with labels and values
static inline void printFourPatternsSideBySide(const std::string& label1, uint64_t pattern1,
                                               const std::string& label2, uint64_t pattern2,
                                               const std::string& label3, uint64_t pattern3,
                                               const std::string& label4, uint64_t pattern4) {
  auto lines1 = getPatternLines(pattern1);
  auto lines2 = getPatternLines(pattern2);
  auto lines3 = getPatternLines(pattern3);
  auto lines4 = getPatternLines(pattern4);

  // Print headers with fixed width (23 = 19 + 4 extra spaces)
  std::cout << std::left << std::setw(23) << label1
            << std::setw(23) << label2
            << std::setw(23) << label3
            << std::setw(23) << label4 << "\n";

  // Print pattern lines
  for (int row = 0; row < 8; row++) {
    std::cout << std::left << std::setw(23) << lines1[row]
              << std::setw(23) << lines2[row]
              << std::setw(23) << lines3[row]
              << std::setw(23) << lines4[row] << "\n";
  }

  // Print values
  std::cout << std::left << std::setw(23) << pattern1
            << std::setw(23) << pattern2
            << std::setw(23) << pattern3
            << std::setw(23) << pattern4 << "\n\n";
}

#endif