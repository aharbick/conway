#ifndef _DISPLAY_UTILS_H_
#define _DISPLAY_UTILS_H_

#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <iomanip>

// Convert a 64-bit number to its binary string representation
constexpr int BINARY_STRING_SIZE = 65;

static inline void asBinary(uint64_t number, char *buf) {
  for (int i = 0; i < 64; ++i) {
    buf[i] = (number >> (63 - i)) & 1 ? '1' : '0';
  }
  buf[64] = '\0';  // Ensure null termination
}


// Print a 64-bit pattern as an 8x8 grid with compact formatting (. for 0, single space)
static inline void printPattern(uint64_t number) {
  for (int row = 0; row < 8; row++) {
    for (int col = 0; col < 8; col++) {
      int bitPos = row * 8 + col;
      char bit = (number & (1ULL << bitPos)) ? '1' : '.';
      std::cout << bit;
      if (col < 7) std::cout << " ";
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