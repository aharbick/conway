#ifndef _DISPLAY_UTILS_H_
#define _DISPLAY_UTILS_H_

#include <array>
#include <cstdint>
#include <iostream>

// Convert a 64-bit number to its binary string representation
constexpr int BINARY_STRING_SIZE = 65;

static inline void asBinary(uint64_t number, char *buf) {
  for (int i = 0; i < 64; ++i) {
    buf[i] = (number >> (63 - i)) & 1 ? '1' : '0';
  }
  buf[64] = '\0';  // Ensure null termination
}

// Print a 64-bit pattern as an 8x8 grid
static inline void printPattern(uint64_t number) {
  std::array<char, BINARY_STRING_SIZE> pat{};
  asBinary(number, pat.data());
  for (int i = 0; i < 64; ++i) {
    std::cout << " " << pat[i] << " ";
    if ((i + 1) % 8 == 0) {
      std::cout << "\n";
    }
  }
  std::cout << "\n";
}

#endif