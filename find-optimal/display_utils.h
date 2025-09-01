#ifndef _DISPLAY_UTILS_H_
#define _DISPLAY_UTILS_H_

#include <stdio.h>
#include "types.h"

// Convert a 64-bit number to its binary string representation
static inline void asBinary(ulong64 number, char *buf) {
  for (int i = 0; i < 64; i++) {
    buf[i] = (number >> (63 - i)) & 1 ? '1' : '0';
  }
  buf[64] = '\0';  // Ensure null termination
}

// Print a 64-bit pattern as an 8x8 grid
static inline void printPattern(ulong64 number) {
  char pat[65] = {'\0'};
  asBinary(number, pat);
  for (int i = 0; i < 64; i++) {
    printf(" %c ", pat[i]);
    if ((i+1) % 8 == 0) {
      printf("\n");
    }
  }
  printf("\n");
}

#endif 