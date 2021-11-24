#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <pthread.h>
#include <limits.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include "mt.h"

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IONBF, 0);

  unsigned long beginAt = 0;
  unsigned long endAt = ULONG_MAX;

  if (argc == 3) {
    char *end;
    beginAt = strtoul(argv[1], &end, 10);
    endAt = strtoul(argv[2], &end, 10);
  }

}
