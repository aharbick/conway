#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <pthread.h>
#include <limits.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include "mt.h"

//#include "bloom.h"

#include <set>
using namespace std;

static const unsigned long gNeighborFilters[64] = {
    // Row 0 pixels
    (unsigned long) 770,
    (unsigned long) 1797 << 0,
    (unsigned long) 1797 << 1,
    (unsigned long) 1797 << 2,
    (unsigned long) 1797 << 3,
    (unsigned long) 1797 << 4,
    (unsigned long) 1797 << 5,
    (unsigned long) 49216,

    // Row 1 pixels
    (unsigned long) 197123,
    (unsigned long) 460039 << 0,
    (unsigned long) 460039 << 1,
    (unsigned long) 460039 << 2,
    (unsigned long) 460039 << 3,
    (unsigned long) 460039 << 4,
    (unsigned long) 460039 << 5,
    (unsigned long) 12599488,


    // Row 2 pixels
    (unsigned long) 197123 << 8,
    (unsigned long) 460039 << 8 << 0,
    (unsigned long) 460039 << 8 << 1,
    (unsigned long) 460039 << 8 << 2,
    (unsigned long) 460039 << 8 << 3,
    (unsigned long) 460039 << 8 << 4,
    (unsigned long) 460039 << 8 << 5,
    (unsigned long) 12599488 << 8,

    // Row 3 pixels
    (unsigned long) 197123 << 16,
    (unsigned long) 460039 << 16 << 0,
    (unsigned long) 460039 << 16 << 1,
    (unsigned long) 460039 << 16 << 2,
    (unsigned long) 460039 << 16 << 3,
    (unsigned long) 460039 << 16 << 4,
    (unsigned long) 460039 << 16 << 5,
    (unsigned long) 12599488 << 16,

    // Row 4 pixels
    (unsigned long) 197123 << 24,
    (unsigned long) 460039 << 24 << 0,
    (unsigned long) 460039 << 24 << 1,
    (unsigned long) 460039 << 24 << 2,
    (unsigned long) 460039 << 24 << 3,
    (unsigned long) 460039 << 24 << 4,
    (unsigned long) 460039 << 24 << 5,
    (unsigned long) 12599488 << 24,

    // Row 5 pixels
    (unsigned long) 197123 << 32,
    (unsigned long) 460039 << 32 << 0,
    (unsigned long) 460039 << 32 << 1,
    (unsigned long) 460039 << 32 << 2,
    (unsigned long) 460039 << 32 << 3,
    (unsigned long) 460039 << 32 << 4,
    (unsigned long) 460039 << 32 << 5,
    (unsigned long) 12599488 << 32,

    // Row 6 pixels
    (unsigned long) 197123 << 40,
    (unsigned long) 460039 << 40 << 0,
    (unsigned long) 460039 << 40 << 1,
    (unsigned long) 460039 << 40 << 2,
    (unsigned long) 460039 << 40 << 3,
    (unsigned long) 460039 << 40 << 4,
    (unsigned long) 460039 << 40 << 5,
    (unsigned long) 12599488 << 40,

    // Row 7 pixels
    (unsigned long) 515 << 48,
    (unsigned long) 1287 << 48 << 0,
    (unsigned long) 1287 << 48 << 1,
    (unsigned long) 1287 << 48 << 2,
    (unsigned long) 1287 << 48 << 3,
    (unsigned long) 1287 << 48 << 4,
    (unsigned long) 1287 << 48 << 5,
    (unsigned long) 16576 << 48
};

unsigned long  gBestPattern = 0;
unsigned long gBestGenerations = 0;
pthread_mutex_t gMutex = PTHREAD_MUTEX_INITIALIZER;

void asBinary(unsigned long number, char *buf) {
    for (int i = 63; i >= 0; i--) {
        buf[-i+63] = (number >> i) & 1 ? '1' : '0';
    }
}

void printPattern(unsigned long number) {
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


unsigned long computeNextGeneration(unsigned long currentGeneration) {
    unsigned long nextGeneration = currentGeneration;
    for (int i = 0; i < 64; i++) {
        unsigned long neighbors = __builtin_popcountll(currentGeneration & gNeighborFilters[i]);
        if (currentGeneration & (1UL << i)) {
            // Currently alive...
            if (neighbors <= 1) {
                // DIE - lonely
                nextGeneration &= ~(1UL << i);
            }
            else if (neighbors >= 4) {
                // DIE - too crowded
                nextGeneration &= ~(1UL << i);
            }
        }
        else {
            // Currently dead
            if (neighbors == 3) {
                // BIRTH - perfect number of neighbors
                nextGeneration |= 1UL << i;
            }
        }
    }
    return nextGeneration;
}

typedef struct SearchRange {
    int threadId;
    bool random;
    unsigned long beginAt;
    unsigned long endAt;
} SearchRange;

void *search(void *range) {
    SearchRange *r = (SearchRange*)range;

    if (r->random) {
        // Initialize Random number generator
        init_genrand64((unsigned long long) time(NULL));
    }

    printf("\n[Thread %d] %s range %lu - %lu\n", r->threadId, r->random ? "RANDOMLY searching" : "searching ALL", r->beginAt, r->endAt);
    sleep(1);
    for (unsigned long pattern = r->beginAt; pattern <= r->endAt; pattern++) {
        if (pattern % 10000000 == 0) {
            printf(".");
        }

        // Change to random pattern
        if (r->random) {
            pattern = genrand64_int64() % (r->endAt + 1 - r->beginAt) + r->beginAt;
        }

        bool ended = false;
        unsigned long generations = 0;
        unsigned long currentGen = pattern;
        set<unsigned long> visitedPatterns;

        // Loop until we cycle to a number we've already visited
        do {
            // Mark the pattern as seen, keep track of the number of generations, and compute the next generation
            visitedPatterns.insert(currentGen);
            generations++;
            unsigned long nextGen = computeNextGeneration(currentGen);

            // We found a terminal pattern if we found a pattern that doesn't change with computeNextGeneration()
            if (nextGen == 0 || currentGen == nextGen)  {
                ended = true;
                break;
            }

            // Safety valve... we assume that an 8x8 grid is either infinite or terminal in less than 1000 steps
            if (generations > 1000) {
                printf("\n[Thread %d] WARN found pattern %lu with more than 1000 generations\n", r->threadId, pattern);
                break;
            }

            // Advance the current gen to what we just checked with nextGen
            currentGen = nextGen;
        } while (visitedPatterns.find(currentGen) == visitedPatterns.end());

        if (ended) {
            pthread_mutex_lock(&gMutex);
            if (gBestGenerations < generations) {
                char bin[65] = {'\0'};
                asBinary(pattern, bin);
                printf("[Thread %d] %lu generations : %lu : %s\n", r->threadId, generations, pattern, bin);
                gBestPattern = pattern;
                gBestGenerations = generations;
            }
            pthread_mutex_unlock(&gMutex);
        }
    }

    return (void*) gBestPattern;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    if (argc != 3) {
        printf("./find-optimal <test | searchall | searchrandom> <num-threads or test-number>\n");
        return 1;
    }

    bool testMode = false;
    bool random = false;
    int numThreads = 1;
    unsigned long testNumber;
    if (strncmp(argv[1], "test", 4) == 0) {
        char *end;
        testMode = true;
        testNumber = strtoul(argv[2], &end, 10);
    }
    else {
        if (strncmp(argv[1], "searchrandom", 12) == 0) {
            random = true;
        }
        numThreads = atoi(argv[2]);
    }

    // Allocate an array of threads
    pthread_t *threads = (pthread_t *) malloc(sizeof(pthread_t) * numThreads);

    unsigned long numPer = ULONG_MAX / numThreads;
    for (int i = 0; i < numThreads; i++) {
        SearchRange *r = (SearchRange *) malloc(sizeof(SearchRange));
        r->random = random;
        r->threadId = i+1;
        if (testMode) {
            r->beginAt = r->endAt = testNumber;
        }
        else {
            r->beginAt = (i * numPer) + 1;
            r->endAt = r->beginAt + numPer - 1;
        }
        pthread_create(&threads[i], NULL, search, (void*) r);
    }

    for (int i = 0; i < numThreads; i++) {
        printf("\nWaiting on thread %i\n", i);
        pthread_join(threads[i], NULL);
    }
}
