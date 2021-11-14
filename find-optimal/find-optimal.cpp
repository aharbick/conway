#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <pthread.h>
#include <limits.h>
#include <unistd.h>
#include <time.h>

//#include "bloom.h"

#include <set>
#include <random>
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
        buf[i] = (number >> i) & 1 ? '1' : '0';
    }
}

void printPattern(unsigned long number) {
    for (int i = 0; i < 64; i++) {
        printf(" %lu ", (number >> i) & 1);
        if ((i+1) % 8 == 0) {
            printf("\n");
        }
    }
}


unsigned long computeNextGeneration(unsigned long currentGeneration) {
    unsigned long nextGeneration = currentGeneration;
    for (int i = 0; i < 64; i++) {
        unsigned long neighbors = __builtin_popcount(currentGeneration & gNeighborFilters[i]);
        if (currentGeneration & (1 << i)) {
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
    unsigned long beginAt;
    unsigned long endAt;
} SearchRange;

void *search(void *range) {
    SearchRange *r = (SearchRange*)range;

#ifndef RANDOM
    printf("\n[Thread %d] searching range %lu - %lu\n", r->threadId, r->beginAt, r->endAt);
    sleep(1);
    for (unsigned long pattern = r->beginAt; pattern < r->endAt; pattern++) {
#else
    printf("\n[Thread %d] RANDOMLY searching range %lu - %lu\n", r->threadId, r->beginAt, r->endAt);
    sleep(1);

    // Random number generator
    random_device seeder;
    mt19937 rng(seeder());
    uniform_int_distribution<> randomPattern(r->beginAt, r->endAt); // uniform, unbiased

    unsigned long pattern = 0;
    for (unsigned long i = r->beginAt; i < r->endAt; i++) {
        pattern = randomPattern(rng);
#endif
        if (pattern % 10000000 == 0) {
            printf(".");
        }

        // Don't check starting patterns with less than 6 bits or more than 32 bits
        if (__builtin_popcount(pattern) < 6 || __builtin_popcount(pattern) > 32) {
            continue;
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
                /*
                int visitPos = 1; int visitTotal = visitedPatterns.size();
                for (unsigned long p : visitedPatterns) {
                    printf("VISITED: %lu : %d : %d\n", p, visitPos++, visitTotal);
                }
                */
                ended = true;
                break;
            }

            // Safety valve... we assume that an 8x8 grid is either infinite or terminal in less than 1000 steps
            if (generations > 80) {
                printf("\n[Thread %d] WARN found pattern %lu with more than 1000 generations\n", r->threadId, pattern);
                break;
            }

            // Advance the current gen to what we just checked with nextGen
            currentGen = nextGen;
        } while (visitedPatterns.find(currentGen) == visitedPatterns.end());

        if (ended) {
            if (gBestGenerations < generations) {
                pthread_mutex_lock(&gMutex);
                char bin[65] = {'\0'};
                asBinary(pattern, bin);
                printf("[Thread %d] %lu generations : %lu : %s\n", r->threadId, generations, pattern, bin);
                gBestPattern = pattern;
                gBestGenerations = generations;
                pthread_mutex_unlock(&gMutex);
            }
        }
    }

    return (void*) gBestPattern;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    if (argc != 2) {
        printf("./find-optimal <num-threads>\n");
        return 1;
    }

    // Allocate an array of threads
    int numThreads = atoi(argv[1]);
    pthread_t *threads = (pthread_t *) malloc(sizeof(pthread_t) * numThreads);

    unsigned long numPer = ULONG_MAX / numThreads;
    for (int i = 0; i < numThreads; i++) {
        SearchRange *r = (SearchRange *) malloc(sizeof(SearchRange));
        r->threadId = i+1;
        r->beginAt = (i * numPer) + 1;
        r->endAt = r->beginAt + numPer -1;
        pthread_create(&threads[i], NULL, search, (void*) r);
    }

    for (int i = 0; i < numThreads; i++) {
        printf("\nWaiting on thread %i\n", i);
        pthread_join(threads[i], NULL);
    }
}
