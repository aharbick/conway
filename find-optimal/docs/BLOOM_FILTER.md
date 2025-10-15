# Bloom Filter

## Description 

We have roughtly 19m precomputed patterns that have bits only present in one of the four
7x7 subgrids of the 8x8 full grid.  If we could feed those into the findKernelCandidates
CUDA kernal and use it to stop calling computeNextGeneration early then we could essentially
short circuit and run faster.

We created a kernel called step6GenerationsAndCheckWithBloom ... After computing each of the
6 step generations we ran this code:

```cpp
  if (isCoverableBy7x7(*g1) && bloomFilterMayContain(*g1, bloomBits, bloomNumBits, bloomNumHashes)) {
    uint64_t idx = getNextCandidateIndex(numCandidates);
    candidates[idx] = pattern;
    *generations = 0;
    return true;
  }
```

The isCoverableBy7x7 function runs in about 4 bit operations and is true 97% of the time.  In hindsight
we probably don't even need to check for the 3% when it's not true.

The bloomFilterMayContain function computes two hashes using splitmix64 (a total of 6 bit operations and
two multiplies) and then computes 3 hashes using Kirsch & Mitzenmacher double hashing... There's probably
no point in doing double hashing with 3 hashes and that it might be just as fast to compute a 3rd hash.
Regardless the current implementation with k=3 does about 30 bit operations, addtions or multiplications.

The observed performance of the the bloom filter is about 7M checks/second which is about 150ns.
The observed performance of computeNextGeneration is almost 1T/second which is 6 orders of magnitude faster.

While 150ns is slower than the reported performance online it's not 6 orders of magnitude slower.

When we measured the results of computing one frameIdx/kernelIdx with the bloom filter it was about 3 orders
of magnitude slower.

## TL;DR

It's not feasible to use a bloom filter lookup to short-circuit computeNextGeneration faster than just 
running the computations themselves.

## Analysis

We built a program that read in our cache data, tested all cache entries for false negatives,
tested random numbers for false postives, and measured performance.  The results are below.

```bash
aharbick@Conway:~/conway/find-optimal$ ./build/test-bloom-filter ./data/7x7subgrid-cache.json
Loading patterns from ./data/7x7subgrid-cache.json...
Loaded 19676112 patterns

Creating Bloom filter:
  n (elements): 19676112
  p (false positive rate): 1e-08 (1 in 1e+08)
  k (hash functions): 3
  m (bits): 27369001748
  Memory: 3262.64 MB

Adding all patterns to Bloom filter...
Done adding patterns

=== Test 1: Checking for false negatives ===
All inserted patterns should return true (no false negatives allowed)
False negatives: 0 / 19676112
PASS: No false negatives detected

=== Test 2: Checking false positive rate ===
Testing with random patterns that were NOT inserted
Generating and testing 1000000000 random patterns...
(This may take a few minutes...)
  Progress: 10000000 / 1000000000 (1.0%), FPs so far: 0
  Progress: 20000000 / 1000000000 (2.0%), FPs so far: 0
  Progress: 30000000 / 1000000000 (3.0%), FPs so far: 0
  Progress: 40000000 / 1000000000 (4.0%), FPs so far: 1
  Progress: 50000000 / 1000000000 (5.0%), FPs so far: 1
  Progress: 60000000 / 1000000000 (6.0%), FPs so far: 1
  Progress: 70000000 / 1000000000 (7.0%), FPs so far: 1
  Progress: 80000000 / 1000000000 (8.0%), FPs so far: 1
  Progress: 90000000 / 1000000000 (9.0%), FPs so far: 1
  Progress: 100000000 / 1000000000 (10.0%), FPs so far: 1
  Progress: 110000000 / 1000000000 (11.0%), FPs so far: 1
  Progress: 120000000 / 1000000000 (12.0%), FPs so far: 1
  Progress: 130000000 / 1000000000 (13.0%), FPs so far: 1
  Progress: 140000000 / 1000000000 (14.0%), FPs so far: 2
  Progress: 150000000 / 1000000000 (15.0%), FPs so far: 2
  Progress: 160000000 / 1000000000 (16.0%), FPs so far: 3
  Progress: 170000000 / 1000000000 (17.0%), FPs so far: 4
  Progress: 180000000 / 1000000000 (18.0%), FPs so far: 5
  Progress: 190000000 / 1000000000 (19.0%), FPs so far: 5
  Progress: 200000000 / 1000000000 (20.0%), FPs so far: 5
  Progress: 210000000 / 1000000000 (21.0%), FPs so far: 5
  Progress: 220000000 / 1000000000 (22.0%), FPs so far: 5
  Progress: 230000000 / 1000000000 (23.0%), FPs so far: 6
  Progress: 240000000 / 1000000000 (24.0%), FPs so far: 6
  Progress: 250000000 / 1000000000 (25.0%), FPs so far: 6
  Progress: 260000000 / 1000000000 (26.0%), FPs so far: 6
  Progress: 270000000 / 1000000000 (27.0%), FPs so far: 7
  Progress: 280000000 / 1000000000 (28.0%), FPs so far: 7
  Progress: 290000000 / 1000000000 (29.0%), FPs so far: 7
  Progress: 300000000 / 1000000000 (30.0%), FPs so far: 7
  Progress: 310000000 / 1000000000 (31.0%), FPs so far: 7
  Progress: 320000000 / 1000000000 (32.0%), FPs so far: 7
  Progress: 330000000 / 1000000000 (33.0%), FPs so far: 8
  Progress: 340000000 / 1000000000 (34.0%), FPs so far: 8
  Progress: 350000000 / 1000000000 (35.0%), FPs so far: 8
  Progress: 360000000 / 1000000000 (36.0%), FPs so far: 8
  Progress: 370000000 / 1000000000 (37.0%), FPs so far: 8
  Progress: 380000000 / 1000000000 (38.0%), FPs so far: 8
  Progress: 390000000 / 1000000000 (39.0%), FPs so far: 8
  Progress: 400000000 / 1000000000 (40.0%), FPs so far: 8
  Progress: 410000000 / 1000000000 (41.0%), FPs so far: 8
  Progress: 420000000 / 1000000000 (42.0%), FPs so far: 9
  Progress: 430000000 / 1000000000 (43.0%), FPs so far: 9
  Progress: 440000000 / 1000000000 (44.0%), FPs so far: 9
  Progress: 450000000 / 1000000000 (45.0%), FPs so far: 10
  Progress: 460000000 / 1000000000 (46.0%), FPs so far: 10
  Progress: 470000000 / 1000000000 (47.0%), FPs so far: 10
  Progress: 480000000 / 1000000000 (48.0%), FPs so far: 10
  Progress: 490000000 / 1000000000 (49.0%), FPs so far: 10
  Progress: 500000000 / 1000000000 (50.0%), FPs so far: 10
  Progress: 510000000 / 1000000000 (51.0%), FPs so far: 10
  Progress: 520000000 / 1000000000 (52.0%), FPs so far: 10
  Progress: 530000000 / 1000000000 (53.0%), FPs so far: 10
  Progress: 540000000 / 1000000000 (54.0%), FPs so far: 10
  Progress: 550000000 / 1000000000 (55.0%), FPs so far: 10
  Progress: 560000000 / 1000000000 (56.0%), FPs so far: 10
  Progress: 570000000 / 1000000000 (57.0%), FPs so far: 10
  Progress: 580000000 / 1000000000 (58.0%), FPs so far: 10
  Progress: 590000000 / 1000000000 (59.0%), FPs so far: 10
  Progress: 600000000 / 1000000000 (60.0%), FPs so far: 10
  Progress: 610000000 / 1000000000 (61.0%), FPs so far: 10
  Progress: 620000000 / 1000000000 (62.0%), FPs so far: 10
  Progress: 630000000 / 1000000000 (63.0%), FPs so far: 10
  Progress: 640000000 / 1000000000 (64.0%), FPs so far: 10
  Progress: 650000000 / 1000000000 (65.0%), FPs so far: 10
  Progress: 660000000 / 1000000000 (66.0%), FPs so far: 10
  Progress: 670000000 / 1000000000 (67.0%), FPs so far: 10
  Progress: 680000000 / 1000000000 (68.0%), FPs so far: 11
  Progress: 690000000 / 1000000000 (69.0%), FPs so far: 11
  Progress: 700000000 / 1000000000 (70.0%), FPs so far: 11
  Progress: 710000000 / 1000000000 (71.0%), FPs so far: 11
  Progress: 720000000 / 1000000000 (72.0%), FPs so far: 12
  Progress: 730000000 / 1000000000 (73.0%), FPs so far: 12
  Progress: 740000000 / 1000000000 (74.0%), FPs so far: 12
  Progress: 750000000 / 1000000000 (75.0%), FPs so far: 12
  Progress: 760000000 / 1000000000 (76.0%), FPs so far: 12
  Progress: 770000000 / 1000000000 (77.0%), FPs so far: 12
  Progress: 780000000 / 1000000000 (78.0%), FPs so far: 12
  Progress: 790000000 / 1000000000 (79.0%), FPs so far: 12
  Progress: 800000000 / 1000000000 (80.0%), FPs so far: 12
  Progress: 810000000 / 1000000000 (81.0%), FPs so far: 12
  Progress: 820000000 / 1000000000 (82.0%), FPs so far: 12
  Progress: 830000000 / 1000000000 (83.0%), FPs so far: 12
  Progress: 840000000 / 1000000000 (84.0%), FPs so far: 12
  Progress: 850000000 / 1000000000 (85.0%), FPs so far: 12
  Progress: 860000000 / 1000000000 (86.0%), FPs so far: 12
  Progress: 870000000 / 1000000000 (87.0%), FPs so far: 12
  Progress: 880000000 / 1000000000 (88.0%), FPs so far: 12
  Progress: 890000000 / 1000000000 (89.0%), FPs so far: 12
  Progress: 900000000 / 1000000000 (90.0%), FPs so far: 13
  Progress: 910000000 / 1000000000 (91.0%), FPs so far: 13
  Progress: 920000000 / 1000000000 (92.0%), FPs so far: 13
  Progress: 930000000 / 1000000000 (93.0%), FPs so far: 14
  Progress: 940000000 / 1000000000 (94.0%), FPs so far: 14
  Progress: 950000000 / 1000000000 (95.0%), FPs so far: 14
  Progress: 960000000 / 1000000000 (96.0%), FPs so far: 14
  Progress: 970000000 / 1000000000 (97.0%), FPs so far: 14
  Progress: 980000000 / 1000000000 (98.0%), FPs so far: 14
  Progress: 990000000 / 1000000000 (99.0%), FPs so far: 14
  Progress: 1000000000 / 1000000000 (100.0%), FPs so far: 14

Results:
  False positives: 14 / 1000000000
  Observed false positive rate: 1.40e-08 (1 in 71428571)
  Expected false positive rate: 1.00e-08 (1 in 100000000)
  Ratio (observed/expected): 1.40

Performance:
  Time elapsed: 135.04 seconds
  Patterns per second: 7405142

PASS: False positive rate is within acceptable range

=== All tests completed ===
```

## Implemetation
```cpp
#ifndef _BLOOM_FILTER_H_
#define _BLOOM_FILTER_H_

#include <cstdint>
#include <cstring>
#include <cmath>

#include "cuda_utils.h"

// SplitMix64 hash function
__host__ __device__ static inline uint64_t splitmix64_hash(uint64_t x) {
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// CUDA-callable bloom filter search
// Parameters:
//   element: the value to check
//   bits: pointer to the bit array
//   m: number of bits in the filter
//   k: number of hash functions
__host__ __device__ inline bool bloomFilterMayContain(uint64_t element, uint64_t* bits, size_t m, int k) {
  // Use double hashing for Bloom filters (Kirsch & Mitzenmacher, 2006)
  //    h_i(x) = (h1(x) + i * h2(x)) mod m
  // to generate k independent hash functions from 2 base hashes
  uint64_t h1 = splitmix64_hash(element);
  uint64_t h2 = splitmix64_hash(element ^ 0x9e3779b97f4a7c15ULL);

  for (int i = 0; i < k; i++) {
    size_t bit_index = (h1 + i * h2) % m;
    size_t word_index = bit_index / 64;
    size_t bit_offset = bit_index % 64;
    if ((bits[word_index] & (1ULL << bit_offset)) == 0) {
      return false;
    }
  }
  return true;
}

// Simple Bloom filter optimized for CUDA device queries
class BloomFilter {
 private:
  uint64_t* bits_;           // Bit array (host memory only)
  size_t m_;                 // Number of bits in the filter
  size_t n_;                 // Expected number of elements
  double p_;                 // False positive probability
  int k_;                    // Number of hash functions
  size_t num_words_;         // Number of uint64_t words (m_ / 64, rounded up)

 public:
  // Bloom filter constructor
  // Parameters:
  //   n: expected number of elements to insert
  //   p: desired false positive probability (e.g., 0.01 = 1%)
  //   k: number of hash functions to use
  // Calculates required bit array size m using: m = -k*n / ln(1 - p^(1/k))
  // See: https://hur.st/bloomfilter
  BloomFilter(size_t n, double p, int k)
      : bits_(nullptr), n_(n), p_(p), k_(k) {
    // Calculate required bits: m = -k*n / ln(1 - p^(1/k))
    double p_root = pow(p, 1.0 / k);
    double m_calc = -(double)k * n / log(1.0 - p_root);
    m_ = (size_t)ceil(m_calc);
    num_words_ = (m_ + 63) / 64;  // Round up to nearest word

    // Allocate host memory (zeroed)
    bits_ = new uint64_t[num_words_]();
  }

  ~BloomFilter() {
    if (bits_) {
      delete[] bits_;
    }
  }

  // Delete copy/move
  BloomFilter(const BloomFilter&) = delete;
  BloomFilter& operator=(const BloomFilter&) = delete;

  // Add element to filter (host only)
  void add(uint64_t element) {
    // Double hashing: h_i(x) = (h1(x) + i * h2(x)) mod m
    uint64_t h1 = splitmix64_hash(element);
    uint64_t h2 = splitmix64_hash(element ^ 0x9e3779b97f4a7c15ULL);  // Different seed

    for (int i = 0; i < k_; i++) {
      size_t bit_index = (h1 + i * h2) % m_;
      size_t word_index = bit_index / 64;
      size_t bit_offset = bit_index % 64;
      bits_[word_index] |= (1ULL << bit_offset);
    }
  }

  // Check if element might be in filter...
  bool mayContain(uint64_t element) const {
    return bloomFilterMayContain(element, bits_, m_, k_);
  }

  // Get memory size
  size_t memoryBytes() const { return num_words_ * sizeof(uint64_t); }

  // Get parameters
  uint64_t* bits() const { return bits_; }
  size_t numBits() const { return m_; }
  size_t numWords() const { return num_words_; }
  int numHashes() const { return k_; }
};

#endif
```
