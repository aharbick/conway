// Validate the 7x7 subgrid cache, because the strip search's early-outs are only as sound
// as this file.
//
//   cat data/7x7subgrid-cache.json.gz.part_* | gzip -dc \
//     | ./build/validate-subgrid-cache [sliceStart] [sliceLength]
//
// Three checks:
//
//  1. Every entry recomputes. Each stored key is re-simulated with 8x8 box dynamics and its
//     generation count compared against the file. Also confirms each key really fits inside
//     a 7x7 box, which is what makes it reachable by a lookup at all.
//
//  2. Nothing above the cache's maximum exists. Tier 1 of the search discards a pattern
//     outright once it fits a 7x7 box early, on the grounds that no coverable state can
//     exceed that maximum. This re-derives the maximum from the recomputation.
//
//  3. Completeness, on a slice. The cache claims to cover all 2^49 7x7 patterns in each of
//     4 positions. This re-enumerates a contiguous slice of that space from scratch and
//     checks that every state it finds at or above the threshold is present in the file
//     with the same count. A systematic gap in the original enumeration shows up here.
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "constants.h"
#include "gol_core.h"
#include "subgrid_bloom.h"

#define CHECK(x)                                                                    \
  do {                                                                              \
    cudaError_t e = (x);                                                            \
    if (e != cudaSuccess) {                                                         \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
      exit(1);                                                                      \
    }                                                                               \
  } while (0)

// Open-addressing table over the cache keys, so the slice check can look them up
#define TABLE_BITS 26
#define TABLE_SIZE (1ull << TABLE_BITS)
#define TABLE_EMPTY 0ull

struct Entry {
  uint64_t key;
  uint32_t gens;
};

__host__ __device__ static inline uint64_t tableHash(uint64_t k) {
  return subgridBloomHash(k) & (TABLE_SIZE - 1);
}

__global__ void k_fillTable(const Entry* entries, uint64_t n, uint64_t* keys, uint32_t* vals) {
  for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n;
       i += (uint64_t)blockDim.x * gridDim.x) {
    uint64_t key = entries[i].key;
    uint64_t slot = tableHash(key);
    for (int probe = 0; probe < 512; probe++) {
      uint64_t idx = (slot + probe) & (TABLE_SIZE - 1);
      uint64_t old = atomicCAS((unsigned long long*)&keys[idx], TABLE_EMPTY, key);
      if (old == TABLE_EMPTY || old == key) {
        vals[idx] = entries[i].gens;  // duplicates carry the same count
        break;
      }
    }
  }
}

__device__ static inline int tableLookup(const uint64_t* keys, const uint32_t* vals, uint64_t key) {
  uint64_t slot = tableHash(key);
  for (int probe = 0; probe < 512; probe++) {
    uint64_t idx = (slot + probe) & (TABLE_SIZE - 1);
    uint64_t k = keys[idx];
    if (k == key) return (int)vals[idx];
    if (k == TABLE_EMPTY) return -1;
  }
  return -1;
}

// Check 1 and 2: recompute every stored entry
__global__ void k_recheckEntries(const Entry* entries, uint64_t n, unsigned long long* stats) {
  for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n;
       i += (uint64_t)blockDim.x * gridDim.x) {
    uint64_t key = entries[i].key;
    uint32_t stored = entries[i].gens;

    if (!isCoverableBy7x7(key)) atomicAdd(&stats[0], 1ULL);

    int gens = countGenerations(key, computeNextGeneration8x8, CYCLE_DETECTION_FLOYD);
    if (gens != (int)stored) atomicAdd(&stats[1], 1ULL);
    if (gens < SUBGRID_MIN_GENERATIONS) atomicAdd(&stats[2], 1ULL);
    atomicMax(&stats[3], (unsigned long long)gens);
  }
}

// Check 3: re-enumerate a slice of the 7x7 space and look everything up
__global__ void k_sliceCompleteness(uint64_t sliceStart, uint64_t sliceLen, const uint64_t* keys,
                                    const uint32_t* vals, unsigned long long* stats) {
  for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < sliceLen;
       i += (uint64_t)blockDim.x * gridDim.x) {
    uint64_t pattern7x7 = sliceStart + i;

    for (int pos = 0; pos < 4; pos++) {
      uint64_t pattern8x8 = expand7x7To8x8(pattern7x7, (pos >= 2) ? 1 : 0, (pos & 1) ? 1 : 0);
      int gens = countGenerations(pattern8x8, computeNextGeneration8x8, CYCLE_DETECTION_FLOYD);
      if (gens < SUBGRID_MIN_GENERATIONS) continue;

      atomicAdd(&stats[0], 1ULL);  // states the slice says belong in the cache
      int found = tableLookup(keys, vals, pattern8x8);
      if (found < 0) {
        atomicAdd(&stats[1], 1ULL);  // missing from the cache
        if (atomicAdd(&stats[3], 1ULL) < 4) {
          printf("    MISSING: pattern=%llu gens=%d (7x7 index %llu, position %d)\n",
                 (unsigned long long)pattern8x8, gens, (unsigned long long)pattern7x7, pos);
        }
      } else if (found != gens) {
        atomicAdd(&stats[2], 1ULL);  // present with the wrong count
        if (atomicAdd(&stats[4], 1ULL) < 4) {
          printf("    MISMATCH: pattern=%llu cache=%d recomputed=%d\n",
                 (unsigned long long)pattern8x8, found, gens);
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  uint64_t sliceStart = (argc > 1) ? strtoull(argv[1], nullptr, 10) : 0;
  uint64_t sliceLen = (argc > 2) ? strtoull(argv[2], nullptr, 10) : (1ull << 33);

  printf("reading cache from stdin...\n");
  std::vector<Entry> entries;
  entries.reserve(20u << 20);
  uint32_t minGens = UINT32_MAX, maxGens = 0;
  char* line = nullptr;
  size_t cap = 0;
  while (getline(&line, &cap, stdin) > 0) {
    const char* g = strstr(line, "\"generations\":");
    const char* p = strstr(line, "\"pattern\":");
    if (!g || !p) continue;
    Entry e;
    e.gens = (uint32_t)strtoul(g + 14, nullptr, 10);
    e.key = strtoull(p + 10, nullptr, 10);
    if (!e.gens || !e.key) continue;
    if (e.gens < minGens) minGens = e.gens;
    if (e.gens > maxGens) maxGens = e.gens;
    entries.push_back(e);
  }
  free(line);
  printf("cache: %zu entries, generations %u..%u\n\n", entries.size(), minGens, maxGens);
  if (entries.empty()) return 1;

  Entry* d_entries = nullptr;
  CHECK(cudaMalloc(&d_entries, entries.size() * sizeof(Entry)));
  CHECK(cudaMemcpy(d_entries, entries.data(), entries.size() * sizeof(Entry), cudaMemcpyHostToDevice));

  unsigned long long* d_stats = nullptr;
  CHECK(cudaMalloc(&d_stats, 8 * sizeof(unsigned long long)));

  // ---- checks 1 and 2 ----
  printf("[1] recomputing every entry with 8x8 box dynamics...\n");
  CHECK(cudaMemset(d_stats, 0, 8 * sizeof(unsigned long long)));
  k_recheckEntries<<<4096, 256>>>(d_entries, entries.size(), d_stats);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());
  unsigned long long st[8];
  CHECK(cudaMemcpy(st, d_stats, sizeof(st), cudaMemcpyDeviceToHost));
  printf("    keys that do not fit a 7x7 box: %llu %s\n", st[0], st[0] ? "<- BAD" : "(required: 0)");
  printf("    entries whose count differs:    %llu %s\n", st[1], st[1] ? "<- BAD" : "(required: 0)");
  printf("    entries below the threshold:    %llu %s\n", st[2], st[2] ? "<- BAD" : "(required: 0)");
  printf("    highest recomputed lifetime:    %llu (file says %u)\n", st[3], maxGens);
  bool ok = (st[0] == 0 && st[1] == 0 && st[2] == 0 && st[3] == maxGens);

  // ---- build the lookup table ----
  uint64_t* d_keys = nullptr;
  uint32_t* d_vals = nullptr;
  CHECK(cudaMalloc(&d_keys, TABLE_SIZE * sizeof(uint64_t)));
  CHECK(cudaMalloc(&d_vals, TABLE_SIZE * sizeof(uint32_t)));
  CHECK(cudaMemset(d_keys, 0, TABLE_SIZE * sizeof(uint64_t)));
  k_fillTable<<<4096, 256>>>(d_entries, entries.size(), d_keys, d_vals);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());

  // ---- check 3 ----
  printf("\n[3] re-enumerating 7x7 patterns [%" PRIu64 ", %" PRIu64 ") x 4 positions "
         "(%.2f%% of the space)...\n",
         sliceStart, sliceStart + sliceLen, 100.0 * sliceLen / (double)SUBGRID_TOTAL_PATTERNS);
  CHECK(cudaMemset(d_stats, 0, 8 * sizeof(unsigned long long)));
  cudaEvent_t a, b;
  CHECK(cudaEventCreate(&a));
  CHECK(cudaEventCreate(&b));
  CHECK(cudaEventRecord(a));
  k_sliceCompleteness<<<8192, 256>>>(sliceStart, sliceLen, d_keys, d_vals, d_stats);
  CHECK(cudaEventRecord(b));
  CHECK(cudaEventSynchronize(b));
  CHECK(cudaGetLastError());
  float ms = 0;
  CHECK(cudaEventElapsedTime(&ms, a, b));
  CHECK(cudaMemcpy(st, d_stats, sizeof(st), cudaMemcpyDeviceToHost));
  printf("    %.1f s, states found at or above the threshold: %llu\n", ms / 1000.0, st[0]);
  printf("    missing from the cache:      %llu %s\n", st[1], st[1] ? "<- INCOMPLETE" : "(required: 0)");
  printf("    present with a wrong count:  %llu %s\n", st[2], st[2] ? "<- BAD" : "(required: 0)");
  ok = ok && st[1] == 0 && st[2] == 0;
  if (st[0] == 0) {
    printf("    NOTE: the slice contained nothing above the threshold, so this proved little\n");
    ok = false;
  }

  printf("\n%s\n", ok ? "cache validated" : "VALIDATION FAILED");
  return ok ? 0 : 1;
}
