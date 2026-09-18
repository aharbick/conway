// Benchmark harness for strip search: times each phase on real middle blocks and
// compares phase-2 kernel variants against the production kernel, checking that every
// variant produces an identical candidate set.
//
//   ./build/bench-strip-search [centerIdx] [middleIdx] [numBlocks] [threads] [xblocks] [hist] [onlyVariant]
//
// e.g.  ./build/bench-strip-search 836 380 3 32 1 1          # all variants + lifetime histogram
//       ./build/bench-strip-search 836 380 1 64 4 0 v2       # sweep one variant's launch config
//
// Variants: v0 = legacy kernel (one thread per top strip), v1 = 2D grid with one pair per
// thread, v2 = 2D grid + lane repacking (this is --strip-kernel=fast), v3xN = v2 plus
// N-wide ILP. Measured on a 5090: v2 is 4.8x-54x faster than v0 depending on how many
// unique top strips the middle block has.
//
// NOTE: the kernels here are copies of the ones in gol_cuda_strips.cu, kept local so this
// harness links without the Google Sheets/curl dependencies. If you change a production
// kernel, mirror it here.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <vector>
#include <string>

#include "center4x4_utils.h"
#include "constants.h"

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)

// ---------------------------------------------------------------- phase 1 --
__device__ static inline void addIfUnique(uint16_t strip, uint32_t signature, uint32_t* hashTable,
                                          uint16_t* uniqueStrips, uint32_t* numUnique) {
  uint32_t hash = hashSignature(signature);
  uint32_t bucket = hash & STRIP_HASH_TABLE_MASK;
  for (int probe = 0; probe < STRIP_MAX_PROBE_LENGTH; probe++) {
    uint32_t probeBucket = (bucket + probe) & STRIP_HASH_TABLE_MASK;
    uint32_t old = atomicCAS(&hashTable[probeBucket], 0xFFFFFFFF, signature);
    if (old == 0xFFFFFFFF) {
      uint32_t idx = atomicAdd(numUnique, 1);
      if (idx < STRIP_SEARCH_MAX_VALID_STRIPS) uniqueStrips[idx] = strip;
      return;
    } else if (old == signature) {
      return;
    }
  }
}

__global__ void findUniqueTopStrips(uint32_t middleBlock, uint16_t* uniqueStrips, uint32_t* numUnique,
                                    uint32_t* hashTable) {
  uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t totalThreads = gridDim.x * blockDim.x;
  for (uint32_t strip = threadId; strip < STRIP_SEARCH_TOTAL_STRIPS; strip += totalThreads) {
    uint64_t pattern = ((uint64_t)strip) | ((uint64_t)middleBlock << 16);
    addIfUnique((uint16_t)strip, computeStripSignature(pattern, true), hashTable, uniqueStrips, numUnique);
  }
}

__global__ void findUniqueBottomStrips(uint32_t middleBlock, uint16_t* uniqueStrips, uint32_t* numUnique,
                                       uint32_t* hashTable) {
  uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t totalThreads = gridDim.x * blockDim.x;
  for (uint32_t strip = threadId; strip < STRIP_SEARCH_TOTAL_STRIPS; strip += totalThreads) {
    uint64_t pattern = ((uint64_t)middleBlock << 16) | ((uint64_t)strip << 48);
    addIfUnique((uint16_t)strip, computeStripSignature(pattern, false), hashTable, uniqueStrips, numUnique);
  }
}

// ------------------------------------------------------- phase 2 variants --

// v0: current production kernel, verbatim.
__global__ void v0_baseline(uint32_t middleBlock, const uint16_t* uniqueTopStrips,
                            const uint16_t* uniqueBottomStrips, uint32_t numUniqueTop,
                            uint32_t numUniqueBottom, uint64_t* candidates, uint64_t* numCandidates) {
  uint32_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t totalThreads = gridDim.x * blockDim.x;

  for (uint32_t topIdx = threadId; topIdx < numUniqueTop; topIdx += totalThreads) {
    uint64_t topBits = (uint64_t)uniqueTopStrips[topIdx];
    for (uint32_t bottomIdx = 0; bottomIdx < numUniqueBottom; bottomIdx++) {
      uint64_t pattern = topBits | ((uint64_t)middleBlock << 16) |
                         ((uint64_t)uniqueBottomStrips[bottomIdx] << 48);
      uint64_t g1 = pattern;
      uint16_t generations = 0;
      while (generations < FAST_SEARCH_MAX_GENERATIONS) {
        generations += 6;
        uint64_t g2 = computeNextGeneration8x8(g1);
        uint64_t g3 = computeNextGeneration8x8(g2);
        uint64_t g4 = computeNextGeneration8x8(g3);
        uint64_t g5 = computeNextGeneration8x8(g4);
        uint64_t g6 = computeNextGeneration8x8(g5);
        g1 = computeNextGeneration8x8(g6);
        if ((g1 == g2) || (g1 == g3) || (g1 == g4)) break;
        if (generations >= MIN_CANDIDATE_GENERATIONS) {
          uint64_t idx = atomicAdd((unsigned long long*)numCandidates, 1ULL);
          if (idx < STRIP_SEARCH_MAX_CANDIDATES) candidates[idx] = pattern;
          break;
        }
      }
    }
  }
}

// v1: 2D grid, one (top,bottom) pair per thread. blockIdx.y = top strip index.
__global__ void v1_pair_per_thread(uint32_t middleBlock, const uint16_t* uniqueTopStrips,
                                   const uint16_t* uniqueBottomStrips, uint32_t numUniqueTop,
                                   uint32_t numUniqueBottom, uint64_t* candidates,
                                   uint64_t* numCandidates) {
  uint32_t topIdx = blockIdx.y;
  if (topIdx >= numUniqueTop) return;
  uint64_t base = (uint64_t)uniqueTopStrips[topIdx] | ((uint64_t)middleBlock << 16);

  for (uint32_t bottomIdx = blockIdx.x * blockDim.x + threadIdx.x; bottomIdx < numUniqueBottom;
       bottomIdx += gridDim.x * blockDim.x) {
    uint64_t pattern = base | ((uint64_t)uniqueBottomStrips[bottomIdx] << 48);
    uint64_t g1 = pattern;
    uint16_t generations = 0;
    while (generations < FAST_SEARCH_MAX_GENERATIONS) {
      generations += 6;
      uint64_t g2 = computeNextGeneration8x8(g1);
      uint64_t g3 = computeNextGeneration8x8(g2);
      uint64_t g4 = computeNextGeneration8x8(g3);
      uint64_t g5 = computeNextGeneration8x8(g4);
      uint64_t g6 = computeNextGeneration8x8(g5);
      g1 = computeNextGeneration8x8(g6);
      if ((g1 == g2) || (g1 == g3) || (g1 == g4)) break;
      if (generations >= MIN_CANDIDATE_GENERATIONS) {
        uint64_t idx = atomicAdd((unsigned long long*)numCandidates, 1ULL);
        if (idx < STRIP_SEARCH_MAX_CANDIDATES) candidates[idx] = pattern;
        break;
      }
    }
  }
}

// v2: 2D grid + repacking. A lane that finishes its pattern immediately picks up the
// next one instead of idling while the rest of the warp finishes (divergence fix).
__global__ void v2_repack(uint32_t middleBlock, const uint16_t* uniqueTopStrips,
                          const uint16_t* uniqueBottomStrips, uint32_t numUniqueTop,
                          uint32_t numUniqueBottom, uint64_t* candidates, uint64_t* numCandidates) {
  uint32_t topIdx = blockIdx.y;
  if (topIdx >= numUniqueTop) return;
  uint64_t base = (uint64_t)uniqueTopStrips[topIdx] | ((uint64_t)middleBlock << 16);

  uint32_t stride = gridDim.x * blockDim.x;
  uint32_t bottomIdx = blockIdx.x * blockDim.x + threadIdx.x;

  uint64_t pattern = 0, g1 = 0;
  uint16_t generations = 0;
  bool active = false;

  while (true) {
    if (!active) {
      if (bottomIdx >= numUniqueBottom) break;
      pattern = base | ((uint64_t)uniqueBottomStrips[bottomIdx] << 48);
      bottomIdx += stride;
      g1 = pattern;
      generations = 0;
      active = true;
    }
    generations += 6;
    uint64_t g2 = computeNextGeneration8x8(g1);
    uint64_t g3 = computeNextGeneration8x8(g2);
    uint64_t g4 = computeNextGeneration8x8(g3);
    uint64_t g5 = computeNextGeneration8x8(g4);
    uint64_t g6 = computeNextGeneration8x8(g5);
    g1 = computeNextGeneration8x8(g6);

    if ((g1 == g2) || (g1 == g3) || (g1 == g4)) {
      active = false;
    } else if (generations >= MIN_CANDIDATE_GENERATIONS) {
      uint64_t idx = atomicAdd((unsigned long long*)numCandidates, 1ULL);
      if (idx < STRIP_SEARCH_MAX_CANDIDATES) candidates[idx] = pattern;
      active = false;
    } else if (generations >= FAST_SEARCH_MAX_GENERATIONS) {
      active = false;
    }
  }
}

// v3: repacking + 2-wide ILP (two independent patterns per thread interleaved).
template <int W>
__global__ void v3_repack_ilp(uint32_t middleBlock, const uint16_t* uniqueTopStrips,
                              const uint16_t* uniqueBottomStrips, uint32_t numUniqueTop,
                              uint32_t numUniqueBottom, uint64_t* candidates, uint64_t* numCandidates) {
  uint32_t topIdx = blockIdx.y;
  if (topIdx >= numUniqueTop) return;
  uint64_t base = (uint64_t)uniqueTopStrips[topIdx] | ((uint64_t)middleBlock << 16);

  uint32_t stride = gridDim.x * blockDim.x;
  uint32_t next = blockIdx.x * blockDim.x + threadIdx.x;

  uint64_t pattern[W], g1[W];
  uint16_t gens[W];
  bool active[W];
#pragma unroll
  for (int w = 0; w < W; w++) { active[w] = false; pattern[w] = 0; g1[w] = 0; gens[w] = 0; }

  int live = 0;
  while (true) {
    live = 0;
#pragma unroll
    for (int w = 0; w < W; w++) {
      if (!active[w] && next < numUniqueBottom) {
        pattern[w] = base | ((uint64_t)uniqueBottomStrips[next] << 48);
        next += stride;
        g1[w] = pattern[w];
        gens[w] = 0;
        active[w] = true;
      }
      if (active[w]) live++;
    }
    if (live == 0) break;

    uint64_t g2[W], g3[W], g4[W];
#pragma unroll
    for (int w = 0; w < W; w++) {
      g2[w] = computeNextGeneration8x8(g1[w]);
    }
#pragma unroll
    for (int w = 0; w < W; w++) g3[w] = computeNextGeneration8x8(g2[w]);
#pragma unroll
    for (int w = 0; w < W; w++) g4[w] = computeNextGeneration8x8(g3[w]);
#pragma unroll
    for (int w = 0; w < W; w++) g1[w] = computeNextGeneration8x8(g4[w]);
#pragma unroll
    for (int w = 0; w < W; w++) g1[w] = computeNextGeneration8x8(g1[w]);
#pragma unroll
    for (int w = 0; w < W; w++) g1[w] = computeNextGeneration8x8(g1[w]);

#pragma unroll
    for (int w = 0; w < W; w++) {
      if (!active[w]) continue;
      gens[w] += 6;
      if ((g1[w] == g2[w]) || (g1[w] == g3[w]) || (g1[w] == g4[w])) {
        active[w] = false;
      } else if (gens[w] >= MIN_CANDIDATE_GENERATIONS) {
        uint64_t idx = atomicAdd((unsigned long long*)numCandidates, 1ULL);
        if (idx < STRIP_SEARCH_MAX_CANDIDATES) candidates[idx] = pattern[w];
        active[w] = false;
      } else if (gens[w] >= FAST_SEARCH_MAX_GENERATIONS) {
        active[w] = false;
      }
    }
  }
}

// Phase 3, as it exists in production: re-run each candidate from generation 0.
__global__ void processCandidates(uint64_t* candidates, uint64_t* numCandidates, uint64_t* bestPattern,
                                  uint64_t* bestGenerations, CycleDetectionAlgorithm algorithm) {
  for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < *numCandidates;
       i += blockDim.x * gridDim.x) {
    uint64_t generations = countGenerations(candidates[i], computeNextGeneration8x8, algorithm);
    if (generations > 0) {
      uint64_t old = atomicMax((unsigned long long*)bestGenerations, (unsigned long long)generations);
      if (old < generations) *bestPattern = candidates[i];
    }
  }
}

// Instrumentation: histogram of how many generations each pattern survives.
__global__ void k_lifetimes(uint32_t middleBlock, const uint16_t* uniqueTopStrips,
                            const uint16_t* uniqueBottomStrips, uint32_t numUniqueTop,
                            uint32_t numUniqueBottom, unsigned long long* hist,
                            unsigned long long* totalSteps) {
  uint32_t topIdx = blockIdx.y;
  if (topIdx >= numUniqueTop) return;
  uint64_t base = (uint64_t)uniqueTopStrips[topIdx] | ((uint64_t)middleBlock << 16);
  unsigned long long mySteps = 0;

  for (uint32_t bottomIdx = blockIdx.x * blockDim.x + threadIdx.x; bottomIdx < numUniqueBottom;
       bottomIdx += gridDim.x * blockDim.x) {
    uint64_t g1 = base | ((uint64_t)uniqueBottomStrips[bottomIdx] << 48);
    uint16_t generations = 0;
    while (generations < FAST_SEARCH_MAX_GENERATIONS) {
      generations += 6;
      uint64_t g2 = computeNextGeneration8x8(g1);
      uint64_t g3 = computeNextGeneration8x8(g2);
      uint64_t g4 = computeNextGeneration8x8(g3);
      uint64_t g5 = computeNextGeneration8x8(g4);
      uint64_t g6 = computeNextGeneration8x8(g5);
      g1 = computeNextGeneration8x8(g6);
      if ((g1 == g2) || (g1 == g3) || (g1 == g4)) break;
      if (generations >= MIN_CANDIDATE_GENERATIONS) break;
    }
    mySteps += generations;
    atomicAdd(&hist[generations / 6], 1ULL);
  }
  atomicAdd(totalSteps, mySteps);
}

// ------------------------------------------------------------------ host ---
struct Buffers {
  uint16_t *d_top, *d_bottom;
  uint32_t *d_numUnique, *d_hash;
  uint64_t *d_candidates, *d_numCandidates;
};

static void phase1(Buffers& b, uint32_t middleBlock, uint32_t& nTop, uint32_t& nBottom) {
  uint32_t zero = 0;
  CHECK(cudaMemcpy(b.d_numUnique, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));
  CHECK(cudaMemset(b.d_hash, 0xFF, STRIP_HASH_TABLE_SIZE * sizeof(uint32_t)));
  findUniqueTopStrips<<<STRIP_SEARCH_UNIQUE_GRID_SIZE, STRIP_SEARCH_UNIQUE_THREADS_PER_BLOCK>>>(
      middleBlock, b.d_top, b.d_numUnique, b.d_hash);
  CHECK(cudaMemcpy(&nTop, b.d_numUnique, sizeof(uint32_t), cudaMemcpyDeviceToHost));

  CHECK(cudaMemcpy(b.d_numUnique, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));
  CHECK(cudaMemset(b.d_hash, 0xFF, STRIP_HASH_TABLE_SIZE * sizeof(uint32_t)));
  findUniqueBottomStrips<<<STRIP_SEARCH_UNIQUE_GRID_SIZE, STRIP_SEARCH_UNIQUE_THREADS_PER_BLOCK>>>(
      middleBlock, b.d_bottom, b.d_numUnique, b.d_hash);
  CHECK(cudaMemcpy(&nBottom, b.d_numUnique, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  CHECK(cudaDeviceSynchronize());
}

static std::vector<uint64_t> runVariant(const std::string& name, Buffers& b, uint32_t middleBlock,
                                        uint32_t nTop, uint32_t nBottom, float& ms, int threads,
                                        int xblocks) {
  uint64_t zero64 = 0;
  CHECK(cudaMemcpy(b.d_numCandidates, &zero64, sizeof(uint64_t), cudaMemcpyHostToDevice));
  cudaEvent_t t0, t1;
  CHECK(cudaEventCreate(&t0));
  CHECK(cudaEventCreate(&t1));
  CHECK(cudaEventRecord(t0));

  dim3 grid2d((unsigned)xblocks, nTop);
  if (name == "v0") {
    v0_baseline<<<STRIP_SEARCH_COMBO_GRID_SIZE, STRIP_SEARCH_COMBO_THREADS_PER_BLOCK>>>(
        middleBlock, b.d_top, b.d_bottom, nTop, nBottom, b.d_candidates, b.d_numCandidates);
  } else if (name == "v1") {
    dim3 g((nBottom + threads - 1) / threads, nTop);
    v1_pair_per_thread<<<g, threads>>>(middleBlock, b.d_top, b.d_bottom, nTop, nBottom,
                                       b.d_candidates, b.d_numCandidates);
  } else if (name == "v2") {
    v2_repack<<<grid2d, threads>>>(middleBlock, b.d_top, b.d_bottom, nTop, nBottom, b.d_candidates,
                                   b.d_numCandidates);
  } else if (name == "v3x2") {
    v3_repack_ilp<2><<<grid2d, threads>>>(middleBlock, b.d_top, b.d_bottom, nTop, nBottom,
                                          b.d_candidates, b.d_numCandidates);
  } else if (name == "v3x4") {
    v3_repack_ilp<4><<<grid2d, threads>>>(middleBlock, b.d_top, b.d_bottom, nTop, nBottom,
                                          b.d_candidates, b.d_numCandidates);
  }
  CHECK(cudaEventRecord(t1));
  CHECK(cudaEventSynchronize(t1));
  CHECK(cudaGetLastError());
  CHECK(cudaEventElapsedTime(&ms, t0, t1));
  CHECK(cudaEventDestroy(t0));
  CHECK(cudaEventDestroy(t1));

  uint64_t n = 0;
  CHECK(cudaMemcpy(&n, b.d_numCandidates, sizeof(uint64_t), cudaMemcpyDeviceToHost));
  std::vector<uint64_t> out(n);
  if (n) CHECK(cudaMemcpy(out.data(), b.d_candidates, n * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  std::sort(out.begin(), out.end());
  return out;
}

int main(int argc, char** argv) {
  uint32_t centerIdx = (argc > 1) ? atoi(argv[1]) : 836;
  uint32_t middleIdx = (argc > 2) ? atoi(argv[2]) : 380;
  int nBlocks = (argc > 3) ? atoi(argv[3]) : 4;     // middle blocks to sample
  int threads = (argc > 4) ? atoi(argv[4]) : 256;   // threads/block for new variants
  int xblocks = (argc > 5) ? atoi(argv[5]) : 8;     // x-dim blocks for repacking variants
  bool doHist = (argc > 6) ? atoi(argv[6]) != 0 : false;
  std::string only = (argc > 7) ? argv[7] : "";      // run just one variant, e.g. "v2"

  initializeUnique4x4Centers();
  uint16_t center4x4 = get4x4CenterByIndex(centerIdx);

  Buffers b{};
  CHECK(cudaMalloc(&b.d_top, STRIP_SEARCH_MAX_VALID_STRIPS * sizeof(uint16_t)));
  CHECK(cudaMalloc(&b.d_bottom, STRIP_SEARCH_MAX_VALID_STRIPS * sizeof(uint16_t)));
  CHECK(cudaMalloc(&b.d_numUnique, sizeof(uint32_t)));
  CHECK(cudaMalloc(&b.d_hash, STRIP_HASH_TABLE_SIZE * sizeof(uint32_t)));
  CHECK(cudaMalloc(&b.d_candidates, (1ULL << 22) * sizeof(uint64_t)));  // 4M is plenty per block
  CHECK(cudaMalloc(&b.d_numCandidates, sizeof(uint64_t)));

  const char* variants[] = {"v0", "v1", "v2", "v3x2", "v3x4"};
  double totals[5] = {0, 0, 0, 0, 0};
  double phase1Total = 0, phase3Total = 0;

  printf("centerIdx=%u middleIdx=%u blocks=%d threads=%d xblocks=%d\n", centerIdx, middleIdx,
         nBlocks, threads, xblocks);

  uint32_t blockStart = middleIdx * STRIP_SEARCH_MIDDLE_BLOCKS_PER_REPORT;
  for (int i = 0; i < nBlocks; i++) {
    uint32_t blockOffset = blockStart + i;
    uint16_t leftEar = blockOffset / CENTER_4X4_TOTAL_EAR_VALUES;
    uint16_t rightEar = blockOffset % CENTER_4X4_TOTAL_EAR_VALUES;
    uint32_t middleBlock = reconstructMiddleBlock(center4x4, (uint8_t)leftEar, (uint8_t)rightEar);

    uint32_t nTop = 0, nBottom = 0;
    cudaEvent_t p0, p1;
    CHECK(cudaEventCreate(&p0));
    CHECK(cudaEventCreate(&p1));
    CHECK(cudaEventRecord(p0));
    phase1(b, middleBlock, nTop, nBottom);
    CHECK(cudaEventRecord(p1));
    CHECK(cudaEventSynchronize(p1));
    float p1ms = 0;
    CHECK(cudaEventElapsedTime(&p1ms, p0, p1));
    phase1Total += p1ms;

    printf("\nmiddleBlock=0x%08X  uniqueTop=%u uniqueBottom=%u pairs=%.1fM  phase1=%.2fms\n",
           middleBlock, nTop, nBottom, (double)nTop * nBottom / 1e6, p1ms);

    std::vector<uint64_t> ref;
    for (int v = 0; v < 5; v++) {
      if (!only.empty() && only != variants[v]) continue;
      float ms = 0;
      std::vector<uint64_t> got = runVariant(variants[v], b, middleBlock, nTop, nBottom, ms, threads, xblocks);
      totals[v] += ms;
      if (ref.empty()) ref = got;
      const char* ok = (got == ref) ? "ok" : "MISMATCH";
      printf("  %-5s %8.2f ms  candidates=%zu  %s\n", variants[v], ms, got.size(), ok);
    }

    {  // phase 3 cost, using the candidate set left in the buffer by the last variant
      uint64_t *d_best, *d_bestGen;
      CHECK(cudaMalloc(&d_best, sizeof(uint64_t)));
      CHECK(cudaMalloc(&d_bestGen, sizeof(uint64_t)));
      CHECK(cudaMemset(d_best, 0, sizeof(uint64_t)));
      CHECK(cudaMemset(d_bestGen, 0, sizeof(uint64_t)));
      cudaEvent_t c0, c1;
      CHECK(cudaEventCreate(&c0)); CHECK(cudaEventCreate(&c1));
      CHECK(cudaEventRecord(c0));
      processCandidates<<<STRIP_SEARCH_COMBO_GRID_SIZE, STRIP_SEARCH_COMBO_THREADS_PER_BLOCK>>>(
          b.d_candidates, b.d_numCandidates, d_best, d_bestGen, CYCLE_DETECTION_FLOYD);
      CHECK(cudaEventRecord(c1));
      CHECK(cudaEventSynchronize(c1));
      float cms = 0; CHECK(cudaEventElapsedTime(&cms, c0, c1));
      uint64_t bg = 0; CHECK(cudaMemcpy(&bg, d_bestGen, sizeof(uint64_t), cudaMemcpyDeviceToHost));
      printf("  phase3 %8.2f ms  bestGenerations=%llu\n", cms, (unsigned long long)bg);
      phase3Total += cms;
      CHECK(cudaFree(d_best)); CHECK(cudaFree(d_bestGen));
    }

    if (doHist) {
      unsigned long long *d_hist, *d_steps;
      CHECK(cudaMalloc(&d_hist, 64 * sizeof(unsigned long long)));
      CHECK(cudaMalloc(&d_steps, sizeof(unsigned long long)));
      CHECK(cudaMemset(d_hist, 0, 64 * sizeof(unsigned long long)));
      CHECK(cudaMemset(d_steps, 0, sizeof(unsigned long long)));
      dim3 g((nBottom + 255) / 256, nTop);
      k_lifetimes<<<g, 256>>>(middleBlock, b.d_top, b.d_bottom, nTop, nBottom, d_hist, d_steps);
      CHECK(cudaDeviceSynchronize());
      unsigned long long hist[64], steps;
      CHECK(cudaMemcpy(hist, d_hist, sizeof(hist), cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(&steps, d_steps, sizeof(steps), cudaMemcpyDeviceToHost));
      uint64_t pairs = (uint64_t)nTop * nBottom;
      printf("  lifetime: mean=%.1f gens; buckets(gens:pct):", (double)steps / pairs);
      for (int k = 0; k < 64; k++)
        if (hist[k] * 1000ull / pairs > 0) printf(" %d:%.1f%%", k * 6, 100.0 * hist[k] / pairs);
      printf("\n");
      CHECK(cudaFree(d_hist));
      CHECK(cudaFree(d_steps));
    }
  }

  printf("\n=== totals over %d middle blocks (phase1 %.1f ms, phase3 %.1f ms) ===\n", nBlocks, phase1Total, phase3Total);
  for (int v = 0; v < 5; v++)
    if (totals[v] > 0)
      printf("  %-5s %9.1f ms   speedup vs v0: %.2fx\n", variants[v], totals[v], totals[0] / totals[v]);
  return 0;
}
