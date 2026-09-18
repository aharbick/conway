// Build the 7x7 subgrid cache's Bloom filter, the artifact the strip search loads to prune
// patterns that cannot reach its target. Reads the cache as JSON lines on stdin:
//
//   cat data/7x7subgrid-cache.json.gz.part_* | gzip -dc \
//     | ./build/build-subgrid-bloom data/7x7subgrid-bloom.bin
//
// Writes a header carrying the cache's min and max generation counts, so the bounds the
// search derives from the filter travel with it instead of being hardcoded.
//
// Self-checks before writing: every key must probe positive (a false negative would make
// the search discard a real candidate), and the false positive rate is measured against
// random keys and compared with the prediction from the fill ratio.
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gol_core.h"
#include "subgrid_bloom.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <out.bin>   (cache JSON lines on stdin)\n", argv[0]);
    return 2;
  }
  const char* outPath = argv[1];

  std::vector<uint64_t> keys;
  keys.reserve(20u << 20);
  uint32_t minGens = UINT32_MAX, maxGens = 0;
  uint64_t lines = 0, malformed = 0, notCoverable = 0;

  // {"generations":188,"pattern":11223022910976}
  char* line = nullptr;
  size_t cap = 0;
  ssize_t len;
  while ((len = getline(&line, &cap, stdin)) > 0) {
    const char* g = strstr(line, "\"generations\":");
    const char* p = strstr(line, "\"pattern\":");
    if (!g || !p) {
      malformed++;
      continue;
    }
    uint32_t gens = (uint32_t)strtoul(g + 14, nullptr, 10);
    uint64_t pattern = strtoull(p + 10, nullptr, 10);
    if (gens == 0 || pattern == 0) {
      malformed++;
      continue;
    }

    // Every key must be a pattern that fits inside a 7x7 box, or the search would be
    // looking up states the cache never enumerated
    if (!isCoverableBy7x7(pattern)) notCoverable++;

    if (gens < minGens) minGens = gens;
    if (gens > maxGens) maxGens = gens;
    keys.push_back(pattern);
    lines++;
    if ((lines % 5000000) == 0) fprintf(stderr, "  read %" PRIu64 " entries...\n", lines);
  }
  free(line);

  if (keys.empty()) {
    fprintf(stderr, "no cache entries on stdin\n");
    return 1;
  }
  printf("cache: %zu entries, generations %u..%u\n", keys.size(), minGens, maxGens);
  if (malformed) printf("  WARNING: %" PRIu64 " malformed lines skipped\n", malformed);
  if (notCoverable) {
    printf("  ERROR: %" PRIu64 " keys are not 7x7-coverable\n", notCoverable);
    return 1;
  }
  printf("  all keys are 7x7-coverable\n");

  std::vector<uint32_t> filter(SUBGRID_BLOOM_WORDS, 0u);
  for (uint64_t k : keys) subgridBloomInsert(filter.data(), k);

  uint64_t bitsSet = 0;
  for (uint32_t w : filter) bitsSet += (uint64_t)__builtin_popcount(w);
  double fill = (double)bitsSet / (SUBGRID_BLOOM_WORDS * 32.0);

  // A false negative would be a correctness bug in the search, so check all of them
  uint64_t falseNegatives = 0;
  for (uint64_t k : keys) {
    if (!subgridBloomMaybe(filter.data(), k)) falseNegatives++;
  }

  // Measure the false positive rate on random keys
  uint64_t probes = 10000000, positives = 0, state = 0x243F6A8885A308D3ULL;
  for (uint64_t i = 0; i < probes; i++) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    if (subgridBloomMaybe(filter.data(), subgridBloomHash(state))) positives++;
  }

  printf("filter: %.0f MB, %u bits per block, k=%u, %.1f%% of bits set\n",
         SUBGRID_BLOOM_BYTES / 1e6, SUBGRID_BLOOM_BLOCK_BITS, SUBGRID_BLOOM_K, 100.0 * fill);
  printf("  false negatives: %" PRIu64 " %s\n", falseNegatives,
         falseNegatives == 0 ? "(required: 0)" : "<- BROKEN, do not use");
  printf("  false positives: %" PRIu64 "/%" PRIu64 " = %.4f%% (predicted %.4f%% from fill)\n",
         positives, probes, 100.0 * positives / probes, 100.0 * pow(fill, SUBGRID_BLOOM_K));
  if (falseNegatives != 0) return 1;

  SubgridBloomHeader h{};
  memcpy(h.magic, SUBGRID_BLOOM_MAGIC, 8);
  h.version = SUBGRID_BLOOM_VERSION;
  h.bitsLog = SUBGRID_BLOOM_BITS_LOG;
  h.blockBits = SUBGRID_BLOOM_BLOCK_BITS;
  h.k = SUBGRID_BLOOM_K;
  h.numKeys = keys.size();
  h.minGenerations = minGens;
  h.maxGenerations = maxGens;
  h.bitsSet = bitsSet;

  FILE* out = fopen(outPath, "wb");
  if (!out) {
    fprintf(stderr, "cannot write %s\n", outPath);
    return 1;
  }
  fwrite(&h, sizeof(h), 1, out);
  fwrite(filter.data(), sizeof(uint32_t), SUBGRID_BLOOM_WORDS, out);
  fclose(out);

  printf("wrote %s (%.0f MB)\n", outPath, (sizeof(h) + SUBGRID_BLOOM_BYTES) / 1e6);
  printf("\nbounds this filter implies, for a target of T generations:\n");
  for (uint32_t t : {215u, 216u, 220u}) {
    printf("  T=%u: discard a coverable state with no lookup up to generation %u, "
           "and on a filter miss up to generation %u\n",
           t, subgridBloomTier1Max(h, t), subgridBloomTier2Max(h, t));
  }
  return 0;
}
