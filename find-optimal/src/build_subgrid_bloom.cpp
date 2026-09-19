// Build the 7x7 subgrid cache's Bloom filters, the artifact the strip search loads to prune
// patterns that cannot reach its target. Reads the cache as JSON lines on stdin:
//
//   cat data/7x7subgrid-cache.json.gz.part_* | gzip -dc \
//     | ./build/build-subgrid-bloom data/7x7subgrid-bloom.bin
//
// Emits one filter per threshold, each over the cache states at or above it, so a lookup
// can use the most selective one that still answers the question. A pattern covered early
// needs a state that lives almost as long as the target, and there are very few of those -
// the filter for that question is kilobytes rather than tens of megabytes. See
// include/subgrid_bloom.h.
//
// The header carries the cache's min and max generation counts, so the bounds the search
// derives travel with the artifact instead of being hardcoded.
//
// Self-checks before writing: every key must probe positive in every tier that should hold
// it (a false negative would make the search discard a real candidate), and each tier's
// false positive rate is measured against random keys.
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gol_core.h"
#include "subgrid_bloom.h"

namespace {

// Thresholds to build, ascending. The spacing is finer near the top because that is where
// the counts collapse - and where almost every lookup lands, since most patterns fit a 7x7
// box within a dozen generations.
const uint32_t kThresholds[] = {180, 188, 192, 196, 200, 204};

uint32_t roundUpPow2(uint64_t v) {
  uint32_t p = 1;
  while (p < v) p <<= 1;
  return p;
}

uint32_t blocksForKeys(uint64_t keys, uint32_t bitsPerKey) {
  uint64_t bits = keys * bitsPerKey;
  uint64_t blocks = (bits + SUBGRID_BLOOM_BLOCK_BITS - 1) / SUBGRID_BLOOM_BLOCK_BITS;
  if (blocks < SUBGRID_BLOOM_MIN_BLOCKS) blocks = SUBGRID_BLOOM_MIN_BLOCKS;
  return roundUpPow2(blocks);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <out.bin> [thresholds]   (cache JSON lines on stdin)\n", argv[0]);
    fprintf(stderr, "  thresholds: comma separated and ascending, e.g. 180,196,204\n");
    return 2;
  }
  const char* outPath = argv[1];

  // Which tiers to build. More tiers means each lookup can use a smaller, more selective
  // filter, at the cost of a bigger artifact; one tier reproduces a single flat filter.
  std::vector<uint32_t> thresholds(std::begin(kThresholds), std::end(kThresholds));
  if (argc > 2) {
    thresholds.clear();
    for (const char* p = argv[2]; *p;) {
      thresholds.push_back((uint32_t)strtoul(p, (char**)&p, 10));
      if (*p == ',') p++;
    }
  }

  std::vector<uint64_t> keys;
  std::vector<uint32_t> gens;
  keys.reserve(20u << 20);
  gens.reserve(20u << 20);
  uint32_t minGens = UINT32_MAX, maxGens = 0;
  uint64_t lines = 0, malformed = 0, notCoverable = 0;

  // {"generations":188,"pattern":11223022910976}
  char* line = nullptr;
  size_t cap = 0;
  while (getline(&line, &cap, stdin) > 0) {
    const char* g = strstr(line, "\"generations\":");
    const char* p = strstr(line, "\"pattern\":");
    if (!g || !p) {
      malformed++;
      continue;
    }
    uint32_t generations = (uint32_t)strtoul(g + 14, nullptr, 10);
    uint64_t pattern = strtoull(p + 10, nullptr, 10);
    if (generations == 0 || pattern == 0) {
      malformed++;
      continue;
    }

    // Every key must be a pattern that fits inside a 7x7 box, or the search would be
    // looking up states the cache never enumerated
    if (!isCoverableBy7x7(pattern)) notCoverable++;

    if (generations < minGens) minGens = generations;
    if (generations > maxGens) maxGens = generations;
    keys.push_back(pattern);
    gens.push_back(generations);
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
  printf("  all keys are 7x7-coverable\n\n");

  // ---- lay the tiers out ----
  SubgridBloomHeader h{};
  memcpy(h.magic, SUBGRID_BLOOM_MAGIC, 8);
  h.version = SUBGRID_BLOOM_VERSION;
  h.blockBits = SUBGRID_BLOOM_BLOCK_BITS;
  h.k = SUBGRID_BLOOM_K;
  h.minGenerations = minGens;
  h.maxGenerations = maxGens;

  uint64_t wordOffset = 0;
  for (uint32_t t : thresholds) {
    if (t < minGens || t > maxGens) continue;  // a threshold the cache cannot answer
    if (h.numTiers >= SUBGRID_BLOOM_MAX_TIERS) break;

    uint64_t count = 0;
    for (uint32_t g : gens) {
      if (g >= t) count++;
    }
    if (count == 0) continue;

    uint32_t bitsPerKey =
        (t == minGens) ? SUBGRID_BLOOM_WIDE_BITS_PER_KEY : SUBGRID_BLOOM_BITS_PER_KEY;
    SubgridBloomTier& tier = h.tiers[h.numTiers++];
    tier.threshold = t;
    tier.keys = count;
    tier.blocks = blocksForKeys(count, bitsPerKey);
    tier.wordOffset = wordOffset;
    wordOffset += (uint64_t)tier.blocks * SUBGRID_BLOOM_WORDS_PER_BLOCK;
  }
  h.totalWords = wordOffset;
  h.numKeys = h.tiers[0].keys;

  std::vector<uint32_t> filters(h.totalWords, 0u);
  for (uint32_t i = 0; i < h.numTiers; i++) {
    const SubgridBloomTier& tier = h.tiers[i];
    uint32_t* base = filters.data() + tier.wordOffset;
    for (size_t j = 0; j < keys.size(); j++) {
      if (gens[j] >= tier.threshold) subgridBloomInsert(base, tier.blocks - 1u, keys[j]);
    }
  }

  // ---- check every tier ----
  printf("tier  threshold      keys       size   bits set   false positives   false negatives\n");
  uint64_t totalFalseNegatives = 0;
  uint64_t state = 0x243F6A8885A308D3ULL;
  for (uint32_t i = 0; i < h.numTiers; i++) {
    const SubgridBloomTier& tier = h.tiers[i];
    const uint32_t* base = filters.data() + tier.wordOffset;
    uint64_t words = (uint64_t)tier.blocks * SUBGRID_BLOOM_WORDS_PER_BLOCK;

    uint64_t falseNegatives = 0;
    for (size_t j = 0; j < keys.size(); j++) {
      if (gens[j] >= tier.threshold && !subgridBloomMaybe(base, tier.blocks - 1u, keys[j])) {
        falseNegatives++;
      }
    }
    totalFalseNegatives += falseNegatives;

    uint64_t bitsSet = 0;
    for (uint64_t w = 0; w < words; w++) bitsSet += (uint64_t)__builtin_popcount(base[w]);

    uint64_t probes = 2000000, positives = 0;
    for (uint64_t j = 0; j < probes; j++) {
      state = state * 6364136223846793005ULL + 1442695040888963407ULL;
      if (subgridBloomMaybe(base, tier.blocks - 1u, subgridBloomHash(state))) positives++;
    }

    printf("  %u      >=%3u  %10" PRIu64 "  %7.2f MB     %5.1f%%          %6.4f%%   %10" PRIu64 " %s\n",
           i, tier.threshold, tier.keys, words * 4.0 / 1e6,
           100.0 * bitsSet / (words * 32.0), 100.0 * positives / probes, falseNegatives,
           falseNegatives == 0 ? "" : "<- BROKEN");
  }

  if (totalFalseNegatives != 0) {
    printf("\nrefusing to write: a false negative would make the search discard a real candidate\n");
    return 1;
  }

  FILE* out = fopen(outPath, "wb");
  if (!out) {
    fprintf(stderr, "cannot write %s\n", outPath);
    return 1;
  }
  fwrite(&h, sizeof(h), 1, out);
  fwrite(filters.data(), sizeof(uint32_t), h.totalWords, out);
  fclose(out);
  printf("\nwrote %s (%.1f MB, %u tiers)\n", outPath,
         (sizeof(h) + h.totalWords * 4.0) / 1e6, h.numTiers);

  // ---- show which tier each generation would use ----
  for (uint32_t target : {215u, 216u}) {
    OracleTierTable table;
    subgridBloomBuildTierTable(h, target, &table);
    printf("\nat target %u: discard with no lookup up to generation %u\n", target, table.tier1Max);
    uint32_t runStart = table.tier1Max + 1;
    for (uint32_t g = table.tier1Max + 1; g <= table.tier2Max + 1; g++) {
      int tier = (g <= table.tier2Max) ? subgridBloomTierForGeneration(table, g) : -1;
      int prev = subgridBloomTierForGeneration(table, runStart);
      if (g > table.tier2Max || tier != prev) {
        if (prev >= 0) {
          const SubgridBloomTier& t = h.tiers[prev];
          printf("  generations %2u-%2u: the >=%u filter, %" PRIu64 " keys, %.2f MB\n", runStart,
                 g - 1, t.threshold, t.keys,
                 (double)t.blocks * SUBGRID_BLOOM_WORDS_PER_BLOCK * 4.0 / 1e6);
        }
        runStart = g;
      }
    }
  }
  return 0;
}
