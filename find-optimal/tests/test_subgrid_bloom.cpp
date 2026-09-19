#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "gol_core.h"
#include "subgrid_bloom.h"

namespace {

// The shipped artifact: 19,676,112 states, 180..206 generations, tiered
SubgridBloomHeader realCacheHeader() {
  SubgridBloomHeader h{};
  h.numKeys = 19676112;
  h.minGenerations = 180;
  h.maxGenerations = 206;
  const uint32_t thresholds[] = {180, 188, 192, 196, 200, 204};
  const uint64_t keys[] = {19676112, 1174848, 348416, 116408, 26288, 368};
  uint64_t offset = 0;
  for (uint32_t i = 0; i < 6; i++) {
    h.tiers[i].threshold = thresholds[i];
    h.tiers[i].keys = keys[i];
    h.tiers[i].blocks = 1024;  // only the offsets and thresholds matter here
    h.tiers[i].wordOffset = offset;
    offset += 1024 * SUBGRID_BLOOM_WORDS_PER_BLOCK;
  }
  h.numTiers = 6;
  h.totalWords = offset;
  return h;
}

// A standalone filter for the insert/query tests
struct TestFilter {
  explicit TestFilter(uint32_t blocks) : blocks(blocks), words(blocks * SUBGRID_BLOOM_WORDS_PER_BLOCK, 0u) {}
  void insert(uint64_t key) { subgridBloomInsert(words.data(), blocks - 1u, key); }
  bool maybe(uint64_t key) const { return subgridBloomMaybe(words.data(), blocks - 1u, key); }
  uint32_t blocks;
  std::vector<uint32_t> words;
};

}  // namespace

// The oracle runs isCoverableBy7x7OrEmpty on every generation of the hot path. It must
// agree with isCoverableBy7x7 on every non-empty pattern, because the cache only contains
// states the original accepts - a pattern the fast test wrongly called coverable could be
// discarded on the strength of a lookup that was never valid for it.
TEST(SubgridBloom, FastCoverabilityMatchesTheOriginalWhenNotEmpty) {
  std::mt19937_64 rng(12345);
  uint64_t checked = 0, coverable = 0;

  for (int i = 0; i < 300000; i++) {
    uint64_t pattern = rng();
    // Sparse patterns too, since dense random ones are almost never coverable
    if (i % 3 == 1) pattern &= rng();
    if (i % 3 == 2) pattern &= rng() & rng();
    if (pattern == 0) continue;
    ASSERT_EQ(isCoverableBy7x7(pattern), isCoverableBy7x7OrEmpty(pattern))
        << "disagreement on pattern " << pattern;
    checked++;
    if (isCoverableBy7x7(pattern)) coverable++;
  }
  // Make sure the comparison actually exercised both answers
  EXPECT_GT(checked, 200000u);
  EXPECT_GT(coverable, 1000u) << "the sample never produced a coverable pattern";
}

// The single intended difference. A dead pattern cannot reach a positive target, so the
// oracle treating it as coverable discards it - which is sound, and a generation earlier
// than the cycle test would manage.
TEST(SubgridBloom, EmptyPatternIsTheOneIntendedDifference) {
  EXPECT_FALSE(isCoverableBy7x7(0));
  EXPECT_TRUE(isCoverableBy7x7OrEmpty(0));
}

TEST(SubgridBloom, FastCoverabilityEdgeCases) {
  EXPECT_TRUE(isCoverableBy7x7OrEmpty(1ULL));       // single cell at row 0, col 0
  EXPECT_FALSE(isCoverableBy7x7OrEmpty(~0ULL));     // every cell live: needs all 8 rows
  // Rows 0-6 but all 8 columns: needs an 8-wide box
  EXPECT_FALSE(isCoverableBy7x7OrEmpty(~0ULL >> 8));
  // Live cells in every row and both edge columns
  EXPECT_FALSE(isCoverableBy7x7OrEmpty(0x8181818181818181ULL));
  // Rows 1-7 and columns 1-7 only: fits the opposite corner
  EXPECT_TRUE(isCoverableBy7x7OrEmpty(0xFEFEFEFEFEFEFE00ULL));
  // Rows 0-6 and columns 0-6: fits position 0
  EXPECT_TRUE(isCoverableBy7x7OrEmpty(0x007F7F7F7F7F7F7FULL));
}

TEST(SubgridBloom, NoFalseNegatives) {
  TestFilter filter(8192);
  std::mt19937_64 rng(999);

  std::vector<uint64_t> keys;
  for (int i = 0; i < 200000; i++) keys.push_back(rng());
  for (uint64_t k : keys) filter.insert(k);

  // A false negative would make the search discard a pattern that could beat the record
  for (uint64_t k : keys) {
    ASSERT_TRUE(filter.maybe(k)) << "false negative for key " << k;
  }
}

TEST(SubgridBloom, FalsePositiveRateIsLowAtTheDesignedLoad) {
  // Sized the way the builder sizes a tier: SUBGRID_BLOOM_BITS_PER_KEY per key
  const uint64_t n = 200000;
  uint32_t blocks = 1;
  while ((uint64_t)blocks * SUBGRID_BLOOM_BLOCK_BITS < n * SUBGRID_BLOOM_BITS_PER_KEY) blocks <<= 1;
  TestFilter filter(blocks);

  std::mt19937_64 rng(4242);
  for (uint64_t i = 0; i < n; i++) filter.insert(rng());

  uint64_t probes = 200000, positives = 0;
  for (uint64_t i = 0; i < probes; i++) {
    if (filter.maybe(rng())) positives++;
  }
  double rate = (double)positives / probes;
  EXPECT_LT(rate, 0.01) << "false positive rate " << rate << " is high enough to cost speed";
}

// Each probe must use the most selective tier that still proves the pattern cannot reach
// the target: covered early means needing a near-maximum state, and there are few of those.
TEST(SubgridBloom, TierSelectionPicksTheSmallestFilterThatAnswers) {
  SubgridBloomHeader h = realCacheHeader();
  OracleTierTable table;
  subgridBloomBuildTierTable(h, 215, &table);

  EXPECT_EQ(table.tier1Max, 8u);
  EXPECT_EQ(table.tier2Max, 34u);

  // A pattern covered at generation 9 needs a state living 206 more generations, so the
  // >=204 tier answers it; by generation 28 it needs only 187, so it falls to the widest.
  EXPECT_EQ(h.tiers[subgridBloomTierForGeneration(table, 9)].threshold, 204u);
  EXPECT_EQ(h.tiers[subgridBloomTierForGeneration(table, 12)].threshold, 200u);
  EXPECT_EQ(h.tiers[subgridBloomTierForGeneration(table, 16)].threshold, 196u);
  EXPECT_EQ(h.tiers[subgridBloomTierForGeneration(table, 20)].threshold, 192u);
  EXPECT_EQ(h.tiers[subgridBloomTierForGeneration(table, 28)].threshold, 180u);

  // Past the window no tier can decide anything
  EXPECT_EQ(subgridBloomTierForGeneration(table, table.tier2Max + 1), -1);
  EXPECT_EQ(subgridBloomTierForGeneration(table, 100), -1);
}

// Soundness of the selection: whatever tier a generation picks, a miss in it must prove the
// pattern cannot reach the target. That means threshold <= target - generation.
TEST(SubgridBloom, EveryTierChoiceIsSound) {
  SubgridBloomHeader h = realCacheHeader();
  for (uint32_t target : {215u, 216u, 220u, 260u}) {
    OracleTierTable table;
    subgridBloomBuildTierTable(h, target, &table);
    for (uint32_t g = 1; g < 128; g++) {
      int tier = subgridBloomTierForGeneration(table, g);
      if (tier < 0) continue;
      ASSERT_LE(g, table.tier2Max) << "a tier was offered past the decidable window";
      uint32_t threshold = h.tiers[tier].threshold;
      ASSERT_LE(threshold + g, target)
          << "target " << target << ", generation " << g << ": a miss in the >=" << threshold
          << " tier would not prove the pattern falls short";
    }
  }
}

TEST(SubgridBloom, TierBoundsForTheShippedCache) {
  SubgridBloomHeader h = realCacheHeader();

  // Beating 214: a state coverable by generation 8 caps out at 8 + 206 = 214, so it cannot
  // reach 215; a filter miss means under 180 more, so anything covered by generation 34.
  EXPECT_EQ(subgridBloomTier1Max(h, 215), 8u);
  EXPECT_EQ(subgridBloomTier2Max(h, 215), 34u);

  // A higher bar prunes harder
  EXPECT_EQ(subgridBloomTier1Max(h, 220), 13u);
  EXPECT_EQ(subgridBloomTier2Max(h, 220), 39u);

  // Targets at or below the cache's own maximum cannot use the no-lookup tier
  EXPECT_EQ(subgridBloomTier1Max(h, 207), 0u);
  EXPECT_EQ(subgridBloomTier1Max(h, 190), 0u);
  EXPECT_EQ(subgridBloomTier2Max(h, 190), 9u);

  // And at or below the threshold neither tier can say anything
  EXPECT_EQ(subgridBloomTier1Max(h, 181), 0u);
  EXPECT_EQ(subgridBloomTier2Max(h, 181), 0u);
}

// The bound tier 1 relies on: nothing 7x7-coverable outlives the cache's maximum, so a
// pattern covered at generation g is capped at g + max. Check the arithmetic is the one the
// kernel needs, at the boundary.
TEST(SubgridBloom, Tier1BoundIsExactAtTheBoundary) {
  SubgridBloomHeader h = realCacheHeader();
  const uint32_t target = 215;
  uint32_t tier1Max = subgridBloomTier1Max(h, target);

  // Discarding at tier1Max is sound: the best possible total is still short of the target
  EXPECT_LT(tier1Max + h.maxGenerations, target);
  // One generation later it is not sound any more, so the kernel must stop there
  EXPECT_GE(tier1Max + 1 + h.maxGenerations, target);
}
