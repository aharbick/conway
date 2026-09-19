#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "gol_core.h"
#include "subgrid_bloom.h"

namespace {

SubgridBloomHeader realCacheHeader() {
  // The shipped filter: 19,676,112 states, 180..206 generations
  SubgridBloomHeader h{};
  h.numKeys = 19676112;
  h.minGenerations = 180;
  h.maxGenerations = 206;
  return h;
}

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
  std::vector<uint32_t> filter(SUBGRID_BLOOM_WORDS, 0u);
  std::mt19937_64 rng(999);

  std::vector<uint64_t> keys;
  for (int i = 0; i < 200000; i++) keys.push_back(rng());
  for (uint64_t k : keys) subgridBloomInsert(filter.data(), k);

  // A false negative would make the search discard a pattern that could beat the record
  for (uint64_t k : keys) {
    ASSERT_TRUE(subgridBloomMaybe(filter.data(), k)) << "false negative for key " << k;
  }
}

TEST(SubgridBloom, FalsePositiveRateIsLowAtTheRealLoad) {
  // Same key count as the shipped cache would give, scaled down 100x along with the filter
  // it is measured against would be wrong, so insert the real count instead
  std::vector<uint32_t> filter(SUBGRID_BLOOM_WORDS, 0u);
  std::mt19937_64 rng(4242);
  for (uint64_t i = 0; i < 19676112; i++) subgridBloomInsert(filter.data(), rng());

  uint64_t probes = 200000, positives = 0;
  for (uint64_t i = 0; i < probes; i++) {
    if (subgridBloomMaybe(filter.data(), rng())) positives++;
  }
  double rate = (double)positives / probes;
  EXPECT_LT(rate, 0.01) << "false positive rate " << rate << " is high enough to cost speed";
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
