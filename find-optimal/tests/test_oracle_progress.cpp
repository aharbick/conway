#include <gtest/gtest.h>

#include <cstdio>
#include <string>

#include "oracle_progress.h"

namespace {

std::string tempPath(const char* name) {
  return std::string("/tmp/find-optimal-test-") + name + "-" + std::to_string(getpid()) + ".bin";
}

struct TempFile {
  explicit TempFile(const char* name) : path(tempPath(name)) { remove(path.c_str()); }
  ~TempFile() {
    remove(path.c_str());
    remove((path + ".tmp").c_str());
  }
  std::string path;
};

}  // namespace

TEST(OracleProgress, StartsEmptyWhenThereIsNoFile) {
  TempFile f("fresh");
  OracleProgress p;
  ASSERT_TRUE(p.load(f.path, 215));
  EXPECT_EQ(p.completedCount(), 0u);
  EXPECT_FALSE(p.isComplete(839, 200));
}

TEST(OracleProgress, RoundTripsThroughAFile) {
  TempFile f("roundtrip");
  {
    OracleProgress p;
    ASSERT_TRUE(p.load(f.path, 215));
    p.markComplete(0, 0, 215);
    p.markComplete(839, 200, 215);
    p.markComplete(8547, 511, 215);  // the last interval
    EXPECT_EQ(p.completedCount(), 3u);
    ASSERT_TRUE(p.save());
  }
  {
    OracleProgress p;
    ASSERT_TRUE(p.load(f.path, 215));
    EXPECT_EQ(p.completedCount(), 3u);
    EXPECT_TRUE(p.isComplete(0, 0));
    EXPECT_TRUE(p.isComplete(839, 200));
    EXPECT_TRUE(p.isComplete(8547, 511));
    EXPECT_FALSE(p.isComplete(839, 201));
    EXPECT_FALSE(p.isComplete(838, 200));
  }
}

TEST(OracleProgress, MarkingTwiceCountsOnce) {
  TempFile f("idempotent");
  OracleProgress p;
  ASSERT_TRUE(p.load(f.path, 215));
  p.markComplete(100, 7, 215);
  p.markComplete(100, 7, 215);
  EXPECT_EQ(p.completedCount(), 1u);
}

TEST(OracleProgress, IgnoresOutOfRangeIntervals) {
  TempFile f("range");
  OracleProgress p;
  ASSERT_TRUE(p.load(f.path, 215));
  p.markComplete(CENTER_4X4_TOTAL_UNIQUE, 0, 215);
  p.markComplete(0, STRIP_SEARCH_TOTAL_MIDDLE_IDX, 215);
  EXPECT_EQ(p.completedCount(), 0u);
  EXPECT_FALSE(p.isComplete(CENTER_4X4_TOTAL_UNIQUE, 0));
}

// A bit set while hunting for 216 means "nothing here reaches 216". Reusing that file for a
// target of 215 would skip intervals that might hold a 215, so it has to be refused.
TEST(OracleProgress, RefusesToReuseAFileFromAHigherTarget) {
  TempFile f("target");
  {
    OracleProgress p;
    ASSERT_TRUE(p.load(f.path, 216));
    p.markComplete(10, 10, 216);
    ASSERT_TRUE(p.save());
  }
  {
    OracleProgress lower;
    EXPECT_FALSE(lower.load(f.path, 215)) << "a lower target must not reuse this file";
  }
  {
    // The same target, or a higher one, is fine: those intervals hold nothing that high
    OracleProgress same;
    EXPECT_TRUE(same.load(f.path, 216));
    OracleProgress higher;
    EXPECT_TRUE(higher.load(f.path, 220));
    EXPECT_TRUE(higher.isComplete(10, 10));
  }
}

TEST(OracleProgress, RefusesAFileThatIsNotOne) {
  TempFile f("garbage");
  FILE* out = fopen(f.path.c_str(), "wb");
  ASSERT_NE(out, nullptr);
  const char junk[] = "this is not an oracle progress file, not even close";
  fwrite(junk, 1, sizeof(junk), out);
  fclose(out);

  OracleProgress p;
  EXPECT_FALSE(p.load(f.path, 215));
}

TEST(OracleProgress, RefusesATruncatedFile) {
  TempFile f("truncated");
  {
    OracleProgress p;
    ASSERT_TRUE(p.load(f.path, 215));
    p.markComplete(1, 1, 215);
    ASSERT_TRUE(p.save());
  }
  // Keep the header, drop most of the bitmap
  FILE* in = fopen(f.path.c_str(), "rb");
  ASSERT_NE(in, nullptr);
  std::vector<char> head(sizeof(OracleProgressHeader) + 64);
  ASSERT_EQ(fread(head.data(), 1, head.size(), in), head.size());
  fclose(in);
  FILE* out = fopen(f.path.c_str(), "wb");
  ASSERT_NE(out, nullptr);
  fwrite(head.data(), 1, head.size(), out);
  fclose(out);

  OracleProgress p;
  EXPECT_FALSE(p.load(f.path, 215));
}

// The bitmap has to cover every interval the search can visit, and nothing more
TEST(OracleProgress, CoversExactlyTheSearchSpace) {
  EXPECT_EQ(ORACLE_PROGRESS_TOTAL_INTERVALS,
            (uint64_t)CENTER_4X4_TOTAL_UNIQUE * STRIP_SEARCH_TOTAL_MIDDLE_IDX);
  EXPECT_EQ(ORACLE_PROGRESS_BYTES, (ORACLE_PROGRESS_TOTAL_INTERVALS + 7) / 8);

  TempFile f("coverage");
  OracleProgress p;
  ASSERT_TRUE(p.load(f.path, 215));
  // Distinct intervals must not collide in the bitmap
  p.markComplete(1, 0, 215);
  EXPECT_TRUE(p.isComplete(1, 0));
  EXPECT_FALSE(p.isComplete(0, STRIP_SEARCH_TOTAL_MIDDLE_IDX - 1));
  EXPECT_FALSE(p.isComplete(0, 1));
}
