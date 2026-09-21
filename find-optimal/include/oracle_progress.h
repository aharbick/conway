#ifndef _ORACLE_PROGRESS_H_
#define _ORACLE_PROGRESS_H_

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "constants.h"
#include "logging.h"

// Completion tracking for oracle-mode strip search, kept separate from the Google Sheets
// bitmap on purpose.
//
// The two mean different things. A bit in the Sheets bitmap means "this interval was
// searched exhaustively at the candidate threshold, and its best was recorded" - that is
// what feeds the histogram, and what a full run resumes from. A bit here means only "this
// interval contains nothing at or above the target", which is all the oracle can establish
// because it discards everything below the target unexamined.
//
// Letting oracle runs set bits in the Sheets bitmap would quietly destroy the distinction:
// a later exhaustive run would skip those intervals as done, and their distribution data
// would be gone for good. So oracle intervals land here, exact intervals land in both (an
// exhaustive search satisfies both claims), and the exhaustive record stays authoritative.
//
// The file is a header plus one bit per centerIdx:middleIdx, 547KB, and is written
// atomically. Merging two of them is a bitwise OR, so partitioned workers can each keep
// their own and combine later.

#define ORACLE_PROGRESS_MAGIC "GOLORACL"
#define ORACLE_PROGRESS_VERSION 1u
#define ORACLE_PROGRESS_TOTAL_INTERVALS \
  ((uint64_t)CENTER_4X4_TOTAL_UNIQUE * STRIP_SEARCH_TOTAL_MIDDLE_IDX)
#define ORACLE_PROGRESS_BYTES ((ORACLE_PROGRESS_TOTAL_INTERVALS + 7) / 8)

struct OracleProgressHeader {
  char magic[8];
  uint32_t version;
  uint32_t maxTarget;        // the highest target any set bit was searched at
  uint64_t totalIntervals;
  uint64_t completed;
  uint64_t reserved;
};

class OracleProgress {
 public:
  OracleProgress() : bitmap_(ORACLE_PROGRESS_BYTES, 0), maxTarget_(0), completed_(0), dirty_(false) {}

  // Load an existing file, or start empty if there is none. Returns false only when the
  // file exists but cannot be trusted.
  bool load(const std::string& path, uint32_t target) {
    path_ = path;

    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
      Logger::out() << "Oracle progress: starting a new file at " << path << "\n";
      maxTarget_ = target;
      return true;
    }

    OracleProgressHeader h{};
    if (fread(&h, sizeof(h), 1, f) != 1 || memcmp(h.magic, ORACLE_PROGRESS_MAGIC, 8) != 0 ||
        h.version != ORACLE_PROGRESS_VERSION || h.totalIntervals != ORACLE_PROGRESS_TOTAL_INTERVALS) {
      Logger::out() << "[FATAL] " << path << " is not an oracle progress file for this search\n";
      fclose(f);
      return false;
    }
    if (fread(bitmap_.data(), 1, ORACLE_PROGRESS_BYTES, f) != ORACLE_PROGRESS_BYTES) {
      Logger::out() << "[FATAL] " << path << " is truncated\n";
      fclose(f);
      return false;
    }
    fclose(f);

    completed_ = countBits();

    // Bits cleared at a higher target are still worth having. One set while hunting 215
    // means nothing there reaches 215, which settles the question this search exists to
    // answer whatever its own target is; all it fails to say is whether a 214 lives there,
    // so those intervals are skipped and any pattern merely matching the record inside one
    // goes unseen. Nothing that could beat the record is missed, and redoing tens of
    // thousands of intervals to catch ties in ground already swept is the worse trade.
    //
    // The file keeps the weakest claim it contains, so a later run at the higher target can
    // still trust every bit: clearing at 214 implies clearing at 215, but not the reverse.
    maxTarget_ = (target > h.maxTarget) ? target : h.maxTarget;
    Logger::out() << "Oracle progress: " << completed_ << " intervals already cleared at target "
                  << h.maxTarget << " (" << path << ")\n";
    if (target < h.maxTarget) {
      Logger::out() << "  those were cleared for a target of " << h.maxTarget
                    << " and are skipped, so a pattern matching " << target
                    << " inside one will not be reported; nothing above " << h.maxTarget
                    << " can be hiding there\n";
    }
    return true;
  }

  bool isComplete(uint32_t centerIdx, uint32_t middleIdx) const {
    if (!inRange(centerIdx, middleIdx)) return false;
    uint64_t idx = linearIndex(centerIdx, middleIdx);
    return (bitmap_[idx / 8] & (1u << (idx % 8))) != 0;
  }

  void markComplete(uint32_t centerIdx, uint32_t middleIdx, uint32_t target) {
    if (!inRange(centerIdx, middleIdx)) return;
    uint64_t idx = linearIndex(centerIdx, middleIdx);
    if (!(bitmap_[idx / 8] & (1u << (idx % 8)))) {
      bitmap_[idx / 8] |= (uint8_t)(1u << (idx % 8));
      completed_++;
    }
    if (target > maxTarget_) maxTarget_ = target;
    dirty_ = true;
  }

  // Write through a temporary file so a crash mid-write cannot leave a damaged one
  bool save() {
    if (!dirty_ || path_.empty()) return true;

    std::string tmp = path_ + ".tmp";
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f) {
      Logger::out() << "[WARNING] cannot write oracle progress to " << tmp << "\n";
      return false;
    }

    OracleProgressHeader h{};
    memcpy(h.magic, ORACLE_PROGRESS_MAGIC, 8);
    h.version = ORACLE_PROGRESS_VERSION;
    h.maxTarget = maxTarget_;
    h.totalIntervals = ORACLE_PROGRESS_TOTAL_INTERVALS;
    h.completed = completed_;

    bool ok = fwrite(&h, sizeof(h), 1, f) == 1 &&
              fwrite(bitmap_.data(), 1, ORACLE_PROGRESS_BYTES, f) == ORACLE_PROGRESS_BYTES;
    fclose(f);
    if (!ok) {
      Logger::out() << "[WARNING] failed writing oracle progress\n";
      remove(tmp.c_str());
      return false;
    }
    if (rename(tmp.c_str(), path_.c_str()) != 0) {
      Logger::out() << "[WARNING] failed replacing " << path_ << "\n";
      return false;
    }
    dirty_ = false;
    return true;
  }

  uint64_t completedCount() const { return completed_; }

 private:
  // Both components must be checked, not just the linear index: middleIdx 512 of center 0
  // lands exactly on center 1's first interval.
  static bool inRange(uint32_t centerIdx, uint32_t middleIdx) {
    return centerIdx < CENTER_4X4_TOTAL_UNIQUE && middleIdx < STRIP_SEARCH_TOTAL_MIDDLE_IDX;
  }

  static uint64_t linearIndex(uint32_t centerIdx, uint32_t middleIdx) {
    return (uint64_t)centerIdx * STRIP_SEARCH_TOTAL_MIDDLE_IDX + middleIdx;
  }

  uint64_t countBits() const {
    uint64_t n = 0;
    for (uint8_t b : bitmap_) n += (uint64_t)__builtin_popcount(b);
    return n;
  }

  std::vector<uint8_t> bitmap_;
  std::string path_;
  uint32_t maxTarget_;
  uint64_t completed_;
  bool dirty_;
};

#endif
