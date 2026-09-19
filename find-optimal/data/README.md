## 7x7 Cache Data

See subgrid_cache.h / computeSubgridCache

This is data represents 8x8 patterns with only bits in one of the four 7x7 subgrids
that have simulations lasting 180 or longer generations.

We use this data with --subgrid-cache-file to speed up searching.

To use this data:
```
cat 7x7subgrid-cache.json.gz.part_* | gzip -dc > 7x7subgrid-cache.json
```

## 7x7 Bloom filter (7x7subgrid-bloom.bin)

A 34MB blocked Bloom filter over the cache keys, used by strip search to discard patterns
that cannot reach its target. Rebuild it whenever the cache changes:

```
cat 7x7subgrid-cache.json.gz.part_* | gzip -dc \
  | ../build/build-subgrid-bloom 7x7subgrid-bloom.bin
```

The tool refuses to write a filter with any false negative, and the header carries the
cache's min and max generation counts (180 and 206) so the bounds the search derives travel
with the artifact instead of being hardcoded. See include/subgrid_bloom.h for why those two
numbers are all the search needs.

## Oracle progress (strip-completion-oracle.bin)

Not in the repo, and not something to build: `--oracle` creates it on first run and writes
it after every interval. It records which intervals were cleared of patterns at or above the
target, which is the oracle's much weaker claim than the exhaustive completion bitmap in
Google Sheets - see include/oracle_progress.h for why the two are kept apart.

It is machine- and target-specific local state, so it is gitignored. Delete it to re-run
intervals, or merge two of them with a bitwise OR if you partition the search across
workers. The Bloom filter next to it is the opposite: a build artifact, committed, and
required.

### Validating the cache

The search's early-outs are only as sound as this cache, so it can be re-checked:

```
cat 7x7subgrid-cache.json.gz.part_* | gzip -dc \
  | ../build/validate-subgrid-cache 281474976710656 68719476736
```

This recomputes every entry with 8x8 box dynamics, re-derives the maximum, and
re-enumerates a contiguous slice of the 2^49 space from scratch to check that everything it
finds above the threshold is in the cache. Last run (2026-09-18), on the slice above:

```
    keys that do not fit a 7x7 box: 0
    entries whose count differs:    0
    entries below the threshold:    0
    highest recomputed lifetime:    206 (file says 206)
    states found at or above the threshold: 570
    missing from the cache:      0
    present with a wrong count:  0
```

The slice is 0.01% of the space and takes about 3 minutes; pass a larger length for more
confidence. It uses a different work division than the cache builder, so an error in the
builder's per-thread arithmetic would show up here.

## Progress Data

The progress-all-20251114.csv is a snapshot from my google sheet.  I ran it for a period of time
and found a bug.  It is all of the data before I decided to restart.

This was the bug:

```diff
diff --git a/find-optimal/src/gol_cuda.cu b/find-optimal/src/gol_cuda.cu
index e41dd3d..f5f21fa 100644
--- a/find-optimal/src/gol_cuda.cu
+++ b/find-optimal/src/gol_cuda.cu
@@ -82,10 +82,10 @@ __global__ void findCandidatesInKernel(uint64_t kernel, uint64_t *candidates, ui
   startingPattern += ((uint64_t)(blockIdx.x >> 6)) << 50;   // set the upper row of 4 'B' bits
 
   uint64_t endAt = startingPattern +
-                   ((1ULL << FRAME_SEARCH_NUM_P_BITS) << 23);  // 2^16 = 65536 increments for the P bits (bits 23-38)
+                   ((1ULL << FRAME_SEARCH_NUM_P_BITS) << 24);  // 2^16 = 65536 increments for the P bits (bits 24-39)
   uint64_t beginAt = startingPattern;
 
-  for (uint64_t pattern = beginAt; pattern < endAt; pattern += (1ULL << 23)) {
+  for (uint64_t pattern = beginAt; pattern != endAt; pattern += (1ULL << 24)) {
     uint64_t g1 = pattern;
     uint16_t generations = 0;
 
@@ -108,10 +108,10 @@ __global__ void findCandidatesInKernelWithCache(uint64_t kernel, uint64_t *candi
   startingPattern += ((uint64_t)(blockIdx.x >> 6)) << 50;   // set the upper row of 4 'B' bits
 
   uint64_t endAt = startingPattern +
-                   ((1ULL << FRAME_SEARCH_NUM_P_BITS) << 23);  // 2^16 = 65536 increments for the P bits (bits 23-38)
+                   ((1ULL << FRAME_SEARCH_NUM_P_BITS) << 24);  // 2^16 = 65536 increments for the P bits (bits 24-39)
   uint64_t beginAt = startingPattern;
 
-  for (uint64_t pattern = beginAt; pattern < endAt; pattern += (1ULL << 23)) {
+  for (uint64_t pattern = beginAt; pattern != endAt; pattern += (1ULL << 24)) {
     uint64_t g1 = pattern;
     uint16_t generations = 0;
```
