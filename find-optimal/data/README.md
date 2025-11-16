## 7x7 Cache Data

See subgrid_cache.h / computeSubgridCache

This is data represents 8x8 patterns with only bits in one of the four 7x7 subgrids
that have simulations lasting 180 or longer generations.

We use this data with --subgrid-cache-file to speed up searching.

To use this data:
```
cat 7x7subgrid-cache.json.gz.part_* | gzip -dc > 7x7subgrid-cache.json
```

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
