## Algorithm Design

This document describes the core algorithms involved in the "frame search" solution to exhaustively search for the longest-lived pattern in an 8x8 Game of Life grid. 

### Key Optimizations

The algorithm achieves tractability through four major optimizations:

#### 1. Bitsliced Next-Generation Computation (Tom Rokicki's Algorithm)

Instead of looping over 64 cells individually, we compute all 64 cells in parallel using bitwise operations. The entire 8x8 grid is packed into a single 64-bit integer, and the Game of Life rules are implemented as a Boolean circuit using only 19 bitwise operations plus shifts.

```
void add2(a, b, &s0, &s1) { s0 = a ^ b; s1 = a & b; }
void add3(a, b, c, &s0, &s1) { add2(a,b,t0,t1); add2(t0,c,s0,t2); s1 = t1^t2; }

next_gen = (x^y^sh2^slh) & ((x|y)^(sh2|slh)) & (sll|a)
```

#### 2. Frame-Based Symmetry Deduplication (~8x speedup)

Patterns that are rotations or reflections of each other will have identical lifespans. We exploit this by defining a "frame" - the 24 corner cells of the grid:

```
FFFooFFF
FFooooFF
FooooooF
oooooooo
oooooooo
FooooooF
FFooooFF
FFFooFFF
```

Of the 2^24 possible frames, only 2,102,800 are "minimal" (lexicographically smallest among all 8 rotations/reflections). We only search patterns with minimal frames, reducing the search space by a factor of ~8.

**Search space reduction:**
- No deduplication: 2^64 patterns
- Frame deduplication: ~2^61 patterns (7.98x reduction)

#### 3. Two-Phase Candidate Filtering

To fully evaluate a pattern we have to handle cycle detection which can cause warp divergence where most threads on the GPU are idle while a few are busy.  To avoid this problem we split the search into two phases:

**Phase 1 (GPU):** Run each pattern for up to 300 generations, checking every 6 generations for:
- Death (pattern becomes empty)
- Short-period cycles (comparing g1 with g2, g3, g4)

Patterns surviving 180+ generations without stabilizing are saved as "candidates."

**Phase 2 (GPU):** Candidates undergo full cycle detection using Floyd's or Nivasch's algorithm to determine their exact lifespan.

This two-phase approach avoids expensive cycle detection for the vast majority of patterns that die quickly.

#### 4. Efficient CUDA Work Distribution

The 2^40 interior bits (non-frame cells) are distributed across GPU resources:

```
FFFKKFFF    F = Frame bits (24 bits, fixed per task)
FFBBBBFF    K = Kernel bits (4 bits, 16 kernel launches)
FBBBBBBF    B = Block bits (10 bits, 1024 blocks)
PPPPPPPP    T = Thread bits (10 bits, 1024 threads)
PPPPPPPP    P = Pattern bits (16 bits, per-thread iteration)
FTTTTTTF
FFTTTTFF
FFFKKFFF
```

The P bits are consecutive, allowing simple iteration: `pattern += 0x1000000`. Each thread processes 2^16 patterns, with threads advancing to new patterns as they complete - avoiding warp divergence by ensuring all threads in a warp do similar amounts of work.


### Coaching from Adam Goucher

Below is an email thread with Adam Goucher (https://gitlab.com/apgoucher/) where with his coaching and support we built the algorithm included in find-optimal.  I include this here for historical record.  You can also find additional discussion here: https://conwaylife.com/forums/viewtopic.php?f=7&t=5489

-----

Dear Andrew,

I'm perplexed as to your point (1) here: when generations >= 180, it should reset the generation count to 0:

else if (generations >= 180) {
        // pattern is potentially interesting; save out for further analysis:
        unsigned int outputIdx = atomicAdd(resultCounter, 1);
        results[outputIdx] = pattern;
        // reset counter ready to advance to next pattern:
        generations = 0;
    }

which then immediately triggers the following conditional block:

    if (generations == 0) {
        // we reset the generation counter, so load the next pattern:
        pattern += blockDim.x * gridDim.x;
        g1 = pattern;
    }

which moves on to the next pattern allocated to the thread. (Did you add a 'continue' or something which caused this to be skipped? Otherwise I can't see why it would save the same pattern more than once.) In any case, saving the pattern multiple times is indicative of a bug, so you should determine why that's happening before proceeding.

As for taking advantage of rotations/reflections, we'll define a 'frame' to be one of the 2^24 ways to assign 0s and 1s to the following 24 cells marked 'F' in the 8x8 grid below:

FFFooFFF
FFooooFF
FooooooF
oooooooo
oooooooo
FooooooF
FFooooFF
FFFooFFF

Whilst there are 2^24 different frames in total, you can deduplicate the ones that are rotations/reflections of each other -- let's say by only considering those that are lexicographically minimal amongst all rotations/reflections. That should reduce the total number of frames to consider to 2102800 (slightly more than 2^21).

The idea is to split the search into 2102800 tasks, one for each distinct frame, and then (within each task) brute-force the 2^40 different ways to assign 0s and 1s to the 40 cells in the octagon marked 'o'.

There will still be a few patterns that we 'overcount' in multiple orientations, specifically certain patterns with symmetrical frames, but they constitute such a small proportion of the overall set of patterns that it's not worth any further optimisation.

In particular, here are the numbers of patterns that you'd need to search with each of:

no deduplication: 18446744073709551616 (= 2^64)
frame deduplication: 2312053050887372800
full deduplication: 2305843028004192256

That is to say, full deduplication would only reduce the number of patterns to search by a further 0.3% (beyond frame deduplication), but it's a lot more effort to implement than frame deduplication so it's not worthwhile. Frame deduplication has already reduced our workload by a factor of 7.9785.

Anyway, 2^40 (the number of patterns per task) is quite a big number, so we'll split it as follows:

(2^4 kernel invocations per task) * (2^10 blocks per kernel) * (2^10 threads per block) * (2^16 patterns per thread)

in specifically the following way:

  FFFKKFFF
  FFBBBBFF
  FBBBBBBF
  PPPPPPPP
  PPPPPPPP
  FTTTTTTF
  FFTTTTFF
  FFFKKFFF

(Note that this enumeration strategy enforces the <<<1024, 1024>>> kernel parameters that you were using before; you can no longer adjust blockDim.x and gridDim.x.)

The important property is that the 16 bits corresponding to the 'patterns per thread' occupy 16 consecutive bits in the 64-bit word, so you can advance to the next pattern using a single addition instruction:

  pattern += 0x1000000;

To determine your thread's starting pattern, you'll need to define your evaluateRange to accept a single 64-bit integer 'kernel_id' of the following shape, containing both the frame bits and the kernel bits:

  FFFKKFFF
  FF0000FF
  F000000F
  00000000
  00000000
  F000000F
  FF0000FF
  FFFKKFFF

In other words, you'll have something like this in your host code which launches the 16 kernels for a given task:

  for (int i = 0; i < 16; i++) {
    uint64_t kernel_id = frame; // this contains the 24 'F' bits
    kernel_id += ((uint64_t) (i & 3)) << 3; // set the lower pair of 'K' bits
    kernel_id += ((uint64_t) (i >> 2)) << 59; // set the upper pair of 'K' bits
    evaluateRange<<<1024, 1024>>>(kernel_id, other_parameters);
  }

(Each loop iteration processes 2^36 patterns, for a total of 2^40.)

Then, at the beginning of your evaluateRange kernel you can incorporate the thread and block bits into the starting pattern:

  uint64_t startingPattern = kernel_id;
  startingPattern += ((uint64_t) (threadIdx.x & 15)) << 10; // set the lower row of 4 'T' bits
  startingPattern += ((uint64_t) (threadIdx.x >> 4)) << 17; // set the upper row of 6 'T' bits
  startingPattern += ((uint64_t) (blockIdx.x & 63)) << 41; // set the lower row of 6 'B' bits
  startingPattern += ((uint64_t) (blockIdx.x >> 6)) << 50; // set the upper row of 4 'B' bits
  uint64_t endAt = startingPattern + 0x10000000000ull;

By counting from startingPattern to endAt (excluding the latter endpoint) in increments of 0x1000000 (2^24), you'll iterate over the 2^16 different choices of the middle 16 bits whilst holding the other 48 bits constant.

Very importantly, your main loop condition needs to be:

  while (pattern != endAt)

and not:

  while (pattern < endAt)

because the following assignment can overflow the 64-bit integer (wrapping round modulo 2^64), thereby causing endAt to be numerically less than startingPattern:

  uint64_t endAt = startingPattern + 0x10000000000ull;


Before you do spend 1.3 years running this, I'd recommend writing a bunch of unit tests to make sure that all of your functions are implemented correctly (lest it would be a very expensive and time-consuming mistake!). Here's the unit-testing library that I use for C++ and CUDA:

https://github.com/google/googletest



Best wishes,


Adam P. Goucher

-----

It took me several days to execute your ideas primarily because I didn’t give enough memory to the GPU to store candidates (and lazily didn’t check bounds) and when it ran off the end of device memory it never failed catastrophically but instead functioned as if it were working but the answers were just messed up and wrong.

My under allocation of memory came from two things:
In your code below “generations >= 180” should be “generations == 180” (or some multiple of 6)… Otherwise any pattern that goes beyond 180 gets saved multiple times.
I kinda ignorantly assumed that 1M patterns would be plenty to store candidates and since I was originally messing around on a Jetson I didn’t have much memory (tbh… not totally sure how much it has but allocating 1GB made it behave *weird*)

After working through those issues and doing more testing I’m happy to announce that as you predicted I saw a 4.9x speed improvement!!  That takes the current code down to 10.4 years to process all possible candidates.

I don’t know how efficiently it is possible to eliminate rotations/reflections but as you suggested taking the search space down to 2^61 would mean that the current code could do a complete search in 1.3years. You’ve been brilliant so far… is it possible to efficiently rule out rotations?

Thank you so much for helping me on this quirky project.  I completely appreciate the depth and clarity of your help.  I would love to get you a beer or beverage of choice somehow.  Maybe some day our paths will cross.  But until then, thanks from the bottom of my nerd heart.

Andy

P.S. Here’s the new longest pattern I also stumbled onto at 209 generations

#CXRLE Pos=-4,-4
x = 8, y = 8, rule = B3/S23:P8,8
o7b$o2bo2b2o$o2bobo2b$2bo3b2o$o5bob$b2ob2obo$o3b4o$2o2bobob!

-----

Dear Andrew,

Thinking about this some more, I think that you can completely avoid reordering work between threads, and still have little or no warp divergence.

The idea is as follows: instead of having 2 nested loops in evaluateRange() (namely an outer loop iterating over starting patterns, and an inner loop iterating over generations) where the inner loop has variable length, collapse this into a single loop of the following form:


uint64_t pattern = beginAt + (blockIdx.x * blockDim.x + threadIdx.x);
uint64_t g1 = pattern;
int generations = 0;

while (pattern < endAt) {

    generations += 6;
    g2 = computeNextGeneration(g1);
    g3 = computeNextGeneration(g2);
    g4 = computeNextGeneration(g3);
    g5 = computeNextGeneration(g4);
    g6 = computeNextGeneration(g5);
    g1 = computeNextGeneration(g6);

    if ((g1 == g2) || (g1 == g3) || (g1 == g4)) {
        // pattern is boring; reset counter ready to advance to next pattern:
        generations = 0;
    } else if (generations >= 180) {
        // pattern is potentially interesting; save out for further analysis:
        unsigned int outputIdx = atomicAdd(resultCounter, 1);
        results[outputIdx] = pattern;
        // reset counter ready to advance to next pattern:
        generations = 0;
    }

    if (generations == 0) {
        // we reset the generation counter, so load the next pattern:
        pattern += blockDim.x * gridDim.x;
        g1 = pattern;
    }
}


This code will save out all of the interesting patterns (those that last >= 180 generations without stabilising into a low-period cycle) into an output array 'results', where they can be further analysed (either on the GPU using another kernel, or on the CPU). The kernel then just acts as a very simple fast prefilter for eliminating the vast majority of patterns, and all of the complicated cycle detection belongs elsewhere (in another kernel or on the CPU).

resultCounter should be a pointer to a device-resident global unsigned int which specifies the number of patterns that we've saved into the results array. This needs to be set to 0 before each time you launch the kernel; after the kernel finishes, this tells you how many patterns to memcpy back to the host for further analysis.

For this to be effective, we should set the range (between beginAt and endAt) to be large -- let's say iterate over 1024 patterns per thread, so with the <<<1024, 1024>>> parameters we'll need to set endAt = beginAt + 1073741824. The reason for making this range large is so that the threads in a warp all do a similar amount of work (by virtue of the law of large numbers), minimising the amount of idleness.

This approach should be far simpler and faster than my earlier suggestion of reordering of patterns within a block.


Best wishes,


Adam P. Goucher

-----

I think I understand what you’re saying at a conceptual level.  In a nutshell I’m spinning up 1024 threads, some fraction take a long time (the very ones I’m hunting for), when one of those ends up in a warp of 31 otherwise fast threads, the 31 other are idle.  To solve this problem don’t let each thread work all the way to completion, but instead only advance 6 generations and then reorder the threads so active ones are at the beginning of the block which will mean that you will eventually have a warp full of long-running patterns and no idle time.

However I don’t understand how the “reorder the patterns between threads” step would work.

Currently my big evaluateRange<<<1024, 1024>>> kernel iterates over a range with each thread getting one entire pattern and then advancing the loop by blockDim.x * gridDim.x.  I started down the path of a __device__ function nextSixGenerations() but the only way that I could come up with to “reorder” was to create a batch of patterns, call my function, sort the batch and repeat.

That didn’t seem right.  So I started noodling down changing to a kernel like advanceSixGenerations<<<1024, 1024>>>(patterns) where patterns is a 1024 array of ulonglong2 for a pattern and the generation count.  After each call to the kernel I would swap patterns into a new array, add new patterns to replace those that finished, and iteratively repeat.

This felt more promising, but at the same time I was feeling out of my depths.  So I thought I would reach out to see if I’m thinking about this correctly or if there are some fundamentals that I’m just plain missing.

Thanks again for your help!

Andy

-----

Dear Andrew,

I've run the visual profiler, and there's exactly one major issue (which I should have foreseen, actually).

Each pattern lasts a different amount of time, so you have pretty bad warp divergence here: all 32 threads in the warp have to wait until the last thread finishes. According to the profiler, your warp efficiency is only 25%, which means that on average three quarters of the warp is doing no useful work.

So there is a theoretical 4x speedup beyond what you have at the moment (thereby reducing the time from 7 years to 2 years). There's a standard technique here:

1. ensure that blocks are as large as possible (1024 threads);
2. every 6 generations (or whatever), reorder the patterns between the threads so that the 'active simulations' are in the first n threads in the block, and the 'finished simulations' are in the last 1024-n threads in the block.

This way, you'll only be running ceil(n/32) warps; the other floor((1024-n)/32) warps will be waiting without consuming resources.


Best wishes,


Adam P. Goucher

-----

Dear Andrew,

Looks great so far!

I haven't profiled your code yet (I'm updating the CUDA installation on my laptop), but here's a small easy minor improvement:


diff --git find-optimal/gol.h find-optimal/gol.h
index 19b187f..f27d37f 100644
--- find-optimal/gol.h
+++ find-optimal/gol.h
@@ -100,13 +100,9 @@ __device__ int countGenerations(uint64_t pattern) {
     g4 = computeNextGeneration(g3);
     g5 = computeNextGeneration(g4);
     g6 = computeNextGeneration(g5);
-
-    if (g1 == g2 || g1 == g3 || g1 == g4 || g1 == g5 || g1 == g6) {
-      break; // periodic
-    }
-
     g1 = computeNextGeneration(g6);
-    if (g1 == 0 || g2 == 0 || g3 == 0 || g4 == 0 || g5 == 0 || g6 == 0) {
+
+    if (g1 == 0) {
       ended = true; // died out

       // Adjust the age
@@ -118,6 +114,11 @@ __device__ int countGenerations(uint64_t pattern) {

       break;
     }
+
+    if ((g1 == g2) || (g1 == g3) || (g1 == g4)) {
+        // periodic
+        break;
+    }
   }
   while (generations < 300);


In particular, once we run the pattern for 6 generations, we firstly check whether it's zero (in which case it's died out) and otherwise do the periodicity check (we only need to check the current generation versus the generation 3, 4, and 5 generations ago to see if it's settled into a cycle of length <= 5).

I've taken a look at the assembly output as well (cuobjdump -sass build/find-optimal.o) and there's nothing suspicious there, so I suspect that there are no further substantial speed improvements (although, again, I'll need to check with the visual profiler to be sure of that).


Best wishes,


Adam P. Goucher

-----

I implemented two of your suggestions working.  I used the bit slicing code with your suggested masking and I implemented period checking without Floyd’s cycle detection before falling back to that.That code running on a machine with 8GPUs is able to run patterns to stable at a rate of ~10B patterns/second; up from ~500M so not quite two orders of magnitude but amazing improvement.

FWIW I ran it for about 7 hours randomly searching and increased the longest pattern I’ve found to 208 generations:

#CXRLE Pos=-4,-4
x = 8, y = 8, rule = B3/S23:P8,8
2o2bob2o$5b2o$o3bob2o$2o3bo$o4b2o$3bobobo$2bob2o$2o4bo!

But… an exhaustive search is still going to take ~60years.

I haven’t looked into how to do fast rotation/reflection to trim the search space…. But even if I could rule out ALL duplicate rotated/reflected patterns for free it would still take about 7years to search exhaustively (using the 2^61 estimate for a plane instead of torus).

I looked a bit into profiling, but I’m working on a Mac with a Jetson nano for prototyping and AWS hosts for real performance testing and on first glance it wasn’t clear to me how to use nvvp in such a setup (or whether it would be practicable).

That said nvprof on a run checking 500M candidates in an arbitrarily chosen range produced this:

[Thread 0] searching ALL 11605407783469538019 - 11605407783969538018
==13860== NVPROF is profiling process 13860, command: ./build/find-optimal -b11605407783469538019 -e11605407783969538019
==13860== Warning: Unified Memory Profiling is not supported on the underlying platform. System requirements for unified memory can be found at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-requirements
[Thread 0] 177 generations : 11605407783469538019 : 1010000100001110101100001110010000010111010000010101101011100011
[Thread 0] 184 generations : 11605407783522095703 : 1010000100001110101100001110010000011010011000110101001001010111
[Thread 0] 192 generations : 11605407783688896998 : 1010000100001110101100001110010000100100010101001000000111100110
[Thread 0] COMPLETE
==13860== Profiling application: ./build/find-optimal -b11605407783469538019 -e11605407783969538019
==13860== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  18.7740s        48  391.12ms  272.07ms  493.44ms  evaluateRange(__int64, __int64, __int64*, __int64*)
                    0.00%  71.404us        96     743ns     416ns  2.0320us  [CUDA memcpy DtoH]
                    0.00%  17.442us        48     363ns     208ns  5.6250us  [CUDA memcpy HtoD]
      API calls:   98.14%  18.7887s       144  130.48ms  35.366us  493.55ms  cudaMemcpy
                    1.84%  353.10ms         2  176.55ms  21.980us  353.08ms  cudaMalloc
                    0.02%  3.5225ms        48  73.385us  54.741us  173.76us  cudaLaunchKernel
                    0.00%  121.88us        97  1.2560us     625ns  30.835us  cuDeviceGetAttribute
                    0.00%  12.135us         1  12.135us  12.135us  12.135us  cuDeviceTotalMem
                    0.00%  7.6040us         3  2.5340us  1.0940us  3.3850us  cuDeviceGetCount
                    0.00%  3.5940us         2  1.7970us  1.4060us  2.1880us  cuDeviceGet
                    0.00%  2.0310us         1  2.0310us  2.0310us  2.0310us  cuDeviceGetName
                    0.00%     938ns         1     938ns     938ns     938ns  cuDeviceGetUuid

Nothing obviously stuck out to me there.

Thanks again for your generous help!

Andy

-----

Great! Let me know how it progresses!

You might want to upgrade the annotation on your 8x8 iteration function from __device__ to __host__ __device__ (I can't see any GPU-specific things in there) so that you can test it more easily (e.g. to test that the bitsliced implementation matches the old implementation). I personally use googletest for writing C++/CUDA unit tests.

Floyd's algorithm is nice and simple, but you can end up doing a lot more work than necessary: if it takes N generations to become oscillatory (say with a small period -- 1 and 2 are most common), then you'll have done 3N iterations (N with the tortoise and 2N with the hare) when < N+2 iterations were actually needed to prove oscillation. Simpler than implementing Nivasch's algorithm would be to add a check for p2 oscillation, by running the pattern 2 generations and comparing with the original state; if this fails after k generations, then you can fall back to Floyd's algorithm. (The apgsearch code does a similar thing, 6 generations at a time, to catch > 99.99% of all cases, and only then applies more sophisticated algorithms as a fallback.)

Yes, feel free to e-mail me with any followup questions.


Best wishes,


Adam P. Goucher

-----

Wow.  Thank you so much for taking the time to guide me! It sounds like I shouldn’t give up quite yet :)

A few answers to your questions below
Yes that was my current implementation of GOL in CUDA and you correctly point out that it loops the cells. I had seen some references to what you describe but wasn’t able to understand it to apply in my case.  Thank you for the details and reference code!
I’m focusing only on the box topology instead of torus.
I’m using Floyd’s algorithm for cycle detection (it's really easy to implement)… I’ll check out Nivasch algorithm.

If I get stuck moving forward would you mind if I ask a followup question?

Thanks,
Andy

-----

Dear Andy,

Thank you for the e-mail! Is this your current implementation?

https://github.com/aharbick/conway/blob/main/find-optimal/gol_cuda.cu

If so, then you're missing a few very important tricks.

The main one is to use bitsliced logic: instead of doing a loop over the 64 cells, you can instead evaluate all 64 cells in parallel. You'll need to rewrite the GoL rules as a Boolean circuit (with 9 inputs and 1 output) and then implement it using shifts and bitwise operations. I've done the analogous thing for a 64x64 toroidal grid in CUDA (except it uses a whole warp instead of a thread, with each thread in the warp 'owning' a 64x2 strip, stored as a 4-tuple of 32-bit registers), and that code is used as the basis of the GPU algorithm in apgsearch:

https://gitlab.com/apgoucher/lifelib/-/blob/master/cuda2/cudagol.h#L31

Your 8x8 code will be somewhat simpler, because it will run in a single thread (so each thread holds a different 8x8 array) rather than in a warp, so you don't need to deal with shuffles to exchange information between threads. In any case, you'll probably want the same Boolean circuit that I'm using (it was discovered by Tom Rokicki, and it uses 19 bitwise operations and a handful of shifts). Here's the relevant section of the e-mail from Tom:

--- Beginning of quoted e-mail fragment ---

2.  I'm not exactly sure how you are computing the actual next
     generation, but I'm going to share this 19-op trick with you.  It
     lets you compute the next generation with only 19 boolean ops.
     It's cheaper than doing the full arithmetic.  It does require that
     you keep around the three-bit sum of two previous words, and
     the two-bit sum of the previous word.  The code I'm showing here
     just calculates the next gen of the internal 6x6 block from an
     8x8 64-bit block:

void add2(lifeword a, lifeword b, lifeword &s0, lifeword &s1) {
   s0 = a ^ b ;
   s1 = a & b ;
}
void add3(lifeword a, lifeword b, lifeword c, lifeword &s0, lifeword &s1) {
   lifeword t0, t1, t2 ;
   add2(a, b, t0, t1) ;
   add2(t0, c, s0, t2) ;
   s1 = t1 ^ t2 ;
}
lifeword gen2(lifeword a) {
   lifeword s0, sh2, a0, a1, sll, slh ;
   add2(a<<1, a>>1, s0, sh2) ;
   add2(s0, a, a0, a1) ;
   a1 |= sh2 ;
   add3(a0>>8, a0<<8, s0, sll, slh) ;
   lifeword y = a1 >> 8 ;
   lifeword x = a1 << 8 ;
   return (x^y^sh2^slh)&((x|y)^(sh2|slh))&(sll|a) ;
}
Of course instead of >>8 and <<8 you'll use the info from the previous
words instead, if you are doing a sequence of words.

Everything is as normal; we calculate both the horizontal sums of
a<<1 and b>>1 and the sum of a<<1, a, and a>>1 (giving us four
values).  Then, we calculate the full-adder sum of the low-order bits
vertically.  Up to now everything is pretty much normal (except note
we are saving the sum of the left and right bits as well as the sum of
all three).  This is where things diverge.

At this point we have the low order bit of the total neighbor sum (of 8,
not of 9), as well as *four* values of weight 2:  the upper bit of the
horizontal sum of the row above (x), the upper bit of the horizontal
sum of the row below (y), the upper bit of the sum of left/right in
the current row (slh), and the upper bit from the vertical sum of the
low order bits (sh2).
The next generation is alive if and only if exactly one of these is 1.
We can calculate the expression "exactly one of a, b, c, and d is 1"
with ((a^b^c^d) /* sum is odd */ & ((a | b) ^ (c | d))) /* exclude 3 */

Overall this, at 19 logical operations plus the requisite shifts is fewer
ops than doing full sums.  I haven't been able to reduce things any
further.

For your version, you'll need to be more careful about the boundaries. When you're doing the shifts by >>1 and <<1, you'll need to mask by constants such as 0x7f7f7f7f7f7f7f7f. If you're doing a torus (wrapping boundaries) instead of a box (permanently-off cells outside the 8x8 arena), you'll need to do circular shifts instead; I'm sure that you can work out the details.

That should be a relatively straightforward rewriting of your inner function, and it should speed it up by ~ 2 orders of magnitude.

Once you've done that, I'd strongly recommend using the CUDA visual profiler (nvvp) to analyse your code and see what the performance bottlenecks are. There's lots of caveats with efficient GPU programming that don't apply to CPU programming, such as warp divergence, memory coalescing, smem bank conflicts, register spilling, etc., that can make a huge impact to the performance of your application. The nvvp tool is great at diagnosing these once you're familiar with using it (there's a bit of a steep learning curve at the beginning).

Make sure you're calling your compiler (nvcc or clang) with the arguments:

-O3 --ptxas-options=-v -lineinfo

The first of these is for the code generation; the remaining options give you helpful output when it comes to debugging and profiling.

I'm intending to produce an online course describing how to write fast GPU code, including example code and screencasts of how to use the profiling tools and suchlike.


What algorithm are you using for cycle detection? If you haven't already, I'd recommend familiarising yourself with the main algorithms here:

https://en.wikipedia.org/wiki/Cycle_detection

Gosper and Nivasch are, of course, both fellow Life enthusiasts.

I'd probably recommend using Nivasch's algorithm, but you'll want to store the stack in shared memory (using S[i * blockDim.x + threadIdx.x] to store the ith element of the stack, where S is an array of size n * blockDim.x, where n is the maximum stack length) to avoid register spills. If your cycle detection stack overflows (should be rare if your stack size n is large enough), then you can fall back on Floyd's algorithm.


As for reducing the search space, again it matters whether you're using a torus or a box. For boxes, you'll need to search roughly 2^61 configurations (because rotation/reflection); for toruses, you'll only need to search roughly 2^55 configurations (because translations).


Best wishes,


Adam P. Goucher

-----

Hi Adam,

This year I’ve been somewhat obsessed with Life.  It started off with a 3d printing/Arduino art project which you can see here: https://youtu.be/GUKUeunvB_Y. And evolved into the question “what is the longest number of generations that an initial pattern can actively evolve in an 8x8 plane with the default rules for Life (before it dies out or stabilizes)”?

Having studied computer science I knew about Life and how Martin Gardner had popularized John Conway’s idea, but I had no idea the depth of work and thinking that have gone into the area before I started trying to answer my question with my own code and brute force/parallelism (https://github.com/aharbick/conway). I got up to being able to simulate ~500m initial 8x8 patterns to completion per second and realized that even that speed (which required 8GPUs on an AWS instance that costs $16/hr) it would take millennia to exhaustively search for an answer.  So I started googling and hunting for previous work and found the vast amounts of thinking and code that have gone into optimizing Life and cellular automata in general.  Your work with apgmera, lifelib, catagolue, and others is impressive.  I’ve read through some of the code and been trying to figure out whether it makes answering my question feasible.

Recently, I’ve started to come to grips with the fact that 2^64 possible initial starting states is just VAST and the only way I’m going to definitively answer my question is if I can identify one of two (unlikely?) possibilities:

a search algorithm that is 4+ orders of magnitude faster
a way to reduce the search space by several orders of magnitude

I’m hoping that you can help me to assess the feasibility of my quest. I’m really lousy at giving up on tech problems, but it might be the wisest move at this point.  I got into as an art project and then got stuck on this apparently big question.  If it’s not practically feasible to answer I have a cool idea for where to take my art instead. :)

Thanks for your consideration!

Andy Harbick
