## Strip Search Algorithm Design

This document describes the "strip search" algorithm, an optimization for exhaustively searching the 8x8 Game of Life grid that exploits the irreversibility of Life to achieve approximately 14x search space reduction.

The initial idea was proposed here: https://conwaylife.com/forums/viewtopic.php?f=7&t=5489#p221977

### The Core Insight: Life is Irreversible

The Game of Life is irreversible - many different starting configurations can evolve into the same result. vilc observed that this property can be exploited for search space reduction:

> "As the Game of Life is irreversible, lots of different starting patterns produce equivalent trajectories after 2 generations. If we could enumerate equivalence classes and test one representative from each class, we would achieve a significant speedup."

### Speed of Light Causality

In Conway's Game of Life, information propagates at most one cell per generation (the "speed of light"). This means:

- **After 2 generations**: Cells more than 2 rows apart cannot have influenced each other
- **Consequence**: The top 2 rows and bottom 2 rows of an 8x8 grid evolve independently for 2 generations (they're separated by 4 rows)

This allows us to decompose the search:

```
SSSSSSSS  <- Top strip (rows 0-1): 16 bits
SSSSSSSS
BBBBBBBB  <- Middle block (rows 2-5): 32 bits
BBBBBBBB
BBBBBBBB
BBBBBBBB
SSSSSSSS  <- Bottom strip (rows 6-7): 16 bits
SSSSSSSS
```

### The Deduplication Strategy

Instead of testing all 2^64 patterns, we:

1. **Fix the middle 4x8 block** (32 bits, 2^32 iterations)
2. **For each middle block, find unique strips**:
   - Test all 2^16 = 65,536 possible top strips
   - Compute what each strip produces in the top 4 rows after 2 generations (the "signature")
   - Keep only strips that produce *unique* signatures (~17K out of 65K)
   - Do the same for bottom strips
3. **Test only unique combinations**:
   - Instead of 65K x 65K = 4.3 billion combinations
   - We test ~17K x ~17K = ~289 million combinations
   - **Speedup: ~14x**

### Why This Works

Multiple starting strips can produce identical 2-generation outcomes:

```
Strip A: 10110010 01001101  ---> after 2 gens ---> 00101100 11010010
Strip B: 11010010 00110101  ---> after 2 gens ---> 00101100 11010010  (same!)
Strip C: 01110001 10001110  ---> after 2 gens ---> 00101100 11010010  (same!)
```

If strips A, B, and C all produce the same state after 2 generations, then for *any* fixed middle block, patterns using A, B, or C will have identical behavior from generation 2 onward. We only need to test one representative from each equivalence class.

### Grid Layout for CUDA

```
SSSSSSSS  <- Top strip: loaded from passed array
SSSSSSSS
BBBBBBBB  <- Block bits: fixed for entire kernel launch
PPPPPPPP  <- Pattern bits (part of middle block)
PPPPPPPP
TTTTTTTT  <- Thread bits (part of middle block)
SSSSSSSS  <- Bottom strip: iterated in inner loop
SSSSSSSS
```

The bit packing in a 64-bit integer:
- Bits 0-15: Top strip (rows 0-1)
- Bits 16-47: Middle block (rows 2-5)
- Bits 48-63: Bottom strip (rows 6-7)

### Algorithm Implementation

#### Phase 1: Find Unique Strips

For each middle block, we identify which strips produce unique 2-generation signatures:

```cpp
// Kernel: findUniqueTopAndBottomStrips
// Input: 32-bit middle block
// Output: Arrays of unique top/bottom strips (~17K each)

for each of 65,536 possible strips:
    // Test as top strip
    pattern = strip | (middleBlock << 16)  // zeros in bottom
    signature = top4Rows(gen2(gen1(pattern)))
    if signature not in hashTable:
        add strip to validTopStrips

    // Test as bottom strip
    pattern = (middleBlock << 16) | (strip << 48)  // zeros in top
    signature = bottom4Rows(gen2(gen1(pattern)))
    if signature not in hashTable:
        add strip to validBottomStrips
```

The hash table uses `atomicCAS` with linear probing for GPU-friendly deduplication.

#### Phase 2: Test Strip Combinations

With ~17K valid strips for top and bottom, test all ~289M combinations:

```cpp
// Kernel: findCandidatesForStripBlock
// Uses nested loops (more efficient than div/mod on GPU)

for topIdx in validTopStrips (strided across threads):
    for bottomIdx in validBottomStrips:
        pattern = topStrip | (middleBlock << 16) | (bottomStrip << 48)

        // Fast evaluation (same as frame search)
        while generations < 300:
            step 6 generations
            check for death/cycles
            if generations >= 180:
                save as candidate
```

#### Phase 3: Process Candidates

Candidates undergo full cycle detection using Floyd's or Nivasch's algorithm (reuses `processCandidates` kernel from frame search).

### Nested Optimization Strategies

vilc described three nested strategies, each subsuming the previous:

#### Strategy 1: Corner Cell Irrelevance (~1.15x)

Some partial patterns make certain corner cells irrelevant to future evolution. For example, if the top-left 3x3 corner is mostly empty, changing the top-left cell doesn't affect generation 1. Testing all 256 possible 3x3 corners reveals 18 arrangements where the corner cell is irrelevant.

- Applied to all 4 corners: (512-18)^4 / 512^4 = 494^4 / 512^4 ≈ 0.87x search space
- **Speedup: ~1.15x**

#### Strategy 2: Edge Strip Deduplication (~2.12x)

For the top 3x8 strip, only 11,510,370 of 2^24 possible patterns produce unique results after 1 generation.

- Applied to top + bottom strips: (2^24 / 11,510,370)^2
- **Speedup: ~2.12x**

#### Strategy 3: Two-Generation Causality (~14x)

Due to the speed-of-light limit (1 cell/generation), regions separated by 4+ cells cannot interact for 2 generations. This is the algorithm implemented in strip search.

- For each 8x4 middle block, ~17,000 unique top strips and ~17,000 unique bottom strips
- Combined: 17,000 × 17,000 ≈ 289M patterns per block (vs 4.3B naive)
- **Speedup: ~14x**

**Strategy 3 subsumes strategies 1 and 2**, so we only implement the most powerful version.

### Why Symmetry Elimination Doesn't Apply

Unlike frame search, we cannot easily eliminate symmetric middle blocks:

- The middle block is not symmetric with respect to the full grid
- Valid strips depend on the specific middle block configuration
- Each middle block produces different sets of valid strips

We must test all 2^32 middle blocks, but the ~14x reduction from strip deduplication more than compensates.

### Empirical Validation Tool

The `explore-reversibility` program ([explore-reversibility.cpp](../src/explore-reversibility.cpp)) provides empirical measurement of the reduction factors by:

1. **Corner Irrelevance Analysis**: Samples random patterns and counts how many have irrelevant corner cells (cells that don't affect generation 1)
2. **Edge Strip Deduplication**: Measures how many unique 3-row results are produced after 1 generation
3. **Two-Generation Causality Test**: For sample middle blocks, counts unique top/bottom signatures after 2 generations

Run with:
```bash
./explore-reversibility 10000              # Basic analysis
./explore-reversibility 10000 42 --two-gen # Include expensive 2-gen test
```

Sample output confirms the theoretical predictions:
- Average unique top 2-row patterns: ~17,000 (out of 65,536)
- Average unique bottom 2-row patterns: ~17,000 (out of 65,536)
- Combined per-block reduction: ~14x