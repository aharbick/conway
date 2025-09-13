## Claude Analysis

  | GPU          | CUDA Cores | Memory | Bandwidth  | Predicted Performance  | Actual Performance | Cost           |
  |--------------|------------|--------|------------|------------------------|--------------------|----------------|
  | Jetson Orin  | 2,048      | 64GB   | 204 GB/s   | Not predicted          | 260M patterns/sec  | $2,000 (owned) |
  | L4 (Cloud)   | 7,424      | 24GB   | 300 GB/s   | ~400-500M patterns/sec | 5.35B patterns/sec | ~$0.71/hour    |
  | V100 (Cloud) | 5,120      | 16GB   | 900 GB/s   | ~600-800M patterns/sec | 3.16B patterns/sec | ~$1.78/hour    |
  | RTX 5090     | 21,760     | 24GB   | 1,792 GB/s | 12-18B patterns/sec    | Not tested         | $3,500 (owned) |

### Testing on L4 (g2-standard-4: 4vCPU, 2 core, 16GB memory)

```
aharbick@conway-gpu-vm:~/conway/find-optimal$ ./build/find-optimal -f 10000:10001 -v
CUDA devices available: 1
Using 1 GPU(s) with blockSize=1024, threadsPerBlock=1024
Best generations so far: 206
Resuming from frame: 10000
[Thread 0 - 1757123303] Running with CUDA enabled
[Thread 0 - 1757123303] searching ALL in frames (10000 - 10001)
[Thread 0 - 1757123316] frameIdx=10000, kernelIdx=0, chunkIdx=0, bestGenerations=196, bestPattern=121164286024753984, bestPatternBin=0000000110101110011101100100011010110011111110000010001101000000, patternsPerSec=5354807460
[Thread 0 - 1757123329] frameIdx=10000, kernelIdx=1, chunkIdx=0, bestGenerations=197, bestPattern=124482395516969800, bestPatternBin=0000000110111010010000000001010001000101100101000010001101001000, patternsPerSec=5261557652
[Thread 0 - 1757123343] frameIdx=10000, kernelIdx=2, chunkIdx=0, bestGenerations=205, bestPattern=108709153201324880, bestPatternBin=0000000110000010001101100110011000011100011001100000001101010000, patternsPerSec=5201557390
[Thread 0 - 1757123357] frameIdx=10000, kernelIdx=3, chunkIdx=0, bestGenerations=202, bestPattern=113245416975775576, bestPatternBin=0000000110010010010101000001101101010011010010000011011101011000, patternsPerSec=5177582419
[Thread 0 - 1757123370] frameIdx=10000, kernelIdx=4, chunkIdx=0, bestGenerations=199, bestPattern=700888277469893440, bestPatternBin=0000100110111010000011100010110011000101110101000001001101000000, patternsPerSec=5154540927
[Thread 0 - 1757123384] frameIdx=10000, kernelIdx=5, chunkIdx=0, bestGenerations=198, bestPattern=691936324663908168, bestPatternBin=0000100110011010010000000110101111010110110010000001011101001000, patternsPerSec=5060420099
...
```

So best was: 5,354,807,460 (about 10-12x better)

I also implemented and tested Nivasch cycle-detection algorithm (vs. Floyd's) and it was a little slower (both with 4/32 and 10/50 stacks/stackSize)

```
aharbick@conway-gpu-vm:~/conway/find-optimal$ ./build/find-optimal -A 436925
Comparing cycle detection algorithms on frameIdx: 436925
Frame value: 181130247033406370

=== Testing Floyd's cycle detection ===
timestamp=1757735030, frameIdx=436925, kernelIdx=0, bestGenerations=204, bestPattern=198056304427363234, bestPatternBin=0000001010111111101000110010100011010110111010110110101110100010, patternsPerSec=5223775938
timestamp=1757735043, frameIdx=436925, kernelIdx=1, bestGenerations=204, bestPattern=198060702465483690, bestPatternBin=0000001010111111101001110010100011010110011010110110001110101010, patternsPerSec=5274506411
timestamp=1757735056, frameIdx=436925, kernelIdx=2, bestGenerations=204, bestPattern=198060702734968754, bestPatternBin=0000001010111111101001110010100011100110011110110110011110110010, patternsPerSec=5192226371
timestamp=1757735069, frameIdx=436925, kernelIdx=3, bestGenerations=204, bestPattern=198056321933337530, bestPatternBin=0000001010111111101000110010110011101010010110110110001110111010, patternsPerSec=5223034909
timestamp=1757735082, frameIdx=436925, kernelIdx=4, bestGenerations=204, bestPattern=772265274155690914, bestPatternBin=0000101010110111101000110010110011011010011010110110101110100010, patternsPerSec=5182108994
timestamp=1757735095, frameIdx=436925, kernelIdx=5, bestGenerations=204, bestPattern=772265136725648298, bestPatternBin=0000101010110111101000110000110011011010111100110110001110101010, patternsPerSec=5271454125
timestamp=1757735109, frameIdx=436925, kernelIdx=6, bestGenerations=204, bestPattern=772265136986220466, bestPatternBin=0000101010110111101000110000110011101010011110110110011110110010, patternsPerSec=5188355346
timestamp=1757735122, frameIdx=436925, kernelIdx=7, bestGenerations=204, bestPattern=772265136986219450, bestPatternBin=0000101010110111101000110000110011101010011110110110001110111010, patternsPerSec=5174661253
timestamp=1757735135, frameIdx=436925, kernelIdx=8, bestGenerations=196, bestPattern=1348706474598553506, bestPatternBin=0001001010110111100100010110010010010100101101110101011110100010, patternsPerSec=5080777316
timestamp=1757735149, frameIdx=436925, kernelIdx=9, bestGenerations=199, bestPattern=1344264079211324330, bestPatternBin=0001001010100111110010010000111011001101101101010110001110101010, patternsPerSec=5174839423
timestamp=1757735162, frameIdx=436925, kernelIdx=10, bestGenerations=203, bestPattern=1344191948510289842, bestPatternBin=0001001010100111100001110111010010010000111001010101111110110010, patternsPerSec=5121222598
timestamp=1757735175, frameIdx=436925, kernelIdx=11, bestGenerations=197, bestPattern=1335265829167912890, bestPatternBin=0001001010000111110100010011001001110100010001010101011110111010, patternsPerSec=5144499442
timestamp=1757735189, frameIdx=436925, kernelIdx=12, bestGenerations=200, bestPattern=1910640374963065762, bestPatternBin=0001101010000011111101010100110001001011001100010101001110100010, patternsPerSec=5128741425
timestamp=1757735202, frameIdx=436925, kernelIdx=13, bestGenerations=212, bestPattern=1916252347412072362, bestPatternBin=0001101010010111111001010101101101110011100000010100011110101010, patternsPerSec=5232373717
timestamp=1757735215, frameIdx=436925, kernelIdx=14, bestGenerations=198, bestPattern=1916142494574017458, bestPatternBin=0001101010010111100000010111001001011000000111010111101110110010, patternsPerSec=5164919870
timestamp=1757735229, frameIdx=436925, kernelIdx=15, bestGenerations=198, bestPattern=1915112901007599546, bestPatternBin=0001101010010011110110010000100101101001010100110110011110111010, patternsPerSec=5166773209
Floyd's algorithm completed in 212.124 seconds

=== Testing Nivasch's cycle detection ===
timestamp=1757735243, frameIdx=436925, kernelIdx=0, bestGenerations=204, bestPattern=198056304418974626, bestPatternBin=0000001010111111101000110010100011010110011010110110101110100010, patternsPerSec=4724572633
timestamp=1757735257, frameIdx=436925, kernelIdx=1, bestGenerations=204, bestPattern=198056184235385770, bestPatternBin=0000001010111111101000110000110011011010111010110110001110101010, patternsPerSec=4816507304
timestamp=1757735272, frameIdx=436925, kernelIdx=2, bestGenerations=204, bestPattern=198060702399424434, bestPatternBin=0000001010111111101001110010100011010010011110110110011110110010, patternsPerSec=4752233760
timestamp=1757735286, frameIdx=436925, kernelIdx=3, bestGenerations=204, bestPattern=198060702399685562, bestPatternBin=0000001010111111101001110010100011010010011111110110001110111010, patternsPerSec=4783907614
timestamp=1757735301, frameIdx=436925, kernelIdx=4, bestGenerations=204, bestPattern=772265274155690914, bestPatternBin=0000101010110111101000110010110011011010011010110110101110100010, patternsPerSec=4756615691
timestamp=1757735315, frameIdx=436925, kernelIdx=5, bestGenerations=204, bestPattern=772265136725648298, bestPatternBin=0000101010110111101000110000110011011010111100110110001110101010, patternsPerSec=4853607995
timestamp=1757735329, frameIdx=436925, kernelIdx=6, bestGenerations=204, bestPattern=772265136994609074, bestPatternBin=0000101010110111101000110000110011101010111110110110011110110010, patternsPerSec=4802876542
timestamp=1757735343, frameIdx=436925, kernelIdx=7, bestGenerations=204, bestPattern=772265136726172602, bestPatternBin=0000101010110111101000110000110011011010111110110110001110111010, patternsPerSec=4805209032
timestamp=1757735358, frameIdx=436925, kernelIdx=8, bestGenerations=196, bestPattern=1348706474598553506, bestPatternBin=0001001010110111100100010110010010010100101101110101011110100010, patternsPerSec=4732397254
timestamp=1757735372, frameIdx=436925, kernelIdx=9, bestGenerations=199, bestPattern=1344264079211324330, bestPatternBin=0001001010100111110010010000111011001101101101010110001110101010, patternsPerSec=4810838685
timestamp=1757735387, frameIdx=436925, kernelIdx=10, bestGenerations=203, bestPattern=1344191948510289842, bestPatternBin=0001001010100111100001110111010010010000111001010101111110110010, patternsPerSec=4756234339
timestamp=1757735401, frameIdx=436925, kernelIdx=11, bestGenerations=197, bestPattern=1335265829167912890, bestPatternBin=0001001010000111110100010011001001110100010001010101011110111010, patternsPerSec=4773932078
timestamp=1757735416, frameIdx=436925, kernelIdx=12, bestGenerations=200, bestPattern=1910640374963065762, bestPatternBin=0001101010000011111101010100110001001011001100010101001110100010, patternsPerSec=4739388090
timestamp=1757735430, frameIdx=436925, kernelIdx=13, bestGenerations=212, bestPattern=1916252347412072362, bestPatternBin=0001101010010111111001010101101101110011100000010100011110101010, patternsPerSec=4830955044
timestamp=1757735444, frameIdx=436925, kernelIdx=14, bestGenerations=198, bestPattern=1916142494574017458, bestPatternBin=0001101010010111100000010111001001011000000111010111101110110010, patternsPerSec=4785676302
timestamp=1757735458, frameIdx=436925, kernelIdx=15, bestGenerations=198, bestPattern=1915112901007599546, bestPatternBin=0001101010010011110110010000100101101001010100110110011110111010, patternsPerSec=4789578229
Nivasch's algorithm completed in 229.939 seconds

=== Performance Comparison ===
Floyd's algorithm:   212.124 seconds
Nivasch's algorithm: 229.939 seconds
Floyd's was faster by 7.7% (17.815 seconds)
``

### Testing on V100 (n1-highmem-2: 2vCPU, 1 core, 13GB memory)
```
CUDA devices available: 1
Using 1 GPU(s) with blockSize=1024, threadsPerBlock=1024
Resuming from frame: 10000
[Thread 0 - 1757131157] Running with CUDA enabled
[Thread 0 - 1757131157] searching ALL in frames (10000 - 10001)
[Thread 0 - 1757131180] frameIdx=10000, kernelIdx=0, chunkIdx=0, bestGenerations=196, bestPattern=111008893702781760, bestPatternBin=0000000110001010011000100000000000101100000100000011101101000000, patternsPerSec=3150424453
[Thread 0 - 1757131201] frameIdx=10000, kernelIdx=1, chunkIdx=0, bestGenerations=197, bestPattern=124482395516969800, bestPatternBin=0000000110111010010000000001010001000101100101000010001101001000, patternsPerSec=3148282772
[Thread 0 - 1757131223] frameIdx=10000, kernelIdx=2, chunkIdx=0, bestGenerations=205, bestPattern=108709153201324880, bestPatternBin=0000000110000010001101100110011000011100011001100000001101010000, patternsPerSec=3145630332
[Thread 0 - 1757131245] frameIdx=10000, kernelIdx=3, chunkIdx=0, bestGenerations=202, bestPattern=113245416975775576, bestPatternBin=0000000110010010010101000001101101010011010010000011011101011000, patternsPerSec=3160440813
[Thread 0 - 1757131267] frameIdx=10000, kernelIdx=4, chunkIdx=0, bestGenerations=199, bestPattern=700888277469893440, bestPatternBin=0000100110111010000011100010110011000101110101000001001101000000, patternsPerSec=3158670502
[Thread 0 - 1757131289] frameIdx=10000, kernelIdx=5, chunkIdx=0, bestGenerations=198, bestPattern=691936324663908168, bestPatternBin=0000100110011010010000000110101111010110110010000001011101001000, patternsPerSec=3147340574
...
```

So best was: 3,160,440,813 (about 5x better)

## 2024 Testing On AWS

### Testing on AWS g5.48xlarge Instances

The most recent code suffered warp divergence such that it was only getting about 25% efficiency. See code at: e3b2d4bdd2433db6cf673333d084d20679b31008

```
[ec2-user@ip-172-31-18-82 find-optimal]$ for i in 1 2 3 4 5; do HOWMANY=10000000000; T1=`date +%s%N | cut -b1-13`;./build/find-optimal -b1 -e$HOWMANY > /dev/null;T2=`date +%s%N | cut -b1-13`; PERSEC=`echo "$(( $HOWMANY / ($T2-$T1) * 1000 ))"`; printf "%'d per second\n" $PERSEC; done
1,117,318,000 per second
1,132,118,000 per second
1,132,246,000 per second
1,131,861,000 per second
1,131,733,000 per second
AVG = 1,129,055,200 per second
```

But Adam Goucher(https://gitlab.com/apgoucher) helped with some amazing suggestions for improvement which produce these results:

```
[ec2-user@ip-172-31-18-82 find-optimal]$ for i in 1 2 3 4 5; do HOWMANY=10000000000; T1=`date +%s%N | cut -b1-13`;./build/find-optimal -b1 -e$HOWMANY > /dev/null;T2=`date +%s%N | cut -b1-13`; PERSEC=`echo "$(( $HOWMANY / ($T2-$T1) * 1000 ))"`; printf "%'d per second\n" $PERSEC; done
5,390,835,000 per second
5,643,340,000 per second
5,617,977,000 per second
5,599,104,000 per second
5,614,823,000 per second
AVG = 5,573,215,800 per second
```

This represents a 4.9x improvement!!!

Unfortunately, I didn't compare apples to apples, but I wrote a script that uses CUDA_VISIBLE_DEVICES=X and launches find-optimal 8 times processing 1TRILLION patterns on 8 GPUS.  Running on a g5.48xlarge on AWS you should be able to reproduce this result:

```
[ec2-user@ip-172-31-18-82 find-optimal]$ ./run8gpu-perf.sh
Launched: 83710
Launched: 83733
Launched: 83746
Launched: 83760
Launched: 83773
Launched: 83786
Launched: 83799
Launched: 83812
Waiting on 83710
Waiting on 83733
Waiting on 83746
Waiting on 83760
Waiting on 83773
Waiting on 83786
Waiting on 83799
Waiting on 83812
55,898,711,000 per second
```

This represents and average of 6,987,338,875 per GPU or ~25% improvement per GPU over the tests above.  I can't come up with a good reason why the run8gpu-perf.sh test would be actually faster per GPU so I think it's likely that if I ran the tests above with 1TRILLION paterns instead of 10BILLION patterns that we would see similar numbers.

Either way...  Even with such massive performance it would take a LONG time to process all possible patterns on the 8x8 grid:

> 2^64 total patterns / 55,898,711,000 patterns per sec / 86,400 secs per day = 3,819.48 days (or about 10.5 years)

If we were able to ignore rotations without incurring ANY overhead then the task would take a little over a year:

> 2^61 patterns no rotations / 55,898,711,000 patterns per sec / 86,400 secs per day = 477.43 days (or about 1.3 years)

For fun here's the new longest pattern I found while doing the above (in RLE format that you can simulate with Golly: https://sourceforge.net/projects/golly/files):

```
#CXRLE Pos=-4,-4
x = 8, y = 8, rule = B3/S23:P8,8
o7b$o2bo2b2o$o2bobo2b$2bo3b2o$o5bob$b2ob2obo$o3b4o$2o2bobob!
```

## Testing on Jetson

### Jetson Nano 2GB

Before the implementation of Adam's suggestions below

```
aharbick@aharbick-jetson:~/Projects/conway/find-optimal$ for i in 1 2 3 4 5; do HOWMANY=10000000000; T1=`date +%s%N | cut -b1-13`;./build/find-optimal -b1 -e$HOWMANY > /dev/null;T2=`date +%s%N | cut -b1-13`; PERSEC=`echo "$(( $HOWMANY / ($T2-$T1) * 1000 ))"`; printf "%'d per second\n" $PERSEC; done
32,434,000 per second
32,339,000 per second
32,233,000 per second
32,229,000 per second
32,189,000 per second
AVG = 32,284,800 per second
```

### Test in Jetson AGX Orin

Before the implementation of Adam's suggestions below

```
aharbick@jetson-agx-orin:~/Projects/conway/find-optimal$ for i in 1 2 3 4 5; do HOWMANY=10000000000; T1=`date +%s%N | cut -b1-13`;./build/find-optimal -b1 -e$HOWMANY > /dev/null;T2=`date +%s%N | cut -b1-13`; PERSEC=`echo "$(( $HOWMANY / ($T2-$T1) * 1000 ))"`; printf "%'d per second\n" $PERSEC; done
265,681,000 per second
266,865,000 per second
270,431,000 per second
266,752,000 per second
270,460,000 per second
AVG = 268,037,800 per second
```
