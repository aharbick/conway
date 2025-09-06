## Claude Cloud vs. Owned Analysis

  | GPU          | CUDA Cores | Memory | Bandwidth  | Est. Performance       | Cost            |
  |--------------|------------|--------|------------|------------------------|-----------------|
  | Jetson Orin  | 2,048      | 64GB   | 204 GB/s   | 260M patterns/sec      | ~$800 (owned)   |
  | RTX 4070 Ti  | 8,448      | 16GB   | 672 GB/s   | ~800M patterns/sec     | ~$800 (owned)   |
  | L4 (Cloud)   | 7,424      | 24GB   | 300 GB/s   | ~400-500M patterns/sec | ~$0.60/hour     |
  | V100 (Cloud) | 5,120      | 16GB   | 900 GB/s   | ~600-800M patterns/sec | ~$2.50/hour     |
  | RTX 5090     | ~21,760    | 32GB   | 1,792 GB/s | ~2-3B patterns/sec     | ~$2,000 (owned) |

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

### Testing on L4 (g2-standard-4: 4vCPU, 2 core, 16GB memory)

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
