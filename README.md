# Conway's Game of Life art project

## Arduino

TODO documentation

## Find Optimal

[Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) is a simulation that follows some simple rules to manipulate a grid of pixels that are either alive or dead through successive generations until either everything dies OR there is an oscillation where repetition happens perpetually.

In an 8x8 grid such as the art project most random patterns die out or repeat infinitely within less than 40 generations.  An 8x8 grid has 2^64 possible states so I wanted to know if it was possible to answer the question:

> What 8x8 grid pattern produces the most generations for the default rules of Conway's Game of Life but doesn't become an oscillation?

In other words...  I wanted to know a pattern that would evolve and ultimately die but would've lived longer than all other patterns.

The code contained in find-optimal has the code to generate patterns, evolve them according to the default rules, and search for a solution.  It uses pthreading so you can parallelize the search, and it searchs both randomly and systematically.

BUT, as currently written, even on an [AWS c6i.32xlarge](https://aws.amazon.com/ec2/instance-types/c6i/) instance it won't finish for centuries.

So... The solution needs some way to trim the search space (e.g. do all mirrored patterns have the same number of generations and hence you could cut out half of mirrorable patterns) OR have access to a lot more scale (e.g. I've been thinking that it's possible to do this as a mapreduce problem) OR perhaps there's a way to find an answer analytically OR ???

### Seeing it in action

Change into the find-optimal directory and compile...  I only tried this on a mac, but it's just using g++ and should be pretty easy to get running.

```
$ cd find-optimal
$ make
mkdir -p build
g++ -o build/find-optimal build/find-optimal.o -std=c++11
g++ -DRANDOM -o build/find-optimal-random build/find-optimal.o -std=c++11
```

Then you can run it to see it kick out some answers...  Pass in number of threads (in the example below I'm running 8 threads and searching randomly)
```
aharbick-M1:find-optimal aharbick$ ./bin/find-optimal-random 8

[Thread 1] searching range 1 - 2305843009213693951

[Thread 3] searching range 4611686018427387903 - 6917529027641081853

[Thread 2] searching range 2305843009213693952 - 4611686018427387902

[Thread 4] searching range 6917529027641081854 - 9223372036854775804

[Thread 5] searching range 9223372036854775805 - 11529215046068469755

[Thread 6] searching range 11529215046068469756 - 13835058055282163706

[Thread 7] searching range 13835058055282163707 - 16140901064495857657

Waiting on thread 0

[Thread 8] searching range 16140901064495857658 - 18446744073709551608
[Thread 6] 3 generations : 11529215046068469756 : 0011111111111111111111111111111111111111111111111111111111111001
[Thread 3] 5 generations : 4611686018427387967 : 1111110000000000000000000000000000000000000000000000000000000010
[Thread 3] 6 generations : 4611686018427387999 : 1111101000000000000000000000000000000000000000000000000000000010
[Thread 3] 8 generations : 4611686018427388031 : 1111111000000000000000000000000000000000000000000000000000000010
[Thread 4] 3 generations : 6917529027641081854 : 0111111111111111111111111111111111111111111111111111111111111010
[Thread 3] 6 generations : 4611686018427388094 : 0111110100000000000000000000000000000000000000000000000000000010
[Thread 6] 6 generations : 11529215046068469950 : 0111110100000000000000000000000000000000000000000000000000000101
[Thread 4] 8 generations : 6917529027641081983 : 1111111000000000000000000000000000000000000000000000000000000110
[Thread 6] 9 generations : 11529215046068470395 : 1101111001000000000000000000000000000000000000000000000000000101
[Thread 6] 38 generations : 11529215046068470655 : 1111111011000000000000000000000000000000000000000000000000000101
[Thread 1] 40 generations : 7665 : 1000111110111000000000000000000000000000000000000000000000000000
[Thread 1] 43 generations : 15543 : 1110110100111100000000000000000000000000000000000000000000000000
[Thread 1] 52 generations : 32209 : 1000101110111110000000000000000000000000000000000000000000000000
[Thread 1] 54 generations : 80482 : 0100011001011100100000000000000000000000000000000000000000000000
[Thread 1] 57 generations : 130730 : 0101010101111111100000000000000000000000000000000000000000000000
```
