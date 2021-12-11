# Conway's Game of Life art project

## Arduino

TODO documentation

## Find Optimal

[Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) is a simulation that follows some simple rules to manipulate a grid of pixels that are either alive or dead through successive generations until either everything dies OR there is an oscillation where repetition happens perpetually.

In an 8x8 grid such as the art project most random patterns die out or repeat infinitely within less than 200 generations.  An 8x8 grid has 2^64 possible states so I wanted to know if it was possible to answer the question:

> What 8x8 grid pattern produces the most generations for the default rules of Conway's Game of Life but doesn't become an oscillation?

In other words...  I wanted to know a pattern that would evolve and ultimately die or become stable but would've lived longer than all other patterns.

The code contained in find-optimal has the code to generate patterns, evolve them according to the default rules, and search for a solution.  There is code that uses pthreading so you can parallelize the search and code that uses CUDA to parallelize.  Both can randomly and sequentially.

BUT, as currently written, even on expensive AWS instances using multiple GPUs it won't finish for centuries.

So... It seems like a solution needs to improve compute speed or trim the search space by several orders of magnitude.

### Seeing it in action

Change into the find-optimal directory and compile...  This shows it compiling on my mac but it also is possible to build on Linux.  NOTE on a mac you have to use Homebrew to install libargp.

```
$ make
mkdir -p build
g++ -c -o build/explore-cachability.o explore-cachability.cpp -I/opt/homebrew/include
g++ -o build/explore-cachability build/explore-cachability.o -I/opt/homebrew/include
g++ -c -o build/find-optimal.o find-optimal.cpp -I/opt/homebrew/include
g++ -c -o build/gol.o gol.cpp -I/opt/homebrew/include
g++ -c -o build/utils.o utils.cpp -I/opt/homebrew/include
g++ -o build/find-optimal build/find-optimal.o build/gol.o build/utils.o -I/opt/homebrew/include -L/opt/homebrew/lib -largp -lpthread
```

Then you can run it to see it kick out some answers...  Here's a run that looks at 1m random numbers using 8 threads (takes about 3s)
```
$ ./build/find-optimal -r1 -t8 -e 1000000
[Thread 0] RANDOMLY searching 124999 candidates (1 - ULONG_MAX)
[Thread 4] RANDOMLY searching 124999 candidates (1 - ULONG_MAX)
[Thread 7] RANDOMLY searching 124999 candidates (1 - ULONG_MAX)
[Thread 2] RANDOMLY searching 124999 candidates (1 - ULONG_MAX)
[Thread 3] RANDOMLY searching 124999 candidates (1 - ULONG_MAX)
[Thread 4] 3 generations : 8893201257484774076 : 0111101101101010111111110110110011101100011011111110101010111100
[Thread 7] 4 generations : 16576868547380073562 : 1110011000001100110111100001110110100110111011110101010001011010
[Thread 2] 14 generations : 16796954305446088685 : 1110100100011010110001001111001001011100100101010000111111101101
[Thread 0] 24 generations : 14487566997632512707 : 1100100100001110001011010110110100001001100011101111001011000011
[Thread 1] RANDOMLY searching 124999 candidates (1 - ULONG_MAX)
[Thread 2] 26 generations : 6636658169425087701 : 0101110000011010001001010111000101001100010001101100110011010101
[Thread 5] RANDOMLY searching 124999 candidates (1 - ULONG_MAX)
[Thread 0] 35 generations : 8359279736737663972 : 0111010000000010001000001010000110010110010111101000101111100100
[Thread 3] 45 generations : 2783232577693583668 : 0010011010100000000001110100100010000000101110001000100100110100
[Thread 3] 47 generations : 1152573963862380368 : 0000111111111110110000111110100111011101011000011000111101010000
[Thread 3] 68 generations : 14787817465925001821 : 1100110100111000111000011010001011010011110001101110001001011101
[Thread 6] RANDOMLY searching 124999 candidates (1 - ULONG_MAX)
[Thread 2] 72 generations : 15457680787659307064 : 1101011010000100101101101100011100101111110100000011100000111000
[Thread 2] 78 generations : 8567746294949225033 : 0111011011100110101111111101110010000100000111100010001001001001
[Thread 6] 100 generations : 8653740192267176050 : 0111100000011000010000101101101011111010100110110000100001110010
[Thread 3] 101 generations : 17168731094460031243 : 1110111001000011100101011111011101110001011111100110000100001011
[Thread 6] 102 generations : 8743451316802808829 : 0111100101010110111110101010010111011000111100111110101111111101
[Thread 0] 103 generations : 11637480641286450864 : 1010000110000000101000101111110000111000010111010010111010110000
[Thread 4] 107 generations : 15764142000968704142 : 1101101011000101011110111010000010111100100011100010000010001110
[Thread 6] 115 generations : 14904260802548025389 : 1100111011010110100100100011101111010111101001110100100000101101
[Thread 7] 118 generations : 13135864391484927618 : 1011011001001011111101110001000100111000001010010110111010000010
[Thread 6] 128 generations : 759807430909934230 : 0000101010001011011000001101010100110001000101101011101010010110
[Thread 6] 144 generations : 92218752798826419 : 0000000101000111101000000111011110010010001101000001111110110011
[Thread 4] 157 generations : 3252803366839646923 : 0010110100100100010001110110110110010100101010111111101011001011
[Thread 0] COMPLETE
[Thread 1] COMPLETE
[Thread 2] COMPLETE
[Thread 3] COMPLETE
[Thread 4] COMPLETE
[Thread 5] COMPLETE
[Thread 6] COMPLETE
[Thread 7] COMPLETE
```

You can pass --help to see full help.

```
aharbick-M1:find-optimal aharbick$ ./build/find-optimal --help
Usage: find-optimal [OPTION...]
Search for terminal and stable states in an 8x8 bounded Conway's Game of Life
grid

  -b, --beginat=num          Explicit beginAt.
  -c, --cudaconfig=config    CUDA kernel params
                             numgpus:blocksize:threadsperblock (e.g.
                             1:4096:256)
  -e, --endat=num            Explicit endAt.
  -r, --random[=ignorerange] Use random patterns. Default in [beginAt-endAt].
                             -r1 [1-ULONG_MAX].
  -t, --threads=num          Number of CPU threads (if you use more than one
                             GPU you should use matching threads).
  -?, --help                 Give this help list
      --usage                Give a short usage message

Mandatory or optional arguments to long options are also mandatory or optional
for any corresponding short options.
```