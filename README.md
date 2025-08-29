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

Change into the find-optimal directory and run `make`.  NOTE on a mac you have to use Homebrew to install libargp.

Run `./build/find-optimal --help` to see the arguments.
