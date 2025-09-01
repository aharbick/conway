# Conway's Game of Life art project

## 3D-Printed / Arduino Project

Here's the fully assembled and working project:
https://www.youtube.com/watch?v=GUKUeunvB_Y

###
The code in the root of the project powers the arduino.

The hardware:

* Arduino UNO starter kit: https://store-usa.arduino.cc/products/arduino-starter-kit-multi-language
* LCD shield: https://www.adafruit.com/product/772
* 4 Servo sheilds: https://www.adafruit.com/product/1411
* 4 sets of header pins: https://www.adafruit.com/product/85
* 4 sets of angled pins: https://www.adafruit.com/product/816
* 64 servos (or more they die a lot): https://www.amazon.com/dp/B0925TDT2D
* Bunches of cabling like this: https://www.amazon.com/dp/B08BF4C6S2 (I bought it 3 times)

The 3D printed part designs can be found here:

* https://www.tinkercad.com/dashboard/collections/hJs3EecuwYQ/3d

Here an in-progress video:

https://www.youtube.com/watch?v=tJrgmwz3HJU

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
