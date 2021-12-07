#ifndef CONWAY_H_
#define CONWAY_H_

#include <Adafruit_PWMServoDriver.h>

class Conway {
public:
  typedef enum Color {WHITE = 1, BLACK = 0, SPLIT = 2};
  typedef enum LogLevel {QUIET = 0, INFO = 1, DEBUG = 2};

  // Default constructor
  Conway();

  // Pattern which will be passed to initializePixels()
  Conway(char *pattern);

  // Initialize with a random pattern
  Conway(bool randomize);

  // Change all of the pixels to a given color
  void resetToColor(Color color);

  // Set to black, then column by column to white, then reverse back to black.
  void waveColumns();

  // Choose a random pattern...  Each pixel has a 20% change of being alive e.g. black
  void randomizePixels(uint8_t alivePct = 20);

  // Initialize the pixels to a pattern from a string of 64 ones and zeroes.
  // e.g. "0011000011010000110000001101000000110000000000000000000000000000"
  void initializePixels(const char *pattern);

  // Set a specific pixel to a specific color
  void changeServo(uint8_t shield, uint8_t servo, Color color);

  // Actively turn off all servos.  Otherwise they sit constantly under tension. This probably shortens
  // the servo life and also causes miscellaneous chirps and mechanical noises.
  void pauseAllServos();

  // Iterate the current set of pixels and apply the default rules of Conway's Game of Life:
  //   1. If a living cell is too isolated (0 or 1 neighbors) then it dies in the next generation.
  //   2. If a living cell has 2 or 3 neighbors then it remains alive in the next generation.
  //   3. If a living cell has 4 or more neighbors it dies in the next generation.
  //
  //  NOTE representationally 0 = dead (white pixel), 1 = alive (black pixel) and each pixel has a maximum of 8 neighbors.
  //
  // Returns the number of changes made (0 if it wasn't running and this was the first call to nextGeneration())
  int nextGeneration();

  // Find an initial pattern that makes the most changes but ends before 100 iterations (i.e. NOT infinite).
  uint8_t findFittestPattern(char *buf, int maxAttempts, uint8_t targetGenerations = 0);

  void setLogLevel(LogLevel level);
  LogLevel getLogLevel();

private:
  //  Log Level
  LogLevel logLevel = QUIET;
  char logMsg[128]; // One buffer to reduce local variable usage

  // Don't actually change servo state if set
  bool testMode = false;

  // Some constants
  static const int SERVOSTOPPED = 4096; // Turn off the servo...
  static const int SERVOMIN = 120; // This is the 'minimum' pulse length count (out of 4096)
  static const int SERVOHALF = 300;
  static const int SERVOMAX = 600; // This is the 'maximum' pulse length count (out of 4096)
  static const int SERVO_FREQ = 50; // Analog servos run at ~50 Hz updates

  // Our shields
  Adafruit_PWMServoDriver shields[4] = {
    Adafruit_PWMServoDriver(0x43),
    Adafruit_PWMServoDriver(0x42),
    Adafruit_PWMServoDriver(0x40),
    Adafruit_PWMServoDriver(0x41)
  };

  void initializeShields();
  void initializeServos();

  // Cell state
  static const uint8_t ROWS = 8;
  static const uint8_t COLS = 8;
  uint8_t pixels[8][8] = {
    {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0}
  };

  // Turn our pixel array into a string that can be used with initializePixels()
  void pixelsToPattern(char *buf);
};

// See https://rheingoldheavy.com/better-arduino-random-values/
void betterRandomSeed();

#endif
