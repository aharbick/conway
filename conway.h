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

  // Run through a sequence of exercises to test all of the servos
  void burnIn();

  // Set to black, then column by column to white, then reverse back to black.
  void waveColumns();

  // Choose a random pattern...  Each pixel has a 20% change of being alive e.g. black
  void randomizePixels(uint8_t alivePct = 20);

  // Initialize the pixels to a pattern from a string of 64 ones and zeroes.
  // e.g. "0011000011010000110000001101000000110000000000000000000000000000"
  void initializePixels(const char *pattern);

  // Set a specific pixel to a specific color
  void changeServo(uint8_t shield, uint8_t servo, Color color);

  // Iterate the current set of pixels and apply the default rules of Conway's Game of Life:
  //   1. If a living cell is too isolated (0 or 1 neighbors) then it dies in the next generation.
  //   2. If a living cell has 2 or 3 neighbors then it remains alive in the next generation.
  //   3. If a living cell has 4 or more neighbors it dies in the next generation.
  //
  //  NOTE representationally 0 = dead (white pixel), 1 = alive (black pixel) and each pixel has a maximum of 8 neighbors.
  //
  // Returns the number of changes made (0 if it wasn't running and this was the first call to nextGeneration())
  uint8_t nextGeneration();

  // Return our current population
  uint16_t population();

  // Find an initial pattern that makes the most changes...
  uint16_t findFittestPattern(char *buf, uint16_t maxAttempts, uint16_t maxGenerations = 256);

  // Print out the pixels to Serial
  void printPixels(char *msg = NULL);

  void setLogLevel(LogLevel level);
  LogLevel getLogLevel();

private:
  //  Log Level
  LogLevel logLevel = QUIET;
  char logMsg[128]; // One buffer to reduce local variable usage

  // Don't actually change servo state if set
  bool testMode = false;

  // Some constants
  static const uint16_t SERVOMIN = 500;
  static const uint16_t SERVOHALF = 1400;
  static const uint16_t SERVOMAX = 2300;
  static const uint16_t SERVO_FREQ = 50; // Analog servos run at ~50 Hz updates

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
  uint8_t pixels[ROWS][COLS] = {
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
