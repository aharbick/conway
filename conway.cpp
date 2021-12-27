#include "conway.h"

// Default constructor
Conway::Conway() {
  initializeShields();
  initializeServos();
}

// Initialize to the specified pattern
Conway::Conway(char *pattern) {
  initializeShields();
  initializePixels(pattern);
}

// Initialize to a random pattern
Conway::Conway(bool randomize) {
  initializeShields();
  randomizePixels();
}

void Conway::setLogLevel(LogLevel level) {
  this->logLevel = level;
}

Conway::LogLevel Conway::getLogLevel() {
  return this->logLevel;
}

void Conway::printPixels(char *msg) {
  if (msg) {
    Serial.println(msg);
  }

  for(uint8_t i = 0; i < ROWS; i++) {
    Serial.print("\t| ");
    for(uint8_t j = 0; j < COLS; j++) {
      Serial.print(this->pixels[i][j]); Serial.print(" | ");
    }
    Serial.print("\n");
  }
}

void Conway::resetToColor(Color color) {
  for (uint8_t i = 0; i < ROWS; i++) {
    for(uint8_t j = 0; j < COLS; j++) {
      this->pixels[i][j] = color;
    }
  }
  initializeServos();
}

void Conway::burnIn() {
  // Toggle between black and white 10 times
  for (uint8_t i = 0; i < 10; i++) {
    resetToColor((Color) i%2);
    delay(2000);
  }

  // Wave
  waveColumns();

  // Toggle between black and white 10 times
  for (uint8_t i = 0; i < 10; i++) {
    resetToColor((Color) i%2);
    delay(2000);
  }
}

void Conway::waveColumns() {
  // Set to black first
  resetToColor(BLACK);
  delay(2000);

  // Change columns one by one to WHITE
  for (uint8_t j = 0; j < COLS; j++) {
    for (uint8_t i = 0; i < ROWS; i++) {
      uint8_t shield = (i/4)*2+(j/4)+1;
      uint8_t servo = (i%4)*4+(j%4)+1;
      changeServo(shield, servo, WHITE);
    }
    delay(100);
  }

  // Give it a bit before spinning immediately back
  delay(500);

  // Change columns one by one to BLACK in reverse
  for (uint8_t j = 0; j < COLS; j++) {
    // Reverse j
    uint8_t rj = (COLS-1)-j;
    for (uint8_t i = 0; i < ROWS; i++) {
      uint8_t shield = (i/4)*2+(rj/4)+1;
      uint8_t servo = (i%4)*4+(rj%4)+1;
      changeServo(shield, servo, BLACK);
    }
    delay(100);
  }

  // Give it a bit before something else takes over
  delay(500);
}

void Conway::randomizePixels(uint8_t alivePct = 20) {
  for(uint8_t i = 0; i < ROWS; i++) {
    for(uint8_t j = 0; j < COLS; j++) {
      this->pixels[i][j] = (random(100) <= alivePct) ? WHITE : BLACK;
    }
  }
  initializeServos();
}

// Some interesting patterns (on 4x8)
//  50 generations - 01110100101101000100011111111010
//  52 generations - 00000011010111100000010000010000
//  53 generations - 01101000001110010110100101001010
//  55 generations - 01000110100010110110100111011000
//  57 generations - 11010101101100011101000010100111
//
// Some interesting patterns (on 8x8)
//  Cha cha        - 0000000000010100010010000010101111010100000100100010100000000000
//  Octagon        - 0001100000100100010000101000000110000001010000100010010000011000
//  D walker       - 0011000011010000110000001101000000110000000000000000000000000000
//  tic-tac-toe    - 0000000000100100010110100010010000100100010110100010010000000000
//  33 generations - 0001100000110001000000010000010000101000000101000100000000000010
//  52 generations - 0100000000000000011000000000000101000000001010001010001000000000
//  75 generations - 0101110000010010000011000001100010000010000000100001010000100001
//  77 generations - 1000001001110101000000101001000000100100010100100001100000000000
//  99 generations - 1000000100100000000000100001110010001010000000000101001101010000
// 111 generations - 1000000000000000000000000000000000000000000000000111111101001011
// 116 generations - 0010000000000000000000000000000000000000000000111110110101011000
// 125 generations - 0010000000000000000000000000000000000000000001110111011000011010
// 129 generations - 0010000000000000000000000000000000000000000011101110111110000101
// 131 generations - 0010000000000000000000000000000000000000000111101111101000001110
// 136 generations - 0010000000000000000000000000000000000000001011111100011101010000
// 144 generations - 0010000000000000000000000000000000000000010111100110101100100101
// 145 generations - 0011001010101000110001111101000100010100010100101010000011100011
// 147 generations - 0011010111000011011001000000000111001110101011100011111001110000
// 151 generations - 1100111010111100010011001001000110110000100010001110000000010101
// 166 generations - 0001001010010100110100111011001010011111101110100100101110111101
// 167 generations - 0011000101100100001110111010100101101110101000010101000101010001
// 185 generations - 1110010001101001110001100110011001101101001011101001001010010011
// 185 generations - 1100011000101101111110101000001100111010100100101101100001111100
// 186 generations - 0100001111111101011100111100001101101110100101110001010011011011
// 188 generations - 1101101000101101101001001001111110101001110101010000101001110011
// 190 generations - 0111110101010010111011101000011101001001101100110110101000001000
// 191 generations - 1110011101111011101000000110011001011011000001010010101001010110
// 192 generations - 1001001010001000000101110110100000001111010110011101000000101101
// 192 generations - 0000111010111011010100010001001101001100010001101100100010100011
// 193 generations - 1010110110011101001110101101100100001001010101010011011001000100
// 197 generations - 0000000100100101011100011110000110110010101101001010100111010011
// 199 generations - 0010110011000010101000000100010010101001011010010001111111001100
// 199 generations - 0011100100111010001001000101011011000001000110100001100110001010
// 200 generations - 1101111010000111110010001011101010001000001111010100100011000000
// 200 generations - 0101000101110001001111111100001000010101111101001010010000000101
// 200 generations - 0111100010110011010101100010000100010110110110010101000110010011
// 200 generations - 0111101111010001010001101000011111110010101001000110101110101110
// 201 generations - 0010001011110000010110101100100110010011100101000011011011101110
// 201 generations - 0110000000100010011110011100011101000100010000011101110100110000
// 202 generations - 1000110010110111010011110010101111101001000001011000110011010011
// 202 generations - 1001010101100001000110100010010111101010100010001111001110001000
// 202 generations - 1010100100100010111010010100001100111110010100100111001000000101
// 202 generations - 1010110100101011001000100010111111111010100001001100010001110010
// 202 generations - 0001000110010001010010010011110011101011001010110100010011100011
// 203 generations - 0010110001010011110101100001001110011000010101100001001101011010
// 204 generations - 0001111101011000111101110000001110110100100101011110000010011111
// 204 generations - 0000011101011100100000100010011010010000100101011011110001100011
// 204 generations - 0110111000101000010000000100010010010101000010101111100100010110
// 205 generations - 1010110000010101110010111011011000100011110100101000100010000010
// 205 generations - 1111110110000101001101101011010110010001101000010010110110101010
// 205 generations - 0011011100101100111101101010001001010011101000100010111010110011
// 205 generations - 0111110100110000100001111001110101110001010001110010101011011101
// 206 generations - 0010000000001000011011001101100000001110110111110000000101000000
// 206 generations - 0000011111001100110111001101100011000011010011011010011111000000
// 206 generations - 0111010000101001001111100011000001000010000000001010011010001101
// 207 generations - 1110000011010010101000000110111000001111011000001000011111011100
// 207 generations - 0100001101010100110101111011010101001001011101111101000101010110
// 207 generations - 0101011011011010000010010000111010001011101111010001000000000111
// 207 generations - 0110110010100000010001011100100111010011011101001101000001001111
// 207 generations - 0000001111000010001100100110011101000000101100110000001000101100
// 208 generations - 1100101100000110100010111100010010000110000101010010110011000010
// 208 generations - 1000100000011100010100100001100101000101001001101100110111010001

void Conway::initializePixels(const char *pattern) {
  for(uint8_t i = 0; i < ROWS; i++) {
    for(uint8_t j = 0; j < COLS; j++) {
      uint8_t idx = (i*COLS)+j;
      this->pixels[i][j] = (idx+1 > strlen(pattern)) ? 0 : (uint8_t)pattern[idx] - 48;
    }
  }

  if (logLevel > QUIET) {
    printPixels("Initialize pixels to:\n");
  }

  initializeServos();
}

uint8_t Conway::nextGeneration() {
  uint8_t changesMade = 0;
  uint8_t future[ROWS][COLS];
  for (uint8_t i = 0; i < ROWS; i++) {
    for (uint8_t j = 0; j < COLS; j++) {
      //  The i,j cell could have up to 8 neighbors...
      //      ---------------
      //      | NW | N | NE |
      //      ---------------
      //      | W  | ? | E  |
      //      -------------
      //      | SW | S | SE |
      //      -------------

      uint8_t nw = (i > 0 && j > 0) ? pixels[i-1][j-1] : 0;
      uint8_t n = (i > 0) ? pixels[i-1][j] : 0;
      uint8_t ne = (i > 0 && j+1 < COLS) ? pixels[i-1][j+1] : 0;
      uint8_t w = (j > 0) ? pixels[i][j-1] : 0;
      uint8_t e = (j+1 < COLS) ? pixels[i][j+1] : 0;
      uint8_t sw = (i+1 < ROWS && j > 0) ? pixels[i+1][j-1] : 0;
      uint8_t s = (i+1 < ROWS) ? pixels[i+1][j] : 0;
      uint8_t se = (i+1 < ROWS && j+1 < COLS) ? pixels[i+1][j+1] : 0;
      uint8_t aliveNeighbors = nw + n + ne + w + e + sw + s + se;

      // Set our new alive/dead state
      future[i][j] = (aliveNeighbors == 3 || (aliveNeighbors == 2 && pixels[i][j] == 1)) ? 1 : 0;
    }
  }

  // Copy the future into pixels and call changeServo() if it was different
  for (uint8_t i = 0; i < ROWS; i++) {
    for (uint8_t j = 0; j < COLS; j++) {
      if (pixels[i][j] != future[i][j]) {
        changesMade++;

        // Update the value
        pixels[i][j] = future[i][j];

        // Figure out the shield/servo and call changeServo()
        uint8_t shield = (i/4)*2+(j/4)+1;
        uint8_t servo = (i%4)*4+(j%4)+1;
        Color color = (Color) pixels[i][j];
        changeServo(shield, servo, color);
      }
    }
  }

  return changesMade;
}

uint16_t Conway::population() {
  uint16_t pop = 0;
  for (uint8_t i = 0; i < ROWS; i++) {
    for (uint8_t j = 0; j < COLS; j++) {
      pop += pixels[i][j];
    }
  }

  return pop;
}

uint16_t Conway::findFittestPattern(char *bestPattern, uint16_t maxAttempts, uint16_t maxGenerations) {
  testMode = true;
  char testPattern[(ROWS*COLS)+1] = {'\0'};
  uint16_t mostGenerations = 0;
  while (maxAttempts > 0) {
    randomizePixels();
    pixelsToPattern(testPattern); // save the pattern

    uint16_t generations = 0;
    for (generations = 0; generations < maxGenerations; generations++) {
      if (!nextGeneration()) { // no changes stop checking generations
        break;
      }
    }

    // We died out after observing the most generations so far.
    if (population() == 0 && generations > mostGenerations) {
      if (logLevel > QUIET) {
        sprintf(logMsg, "\Best with %d generations\n%s", generations, testPattern);
        Serial.println(logMsg);
      }
      strcpy(bestPattern, testPattern);
      mostGenerations = generations;
    }

    maxAttempts--;
  }
  testMode = false;

  return mostGenerations;
}

void Conway::changeServo(uint8_t shield, uint8_t servo, Color color) {
  if (testMode) {return;}

  if (shield <= 0 || shield > 4 || servo <= 0 || servo > 16) {
    sprintf(logMsg, "Invalid shield %d, servo %d!", shield, servo);
    Serial.println(logMsg);
    return;
  }

  if (logLevel == DEBUG) {
    sprintf(logMsg, "Changing shield %d, servo %d, to %s", shield, servo, color == WHITE ? "white" : "black");
    Serial.println(logMsg);
  }

  shield--; // shield and servo are zero-based
  servo--;

  if (color == WHITE) {
    shields[shield].writeMicroseconds(servo, SERVOMAX);
  }
  else if (color == BLACK) {
    shields[shield].writeMicroseconds(servo, SERVOMIN);
  }
  else if (color == SPLIT) {
    shields[shield].writeMicroseconds(servo, SERVOHALF);
  }
}

///////////////////////////////////////////////////////////////////////////
// Private functions

void Conway::initializeShields() {
  if (testMode) {return;}

  // Initialize our shields...
  // See this: https://github.com/adafruit/Adafruit-PWM-Servo-Driver-Library/blob/master/examples/servo/servo.ino
  for (uint8_t i = 0; i < 4; i++) {
    shields[i].begin();
    shields[i].setOscillatorFrequency(27000000);
    shields[i].setPWMFreq(SERVO_FREQ);
  }
}

/*
  The intended installation looks like an 8x8 grid with dead (0) and live (1) cells.

  ---------------------------------
  | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 |
  | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
  | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
  | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
  | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 |
  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  ---------------------------------

  But there are 4 shields controlling 16 servos and rather than having each
  sheild control 2 "rows" we have each shield control a 4x4 quadrant.  That way
  you could hang less of the piece or in staggered arrangements.  As such
  addressing the correct servo is tricker.

  So this is what we actually have...

  Shield 1                 Shield 2
  ---------------------   ---------------------
  |  1 |  2 |  3 |  4 |   |  1 |  2 |  3 |  4 |
  |  5 |  6 |  7 |  8 |   |  5 |  6 |  7 |  8 |
  |  9 | 10 | 11 | 12 |   |  9 | 10 | 11 | 12 |
  | 13 | 14 | 15 | 16 |   | 13 | 14 | 15 | 16 |
  ---------------------   ---------------------

  Shield 3                 Shield 4
  ---------------------   ---------------------
  |  1 |  2 |  3 |  4 |   |  1 |  2 |  3 |  4 |
  |  5 |  6 |  7 |  8 |   |  5 |  6 |  7 |  8 |
  |  9 | 10 | 11 | 12 |   |  9 | 10 | 11 | 12 |
  | 13 | 14 | 15 | 16 |   | 13 | 14 | 15 | 16 |
  ---------------------   ---------------------

  So given the 8x8 pixels grid, when we iterate using i (rows), j (columns) then the mapping to shield servo is:

  uint8_t shield = (i/4)*2+(j/4)+1;
  uint8_t servo = (i%4)*4+(j%4)+1;

*/

void Conway::initializeServos() {
  if (testMode) {return;}

  // COLS then ROWS here because it looks the same as waveColumns
  for (uint8_t j = 0; j < COLS; j++) {
    for (uint8_t i = 0; i < ROWS; i++) {
      uint8_t shield = (i/4)*2+(j/4)+1;
      uint8_t servo = (i%4)*4+(j%4)+1;
      changeServo(shield, servo, (Color) pixels[i][j]);
      delay(10);
    }
  }
}

void Conway::pixelsToPattern(char *buf) {
  for (uint8_t i = 0; i < ROWS; i++) {
    for (uint8_t j = 0; j < COLS; j++) {
      buf[(i*COLS)+j] = pixels[i][j] == 0 ? '0' : '1';
    }
  }
}

// See https://rheingoldheavy.com/better-arduino-random-values/
void betterRandomSeed() {
  uint8_t  seedBitValue  = 0;
  uint8_t  seedByteValue = 0;
  uint32_t seedWordValue = 0;

  for (uint8_t wordShift = 0; wordShift < 4; wordShift++) {
    for (uint8_t byteShift = 0; byteShift < 8; byteShift++) {
      for (uint8_t bitSum = 0; bitSum <= 8; bitSum++) {
        seedBitValue = seedBitValue + (analogRead(A0) & 0x01);
      }
      delay(1);
      seedByteValue = seedByteValue | ((seedBitValue & 0x01) << byteShift);
      seedBitValue = 0;
    }
    seedWordValue = seedWordValue | (uint32_t)seedByteValue << (8 * wordShift);
    seedByteValue = 0;
  }

  randomSeed(seedWordValue);
}
