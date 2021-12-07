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

void Conway::resetToColor(Color color) {
  for(uint8_t i = 0; i < ROWS; i++) {
    for(uint8_t j = 0; j < COLS; j++) {
      this->pixels[i][j] = color;
    }
  }
  initializeServos();
}

void Conway::waveColumns() {
  // Set to black first
  resetToColor(BLACK);

  // Change columns one by one to WHITE
  for (uint8_t j = 0; j < COLS; j++) {
    for (uint8_t i = 0; i < ROWS; i++) {
      uint8_t shield = (i/4)*2+(j/4)+1;
      uint8_t servo = (i%4)*4+(j%4)+1;
      changeServo(shield, servo, WHITE);
      delay(5);
    }
    delay(100);
  }

  // Give it a bit before spinning immediately back
  delay(300);

  // Change columns one by one to BLACK in reverse
  for (uint8_t j = 0; j < COLS; j++) {
    // Reverse j
    uint8_t rj = (COLS-1)-j;
    for (uint8_t i = 0; i < ROWS; i++) {
      uint8_t shield = (i/4)*2+(rj/4)+1;
      uint8_t servo = (i%4)*4+(rj%4)+1;
      changeServo(shield, servo, BLACK);
      delay(5);
    }
    delay(100);
  }

  // Give it a bit before something else takes over
  delay(300);
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
void Conway::initializePixels(const char *pattern) {
  for(uint8_t i = 0; i < ROWS; i++) {
    for(uint8_t j = 0; j < COLS; j++) {
      uint8_t idx = (i*COLS)+j;
      this->pixels[i][j] = (idx+1 > strlen(pattern)) ? 0 : (uint8_t)pattern[idx] - 48;
    }
  }

  if (logLevel > QUIET) {
    Serial.println("Initialize pixels to:\n");
    for(uint8_t i = 0; i < ROWS; i++) {
      Serial.print("\t| ");
      for(uint8_t j = 0; j < COLS; j++) {
        Serial.print(this->pixels[i][j]); Serial.print(" | ");
      }
      Serial.print("\n");
    }
  }

  initializeServos();
}

int Conway::nextGeneration() {
  int changesMade = 0;

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

      //  Count our alive neighbors
      uint8_t aliveNeighbors = 0;
      for (uint8_t ioff = -1; ioff != 2; ioff++) {
        for (uint8_t joff = -1; joff != 2; joff++) {
          uint8_t ineighbor = i + ioff;
          uint8_t jneighbor = j + joff;
          if ((ineighbor >= 0 && ineighbor < ROWS) && // ineighbor is inbounds
              (jneighbor >= 0 && jneighbor < COLS) && // jneighbor is inbounds
              (ineighbor != i || jneighbor != j)       // NOT comparing to ourself
              ) {
            if (pixels[ineighbor][jneighbor] == 1) {
              aliveNeighbors++;
            }
          }
        }
      }

      if (pixels[i][j] == 1 && aliveNeighbors <= 1) {
        // DIE if we're alive and we have less 0 or 1 neighbors
        future[i][j] = 0;
      }
      else if (pixels[i][j] == 1 && aliveNeighbors >= 4) {
        // DIE if we're alive and we have 4 or more neighbors
        future[i][j] = 0;
      }
      else if (pixels[i][j] == 0 && aliveNeighbors == 3) {
        // BIRTH if we're dead and we have 3 aliveNeighbors
        future[i][j] = 1;
      }
      else {
        // Stay the same...
        future[i][j] = pixels[i][j];
      }
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

uint8_t Conway::findFittestPattern(char *bestPattern, int maxAttempts, uint8_t targetGenerations) {
  // Set ourself to testMode so that we don't actually change the servos.
  testMode = true;

  char testPattern[(ROWS*COLS)+1] = {'\0'};
  uint8_t mostGenerations = 0;
  for (int i = 0; i < maxAttempts; i++) {
    if (logLevel > QUIET && i % 100 == 0) {
      Serial.print(".");
    }

    // Make a random pattern of alive pixels between (20 and 80% alive)
    randomizePixels();
    pixelsToPattern(testPattern); // save the pattern

    if (logLevel == DEBUG) {
      sprintf(logMsg, "Trying pattern: %s", testPattern);
      Serial.println(logMsg);
    }

    int lastChanges = 0;
    uint8_t repeats = 0;
    bool finite = false;
    uint8_t generations = 0;
    while (true) {
      generations++;
      int changes = nextGeneration();
      if (lastChanges == changes) {
        repeats++;
      }
      else {
        repeats = 0;
        lastChanges = changes;
      }

      // Break out of the loop if we repeat 4 (or more) times.
      if (repeats == 4) {
        // It's a finite simulation.
        if (changes == 0) {
          finite = true;
          generations -= 4; // Subtract the repeats
        }
        break;
      }
    }

    if (finite && mostGenerations < generations) {
      if (logLevel > QUIET && targetGenerations > 0) {
        sprintf(logMsg, "\Best with %d generations\n%s", generations, testPattern);
        Serial.println(logMsg);
      }
      strcpy(bestPattern, testPattern);
      mostGenerations = generations;
    }

    if (targetGenerations > 0 && mostGenerations >= targetGenerations) {
      break;
    }
  }

  // Leave test mode
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
    shields[shield].setPWM(servo, 0, SERVOMAX);
  }
  else if (color == BLACK) {
    shields[shield].setPWM(servo, 0, SERVOMIN);
  }
  else if (color == SPLIT) {
    shields[shield].setPWM(servo, 0, SERVOHALF);
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

  for (uint8_t i = 0; i < ROWS; i++) {
    for (uint8_t j = 0; j < COLS; j++) {
      uint8_t shield = (i/4)*2+(j/4)+1;
      uint8_t servo = (i%4)*4+(j%4)+1;
      changeServo(shield, servo, (Color) pixels[i][j]);
      delay(5);
    }
  }

  pauseAllServos();
}

void Conway::pauseAllServos() {
  delay(2000); // Wait long enough for the servos to stop moving
  for (uint8_t shield = 0; shield < 4; shield++) {
    for (uint8_t servo = 0; servo < 16; servo++) {
      shields[shield].setPWM(servo, 0, SERVOSTOPPED);
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
