#include "conway.h"

// Default constructor
Conway::Conway() {
    initializeShields();
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
        }
        delay(100);
    }

    // Give it a bit before spinning immediately back
    delay(250);

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
//  70 generations - 0001100110011001100110011001100111001101010000100010010101010001
//  70 generations - 1100100000000000000000000000000001011101000000110111000101101000
//  70 generations - 1010011000000000000000000000000000011101001000110111000100110000
//  70 generations - 1011100000000000000000000000001011000110010000100010110100111110
//  70 generations - 0110001011111000001000110110110111111111111111111111111111111111
//  71 generations - 1000000000000000000000000000000001000110110010010000110100101001
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
        shields[shield].setPWM(servo, 0, SERVOMIN);
        shields[shield].setPWM(servo, 0, SERVOMAX);
    }
    else if (color == BLACK) {
        shields[shield].setPWM(servo, 0, SERVOMAX);
        shields[shield].setPWM(servo, 0, SERVOMIN);
    }
    else if (color == SPLIT) {
        shields[shield].setPWM(servo, 0, SERVOMIN);
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
        shields[i].setPWMFreq(50);
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
