#include "menu_utils.h"
#include "conway.h"

#include <Adafruit_RGBLCDShield.h>

// Shared globals
Conway *simulation = NULL;
char msg[64];

///////////////////////////////////////////////////////////////////////////
// There's no easy way to reconcile two different systems of control (Serial
// monitor and UP/DOWN/LEFT/RIGHT buttons) so we just set a flag and use
// different code in PRODUCTION mode.
#define PRODUCTION

#ifdef PRODUCTION
// In PRODUCTION we run with LCD menuing/control and have fewer options
Adafruit_RGBLCDShield lcd = Adafruit_RGBLCDShield();
int16_t gLongestLife = 0;
bool gNeedPrompt = false;

unsigned long roundDown(unsigned long n, unsigned long to) {
  return n - n % to;
}

void showIntro() {
  static const char* introMessages[] = {
    "Welcome to the",
    "Game of Life!",

    NULL, // Don't clear display
    NULL, // but wait 1 second.

    "Life follows 3",
    "simple rules to",

    "evolve this 8x8",
    "grid.",

    "On this 8x8 grid",
    "there are...",

    "18,446,744,073,",
    "709,551,616",

    "initial starting",
    "grids.",

    "We want to find",
    "which lives the",

    "longest before",
    "dying out.",

    "Push any button",
    "to search...",

  };

  for (uint8_t i = 0; i < sizeof(introMessages)/sizeof(introMessages[0]); i++) {
    if (introMessages[i]) {
      if (i % 2 == 0) { lcd.clear(); }
      lcd.setCursor(0, i % 2);
      lcd.print(introMessages[i]);
    }
    if (i % 2 == 1) { delay(4000); }
  }
}

void playLife() {
  // Show our searching message
  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("   Searching");
  lcd.setCursor(0,1);
  lcd.print("     ....");

  // Find a long life...
  simulation->waveColumns();
  delay(4000);
  char pattern[65] = {'\0'};
  int16_t years = simulation->findFittestPattern(pattern, 500);
  simulation->initializePixels(pattern);
  delay(4000);

  // Play the simulation...
  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("  Simulating... ");
  for (int16_t i = 1; i <= years; i++) {
    int changes = simulation->nextGeneration();
    sprintf(msg, "     Year %d", i);
    lcd.setCursor(0,1);
    lcd.print(msg);
    delay(500);
  }

  // Show the results to LCD and save
  if (years > gLongestLife) {
    gLongestLife = years;
    lcd.clear();
    lcd.setCursor(0,0);
    lcd.print("New long life!!!");
    sprintf(msg, "   %d years", gLongestLife);
    lcd.setCursor(0,1);
    lcd.print(msg);
  }
  else {
    lcd.clear();
    lcd.setCursor(0,0);
    lcd.print("  Try again...");
    sprintf(msg, " Only %d years", years);
    lcd.setCursor(0,1);
    lcd.print(msg);
  }
  delay(4000);
}

void showPrompt() {
  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("Push any button");
  lcd.setCursor(0,1);
  lcd.print("to search...");
}

void showLongest() {
  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("  Longest life");
  sprintf(msg, "   %d years", gLongestLife);
  lcd.setCursor(0,1);
  lcd.print(msg);
}

void setup() {
  Serial.begin(9600);
  betterRandomSeed();
  simulation = new Conway();

  // set up the LCD
  lcd.begin(16, 2);

  // Show the initial greeting
  showIntro();
}

void loop() {
  uint8_t buttons = lcd.readButtons();
  if (buttons) {
    playLife();
    gNeedPrompt = true;
  }
  else {
    // Every 10s without using delay() which messes up our ability to readButtons()
    unsigned long now = roundDown(millis(), 10);
    if (now % 10000 == 0) {
      if (gNeedPrompt) {
        showPrompt();
        gNeedPrompt = false;
      }
      else if (gLongestLife > 0) {
        showLongest();
        gNeedPrompt = true;
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////
#else
// Not PRODUCTION... Use Serial menuing/control
void setup() {
  Serial.begin(9600);
  betterRandomSeed();
  delay(1000);
  simulation = new Conway();
}

void loop() {
  Serial.println("\n----");
  Serial.println("1. Settings");
  Serial.println("2. Run Conway");

  uint8_t choice = awaitInteger("", 1, 4);
  switch(choice) {
  case 1: {
    bool settings = true;
    while (settings) {
      Serial.println("\n----");
      Serial.println("1. Reset to color");
      Serial.println("2. Burn in");
      Serial.println("3. Set pixel");
      Serial.println("4. Set log level");
      Serial.println("5. Back");
      uint8_t choice = awaitInteger("", 1, 5);
      switch(choice) {
      case 1: {
        Conway::Color color = (Conway::Color) awaitInteger("color", 0, 2);
        simulation->resetToColor(color);
        break;
      }
      case 2: {
        simulation->burnIn();
        break;
      }
      case 3: {
        // Set pixel menu choice
        uint8_t quadrant = awaitInteger("quadrant", 1, 4);
        uint8_t pixel = awaitInteger("pixel", 1, 16);
        Conway::Color color = (Conway::Color) awaitInteger("color", 0, 2);
        simulation->changeServo(quadrant, pixel, color);
        break;
      }
      case 4: {
        Conway::LogLevel logLevel = (Conway::LogLevel) awaitInteger("level", 0, 2);
        simulation->setLogLevel(logLevel);
        break;
      }
      case 5: {
        settings = false;
        break;
      }
      default: {
        Serial.println(String(choice) + " - invalid choice");
      }
      }
    }
    break;
  }
  case 2: {
    bool run = true;
    while(run) {
      Serial.println("\n----");
      Serial.println("1. Enter pattern");
      Serial.println("2. Random pattern");
      Serial.println("3. Run best known");
      Serial.println("4. Simulate");
      Serial.println("5. Run hourly");
      Serial.println("6. Exit");
      uint8_t choice = awaitInteger("", 1, 6);
      switch(choice) {
      case 1: {
        simulation->initializePixels(awaitString("pattern").c_str());
        break;
      }
      case 2: {
        simulation->randomizePixels();
        break;
      }
      case 3: {
        simulation->waveColumns();
        simulation->initializePixels("0011100100111010001001000101011011000001000110100001100110001010");
        delay(2000);
        for (int16_t i = 1; i <= 199; i++) {
          int changes = simulation->nextGeneration();
          if (simulation->getLogLevel() > Conway::QUIET) {
            sprintf(msg, "Generation %d made %d changes", i, changes);
            Serial.println(msg);
          }
          delay(500);
        }
        delay(2000);
        simulation->resetToColor(Conway::BLACK);
        break;
      }
      case 4: {
        int16_t numGenerations = awaitInteger("# generations", 1, 500);
        for (int16_t i = 1; i <= numGenerations; i++) {
          int changes = simulation->nextGeneration();
          if (simulation->getLogLevel() > Conway::QUIET) {
            sprintf(msg, "Generation %d made %d changes", i, changes);
            Serial.println(msg);
          }

          if (changes == 0) {
            if (simulation->getLogLevel() > Conway::QUIET) {
              Serial.println("Randomizing!");
            }
            delay(5000);
            simulation->randomizePixels();
          }
          delay(500);
        }
        break;
      }
      case 5: {
        unsigned long nextHourMillis = (unsigned long) awaitInteger("seconds", 1, 3600) * 1000;
        while(true) {
          unsigned long startTime = millis();

          // Wave the columns left and then right and then wait 3 seconds.
          simulation->waveColumns();

          // Find a simulation that has at least 20 generations
          char pattern[65] = {'\0'};
          if (simulation->getLogLevel() > Conway::QUIET) {
            Serial.println("Finding pattern");
          }
          int16_t generations = simulation->findFittestPattern(pattern, 1000);
          if (simulation->getLogLevel() > Conway::QUIET) {
            sprintf(msg, "Initializing %d generations", generations);
            Serial.println(msg);
          }
          simulation->initializePixels(pattern);
          if (simulation->getLogLevel() > Conway::QUIET) {
            Serial.println("Sleeping 4s");
          }
          delay(4000);

          // Loop until it stops, sleep an hour and then repeat.
          int changes = 0;
          uint8_t iterations = 0;
          do {
            changes = simulation->nextGeneration();
            iterations++;
            delay(500);
          } while (changes > 0 && iterations < 50);

          //  Wait until the top of the our to run again.
          unsigned long delayMillis = nextHourMillis - (millis() - startTime);
          int delaySeconds = (int) (delayMillis/1000);

          if (simulation->getLogLevel() > Conway::QUIET) {
            sprintf(msg, "Sleeping %ds", delaySeconds);
            Serial.println(msg);
          }

          delay(delayMillis);
          nextHourMillis = 3600000;
        }
        break;
      }
      case 6: {
        if (simulation->getLogLevel() > Conway::QUIET) {
          Serial.println("Done");
        }
        run = false;
        break;
      }
      }

    }
    break;
  }
  default:
    Serial.println(String(choice) + " - invalid choice");
  }
}
#endif
