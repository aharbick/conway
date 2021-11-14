#include "menu_utils.h"
#include "conway.h"

Conway *simulation = NULL;

void setup() {
    Serial.begin(9600);
    betterRandomSeed();
    delay(1000);
    simulation = new Conway();
}

void loop() {
    Serial.println("\n----");
    Serial.println("1. Reset");
    Serial.println("2. Set pixel");
    Serial.println("3. Run Conway");
    Serial.println("4. Log level\n");

    uint8_t choice = awaitInteger("", 1, 4);
    switch(choice) {
    case 1: {
        Conway::Color color = (Conway::Color) awaitInteger("color", 0, 2);
        simulation->resetToColor(color);
        break;
    }
    case 2: {
        // Move pixel menu choice
        uint8_t quadrant = awaitInteger("quadrant", 1, 4);
        uint8_t pixel = awaitInteger("pixel", 1, 16);
        Conway::Color color = (Conway::Color) awaitInteger("color", 0, 2);
        simulation->changeServo(quadrant, pixel, color);
        break;
    }
    case 3: {
        bool simulate = true;
        while(simulate) {
            Serial.println("1. Enter pattern");
            Serial.println("2. Random pattern");
            Serial.println("3. Simulate");
            Serial.println("4. Find pattern");
            Serial.println("5. Run hourly");
            Serial.println("6. Exit");
            uint8_t cchoice = awaitInteger("", 1, 6);
            switch(cchoice) {
            case 1: {
                simulation->initializePixels(awaitString("pattern").c_str());
                break;
            }
            case 2: {
                simulation->randomizePixels();
                break;
            }
            case 3: {
                char msg[64];
                uint8_t numGenerations = awaitInteger("# generations", 1, 100);
                uint8_t i = 1;
                while (numGenerations > 0) {
                    int changes = simulation->nextGeneration();
                    if (simulation->getLogLevel() > Conway::QUIET) {
                        sprintf(msg, "Generation %d made %d changes", i, changes);
                        Serial.println(msg);
                    }
                    i++; numGenerations--;

                    if (changes == 0 && numGenerations > 0) {
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
            case 4: {
                char pattern[65] = {'\0'};
                uint8_t target = awaitInteger("# generations", 1, 100);
                simulation->findFittestPattern(pattern, 5000, target);
                break;
            }
            case 5: {
                char msg[32];
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
                    uint8_t generations = simulation->findFittestPattern(pattern, 1000, 24);
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
                simulate = false;
                break;
            }
            }

        }
        break;
    }
    case 4: {
        Conway::LogLevel logLevel = (Conway::LogLevel) awaitInteger("level", 0, 2);
        simulation->setLogLevel(logLevel);
        break;
    }
    default:
        Serial.println(String(choice) + " - invalid choice");
    }
}
