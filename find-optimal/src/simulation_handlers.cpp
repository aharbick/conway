#include "simulation_handlers.h"

#include <cstdlib>
#include <ctime>
#include <iostream>

#include "display_utils.h"
#include "gol.h"
#include "gol_core.h"

int handleSimulationMode(ProgramArgs* cli) {
  std::cout << "Interactive pattern simulation mode (grid mode: "
            << (cli->golGridMode == GOL_GRID_MODE_TORUS ? "torus" : "plane") << ")\n\n";

  while (true) {
    // Get pattern from user
    std::cout << "Please enter an integer pattern to simulate, 'r' for random, or 'q' to quit: ";
    std::string input;
    std::getline(std::cin, input);

    if (input == "q" || input == "Q") {
      std::cout << "Exiting simulation mode.\n";
      break;
    }

    // Handle random pattern request
    uint64_t currentPattern;
    if (input == "r" || input == "R") {
      // Generate a random 64-bit pattern
      static bool seeded = false;
      if (!seeded) {
        std::srand(std::time(nullptr));
        seeded = true;
      }

      // Generate random 64-bit number using two 32-bit randoms
      currentPattern = ((uint64_t)std::rand() << 32) | std::rand();
      std::cout << "Generated random pattern: " << currentPattern << "\n";
    } else {
      // Parse the pattern
      try {
        currentPattern = std::stoull(input);
      } catch (const std::exception&) {
        std::cout << "Invalid pattern. Please enter a valid 64-bit integer, 'r' for random, or 'q' to quit.\n\n";
        continue;
      }
    }

    std::cout << "Starting pattern: " << currentPattern << "\n\n";

    // Simulate the pattern
    int generation = 0;
    char stepInput;

    while (true) {
      std::cout << "Generation " << generation << ":\n";
      printPattern(currentPattern);

      std::cout << "Press ENTER/'n' to continue, 'q' to quit simulation: ";

      // Read a line to handle ENTER
      std::string stepInputStr;
      std::getline(std::cin, stepInputStr);

      // Check for quit
      if (!stepInputStr.empty() && (stepInputStr[0] == 'q' || stepInputStr[0] == 'Q')) {
        std::cout << "Stopping simulation.\n\n";
        break;
      }

      // ENTER (empty string) or 'n'/'N' means continue
      if (stepInputStr.empty() || stepInputStr[0] == 'n' || stepInputStr[0] == 'N') {
        uint64_t nextPattern = computeNextGeneration(currentPattern, cli->golGridMode);

        // Check if pattern died out
        if (nextPattern == 0) {
          generation++;
          std::cout << "\nGeneration " << generation << ":\n";
          printPattern(nextPattern);
          std::cout << "Pattern died out after " << generation << " generations.\n\n";
          break;
        }

        // Check if pattern stabilized
        if (nextPattern == currentPattern) {
          std::cout << "Pattern stabilized after " << generation << " generations.\n\n";
          break;
        }

        currentPattern = nextPattern;
        generation++;
        std::cout << "\n";
      }
    }
  }

  return 0;
}

int handleCompareCycleAlgorithms(ProgramArgs* cli) {
  compareCycleDetectionAlgorithms(cli, cli->compareFrameIdx);
  return 0;
}