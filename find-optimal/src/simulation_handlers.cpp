#include "simulation_handlers.h"

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <vector>
#include <set>
#include <algorithm>
#include <numeric>

#include "display_utils.h"
#include "gol.h"
#include "gol_core.h"
#include "frame_utils.h"
#include "constants.h"

int handlePatternSimulation(ProgramArgs* cli) {
  std::cout << "Interactive pattern simulation mode\n\n";

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
        uint64_t nextPattern = computeNextGeneration8x8(currentPattern);

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

int handleSymmetrySimulation(ProgramArgs* cli) {
  std::cout << "Interactive symmetry visualization mode\n\n";

  // Seed random number generator
  static bool seeded = false;
  if (!seeded) {
    std::srand(std::time(nullptr));
    seeded = true;
  }

  // Track all observed reductions for average calculation
  std::vector<int> allReductions;

  while (true) {
    std::cout << "Press ENTER to show random frame symmetries, or 'q' to quit: ";
    std::string input;
    std::getline(std::cin, input);

    if (input == "q" || input == "Q") {
      std::cout << "Exiting symmetry mode.\n";
      break;
    }

    // Generate random frame index (1 to FRAME_SEARCH_MAX_FRAMES)
    uint64_t frameIdx = 1 + (std::rand() % (FRAME_SEARCH_MAX_FRAMES - 1));
    uint64_t frame = spreadBitsToFrame(frameIdx);

    std::cout << "\n=== Frame " << frameIdx << " (bits: " << frameIdx << ") ===\n";
    std::cout << "Frame value: " << frame << "\n";
    std::cout << "Frame bits set: " << POPCOUNTLL(frame) << "\n\n";

    // Collect all unique transformations
    std::set<uint64_t> allTransformations;
    int originalBits = POPCOUNTLL(frame);

    // Add rotations
    uint64_t rot90 = rotate90(frame);
    uint64_t rot180 = rotate90(rot90);
    uint64_t rot270 = rotate90(rot180);

    allTransformations.insert(frame);
    allTransformations.insert(rot90);
    allTransformations.insert(rot180);
    allTransformations.insert(rot270);

    // Add reflections and their rotations
    uint64_t reflected = reflectHorizontal(frame);
    uint64_t refRot90 = rotate90(reflected);
    uint64_t refRot180 = rotate90(refRot90);
    uint64_t refRot270 = rotate90(refRot180);

    allTransformations.insert(reflected);
    allTransformations.insert(refRot90);
    allTransformations.insert(refRot180);
    allTransformations.insert(refRot270);

    // Show rotations side by side
    std::cout << "Rotations:\n";
    printFourPatternsSideBySide("Original:", frame, "90deg:", rot90, "180deg:", rot180, "270deg:", rot270);

    // Show reflections side by side
    std::cout << "Reflections:\n";
    printFourPatternsSideBySide("Horizontal:", reflected, "90deg:", refRot90, "180deg:", refRot180, "270deg:", refRot270);

    // Find minimum and show analysis
    uint64_t minimum = *std::min_element(allTransformations.begin(), allTransformations.end());
    std::vector<uint64_t> sortedUnique(allTransformations.begin(), allTransformations.end());

    std::cout << "Minimum:\n";
    std::cout << minimum << "\n\n";

    std::cout << "Unique:\n";
    for (uint64_t value : sortedUnique) {
      std::cout << value << "\n";
    }
    std::cout << "\n";

    int currentReduction = static_cast<int>(sortedUnique.size());
    allReductions.push_back(currentReduction);
    double average = std::accumulate(allReductions.begin(), allReductions.end(), 0.0) / allReductions.size();

    std::cout << "Observed reduction: " << currentReduction << " -> 1\n";
    std::cout << "Average reduction (n=" << allReductions.size() << "): " << std::fixed << std::setprecision(2) << average << " -> 1\n";

    std::cout << "\n" << std::string(50, '=') << "\n\n";
  }

  return 0;
}

int handleSimulationMode(ProgramArgs* cli) {
  if (cli->simulateType == SIMULATE_PATTERN) {
    return handlePatternSimulation(cli);
  } else if (cli->simulateType == SIMULATE_SYMMETRY) {
    return handleSymmetrySimulation(cli);
  } else {
    std::cerr << "[ERROR] Invalid simulate type, expected 'pattern' or 'symmetry'\n";
    return 1;
  }
}

int handleCompareCycleAlgorithms(ProgramArgs* cli) {
  compareCycleDetectionAlgorithms(cli, cli->compareFrameIdx);
  return 0;
}