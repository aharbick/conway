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
#include "symmetry_utils.h"
#include "constants.h"

int handlePatternSimulation(ProgramArgs* cli) {
  bool is7x7 = (cli->gridSize == GRID_SIZE_7X7);
  uint64_t maxPattern = is7x7 ? ((1ULL << 49) - 1) : UINT64_MAX;

  std::cout << "Interactive pattern simulation mode (" << (is7x7 ? "7x7" : "8x8") << " grid)\n\n";

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
      // Generate a random pattern
      static bool seeded = false;
      if (!seeded) {
        std::srand(std::time(nullptr));
        seeded = true;
      }

      // Generate random 64-bit number using two 32-bit randoms
      currentPattern = ((uint64_t)std::rand() << 32) | std::rand();

      // Mask to appropriate size for 7x7
      if (is7x7) {
        currentPattern &= maxPattern;
      }

      std::cout << "Generated random pattern: " << currentPattern << "\n";
    } else {
      // Parse the pattern
      try {
        currentPattern = std::stoull(input);
        if (is7x7 && currentPattern > maxPattern) {
          std::cout << "Pattern exceeds 7x7 grid size (max: " << maxPattern << "). Please try again.\n\n";
          continue;
        }
      } catch (const std::exception&) {
        std::cout << "Invalid pattern. Please enter a valid integer, 'r' for random, or 'q' to quit.\n\n";
        continue;
      }
    }

    std::cout << "Starting pattern: " << currentPattern << "\n\n";

    // For 7x7: unpack at start, keep in unpacked format during simulation, display uses compact
    // For 8x8: already in correct format
    uint64_t simulationPattern = is7x7 ? unpack7x7(currentPattern) : currentPattern;
    uint64_t displayPattern = currentPattern;  // Keep compact version for display

    // Simulate the pattern
    int generation = 0;

    while (true) {
      std::cout << "Generation " << generation << ":\n";
      // Display compact format for 7x7, normal format for 8x8
      printPattern(displayPattern, is7x7 ? 7 : 8);

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
        // Use appropriate computation based on grid size
        // For 7x7: work on unpacked format, pack only for display
        uint64_t nextPattern = is7x7 ? computeNextGeneration7x7(simulationPattern) : computeNextGeneration8x8(simulationPattern);
        if (is7x7) {
          displayPattern = pack7x7(nextPattern);
        } else {
          displayPattern = nextPattern;
        }

        // Check if pattern died out
        if (nextPattern == 0) {
          generation++;
          std::cout << "\nGeneration " << generation << ":\n";
          printPattern(0, is7x7 ? 7 : 8);
          std::cout << "Pattern died out after " << generation << " generations.\n\n";
          break;
        }

        // Check if pattern stabilized
        if (nextPattern == simulationPattern) {
          std::cout << "Pattern stabilized after " << generation << " generations.\n\n";
          break;
        }

        simulationPattern = nextPattern;
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