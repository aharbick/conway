#ifndef _LOGGING_H_
#define _LOGGING_H_

#include <fstream>
#include <iostream>
#include <memory>

#include "cli_parser.h"

namespace Logging {
class LogManager {
 private:
  std::unique_ptr<std::ofstream> logFile;
  LogManager() = default;               // Private default constructor
  LogManager(const ProgramArgs* args);  // Private constructor with args

 public:
  static void initialize(const ProgramArgs* args);
  static LogManager& getInstance();
  std::ostream& out();  // Declaration only
  ~LogManager() = default;
};

// Convenience function declaration
std::ostream& out();
}  // namespace Logging

#endif