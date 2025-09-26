#ifndef _LOGGING_H_
#define _LOGGING_H_

#include <fstream>
#include <iostream>
#include <memory>

#include "cli_parser.h"

class Logger {
 private:
  std::unique_ptr<std::ofstream> logFile;

  // Private constructors for singleton
  Logger() = default;
  Logger(const ProgramArgs* args);

 public:
  static void initialize(const ProgramArgs* args);
  static Logger& out();

  template<typename T>
  Logger& operator<<(const T& data) {
    std::cout << data;
    if (logFile && logFile->is_open()) {
      *logFile << data;
      logFile->flush();
    }
    return *this;
  }

  // Handle stream manipulators (endl, flush, etc.)
  Logger& operator<<(std::ostream& (*manip)(std::ostream&)) {
    manip(std::cout);
    if (logFile && logFile->is_open()) {
      manip(*logFile);
    }
    return *this;
  }

  ~Logger() = default;
};

#endif