#include "logging.h"

#include <fstream>
#include <iostream>
#include <memory>

// Private constructor implementation
Logger::Logger(const ProgramArgs* args) {
  if (args && !args->logFilePath.empty()) {
    logFile = std::make_unique<std::ofstream>(args->logFilePath, std::ios::app);
    if (!logFile || !logFile->is_open()) {
      std::cerr << "[ERROR] Could not open log file: " << args->logFilePath << std::endl;
      logFile.reset();  // Clear the failed file pointer
    } else {
      // Enable auto-flush for immediate output
      *logFile << std::unitbuf;
    }
  }
}

// Static shared instance
static std::unique_ptr<Logger> instance;

// Static initialize method
void Logger::initialize(const ProgramArgs* args) {
  instance.reset(new Logger(args));
}

// Static out method (combines getInstance and out)
Logger& Logger::out() {
  if (!instance) {
    // If not initialized, create default instance (console only)
    instance.reset(new Logger());
  }
  return *instance;
}