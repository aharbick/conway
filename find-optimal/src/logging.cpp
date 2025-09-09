#include "logging.h"

#include <fstream>
#include <iostream>
#include <memory>

namespace Logging {

// Private constructor implementation
LogManager::LogManager(const ProgramArgs* args) {
  if (args && !args->logFilePath.empty()) {
    logFile = std::make_unique<std::ofstream>(args->logFilePath, std::ios::app);
    if (!logFile || !logFile->is_open()) {
      std::cerr << "[ERROR] Could not open log file: " << args->logFilePath << std::endl;
      logFile.reset();  // Clear the failed file pointer
    }
  }
}

// Static shared instance
static std::unique_ptr<LogManager> instance;

// Static initialize method
void LogManager::initialize(const ProgramArgs* args) {
  instance.reset(new LogManager(args));
}

// Singleton getInstance implementation
LogManager& LogManager::getInstance() {
  if (!instance) {
    // If not initialized, create default instance (uses std::cout)
    instance.reset(new LogManager());
  }
  return *instance;
}

// out() method implementation
std::ostream& LogManager::out() {
  return (logFile && logFile->is_open()) ? *logFile : std::cout;
}

// Convenience function implementation
std::ostream& out() {
  return LogManager::getInstance().out();
}

}  // namespace Logging