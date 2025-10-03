#ifndef _REQUEST_QUEUE_H_
#define _REQUEST_QUEUE_H_

#include <nlohmann/json.hpp>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "logging.h"

namespace fs = std::filesystem;

// Request handler type: takes payload JSON, returns success/failure
using RequestHandler = std::function<bool(const nlohmann::json&)>;

struct QueuedRequest {
  std::string apiName;
  uint64_t attemptCount;
  uint64_t firstQueuedAt;   // Unix timestamp (seconds)
  uint64_t nextRetryAt;      // Unix timestamp (seconds)
  std::string lastError;
  bool attentionNeeded;      // Set to true after exponential backoff phase
  nlohmann::json payload;

  // Convert to/from JSON
  nlohmann::json toJson() const {
    return nlohmann::json{
        {"apiName", apiName},
        {"attemptCount", attemptCount},
        {"firstQueuedAt", firstQueuedAt},
        {"nextRetryAt", nextRetryAt},
        {"lastError", lastError},
        {"attentionNeeded", attentionNeeded},
        {"payload", payload}
    };
  }

  static QueuedRequest fromJson(const nlohmann::json& j) {
    QueuedRequest req;
    req.apiName = j.value("apiName", "unknown");
    req.attemptCount = j.value("attemptCount", 0);
    req.firstQueuedAt = j.value("firstQueuedAt", 0);
    req.nextRetryAt = j.value("nextRetryAt", 0);
    req.lastError = j.value("lastError", "");
    req.attentionNeeded = j.value("attentionNeeded", false);
    req.payload = j.value("payload", nlohmann::json::object());
    return req;
  }
};

class RequestQueue {
 private:
  fs::path queueDir;
  std::atomic<bool> running;
  std::thread processorThread;
  std::mutex handlersMutex;
  std::map<std::string, RequestHandler> handlers;

  // Retry strategy:
  // Attempts 1-6: Exponential backoff 1, 2, 4, 8, 16, 32 seconds
  // Attempts 7+: Every 3600 seconds (1 hour)
  static constexpr uint64_t MAX_EXPONENTIAL_ATTEMPTS = 6;
  static constexpr uint64_t HOURLY_BACKOFF_SECONDS = 3600;

  uint64_t getCurrentTimestamp() const {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
  }

  uint64_t calculateNextRetryTime(uint64_t attemptCount) const {
    if (attemptCount <= MAX_EXPONENTIAL_ATTEMPTS) {
      // Exponential: 2^(attemptCount-1) = 1, 2, 4, 8, 16, 32
      uint64_t delaySec = 1ULL << (attemptCount - 1);
      return getCurrentTimestamp() + delaySec;
    } else {
      // Hourly retry
      return getCurrentTimestamp() + HOURLY_BACKOFF_SECONDS;
    }
  }

  std::string generateUUID() const {
    // Simple UUID-like string using timestamp + random
    auto now = std::chrono::high_resolution_clock::now();
    auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    std::stringstream ss;
    ss << std::hex << nanos << "-" << std::hex << (rand() % 0xFFFFFF);
    return ss.str();
  }

  fs::path createQueueFilePath(const QueuedRequest& req) const {
    // Format: <nextRetryAt>-<apiName>-<attention|ok>-<uuid>.json
    std::stringstream filename;
    filename << req.nextRetryAt << "-" << req.apiName << "-"
             << (req.attentionNeeded ? "attention" : "ok") << "-"
             << generateUUID() << ".json";
    return queueDir / filename.str();
  }

  bool saveRequest(const QueuedRequest& req, const fs::path& filepath) const {
    try {
      // Write to temp file first, then atomic rename
      fs::path tempPath = filepath;
      tempPath += ".tmp";

      std::ofstream ofs(tempPath);
      if (!ofs.is_open()) {
        std::cerr << "[ERROR] Failed to open temp file for writing: " << tempPath << "\n";
        return false;
      }

      ofs << req.toJson().dump(2);
      ofs.close();

      // Atomic rename
      fs::rename(tempPath, filepath);
      return true;
    } catch (const std::exception& e) {
      std::cerr << "[ERROR] Failed to save request: " << e.what() << "\n";
      return false;
    }
  }

  bool loadRequest(const fs::path& filepath, QueuedRequest& req) const {
    try {
      std::ifstream ifs(filepath);
      if (!ifs.is_open()) {
        return false;
      }

      nlohmann::json j;
      ifs >> j;
      req = QueuedRequest::fromJson(j);
      return true;
    } catch (const std::exception& e) {
      std::cerr << "[ERROR] Failed to load request from " << filepath << ": " << e.what() << "\n";
      return false;
    }
  }

  void processorLoop() {
    while (running) {
      try {
        processQueuedRequests();
      } catch (const std::exception& e) {
        std::cerr << "[ERROR] Queue processor exception: " << e.what() << "\n";
      }

      // Check every second
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }

  void processQueuedRequests() {
    if (!fs::exists(queueDir)) {
      return;
    }

    uint64_t now = getCurrentTimestamp();
    std::vector<fs::path> filesToProcess;

    // Collect files ready for processing
    for (const auto& entry : fs::directory_iterator(queueDir)) {
      if (entry.is_regular_file() && entry.path().extension() == ".json") {
        filesToProcess.push_back(entry.path());
      }
    }

    // Sort by filename (timestamp prefix ensures chronological order)
    std::sort(filesToProcess.begin(), filesToProcess.end());

    for (const auto& filepath : filesToProcess) {
      if (!running) break;

      QueuedRequest req;
      if (!loadRequest(filepath, req)) {
        continue;
      }

      // Check if it's time to retry
      if (req.nextRetryAt > now) {
        continue;
      }

      // Find handler for this API
      RequestHandler handler;
      {
        std::lock_guard<std::mutex> lock(handlersMutex);
        auto it = handlers.find(req.apiName);
        if (it == handlers.end()) {
          std::cerr << "[ERROR] No handler registered for API: " << req.apiName << "\n";
          // Delete orphaned request
          fs::remove(filepath);
          continue;
        }
        handler = it->second;
      }

      // Attempt to process request
      req.attemptCount++;
      bool success = false;

      try {
        success = handler(req.payload);
      } catch (const std::exception& e) {
        req.lastError = e.what();
        std::cerr << "[ERROR] Handler exception for " << req.apiName << ": " << e.what() << "\n";
      }

      if (success) {
        // Success! Remove from queue
        fs::remove(filepath);
      } else {
        // Failed, reschedule with backoff
        req.nextRetryAt = calculateNextRetryTime(req.attemptCount);

        // Mark as attention needed if we've exceeded exponential backoff attempts
        bool newlyNeedsAttention = false;
        if (req.attemptCount > MAX_EXPONENTIAL_ATTEMPTS && !req.attentionNeeded) {
          req.attentionNeeded = true;
          newlyNeedsAttention = true;
        }

        // Delete old file and save with new timestamp in filename
        fs::remove(filepath);
        fs::path newPath = createQueueFilePath(req);
        saveRequest(req, newPath);

        // Log error only when attention is newly needed
        if (newlyNeedsAttention) {
          std::cerr << "[ERROR] Request requires attention after " << req.attemptCount
                    << " attempts: " << req.apiName << "\n";
        }
      }
    }
  }

 public:
  RequestQueue() : running(false) {}

  ~RequestQueue() {
    stop();
  }

  bool initialize(const std::string& directory = "") {
    try {
      if (directory.empty()) {
        queueDir = fs::current_path() / "request-queue";
      } else {
        queueDir = fs::path(directory);
      }

      // Create directory if it doesn't exist
      if (!fs::exists(queueDir)) {
        fs::create_directories(queueDir);
      }

      return true;
    } catch (const std::exception& e) {
      std::cerr << "[ERROR] Failed to initialize queue directory: " << e.what() << "\n";
      return false;
    }
  }

  void registerHandler(const std::string& apiName, RequestHandler handler) {
    std::lock_guard<std::mutex> lock(handlersMutex);
    handlers[apiName] = handler;
  }

  void start() {
    if (queueDir.empty() || running) {
      return;
    }

    running = true;
    processorThread = std::thread(&RequestQueue::processorLoop, this);

    Logger::out() << "Request queue with directory \"" << queueDir << "\" started processing\n";
  }

  void stop() {
    if (!running) {
      return;
    }

    running = false;

    if (processorThread.joinable()) {
      processorThread.join();
    }
  }

  bool enqueue(const std::string& apiName, const nlohmann::json& payload) {
    if (queueDir.empty()) {
      std::cerr << "[ERROR] Queue not initialized, cannot enqueue request\n";
      return false;
    }

    QueuedRequest req;
    req.apiName = apiName;
    req.attemptCount = 0;
    req.firstQueuedAt = getCurrentTimestamp();
    req.nextRetryAt = getCurrentTimestamp();  // Try immediately
    req.lastError = "";
    req.attentionNeeded = false;
    req.payload = payload;

    fs::path filepath = createQueueFilePath(req);
    return saveRequest(req, filepath);
  }

  size_t getPendingCount() const {
    if (!fs::exists(queueDir)) {
      return 0;
    }

    size_t count = 0;
    for (const auto& entry : fs::directory_iterator(queueDir)) {
      if (entry.is_regular_file() && entry.path().extension() == ".json") {
        count++;
      }
    }
    return count;
  }

  bool isEnabled() const {
    return !queueDir.empty();
  }
};

// Global queue instance - using inline to ensure single instance across translation units
inline RequestQueue globalRequestQueue;

#endif
