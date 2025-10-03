#ifndef _GOOGLE_REQUEST_QUEUE_H_
#define _GOOGLE_REQUEST_QUEUE_H_

#include "google_client.h"
#include "request_queue.h"

// Initialize the request queue and register handlers
static bool initGoogleRequestQueue(const std::string& queueDirectory = "") {
  if (!globalRequestQueue.initialize(queueDirectory)) {
    return false;
  }

  // Register handler for sendProgress
  globalRequestQueue.registerHandler("sendProgress", [](const nlohmann::json& payload) -> bool {
    try {
      uint64_t frameIdx = payload.at("frameIdx").get<uint64_t>();
      int kernelIdx = payload.at("kernelIdx").get<int>();
      int bestGenerations = payload.at("bestGenerations").get<int>();
      uint64_t bestPattern = payload.at("bestPattern").get<uint64_t>();
      std::string bestPatternBin = payload.at("bestPatternBin").get<std::string>();

      return sendGoogleProgress(frameIdx, kernelIdx, bestGenerations, bestPattern, bestPatternBin.c_str());
    } catch (const std::exception& e) {
      std::cerr << "[ERROR] sendProgress handler failed: " << e.what() << "\n";
      return false;
    }
  });

  // Register handler for sendSummaryData
  globalRequestQueue.registerHandler("sendSummaryData", [](const nlohmann::json& payload) -> bool {
    try {
      int bestGenerations = payload.at("bestGenerations").get<int>();
      uint64_t bestPattern = payload.at("bestPattern").get<uint64_t>();
      std::string bestPatternBin = payload.at("bestPatternBin").get<std::string>();
      uint64_t completedFrameIdx = payload.value("completedFrameIdx", UINT64_MAX);

      return sendGoogleSummaryData(bestGenerations, bestPattern, bestPatternBin.c_str(), completedFrameIdx);
    } catch (const std::exception& e) {
      std::cerr << "[ERROR] sendSummaryData handler failed: " << e.what() << "\n";
      return false;
    }
  });

  globalRequestQueue.start();
  return true;
}

// Stop the request queue processor
static void stopGoogleRequestQueue() {
  globalRequestQueue.stop();
}

// Queue-based async version of sendGoogleProgress
static void queueGoogleProgress(uint64_t frameIdx, int kernelIdx, int bestGenerations, uint64_t bestPattern,
                                 const char* bestPatternBin) {
  nlohmann::json payload = {
      {"frameIdx", frameIdx},
      {"kernelIdx", kernelIdx},
      {"bestGenerations", bestGenerations},
      {"bestPattern", bestPattern},
      {"bestPatternBin", std::string(bestPatternBin ? bestPatternBin : "")}
  };

  if (!globalRequestQueue.enqueue("sendProgress", payload)) {
    std::cerr << "[ERROR] Failed to enqueue sendProgress request\n";
  }

  // Update cache if frame is complete
  if (kernelIdx == 15) {
    frameCache.setFrameComplete(frameIdx);
  }
}

// Queue-based async version of sendGoogleSummaryData
static void queueGoogleSummaryData(int bestGenerations, uint64_t bestPattern, const char* bestPatternBin,
                                   uint64_t completedFrameIdx = UINT64_MAX) {
  nlohmann::json payload = {
      {"bestGenerations", bestGenerations},
      {"bestPattern", bestPattern},
      {"bestPatternBin", std::string(bestPatternBin ? bestPatternBin : "")}
  };

  if (completedFrameIdx != UINT64_MAX) {
    payload["completedFrameIdx"] = completedFrameIdx;
  }

  if (!globalRequestQueue.enqueue("sendSummaryData", payload)) {
    std::cerr << "[ERROR] Failed to enqueue sendSummaryData request\n";
  }
}

// Get count of pending queued requests
static size_t getGoogleRequestQueueCount() {
  return globalRequestQueue.getPendingCount();
}

// Check if queue is enabled
static bool isGoogleRequestQueueEnabled() {
  return globalRequestQueue.isEnabled();
}

#endif
