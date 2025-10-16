#include "queue_handlers.h"

#include <chrono>
#include <iostream>
#include <thread>

#include "google_request_queue.h"
#include "logging.h"

int handleDrainRequestQueue(ProgramArgs* cli) {
  Logger::out() << "Draining request queue from: " << cli->queueDirectory << "\n";

  // Initialize the request queue
  if (!initGoogleRequestQueue(cli->queueDirectory)) {
    std::cerr << "[ERROR] Failed to initialize request queue\n";
    return 1;
  }

  size_t initialCount = getGoogleRequestQueueCount();
  if (initialCount == 0) {
    Logger::out() << "Request queue is empty. Nothing to process.\n";
    stopGoogleRequestQueue();
    return 0;
  }

  Logger::out() << "Found " << initialCount << " pending requests\n";
  Logger::out() << "Processing requests until queue is empty or only non-retriable entries remain...\n";

  size_t lastCount = initialCount;
  int stableIterations = 0;

  // Process requests until done
  while (true) {
    size_t currentCount = getGoogleRequestQueueCount();

    if (currentCount == 0) {
      Logger::out() << "All requests processed successfully!\n";
      break;
    }

    // If count hasn't changed for 10 seconds, assume remaining are non-retriable
    if (currentCount == lastCount) {
      stableIterations++;
      if (stableIterations >= 5) {  // 5 iterations * 2 seconds = 10 seconds
        Logger::out() << currentCount << " requests remain in queue (non-retriable or awaiting retry)\n";
        break;
      }
    } else {
      stableIterations = 0;
      Logger::out() << "Processing... " << currentCount << " requests remaining\n";
    }

    lastCount = currentCount;

    // Sleep briefly to allow background processing
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }

  stopGoogleRequestQueue();
  return 0;
}
