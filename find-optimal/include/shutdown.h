#ifndef _SHUTDOWN_H_
#define _SHUTDOWN_H_

#include <atomic>

// Cooperative shutdown, so Ctrl-C does not strand progress.
//
// The search only records an interval as complete once it has been uploaded, and those
// uploads go through a disk-backed queue. Dying on the spot leaves whatever is queued to be
// retried by some later run, which is survivable but means the spreadsheet disagrees with
// what was actually searched until then - and the interval in flight is simply lost.
//
// So SIGINT and SIGTERM set a flag, the search finishes the middle block it is on and
// stops at an interval boundary, and the shutdown path drains the queue before exiting. A
// second signal gives up on that and exits immediately, for when the network is the reason
// you are pressing Ctrl-C in the first place.

// Set by the signal handler; polled by the search loops
extern std::atomic<bool> gShutdownRequested;

// Install handlers for SIGINT and SIGTERM. Safe to call once, early in main().
void installShutdownHandler();

inline bool shutdownRequested() {
  return gShutdownRequested.load(std::memory_order_relaxed);
}

#endif
