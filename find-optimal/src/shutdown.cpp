#include "shutdown.h"

#include <signal.h>
#include <unistd.h>

std::atomic<bool> gShutdownRequested{false};

namespace {

// Only async-signal-safe work here: set the flag and write a fixed string. Anything else -
// locks, streams, allocation - risks deadlocking against whatever the interrupted thread
// was holding.
void handleSignal(int) {
  if (gShutdownRequested.exchange(true)) {
    // Already shutting down and the user asked again: stop waiting on the queue drain,
    // which is the only thing that can take a while, and which needs the network.
    const char msg[] = "\nSecond signal: exiting now, queued uploads will be retried next run\n";
    ssize_t ignored = write(STDERR_FILENO, msg, sizeof(msg) - 1);
    (void)ignored;
    _exit(130);
  }

  const char msg[] =
      "\nShutting down: finishing the current middle block, then draining the upload queue."
      " Press Ctrl-C again to exit immediately.\n";
  ssize_t ignored = write(STDERR_FILENO, msg, sizeof(msg) - 1);
  (void)ignored;
}

}  // namespace

void installShutdownHandler() {
  struct sigaction sa;
  sa.sa_handler = handleSignal;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;  // no SA_RESTART: let interruptible waits return so we notice promptly

  sigaction(SIGINT, &sa, nullptr);
  sigaction(SIGTERM, &sa, nullptr);
}
