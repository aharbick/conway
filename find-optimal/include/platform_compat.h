#ifndef _PLATFORM_COMPAT_H_
#define _PLATFORM_COMPAT_H_

/**
 * Platform compatibility macros and definitions
 *
 * This header centralizes all platform-specific code to provide a consistent
 * interface across Windows and Unix/Linux systems.
 */

#ifdef _WIN32
// Windows-specific includes and definitions
#include <windows.h>
#include <process.h>
#include <intrin.h>
#include <chrono>

// Population count (count set bits) - use MSVC intrinsic
#define POPCOUNTLL(x) __popcnt64(x)

// High-resolution time function
inline double getPlatformHighResTime() {
    static LARGE_INTEGER frequency;
    static bool initialized = false;
    if (!initialized) {
        QueryPerformanceFrequency(&frequency);
        initialized = true;
    }

    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
}

#else
// Unix/Linux includes and definitions
#include <pthread.h>
#include <unistd.h>
#include <time.h>

// Population count (count set bits) - use GCC builtin
#define POPCOUNTLL(x) __builtin_popcountll(x)

// High-resolution time function
inline double getPlatformHighResTime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1000000000.0;
}

#endif // _WIN32

#endif // _PLATFORM_COMPAT_H_