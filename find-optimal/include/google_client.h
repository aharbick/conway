#ifndef _GOOGLE_CLIENT_H_
#define _GOOGLE_CLIENT_H_

#include <curl/curl.h>
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cstdint>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "logging.h"

#define CURL_TIMEOUT 30L
#define FRAME_CACHE_TOTAL_FRAMES 2102800ULL
#define FRAME_CACHE_BITMAP_BYTES ((FRAME_CACHE_TOTAL_FRAMES + 7) / 8)  // 262,850 bytes

class CurlResponse {
 public:
  std::string data;

  const char* memory() const {
    return data.c_str();
  }
  size_t size() const {
    return data.size();
  }
  void clear() {
    data.clear();
  }
};

typedef struct {
  const char* webappUrl;
  const char* apiKey;
} GoogleConfig;

typedef enum {
  GOOGLE_SUCCESS,
  GOOGLE_ERROR_CONFIG,
  GOOGLE_ERROR_CURL_INIT,
  GOOGLE_ERROR_CURL_PERFORM,
  GOOGLE_ERROR_HTTP
} GoogleResult;

static size_t WriteMemoryCallback(void* contents, size_t size, size_t nmemb, CurlResponse* response) {
  size_t realsize = size * nmemb;
  try {
    response->data.append(static_cast<char*>(contents), realsize);
    return realsize;
  } catch (const std::bad_alloc&) {
    std::cerr << "[ERROR] Not enough memory (std::string append failed)\n";
    return 0;
  }
}

static GoogleResult googleGetConfig(GoogleConfig* config) {
  config->webappUrl = getenv("GOOGLE_WEBAPP_URL");
  config->apiKey = getenv("GOOGLE_API_KEY");

  if (!config->webappUrl || !config->apiKey) {
    return GOOGLE_ERROR_CONFIG;
  }

  return GOOGLE_SUCCESS;
}

static void googleCleanupResponse(CurlResponse* response) {
  response->clear();
}

static std::string googleUrlEncode(const std::string& value) {
  std::ostringstream encoded;
  for (char c : value) {
    if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
      encoded << c;
    } else {
      encoded << '%' << std::hex << std::uppercase << (unsigned char)c;
    }
  }
  return encoded.str();
}

static GoogleResult googleHttpRequest(const std::string& baseUrl, const std::map<std::string, std::string>& params,
                                      CurlResponse* response, const std::string& apiName = "unknown") {
  CURL* curl = curl_easy_init();
  if (!curl) {
    std::cerr << "[ERROR] Failed to initialize curl\n";
    return GOOGLE_ERROR_CURL_INIT;
  }

  // Build URL with parameters
  std::ostringstream urlStream;
  urlStream << baseUrl;

  if (!params.empty()) {
    urlStream << "?";
    bool first = true;
    for (const auto& param : params) {
      if (!first)
        urlStream << "&";
      urlStream << param.first << "=" << googleUrlEncode(param.second);
      first = false;
    }
  }

  std::string url = urlStream.str();
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);  // Follow redirects
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void*)response);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, CURL_TIMEOUT);

  CURLcode res = curl_easy_perform(curl);
  GoogleResult result = GOOGLE_SUCCESS;

  if (res != CURLE_OK) {
    std::cerr << "[ERROR] curl_easy_perform() failed: " << curl_easy_strerror(res) << "\n";
    result = GOOGLE_ERROR_CURL_PERFORM;
  } else {
    long responseCode;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &responseCode);

    if (responseCode != 200) {
      // Build comma-separated params for logging
      std::ostringstream paramStream;
      bool first = true;
      for (const auto& param : params) {
        if (!first)
          paramStream << ",";
        paramStream << param.first << "=" << param.second;
        first = false;
      }

      std::cerr << "[ERROR] " << apiName << ", code=" << responseCode;
      if (!params.empty()) {
        std::cerr << ", " << paramStream.str();
      }
      std::cerr << "\n";
      result = GOOGLE_ERROR_HTTP;
    }
  }

  curl_easy_cleanup(curl);
  return result;
}

// Frame completion cache for fast lookups
class FrameCompletionCache {
 private:
  uint8_t* bitmap;
  bool loaded;
  uint64_t completedCount;

 public:
  FrameCompletionCache() : bitmap(nullptr), loaded(false), completedCount(0) {
    bitmap = new uint8_t[FRAME_CACHE_BITMAP_BYTES]();  // Initialize to zero
  }

  ~FrameCompletionCache() {
    delete[] bitmap;
  }

  // Load cache from Google Sheets API
  bool loadFromAPI() {
    GoogleConfig config;
    if (googleGetConfig(&config) != GOOGLE_SUCCESS) {
      return false;
    }

    // Build parameters map
    std::map<std::string, std::string> params;
    params["action"] = "getCompleteFrameCache";
    params["apiKey"] = config.apiKey;

    CurlResponse response;
    GoogleResult result = googleHttpRequest(config.webappUrl, params, &response, "getCompleteFrameCache");

    if (result != GOOGLE_SUCCESS) {
      return false;
    }

    if (response.data.empty()) {
      return false;
    }

    // Parse JSON response to get base64 bitmap
    std::string bitmapBase64;
    try {
      nlohmann::json j = nlohmann::json::parse(response.data);
      if (j.contains("bitmap") && j["bitmap"].is_string()) {
        bitmapBase64 = j["bitmap"].get<std::string>();
      } else {
        googleCleanupResponse(&response);
        return false;
      }
    } catch (const nlohmann::json::exception& e) {
      std::cerr << "[ERROR] Failed to parse frame cache JSON: " << e.what() << "\n";
      googleCleanupResponse(&response);
      return false;
    }

    googleCleanupResponse(&response);

    // Decode base64 to bitmap
    if (!decodeBase64(bitmapBase64)) {
      return false;
    }

    // Count completed frames during load
    completedCount = 0;
    for (uint64_t i = 0; i < FRAME_CACHE_TOTAL_FRAMES; i++) {
      const uint64_t byteIdx = i / 8;
      const uint8_t bitIdx = i % 8;
      if (bitmap[byteIdx] & (1 << bitIdx)) {
        completedCount++;
      }
    }

    loaded = true;
    return true;
  }

  // Check if a frame is complete
  bool isFrameComplete(uint64_t frameIdx) const {
    if (!loaded || frameIdx >= FRAME_CACHE_TOTAL_FRAMES) {
      return false;
    }
    const uint64_t byteIdx = frameIdx / 8;
    const uint8_t bitIdx = frameIdx % 8;
    return (bitmap[byteIdx] & (1 << bitIdx)) != 0;
  }

  // Mark a frame as complete
  void setFrameComplete(uint64_t frameIdx) {
    if (frameIdx >= FRAME_CACHE_TOTAL_FRAMES) {
      return;
    }
    const uint64_t byteIdx = frameIdx / 8;
    const uint8_t bitIdx = frameIdx % 8;

    // Only increment count if frame wasn't already complete
    if (!(bitmap[byteIdx] & (1 << bitIdx))) {
      completedCount++;
    }

    bitmap[byteIdx] |= (1 << bitIdx);
  }

  bool isLoaded() const {
    return loaded;
  }

  uint64_t getCompletedCount() const {
    return completedCount;
  }

 private:
  // Base64 decoder using OpenSSL BIO
  bool decodeBase64(const std::string& input) {
    if (input.empty())
      return true;  // Empty input is valid

    // Create BIO chain: base64 decoder -> memory buffer
    BIO* bio = BIO_new_mem_buf(input.c_str(), input.length());
    BIO* b64 = BIO_new(BIO_f_base64());
    BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL);  // No newlines
    bio = BIO_push(b64, bio);

    // Read decoded data
    int decoded_len = BIO_read(bio, bitmap, FRAME_CACHE_BITMAP_BYTES);

    BIO_free_all(bio);  // Cleans up the entire chain

    return decoded_len > 0 && decoded_len <= FRAME_CACHE_BITMAP_BYTES;
  }
};

// Global cache instance
static FrameCompletionCache frameCache;

static bool googleSendProgress(uint64_t frameIdx, int kernelIdx, int bestGenerations, uint64_t bestPattern,
                               const char* bestPatternBin) {
  GoogleConfig config;
  if (googleGetConfig(&config) != GOOGLE_SUCCESS) {
    return false;
  }

  // Build parameters map
  std::map<std::string, std::string> params;
  params["action"] = "sendProgress";
  params["apiKey"] = config.apiKey;
  params["frameIdx"] = std::to_string(frameIdx);
  params["kernelIdx"] = std::to_string(kernelIdx);
  params["bestGenerations"] = std::to_string(bestGenerations);
  params["bestPattern"] = std::string(bestPatternBin) + ":" + std::to_string(bestPattern);

  CurlResponse response;
  GoogleResult result = googleHttpRequest(config.webappUrl, params, &response, "sendProgress");

  googleCleanupResponse(&response);

  // Update cache if frame is complete
  if (result == GOOGLE_SUCCESS && kernelIdx == 15) {
    frameCache.setFrameComplete(frameIdx);
  }

  return (result == GOOGLE_SUCCESS);
}

static std::string googleSendProgressWithResponse(uint64_t frameIdx, int kernelIdx, int bestGenerations,
                                                  uint64_t bestPattern, const char* bestPatternBin) {
  GoogleConfig config;
  if (googleGetConfig(&config) != GOOGLE_SUCCESS) {
    return "{\"error\": \"Configuration failed\"}";
  }

  // Build parameters map
  std::map<std::string, std::string> params;
  params["action"] = "sendProgress";
  params["apiKey"] = config.apiKey;
  params["frameIdx"] = std::to_string(frameIdx);
  params["kernelIdx"] = std::to_string(kernelIdx);
  params["bestGenerations"] = std::to_string(bestGenerations);
  params["bestPattern"] = std::string(bestPatternBin) + ":" + std::to_string(bestPattern);

  CurlResponse response;
  GoogleResult result = googleHttpRequest(config.webappUrl, params, &response, "sendProgressWithResponse");

  std::string responseStr = response.data;
  googleCleanupResponse(&response);

  if (result != GOOGLE_SUCCESS) {
    return "{\"error\": \"HTTP request failed\"}";
  }

  return responseStr;
}

static bool googleInit() {
  CURLcode res = curl_global_init(CURL_GLOBAL_DEFAULT);
  if (res != CURLE_OK) {
    std::cerr << "[ERROR] Failed to initialize libcurl: " << curl_easy_strerror(res) << "\n";
    return false;
  }
  return true;
}

static int googleGetBestResult() {
  GoogleConfig config;
  if (googleGetConfig(&config) != GOOGLE_SUCCESS) {
    return -1;
  }

  // Build parameters map
  std::map<std::string, std::string> params;
  params["action"] = "getBestResult";
  params["apiKey"] = config.apiKey;

  CurlResponse response;
  GoogleResult result = googleHttpRequest(config.webappUrl, params, &response, "getBestResult");

  int bestGenerations = 0;
  if (result == GOOGLE_SUCCESS && !response.data.empty()) {
    try {
      nlohmann::json j = nlohmann::json::parse(response.data);
      if (j.contains("bestGenerations") && j["bestGenerations"].is_number()) {
        bestGenerations = j["bestGenerations"].get<int>();
      }
    } catch (const nlohmann::json::exception& e) {
      std::cerr << "[ERROR] Failed to parse best result JSON: " << e.what() << "\n";
    }
  }

  googleCleanupResponse(&response);
  return bestGenerations;
}

static void googleCleanup() {
  curl_global_cleanup();
}

// Async version of googleSendProgress - "send and forget"
static void googleSendProgressAsync(uint64_t frameIdx, int kernelIdx, int bestGenerations, uint64_t bestPattern,
                                    const char* bestPatternBin) {
  // Copy the bestPatternBin string since the original may be destroyed
  std::string patternBinCopy(bestPatternBin ? bestPatternBin : "");

  // Launch detached thread using C++11 lambda with capture list
  // [=] captures all variables by value (copy) to ensure thread safety
  // The lambda runs googleSendProgress with retry logic in a separate thread, then thread is detached
  std::thread([=]() {
    const int maxRetries = 3;
    bool finalSuccess = false;

    for (int attempt = 0; attempt < maxRetries; ++attempt) {
      bool success = googleSendProgress(frameIdx, kernelIdx, bestGenerations, bestPattern, patternBinCopy.c_str());
      if (success) {
        finalSuccess = true;
        break;  // Success, exit retry loop
      }

      // If not the last attempt, wait with exponential backoff
      if (attempt < maxRetries - 1) {
        int delayMs = 500 * (1 << attempt);  // 500 * 2^attempt: 500, 1000, 2000ms
        std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
      }
    }

    if (!finalSuccess) {
      Logging::out() << "timestamp=" << time(NULL) << ", ERROR=Failed to send progress after " << maxRetries
                     << " attempts (frame " << frameIdx << ")\n";
    }
  }).detach();
}

// Cache management functions
static bool googleLoadFrameCache() {
  return frameCache.loadFromAPI();
}

static uint64_t googleGetFrameCacheCompletedCount() {
  return frameCache.getCompletedCount();
}

static bool googleIsFrameCompleteFromCache(uint64_t frameIdx) {
  // Load cache on first use
  if (!frameCache.isLoaded()) {
    if (!frameCache.loadFromAPI()) {
      // Cache loading failed, assume frame is incomplete to be safe
      return false;
    }
  }
  return frameCache.isFrameComplete(frameIdx);
}

static std::vector<uint64_t> googleGetIncompleteFrames() {
  GoogleConfig config;
  if (googleGetConfig(&config) != GOOGLE_SUCCESS) {
    return {};
  }

  // Build parameters map
  std::map<std::string, std::string> params;
  params["action"] = "getIncompleteFrames";
  params["apiKey"] = config.apiKey;

  CurlResponse response;
  GoogleResult result = googleHttpRequest(config.webappUrl, params, &response, "getIncompleteFrames");

  std::vector<uint64_t> incompleteFrames;
  if (result == GOOGLE_SUCCESS && !response.data.empty()) {
    try {
      // Parse JSON array
      nlohmann::json j = nlohmann::json::parse(response.data);

      // Convert JSON array to vector<uint64_t>
      for (const auto& frameIdx : j) {
        if (frameIdx.is_number_unsigned()) {
          incompleteFrames.push_back(frameIdx.get<uint64_t>());
        }
      }
    } catch (const nlohmann::json::exception& e) {
      std::cerr << "[ERROR] Failed to parse incomplete frames JSON: " << e.what() << "\n";
    }
  }

  googleCleanupResponse(&response);
  return incompleteFrames;
}

#endif