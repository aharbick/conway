#ifndef _GOOGLE_CLIENT_H_
#define _GOOGLE_CLIENT_H_

#include <curl/curl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>


#define CURL_TIMEOUT 30L

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
  const char* webapp_url;
  const char* spreadsheet_id;
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
  config->webapp_url = getenv("GOOGLE_WEBAPP_URL");
  config->spreadsheet_id = getenv("GOOGLE_SPREADSHEET_ID");

  if (!config->webapp_url || !config->spreadsheet_id) {
    return GOOGLE_ERROR_CONFIG;
  }

  return GOOGLE_SUCCESS;
}

static void googleCleanupResponse(CurlResponse* response) {
  response->clear();
}

static GoogleResult googleHttpRequest(const char* url, CurlResponse* response) {
  CURL* curl = curl_easy_init();
  if (!curl) {
    std::cerr << "[ERROR] Failed to initialize curl\n";
    return GOOGLE_ERROR_CURL_INIT;
  }

  curl_easy_setopt(curl, CURLOPT_URL, url);
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
      std::cerr << "[ERROR] Google Sheets API returned HTTP " << responseCode << "\n";
      if (!response->data.empty()) {
        std::cerr << "[ERROR] Response: " << response->memory() << "\n";
      }
      result = GOOGLE_ERROR_HTTP;
    }
  }

  curl_easy_cleanup(curl);
  return result;
}

static const char* googleFindJsonValue(const char* json, const char* key) {
  if (!json || !key)
    return NULL;

  const std::string searchKey = "\"" + std::string(key) + "\":";

  const char* pos = strstr(json, searchKey.c_str());
  if (!pos)
    return NULL;

  pos += searchKey.size();
  while (*pos == ' ' || *pos == '\t')
    pos++;

  return pos;
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

static bool googleSendProgress(bool frameComplete, uint64_t frameIdx, int kernelIdx, int chunkIdx,
                               uint64_t patternsPerSecond, int bestGenerations, uint64_t bestPattern,
                               const char* bestPatternBin, bool isTest, bool randomFrame = false) {
  GoogleConfig config;
  if (googleGetConfig(&config) != GOOGLE_SUCCESS) {
    return false;
  }

  // Build URL with parameters
  std::ostringstream urlStream;
  urlStream << config.webapp_url << "?action=sendProgress";
  urlStream << "&spreadsheetId=" << googleUrlEncode(config.spreadsheet_id);
  if (frameComplete) {
    urlStream << "&frameComplete=true";
  }
  urlStream << "&frameIdx=" << frameIdx;
  urlStream << "&kernelIdx=" << kernelIdx;
  urlStream << "&chunkIdx=" << chunkIdx;
  urlStream << "&patternsPerSecond=" << patternsPerSecond;
  urlStream << "&bestGenerations=" << bestGenerations;
  urlStream << "&bestPattern=" << googleUrlEncode(std::to_string(bestPattern));
  urlStream << "&bestPatternBin=" << googleUrlEncode(bestPatternBin);
  if (isTest) {
    urlStream << "&test=true";
  }
  urlStream << "&randomFrame=" << (randomFrame ? "true" : "false");

  const std::string url = urlStream.str();

  CurlResponse response;
  GoogleResult result = googleHttpRequest(url.c_str(), &response);

  googleCleanupResponse(&response);
  return (result == GOOGLE_SUCCESS);
}

static std::string googleSendProgressWithResponse(bool frameComplete, uint64_t frameIdx, int kernelIdx, int chunkIdx,
                                                  uint64_t patternsPerSecond, int bestGenerations, uint64_t bestPattern,
                                                  const char* bestPatternBin, bool isTest, bool randomFrame = false) {
  GoogleConfig config;
  if (googleGetConfig(&config) != GOOGLE_SUCCESS) {
    return "{\"error\": \"Configuration failed\"}";
  }

  // Build URL with parameters
  std::ostringstream urlStream;
  urlStream << config.webapp_url << "?action=sendProgress";
  urlStream << "&spreadsheetId=" << googleUrlEncode(config.spreadsheet_id);
  if (frameComplete) {
    urlStream << "&frameComplete=true";
  }
  urlStream << "&frameIdx=" << frameIdx;
  urlStream << "&kernelIdx=" << kernelIdx;
  urlStream << "&chunkIdx=" << chunkIdx;
  urlStream << "&patternsPerSecond=" << patternsPerSecond;
  urlStream << "&bestGenerations=" << bestGenerations;
  urlStream << "&bestPattern=" << googleUrlEncode(std::to_string(bestPattern));
  urlStream << "&bestPatternBin=" << googleUrlEncode(bestPatternBin);
  if (isTest) {
    urlStream << "&test=true";
  }
  urlStream << "&randomFrame=" << (randomFrame ? "true" : "false");

  const std::string url = urlStream.str();

  CurlResponse response;
  GoogleResult result = googleHttpRequest(url.c_str(), &response);

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

  std::ostringstream urlStream;
  urlStream << config.webapp_url << "?action=getBestResult";
  urlStream << "&spreadsheetId=" << googleUrlEncode(config.spreadsheet_id);
  const std::string url = urlStream.str();

  CurlResponse response;
  GoogleResult result = googleHttpRequest(url.c_str(), &response);

  int bestGenerations = -1;
  if (result == GOOGLE_SUCCESS && !response.data.empty()) {
    const char* genPos = googleFindJsonValue(response.memory(), "bestGenerations");
    if (genPos) {
      bestGenerations = atoi(genPos);
    } else {
      bestGenerations = 0;
    }
  }

  googleCleanupResponse(&response);
  return bestGenerations;
}

static uint64_t googleGetBestCompleteFrame() {
  GoogleConfig config;
  if (googleGetConfig(&config) != GOOGLE_SUCCESS) {
    return ULLONG_MAX;
  }

  std::ostringstream urlStream;
  urlStream << config.webapp_url << "?action=getBestCompleteFrame";
  urlStream << "&spreadsheetId=" << googleUrlEncode(config.spreadsheet_id);
  const std::string url = urlStream.str();

  CurlResponse response;
  GoogleResult result = googleHttpRequest(url.c_str(), &response);

  uint64_t bestFrameIdx = ULLONG_MAX;
  if (result == GOOGLE_SUCCESS && !response.data.empty()) {
    const char* framePos = googleFindJsonValue(response.memory(), "bestFrameIdx");
    if (framePos) {
      bestFrameIdx = strtoull(framePos, NULL, 10);
    }
  }

  googleCleanupResponse(&response);
  return bestFrameIdx;
}

static bool googleGetIsFrameComplete(uint64_t frameIdx) {
  GoogleConfig config;
  if (googleGetConfig(&config) != GOOGLE_SUCCESS) {
    return false;
  }

  std::ostringstream urlStream;
  urlStream << config.webapp_url << "?action=getIsFrameComplete";
  urlStream << "&spreadsheetId=" << googleUrlEncode(config.spreadsheet_id);
  urlStream << "&frameIdx=" << frameIdx;
  const std::string url = urlStream.str();

  CurlResponse response;
  GoogleResult result = googleHttpRequest(url.c_str(), &response);

  bool isComplete = false;
  if (result == GOOGLE_SUCCESS && !response.data.empty()) {
    const char* completePos = googleFindJsonValue(response.memory(), "isComplete");
    if (completePos) {
      // Check if the value is "true"
      isComplete = (strncmp(completePos, "true", 4) == 0);
    }
  }

  googleCleanupResponse(&response);
  return isComplete;
}

static void googleCleanup() {
  curl_global_cleanup();
}

// Async version of googleSendProgress - "send and forget"
static void googleSendProgressAsync(bool frameComplete, uint64_t frameIdx, int kernelIdx, int chunkIdx,
                                    uint64_t patternsPerSecond, int bestGenerations, uint64_t bestPattern,
                                    const char* bestPatternBin, bool isTest, bool randomFrame = false) {
  // Copy the bestPatternBin string since the original may be destroyed
  std::string patternBinCopy(bestPatternBin ? bestPatternBin : "");

  // Launch detached thread using C++11 lambda with capture list
  // [=] captures all variables by value (copy) to ensure thread safety
  // The lambda runs googleSendProgress in a separate thread, then thread is detached
  std::thread([=]() {
    googleSendProgress(frameComplete, frameIdx, kernelIdx, chunkIdx, patternsPerSecond, bestGenerations, bestPattern,
                       patternBinCopy.c_str(), isTest, randomFrame);
  }).detach();
}

#endif