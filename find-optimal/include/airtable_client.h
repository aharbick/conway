#ifndef _AIRTABLE_CLIENT_H_
#define _AIRTABLE_CLIENT_H_

#include <curl/curl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "types.h"

#define MAX_URL_LENGTH 1024
#define MAX_JSON_LENGTH 2048
#define MAX_RESPONSE_LENGTH 4096
#define CURL_TIMEOUT 30L

typedef struct {
  char* memory;
  size_t size;
} CurlResponse;

typedef struct {
  const char* endpoint;
  const char* api_key;
} AirtableConfig;

typedef enum {
  AIRTABLE_SUCCESS,
  AIRTABLE_ERROR_CONFIG,
  AIRTABLE_ERROR_CURL_INIT,
  AIRTABLE_ERROR_CURL_PERFORM,
  AIRTABLE_ERROR_HTTP
} AirtableResult;

static size_t WriteMemoryCallback(void* contents, size_t size, size_t nmemb, CurlResponse* response) {
  size_t realsize = size * nmemb;
  char* ptr = (char*)realloc(response->memory, response->size + realsize + 1);

  if (!ptr) {
    printf("[ERROR] Not enough memory (realloc returned NULL)\n");
    return 0;
  }

  response->memory = ptr;
  memcpy(&(response->memory[response->size]), contents, realsize);
  response->size += realsize;
  response->memory[response->size] = 0;

  return realsize;
}

static AirtableResult airtableGetConfig(AirtableConfig* config) {
  config->endpoint = getenv("AIRTABLE_END_POINT");
  config->api_key = getenv("AIRTABLE_API_KEY");

  if (!config->endpoint || !config->api_key) {
    printf("[WARN] Airtable environment variables not set\n");
    return AIRTABLE_ERROR_CONFIG;
  }

  return AIRTABLE_SUCCESS;
}

static void airtableCleanupResponse(CurlResponse* response) {
  if (response->memory) {
    free(response->memory);
    response->memory = NULL;
    response->size = 0;
  }
}

static AirtableResult airtableHttpRequest(const char* url, const char* jsonData, const char* apiKey,
                                          CurlResponse* response, bool isPost) {
  CURL* curl = curl_easy_init();
  if (!curl) {
    printf("[ERROR] Failed to initialize curl\n");
    return AIRTABLE_ERROR_CURL_INIT;
  }

  char authHeader[512];
  snprintf(authHeader, sizeof(authHeader), "Authorization: Bearer %s", apiKey);

  struct curl_slist* headers = NULL;
  if (isPost) {
    headers = curl_slist_append(headers, "Content-Type: application/json");
  }
  headers = curl_slist_append(headers, authHeader);

  curl_easy_setopt(curl, CURLOPT_URL, url);
  if (isPost && jsonData) {
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonData);
  }
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void*)response);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, CURL_TIMEOUT);

  CURLcode res = curl_easy_perform(curl);
  AirtableResult result = AIRTABLE_SUCCESS;

  if (res != CURLE_OK) {
    printf("[ERROR] curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
    result = AIRTABLE_ERROR_CURL_PERFORM;
  } else {
    long responseCode;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &responseCode);

    if (responseCode != 200 && responseCode != 201) {
      printf("[ERROR] Airtable API returned HTTP %ld\n", responseCode);
      if (response->memory) {
        printf("[ERROR] Response: %s\n", response->memory);
      }
      result = AIRTABLE_ERROR_HTTP;
    }
  }

  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);

  return result;
}

static const char* airtableFindJsonValue(const char* json, const char* key) {
  if (!json || !key)
    return NULL;

  char searchKey[256];
  snprintf(searchKey, sizeof(searchKey), "\"%s\":", key);

  const char* pos = strstr(json, searchKey);
  if (!pos)
    return NULL;

  pos += strlen(searchKey);
  while (*pos == ' ' || *pos == '\t')
    pos++;

  return pos;
}

static bool airtableSendProgress(bool frameComplete, ulong64 frameIdx, int kernelIdx, int chunkIdx,
                                 ulong64 patternsPerSecond, int bestGenerations, ulong64 bestPattern,
                                 const char* bestPatternBin, bool isTest) {
  AirtableConfig config;
  if (airtableGetConfig(&config) != AIRTABLE_SUCCESS) {
    printf("[WARN] Skipping progress upload\n");
    return false;
  }

  char url[MAX_URL_LENGTH];
  snprintf(url, sizeof(url), "%s/Progress", config.endpoint);

  char jsonData[MAX_JSON_LENGTH];
  time_t currentTime = time(NULL);
  snprintf(jsonData, sizeof(jsonData),
           "{"
           "\"fields\": {"
           "\"timestamp\": %ld,"
           "\"frameComplete\": %s,"
           "\"frameIdx\": %llu,"
           "\"kernelIdx\": %d,"
           "\"chunkIdx\": %d,"
           "\"patternsPerSecond\": %llu,"
           "\"bestGenerations\": %d,"
           "\"bestPattern\": \"%llu\","
           "\"bestPatternBin\": \"%s\","
           "\"test\": %s"
           "}"
           "}",
           currentTime, frameComplete ? "true" : "false", frameIdx, kernelIdx, chunkIdx, patternsPerSecond,
           bestGenerations, bestPattern, bestPatternBin, isTest ? "true" : "false");

  CurlResponse response = {0};
  AirtableResult result = airtableHttpRequest(url, jsonData, config.api_key, &response, true);

  airtableCleanupResponse(&response);
  return (result == AIRTABLE_SUCCESS);
}


static bool airtableInit() {
  CURLcode res = curl_global_init(CURL_GLOBAL_DEFAULT);
  if (res != CURLE_OK) {
    printf("[ERROR] Failed to initialize libcurl: %s\n", curl_easy_strerror(res));
    return false;
  }
  return true;
}

static int airtableGetBestResult() {
  AirtableConfig config;
  if (airtableGetConfig(&config) != AIRTABLE_SUCCESS) {
    printf("[WARN] Cannot query best result\n");
    return -1;
  }

  char url[MAX_URL_LENGTH];
  snprintf(url, sizeof(url),
           "%s/"
           "Progress?sort%%5B0%%5D%%5Bfield%%5D=bestGenerations&sort%%5B0%%5D%%5Bdirection%%5D=desc&maxRecords=1&"
           "filterByFormula=NOT(test)",
           config.endpoint);

  CurlResponse response = {0};
  AirtableResult result = airtableHttpRequest(url, NULL, config.api_key, &response, false);

  int bestGenerations = -1;
  if (result == AIRTABLE_SUCCESS && response.memory) {
    const char* genPos = airtableFindJsonValue(response.memory, "bestGenerations");
    if (genPos) {
      bestGenerations = atoi(genPos);
    } else {
      bestGenerations = 0;
    }
  }

  airtableCleanupResponse(&response);
  return bestGenerations;
}

static ulong64 airtableGetBestCompleteFrame() {
  AirtableConfig config;
  if (airtableGetConfig(&config) != AIRTABLE_SUCCESS) {
    printf("[WARN] Cannot query best complete frame\n");
    return ULLONG_MAX;
  }

  char url[MAX_URL_LENGTH];
  snprintf(url, sizeof(url),
           "%s/"
           "Progress?sort%%5B0%%5D%%5Bfield%%5D=frameIdx&sort%%5B0%%5D%%5Bdirection%%5D=desc&maxRecords=1&"
           "filterByFormula=AND(NOT(test),frameComplete)",
           config.endpoint);

  CurlResponse response = {0};
  AirtableResult result = airtableHttpRequest(url, NULL, config.api_key, &response, false);

  ulong64 bestFrameIdx = ULLONG_MAX;
  if (result == AIRTABLE_SUCCESS && response.memory) {
    const char* framePos = airtableFindJsonValue(response.memory, "frameIdx");
    if (framePos) {
      bestFrameIdx = strtoull(framePos, NULL, 10);
    }
  }

  airtableCleanupResponse(&response);
  return bestFrameIdx;
}

static void airtableCleanup() {
  curl_global_cleanup();
}

#endif
