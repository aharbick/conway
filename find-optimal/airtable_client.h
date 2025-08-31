#ifndef _AIRTABLE_CLIENT_H_
#define _AIRTABLE_CLIENT_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>
#include <time.h>
#include "types.h"

#define MAX_URL_LENGTH 1024
#define MAX_JSON_LENGTH 2048
#define MAX_RESPONSE_LENGTH 4096
#define CURL_TIMEOUT 30L

typedef struct {
    char* memory;
    size_t size;
} curl_response_t;

typedef struct {
    const char* endpoint;
    const char* api_key;
} airtable_config_t;

typedef enum {
    AIRTABLE_SUCCESS,
    AIRTABLE_ERROR_CONFIG,
    AIRTABLE_ERROR_CURL_INIT,
    AIRTABLE_ERROR_CURL_PERFORM,
    AIRTABLE_ERROR_HTTP
} airtable_result_t;

static size_t WriteMemoryCallback(void *contents, size_t size, size_t nmemb, curl_response_t *response) {
    size_t realsize = size * nmemb;
    char *ptr = (char*)realloc(response->memory, response->size + realsize + 1);

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

static airtable_result_t airtable_get_config(airtable_config_t* config) {
    config->endpoint = getenv("AIRTABLE_END_POINT");
    config->api_key = getenv("AIRTABLE_API_KEY");

    if (!config->endpoint || !config->api_key) {
        printf("[WARN] Airtable environment variables not set\n");
        return AIRTABLE_ERROR_CONFIG;
    }

    return AIRTABLE_SUCCESS;
}

static void airtable_cleanup_response(curl_response_t* response) {
    if (response->memory) {
        free(response->memory);
        response->memory = NULL;
        response->size = 0;
    }
}

static airtable_result_t airtable_http_request(const char* url, const char* json_data, const char* api_key, curl_response_t* response, bool is_post) {
    CURL *curl = curl_easy_init();
    if (!curl) {
        printf("[ERROR] Failed to initialize curl\n");
        return AIRTABLE_ERROR_CURL_INIT;
    }

    char auth_header[512];
    snprintf(auth_header, sizeof(auth_header), "Authorization: Bearer %s", api_key);

    struct curl_slist *headers = NULL;
    if (is_post) {
        headers = curl_slist_append(headers, "Content-Type: application/json");
    }
    headers = curl_slist_append(headers, auth_header);

    curl_easy_setopt(curl, CURLOPT_URL, url);
    if (is_post && json_data) {
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_data);
    }
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, CURL_TIMEOUT);

    CURLcode res = curl_easy_perform(curl);
    airtable_result_t result = AIRTABLE_SUCCESS;

    if (res != CURLE_OK) {
        printf("[ERROR] curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        result = AIRTABLE_ERROR_CURL_PERFORM;
    } else {
        long response_code;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);

        if (response_code != 200 && response_code != 201) {
            printf("[ERROR] Airtable API returned HTTP %ld\n", response_code);
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

static const char* airtable_find_json_value(const char* json, const char* key) {
    if (!json || !key) return NULL;

    char search_key[256];
    snprintf(search_key, sizeof(search_key), "\"%s\":", key);

    const char* pos = strstr(json, search_key);
    if (!pos) return NULL;

    pos += strlen(search_key);
    while (*pos == ' ' || *pos == '\t') pos++;

    return pos;
}

static bool airtable_send_progress(ulong64 frame_id, int kernel_id, int chunk_id, double patterns_per_second, bool is_frame_complete, bool is_test) {
    airtable_config_t config;
    if (airtable_get_config(&config) != AIRTABLE_SUCCESS) {
        printf("[WARN] Skipping progress upload\n");
        return false;
    }

    char url[MAX_URL_LENGTH];
    snprintf(url, sizeof(url), "%s/Progress", config.endpoint);

    char json_data[MAX_JSON_LENGTH];
    time_t current_time = time(NULL);
    snprintf(json_data, sizeof(json_data),
        "{"
        "\"fields\": {"
        "\"timestamp\": %ld,"
        "\"frame_id\": %llu,"
        "\"kernel_id\": %d,"
        "\"chunk_id\": %d,"
        "\"patterns_per_second\": %.2f,"
        "\"frame_complete\": %s,"
        "\"test\": %s"
        "}"
        "}",
        current_time, frame_id, kernel_id, chunk_id, patterns_per_second,
        is_frame_complete ? "true" : "false", is_test ? "true" : "false"
    );

    curl_response_t response = {0};
    airtable_result_t result = airtable_http_request(url, json_data, config.api_key, &response, true);

    airtable_cleanup_response(&response);
    return (result == AIRTABLE_SUCCESS);
}

static bool airtable_send_result(int generations, ulong64 pattern, const char* pattern_bin, bool is_test) {
    airtable_config_t config;
    if (airtable_get_config(&config) != AIRTABLE_SUCCESS) {
        printf("[WARN] Skipping result upload\n");
        return false;
    }

    char url[MAX_URL_LENGTH];
    snprintf(url, sizeof(url), "%s/Results", config.endpoint);

    char json_data[MAX_JSON_LENGTH];
    snprintf(json_data, sizeof(json_data),
        "{"
        "\"fields\": {"
        "\"generations\": %d,"
        "\"pattern\": %llu,"
        "\"pattern_bin\": \"%s\","
        "\"test\": %s"
        "}"
        "}",
        generations, pattern, pattern_bin, is_test ? "true" : "false"
    );

    curl_response_t response = {0};
    airtable_result_t result = airtable_http_request(url, json_data, config.api_key, &response, true);

    airtable_cleanup_response(&response);
    return (result == AIRTABLE_SUCCESS);
}

static bool airtable_init() {
    CURLcode res = curl_global_init(CURL_GLOBAL_DEFAULT);
    if (res != CURLE_OK) {
        printf("[ERROR] Failed to initialize libcurl: %s\n", curl_easy_strerror(res));
        return false;
    }
    return true;
}

static int airtable_get_best_result() {
    airtable_config_t config;
    if (airtable_get_config(&config) != AIRTABLE_SUCCESS) {
        printf("[WARN] Cannot query best result\n");
        return -1;
    }

    char url[MAX_URL_LENGTH];
    snprintf(url, sizeof(url), "%s/Results?sort%%5B0%%5D%%5Bfield%%5D=generations&sort%%5B0%%5D%%5Bdirection%%5D=desc&maxRecords=1&filterByFormula=NOT(test)", config.endpoint);

    curl_response_t response = {0};
    airtable_result_t result = airtable_http_request(url, NULL, config.api_key, &response, false);

    int best_generations = -1;
    if (result == AIRTABLE_SUCCESS && response.memory) {
        const char* gen_pos = airtable_find_json_value(response.memory, "generations");
        if (gen_pos) {
            best_generations = atoi(gen_pos);
        } else {
            best_generations = 0;
        }
    }

    airtable_cleanup_response(&response);
    return best_generations;
}

static ulong64 airtable_get_best_complete_frame() {
    airtable_config_t config;
    if (airtable_get_config(&config) != AIRTABLE_SUCCESS) {
        printf("[WARN] Cannot query best complete frame\n");
        return ULLONG_MAX;
    }

    char url[MAX_URL_LENGTH];
    snprintf(url, sizeof(url), "%s/Progress?sort%%5B0%%5D%%5Bfield%%5D=frame_id&sort%%5B0%%5D%%5Bdirection%%5D=desc&maxRecords=1&filterByFormula=AND(NOT(test),frame_complete)", config.endpoint);

    curl_response_t response = {0};
    airtable_result_t result = airtable_http_request(url, NULL, config.api_key, &response, false);

    ulong64 best_frame_id = ULLONG_MAX;
    if (result == AIRTABLE_SUCCESS && response.memory) {
        const char* frame_pos = airtable_find_json_value(response.memory, "frame_id");
        if (frame_pos) {
            best_frame_id = strtoull(frame_pos, NULL, 10);
        }
    }

    airtable_cleanup_response(&response);
    return best_frame_id;
}

static void airtable_cleanup() {
    curl_global_cleanup();
}

#endif
