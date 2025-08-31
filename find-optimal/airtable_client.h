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

typedef struct {
    char* memory;
    size_t size;
} curl_response_t;

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

static bool airtable_send_progress(ulong64 frame_id, int kernel_id, int chunk_id, double patterns_per_second, bool is_frame_complete, bool is_test) {
    const char* endpoint = getenv("AIRTABLE_END_POINT");
    const char* api_key = getenv("AIRTABLE_API_KEY");
    
    if (!endpoint || !api_key) {
        printf("[WARN] Airtable environment variables not set, skipping progress upload\n");
        return false;
    }
    
    CURL *curl;
    CURLcode res;
    bool success = false;
    
    curl = curl_easy_init();
    if (!curl) {
        printf("[ERROR] Failed to initialize curl\n");
        return false;
    }
    
    char url[MAX_URL_LENGTH];
    snprintf(url, sizeof(url), "%s/Progress", endpoint);
    
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
        current_time, frame_id, kernel_id, chunk_id, patterns_per_second, is_frame_complete ? "true" : "false", is_test ? "true" : "false"
    );
    
    char auth_header[512];
    snprintf(auth_header, sizeof(auth_header), "Authorization: Bearer %s", api_key);
    
    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, auth_header);
    
    curl_response_t response = {0};
    
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_data);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    
    res = curl_easy_perform(curl);
    
    if (res != CURLE_OK) {
        printf("[ERROR] curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
    } else {
        long response_code;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
        
        if (response_code == 200 || response_code == 201) {
            success = true;
        } else {
            printf("[ERROR] Airtable API returned HTTP %ld\n", response_code);
            if (response.memory) {
                printf("[ERROR] Response: %s\n", response.memory);
            }
        }
    }
    
    if (response.memory) {
        free(response.memory);
    }
    
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    
    return success;
}

static bool airtable_send_result(int generations, ulong64 pattern, const char* pattern_bin, bool is_test) {
    const char* endpoint = getenv("AIRTABLE_END_POINT");
    const char* api_key = getenv("AIRTABLE_API_KEY");
    
    if (!endpoint || !api_key) {
        printf("[WARN] Airtable environment variables not set, skipping result upload\n");
        return false;
    }
    
    CURL *curl;
    CURLcode res;
    bool success = false;
    
    curl = curl_easy_init();
    if (!curl) {
        printf("[ERROR] Failed to initialize curl\n");
        return false;
    }
    
    char url[MAX_URL_LENGTH];
    snprintf(url, sizeof(url), "%s/Results", endpoint);
    
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
    
    char auth_header[512];
    snprintf(auth_header, sizeof(auth_header), "Authorization: Bearer %s", api_key);
    
    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, auth_header);
    
    curl_response_t response = {0};
    
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_data);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    
    res = curl_easy_perform(curl);
    
    if (res != CURLE_OK) {
        printf("[ERROR] curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
    } else {
        long response_code;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
        
        if (response_code == 200 || response_code == 201) {
            success = true;
        } else {
            printf("[ERROR] Airtable API returned HTTP %ld\n", response_code);
            if (response.memory) {
                printf("[ERROR] Response: %s\n", response.memory);
            }
        }
    }
    
    if (response.memory) {
        free(response.memory);
    }
    
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    
    return success;
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
    const char* endpoint = getenv("AIRTABLE_END_POINT");
    const char* api_key = getenv("AIRTABLE_API_KEY");
    
    if (!endpoint || !api_key) {
        printf("[WARN] Airtable environment variables not set, cannot query best result\n");
        return -1;
    }
    
    CURL *curl;
    CURLcode res;
    int best_generations = -1;
    
    curl = curl_easy_init();
    if (!curl) {
        printf("[ERROR] Failed to initialize curl\n");
        return -1;
    }
    
    // Query Results table sorted by generations descending, limit to 1 record
    // Filter out test records
    char url[MAX_URL_LENGTH];
    snprintf(url, sizeof(url), "%s/Results?sort%%5B0%%5D%%5Bfield%%5D=generations&sort%%5B0%%5D%%5Bdirection%%5D=desc&maxRecords=1&filterByFormula=NOT(test)", endpoint);
    
    char auth_header[512];
    snprintf(auth_header, sizeof(auth_header), "Authorization: Bearer %s", api_key);
    
    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, auth_header);
    
    curl_response_t response = {0};
    
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    
    res = curl_easy_perform(curl);
    
    if (res != CURLE_OK) {
        printf("[ERROR] curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
    } else {
        long response_code;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
        
        if (response_code == 200) {
            if (response.memory) {
                // Simple JSON parsing to find the generations value
                // Look for "generations": followed by a number
                char* gen_pos = strstr(response.memory, "\"generations\":");
                if (gen_pos) {
                    gen_pos += strlen("\"generations\":");
                    // Skip whitespace
                    while (*gen_pos == ' ' || *gen_pos == '\t') gen_pos++;
                    // Parse the number
                    best_generations = atoi(gen_pos);
                } else {
                    // No records found (empty result set)
                    best_generations = 0;
                }
            }
        } else {
            printf("[ERROR] Airtable API returned HTTP %ld\n", response_code);
            if (response.memory) {
                printf("[ERROR] Response: %s\n", response.memory);
            }
        }
    }
    
    if (response.memory) {
        free(response.memory);
    }
    
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    
    return best_generations;
}

static ulong64 airtable_get_best_complete_frame() {
    const char* endpoint = getenv("AIRTABLE_END_POINT");
    const char* api_key = getenv("AIRTABLE_API_KEY");
    
    if (!endpoint || !api_key) {
        printf("[WARN] Airtable environment variables not set, cannot query best complete frame\n");
        return ULLONG_MAX;  // Return special value to indicate error
    }
    
    CURL *curl;
    CURLcode res;
    ulong64 best_frame_id = ULLONG_MAX;  // Use ULLONG_MAX to indicate "not found"
    
    curl = curl_easy_init();
    if (!curl) {
        printf("[ERROR] Failed to initialize curl\n");
        return ULLONG_MAX;
    }
    
    // Query Progress table for non-test records where frame_complete=true, sorted by frame_id descending, limit to 1
    char url[MAX_URL_LENGTH];
    snprintf(url, sizeof(url), "%s/Progress?sort%%5B0%%5D%%5Bfield%%5D=frame_id&sort%%5B0%%5D%%5Bdirection%%5D=desc&maxRecords=1&filterByFormula=AND(NOT(test),frame_complete)", endpoint);
    
    char auth_header[512];
    snprintf(auth_header, sizeof(auth_header), "Authorization: Bearer %s", api_key);
    
    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, auth_header);
    
    curl_response_t response = {0};
    
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    
    res = curl_easy_perform(curl);
    
    if (res != CURLE_OK) {
        printf("[ERROR] curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
    } else {
        long response_code;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
        
        if (response_code == 200) {
            if (response.memory) {
                // Simple JSON parsing to find the frame_id value
                // Look for "frame_id": followed by a number
                char* frame_pos = strstr(response.memory, "\"frame_id\":");
                if (frame_pos) {
                    frame_pos += strlen("\"frame_id\":");
                    // Skip whitespace
                    while (*frame_pos == ' ' || *frame_pos == '\t') frame_pos++;
                    // Parse the number
                    best_frame_id = strtoull(frame_pos, NULL, 10);
                }
                // If frame_pos is NULL, best_frame_id remains ULLONG_MAX (not found)
            }
        } else {
            printf("[ERROR] Airtable API returned HTTP %ld\n", response_code);
            if (response.memory) {
                printf("[ERROR] Response: %s\n", response.memory);
            }
        }
    }
    
    if (response.memory) {
        free(response.memory);
    }
    
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    
    return best_frame_id;
}

static void airtable_cleanup() {
    curl_global_cleanup();
}

#endif
