#include "alert.h"
 
#include <cstdlib>
#include <iostream>
#include <string>
#include <curl/curl.h>
 
bool send_alert(const std::string& message,
                const std::string& severity,
                const std::string& source,
                const std::string& type,
                double confidence,
                const std::string& data_json) {
    const char* token = std::getenv("INTERNAL_ALERT_TOKEN");
    const char* url   = std::getenv("RTCT_INTERNAL_ALERT_URL");
 
    if (!token) {
        std::cerr << "INTERNAL_ALERT_TOKEN not set\n";
        return false;
    }
 
    std::string endpoint = url ? url : "http://api:4000/internal/alert";
 
    std::string payload =
        "{"
        "\"message\":\"" + message + "\","
        "\"severity\":\"" + severity + "\","
        "\"source\":\"" + source + "\","
        "\"type\":\"" + type + "\","
        "\"confidence\":" + std::to_string(confidence) + ","
        "\"data\":" + data_json +
        "}";
 
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Failed to init curl\n";
        return false;
    }
 
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    std::string token_header = std::string("x-internal-token: ") + token;
    headers = curl_slist_append(headers, token_header.c_str());
 
    curl_easy_setopt(curl, CURLOPT_URL, endpoint.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);
 
    CURLcode res = curl_easy_perform(curl);
 
    long status_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status_code);
 
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
 
    if (res != CURLE_OK) {
        std::cerr << "curl error: " << curl_easy_strerror(res) << "\n";
        return false;
    }
 
    if (status_code < 200 || status_code >= 300) {
        std::cerr << "HTTP error: " << status_code << "\n";
        return false;
    }
 
    return true;
}
 