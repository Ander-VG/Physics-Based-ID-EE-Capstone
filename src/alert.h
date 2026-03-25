#ifndef ALERT_H
#define ALERT_H
 
#include <string>
 
bool send_alert(const std::string& message,
                const std::string& severity,
                const std::string& source,
                const std::string& type,
                int confidence,
                const std::string& data_json = "{}");
 
#endif