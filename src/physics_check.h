#ifndef PHYSICS_CHECK_H
#define PHYSICS_CHECK_H

#include <string>
#include <vector>

// Forward declarations for the main classes
class IntrusionDetector;

/**
 * Run intrusion detection analysis on CSV files
 * 
 * @param ROS_data_file Path to battery state CSV
 * @return 0 on success, non-zero on error
 */
int runIntrusionDetection(const std::string& ROS_data_file);

#endif // PHYSICS_CHECK_H