#ifndef PHYSICS_CHECK_H
#define PHYSICS_CHECK_H

#include <string>
#include <vector>

// Forward declarations for the main classes
class IntrusionDetector;

/**
 * Run intrusion detection analysis on CSV files
 * 
 * @param odom_file Path to odometry CSV
 * @param cmd_file Path to command velocity CSV
 * @param imu_file Path to IMU CSV
 * @param joint_file Path to joint states CSV
 * @param error_file Path to tracking error CSV
 * @param battery_file Path to battery state CSV
 * @return 0 on success, non-zero on error
 */
int runIntrusionDetection(
    const std::string& odom_file,
    const std::string& cmd_file,
    const std::string& imu_file,
    const std::string& joint_file,
    const std::string& error_file,
    const std::string& battery_file
);

#endif // PHYSICS_CHECK_H