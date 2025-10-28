#ifndef DATA_TRANSITION_H
#define DATA_TRANSITION_H

#include <string>

/**
 * Convert ROS2 bag file to multiple CSV files
 * 
 * @param bag_path Path to the .db3 bag file
 * @param odom_file Output path for odometry CSV
 * @param cmd_file Output path for command velocity CSV
 * @param imu_file Output path for IMU CSV
 * @param js_file Output path for joint states CSV
 * @param err_file Output path for tracking error CSV
 * @param bat_file Output path for battery state CSV
 * @param ss_file Output path for sensor state CSV
 * @return Total number of messages converted, or -1 on error
 */
int BagtoCSV(
    const std::string& bag_path,
    const std::string& odom_file,
    const std::string& cmd_file,
    const std::string& imu_file,
    const std::string& js_file,
    const std::string& err_file,
    const std::string& bat_file,
    const std::string& ss_file
);

/**
 * Initialize ROS2 for bag processing
 * Must be called before BagtoCSV
 */
void initializeROS(int argc, char** argv);

/**
 * Shutdown ROS2 after bag processing
 * Should be called after all BagtoCSV operations
 */
void shutdownROS();

#endif // DATA_TRANSITION_H