#ifndef DATA_TRANSITION_H
#define DATA_TRANSITION_H

#include <string>

/**
 * Convert ROS2 bag file to multiple CSV files
 * 
 * @param bag_path Path to the .db3 bag file
 * @param output_file Output path for battery state CSV
 * @return Total number of messages converted, or -1 on error
 */
int BagtoCSV(
    const std::string& bag_path,
    const std::string& output_file
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