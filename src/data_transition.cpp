#include "data_transition.h"

#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <iomanip>
#include <chrono>
#include <cmath>

//ROS premade C++ friendly libraries
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/serialization.hpp"
#include "rosbag2_cpp/reader.hpp"
#include "rosbag2_storage/storage_options.hpp"
#include "rosbag2_cpp/storage_options.hpp"

//message types
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "sensor_msgs/msg/battery_state.hpp"

//standard functions (ongoing)
using std::cout;
using std::endl;
using std::string;
using std::ofstream;
using std::unique_ptr;
using std::shared_ptr;
using std::make_unique;
using std::make_shared;
using std::exception;
using std::fixed;
using std::setprecision;


//ROS2 functions
using rosbag2_cpp::Reader;
using rosbag2_storage::StorageOptions;
using rosbag2_cpp::ConverterOptions;
using rosbag2_storage::SerializedBagMessage;
using rclcpp::Serialization;


//The purpose of this program is to parse ROSbags and convert them into CSV files for further analysis
//This program should not be required in the second half of the capstone, for the program should be incorporated
//into the Kubernetes Cluster.
//However, for the purposes of learning ROS architecture and testing the subsystem this program is necessary


//First Function: Creating a reader for the bag. If true, it is open if false then something wrong happened
bool openBag(const string& bag_path, unique_ptr<Reader>& reader)
{
    //Create the reader
    reader = make_unique<Reader>();

    //find the format of the bag
    StorageOptions storage_options;
    storage_options.uri = bag_path;
    storage_options.storage_id = "sqlite3";

    ConverterOptions converter_options;
    converter_options.input_serialization_format = "cdr";
    converter_options.output_serialization_format = "cdr";

    //try statement to check if it works
    try{
        reader->open(storage_options, converter_options);
        cout << "Successful bag opening" << endl;
        return true;
    }

    catch (const exception& e){
        cout << "Error in bag opening: " << e.what() << endl;
        return false;
    }
}

//Second Function: Reading the bag
void showBag(Reader* reader)
{
    cout << "Analyzing bag content" << endl;

    // List all topics
    auto topics_and_types = reader -> get_all_topics_and_types();
    cout << topics_and_types.size() << " topics found:" << endl;

    // //find each topic
    // for (const auto& topic : topics_and_types) {
    //     cout << "Topic: " << topic.name << endl;
    //     cout << "Type: " << topic.type << endl;
    // }
}

bool createCSV(const string& filename, ofstream& csv_file){
    //create a file
    csv_file.open(filename);
    //check
    if (!csv_file.is_open()){
        cout << "Error creating CSV" << endl;
        return false;
    }
    csv_file << "Time,";      // When was data generated (ns)
    csv_file << "X,";         // X position
    csv_file << "Y,";         // Y position
    csv_file << "Theta,";      // Orientation angle (yaw)
    csv_file << "linear_x,";     // Linear velocity
    csv_file << "angular_z,";    // Angular velocity
    csv_file << "imu_accel_x,imu_accel_y,imu_accel_z,"; // IMUs measured by accelerometer
    csv_file << "imu_gyro_x,imu_gyro_y,imu_gyro_z,";  // IMUs measured by gyroscope
    csv_file << "v_R,v_L,";  // Right and Left wheel velocities
    csv_file << "right_encoder,left_encoder,";  // Left and Right wheel encoders
    csv_file << "tracking_error,error_x,error_y,";  // Error in sensors
    csv_file << "battery_voltage,battery_current,battery_percentage,";//Battery states
    csv_file << "temperature";

    csv_file << endl;
    return true;
}

//Identifying if the topic is odometry
bool OdomId(const string& topic_name)
{
    return topic_name.find("odom") != string::npos;
}

//Identifying if the topic is command velocity
bool CmdId(const string& topic_name)
{
    return topic_name.find("cmd_vel") != string::npos;
}

//Identifying if the topic is IMU
bool IMUId(const string& topic_name)
{
    return topic_name.find("imu") != string::npos;
}

//Identifying if the topic is Joint State
bool JSId(const string& topic_name)
{
    return topic_name.find("joint_states") != string::npos;
}

//Identifying if the topic is tracking error
bool ErrorId(const string& topic_name)
{
    return topic_name.find("tracking_error") != string::npos;
}

//Identifying if the topic is Battery State
bool BatId(const string& topic_name)
{
    return topic_name.find("battery_state") != string::npos;
}

int BagtoCSV (const string& bag_path, const string& output_file)
{
    int rows = 0;
    int messages_tot = 0;

    auto start_time = std::chrono::high_resolution_clock::now();
    unique_ptr<Reader> reader;

    if (!openBag(bag_path, reader)) {
        cout << "Error opening Bag" << endl;
        return -1;
    }

    //getting bag values
    showBag(reader.get());

    ofstream csv_file;
    if (!createCSV(output_file, csv_file)) {
        cout << "Could not create CSV file" << endl;
        return -1;
    }

    //keeping track of messages
    int cmd_vel_messages = 0;
    int odom_messages = 0;
    int imu_messages = 0;
    int js_messages = 0;
    int err_messages = 0;
    int bat_messages = 0;

    //Fill-forward values
    double first_timestamp = -1.0;
    double last_x = 0.0, last_y = 0.0, last_theta = 0.0;
    double last_imu_accel_x = 0.0, last_imu_accel_y = 0.0, last_imu_accel_z = 0.0;
    double last_imu_gyro_x = 0.0, last_imu_gyro_y = 0.0, last_imu_gyro_z = 0.0;
    double last_v_R = 0.0, last_v_L = 0.0;
    double last_right_encoder = 0.0, last_left_encoder = 0.0;
    double last_tracking_error = 0.0, last_error_x = 0.0, last_error_y = 0.0;
    double last_battery_voltage = 0, last_battery_current = 0.0;
    double last_battery_percentage = 0.0, last_temperature = 0.0;

    while (reader->has_next())
    {
        auto bag_message = reader->read_next();
        messages_tot++;
        const string& t = bag_message->topic_name;
        //Cmd first so it can lead the timestamp
        if (CmdId(t)) {
            cmd_vel_messages++;
            Serialization<geometry_msgs::msg::Twist> deserializer;
            try {
                //getting binary data
                rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);

                //empty command velocity message
                auto cmd_vel_msg = make_shared<geometry_msgs::msg::Twist>();

                //adding bag data
                deserializer.deserialize_message(&serialized_msg, cmd_vel_msg.get());

                uint64_t timestamp_ns = bag_message->time_stamp;
                double time_in_seconds = timestamp_ns / 1000000000.0;
                
                if (first_timestamp < 0) {
                    first_timestamp = time_in_seconds;
                }
                
                //calculate relative time
                double relative_time = time_in_seconds - first_timestamp;
                //extract cmd values
                double linear_x = cmd_vel_msg->linear.x;
                double angular_z = cmd_vel_msg->angular.z;

                // Rewrite CSV Row
                csv_file << fixed << setprecision(9);
                csv_file << relative_time << ",";
                csv_file << last_x << "," << last_y << "," << last_theta << ",";
                csv_file << linear_x << "," << angular_z << ",";
                csv_file << last_imu_accel_x << "," << last_imu_accel_y << "," << last_imu_accel_z << ",";
                csv_file << last_imu_gyro_x << "," << last_imu_gyro_y << "," << last_imu_gyro_z << ",";
                csv_file << last_v_R << "," << last_v_L << ",";
                csv_file << last_right_encoder << "," << last_left_encoder << ",";
                csv_file << last_tracking_error << "," << last_error_x << "," << last_error_y << ",";
                csv_file << last_battery_voltage << "," << last_battery_current << ",";
                csv_file << last_battery_percentage << "," << last_temperature;
                csv_file << endl;

                rows++;
            }
            catch (const exception& e) {
                cout << "Error processing cmd_vel: " << e.what() << endl;
            }
        }
        //Odom time
        else if (OdomId(t)) {
            odom_messages++;

            // Deserialize the message
            Serialization<nav_msgs::msg::Odometry> deserializer;
            //binary to text
            try {
                //getting binary data
                rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);
                //empty odometry message
                auto odom_msg = make_shared<nav_msgs::msg::Odometry>();
                //adding bag data
                deserializer.deserialize_message(&serialized_msg, odom_msg.get());
                
                //extract X and Y position
                last_x = odom_msg->pose.pose.position.x;
                last_y = odom_msg->pose.pose.position.y;
                
                //convert quaternion to yaw angle (theta)
                double qx = odom_msg->pose.pose.orientation.x;
                double qy = odom_msg->pose.pose.orientation.y;
                double qz = odom_msg->pose.pose.orientation.z;
                double qw = odom_msg->pose.pose.orientation.w;
                
                //calculate yaw (theta) from quaternion
                last_theta = atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));    
            }
            catch (const exception& e) {
                cout << "Error processing Odom: " << e.what() << endl;
            }
        }
        //IMU addition
        else if (IMUId(t)) {
            imu_messages++;
            // Deserialize the message
            Serialization<sensor_msgs::msg::Imu> deserializer;
            //binary to text
            try {

                //getting binary data
                rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);

                //empty IMU message
                auto imu_msg = make_shared<sensor_msgs::msg::Imu>();

                //adding bag data
                deserializer.deserialize_message(&serialized_msg, imu_msg.get());

                last_imu_accel_x = imu_msg->linear_acceleration.x;
                last_imu_accel_y = imu_msg->linear_acceleration.y;
                last_imu_accel_z = imu_msg->linear_acceleration.z;
                last_imu_gyro_x = imu_msg->angular_velocity.x;
                last_imu_gyro_y = imu_msg->angular_velocity.y;
                last_imu_gyro_z = imu_msg->angular_velocity.z;
            }
            catch (const exception& e) {
                cout << "Error processing imu: " << e.what() << endl;
            }
        }
        //Joint States addition
        else if (JSId(t)) {
            js_messages++;
            // Deserialize the message
            Serialization<sensor_msgs::msg::JointState> deserializer;
            //binary to text
            try {

                //getting binary data
                rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);

                //empty joint state message
                auto js_msg = make_shared<sensor_msgs::msg::JointState>();

                //adding bag data
                deserializer.deserialize_message(&serialized_msg, js_msg.get());

                if (js_msg->velocity.size() >= 2 && js_msg->position.size() >= 2){
                    last_v_R = js_msg->velocity[1];
                    last_v_L = js_msg->velocity[0];
                    last_right_encoder = js_msg->position[1];
                    last_left_encoder = js_msg->position[0];
                }
            }
            catch (const exception& e) {
                cout << "Error processing Joint States: " << e.what() << endl;
            }
        }
        else if (ErrorId(t)) {
            err_messages++;
            // Deserialize the message
            Serialization<std_msgs::msg::Float32MultiArray> deserializer;
            //binary to text
            try {

                //getting binary data
                rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);

                //empty error message
                auto err_msg = make_shared<std_msgs::msg::Float32MultiArray>();

                //adding bag data
                deserializer.deserialize_message(&serialized_msg, err_msg.get());

                if (err_msg->data.size() >= 3) {
                    last_tracking_error = err_msg->data[0];
                    last_error_x = err_msg->data[1];
                    last_error_y = err_msg->data[2];
                }
            }
            catch (const exception& e) {
                cout << "Error processing tracking_error: " << e.what() << endl;
            }
        }
        else if (BatId(t)) {
            bat_messages++;
            Serialization<sensor_msgs::msg::BatteryState> deserializer;
            try {

                //getting binary data
                rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);

                //empty joint state message
                auto bat_msg = make_shared<sensor_msgs::msg::BatteryState>();

                //adding bag data
                deserializer.deserialize_message(&serialized_msg, bat_msg.get());

                last_battery_voltage = bat_msg->voltage;
                last_battery_current = bat_msg->current;
                last_battery_percentage = bat_msg->percentage;
                last_temperature = bat_msg->temperature;
            }
            catch (const exception& e) {
                cout << "Error processing battery: " << e.what() << endl;
            }
        }
    }
    //close CSV
    csv_file.close();

    //end timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Print summary
    cout << "\n<--- Unified CSV Conversion Summary --->" << endl;
    cout << "Total messages processed: " << messages_tot << endl;
    cout << "  Command velocity (primary): " << cmd_vel_messages << endl;
    cout << "  Odometry: " << odom_messages << endl;
    cout << "  IMU: " << imu_messages << endl;
    cout << "  Joint States: " << js_messages << endl;
    cout << "  Tracking Error: " << err_messages << endl;
    cout << "  Battery: " << bat_messages << endl;
    cout << "\nCSV rows written: " << rows << endl;
    cout << "Time taken: " << duration.count() << " ms" << endl;

    return rows;
}

static bool ros_initialized = false;

void initializeROS(int argc, char** argv) {
    if (!ros_initialized) {
        rclcpp::init(argc, argv);
        ros_initialized = true;
    }
}

void shutdownROS() {
    if (ros_initialized) {
        rclcpp::shutdown();
        ros_initialized = false;
    }
}