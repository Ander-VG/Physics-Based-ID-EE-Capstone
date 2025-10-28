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
#include "turtlebot3_msgs/msg/sensor_state.hpp"

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

    //find each topic
    for (const auto& topic : topics_and_types) {
        cout << "Topic: " << topic.name << endl;
        cout << "Type: " << topic.type << endl;
    }
}

//Creating a specific CSV file for odometry
bool createOdomCSV(const string& filename, ofstream& csv_file)
{
    cout << "Creating odometry CSV" << endl;
    //Create a file
    csv_file.open(filename);
    //check
    if (!csv_file.is_open()){
        cout << "Error creating CSV" << endl;
        return false;
    }
    //Column headers
    csv_file << "Time,";      // When was data generated (ns)
    csv_file << "X,";         // X position
    csv_file << "Y,";         // Y position
    csv_file << "Theta";      // Orientation angle (yaw)
    csv_file << endl;
    cout << "CSV for Odometry created" << endl;
    return true;
}

//Creating a specific CSV file for command velocity 
bool create_CmdCSV(const string& filename, ofstream& csv_file)
{
    cout << "Creating velocity CSV" << endl;
    //Create a file
    csv_file.open(filename);

    //check
    if (!csv_file.is_open()){
        cout << "Error creating CSV" << endl;
        return false;
    }

    //Column headers
    csv_file << "Time,";  // When was data generated (ns)                     
    csv_file << "linear_x,";     // Linear velocity
    csv_file << "angular_z";    // Angular velocity
    csv_file << endl;

    cout << "CSV for Command Velocity created" << endl;
    return true;
}

//Creating a specific CSV file for IMU
bool create_imuCSV(const string& filename, ofstream& csv_file)
{
    cout << "Creating IMU CSV" << endl;
    //Create a file
    csv_file.open(filename);

    //check
    if (!csv_file.is_open()){
        cout << "Error creating CSV" << endl;
        return false;
    }

    //Column headers
    csv_file << "Time,";  // When was data generated (ns)
    csv_file << "imu_accel_x,imu_accel_y,imu_accel_z,"; // IMUs measured by accelerometer
    csv_file << "imu_gyro_x,imu_gyro_y,imu_gyro_z";  // IMUs measured by gyroscope
    csv_file << endl;

    cout << "CSV for IMU created" << endl;
    return true;
}

//Creating a specific CSV file for Joint State
bool create_JSCSV(const string& filename, ofstream& csv_file)
{
    cout << "Creating Joint State CSV" << endl;
    //Create a file
    csv_file.open(filename);

    //check
    if (!csv_file.is_open()){
        cout << "Error creating CSV" << endl;
        return false;
    }

    //Column headers
    csv_file << "Time,";  // When was data generated (ns)
    csv_file << "v_R,v_L,";  // Right and Left wheel velocities
    csv_file << "right_encoder,left_encoder";  // Left and Right wheel encoders
    csv_file << endl;

    cout << "CSV for Joint State created" << endl;
    return true;
}

//Creating a specific CSV file for Error
bool create_ErrorCSV(const string& filename, ofstream& csv_file)
{
    cout << "Creating Error CSV" << endl;
    //Create a file
    csv_file.open(filename);

    //check
    if (!csv_file.is_open()){
        cout << "Error creating CSV" << endl;
        return false;
    }

    //Column headers
    csv_file << "Time,";  // When was data generated (ns)
    csv_file << "tracking_error,error_x,error_y";  // Error in sensors
    csv_file << endl;

    cout << "CSV for Error created" << endl;
    return true;
}

//Creating a specific CSV file for battery state
bool create_batCSV(const string& filename, ofstream& csv_file)
{
    cout << "Creating Battery CSV" << endl;
    //Create a file
    csv_file.open(filename);

    //check
    if (!csv_file.is_open()){
        cout << "Error creating CSV" << endl;
        return false;
    }

    //Column headers
    csv_file << "Time,";  // When was data generated (ns)
    csv_file << "battery_voltage,battery_current,battery_percentage,";//Battery states
    csv_file << "temperature";
    csv_file << endl;

    cout << "CSV for Battery State created" << endl;
    return true;
}

bool create_SSCSV(const string& filename, ofstream& csv_file)
{
    cout << "Creating Sensor State CSV" << endl;
    //create a file
    csv_file.open(filename);

    //check
    if (!csv_file.is_open()){
        cout << "Error creating CSV" << endl;
        return false;
    }

    //Column headers
    csv_file << "Time,"; // When was data generated (ns)
    csv_file << "right_encoder,left_encoder,"; // encoders
    csv_file << "torque";

    csv_file << endl;

    cout << "CSV for Sensor State created" << endl;
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

//Identifying if the topic is Sensor State
bool SSId(const string& topic_name)
{
    return topic_name.find("sensor_state") != string::npos;
}

//Breaking down timestamp for odometry
bool Deserializing_Odom(shared_ptr<SerializedBagMessage> bag_message, ofstream& csv_file, double& first_timestamp) {
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
        
        //header timestamp extraction - convert to seconds
        uint64_t timestamp_sec = odom_msg->header.stamp.sec;
        uint64_t timestamp_nanosec = odom_msg->header.stamp.nanosec;
        double time_in_seconds = timestamp_sec + (timestamp_nanosec / 1000000000.0);
        
        //set first timestamp if not set (first message)
        if (first_timestamp < 0) {
            first_timestamp = time_in_seconds;
        }
        
        //calculate relative time
        double relative_time = time_in_seconds - first_timestamp;
        
        //extract X and Y position
        double x = odom_msg->pose.pose.position.x;
        double y = odom_msg->pose.pose.position.y;
        
        //convert quaternion to yaw angle (theta)
        double qx = odom_msg->pose.pose.orientation.x;
        double qy = odom_msg->pose.pose.orientation.y;
        double qz = odom_msg->pose.pose.orientation.z;
        double qw = odom_msg->pose.pose.orientation.w;
        
        //calculate yaw (theta) from quaternion
        double theta = atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
        
        //writing to CSV
        csv_file << fixed << setprecision(12);
        csv_file << relative_time << ",";
        csv_file << x << ","; //X position
        csv_file << y << ","; //Y position
        csv_file << theta; //Orientation angle 
        csv_file << endl;
        return true;
    }
    catch (const exception& e) {
        cout << "Error deserializing odometry message: " << e.what() << endl;
        return false;
    }
}

//Breaking down timestamp for command velocity
bool Deserializing_Cmd(shared_ptr<SerializedBagMessage> bag_message, ofstream& csv_file, double& first_timestamp) {
    // Deserialize the message
    Serialization<geometry_msgs::msg::Twist> deserializer;
    //binary to text
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

        //writing to CSV
        csv_file << fixed << setprecision(9);
        csv_file << relative_time << ",";

        csv_file << cmd_vel_msg->linear.x << ","; //v_Cmd
        csv_file << cmd_vel_msg->angular.z; //w_Cmd
        csv_file << endl;
        return true;
    }

    catch (const exception& e) {
        cout << "Error deserializing command velocity message: " << e.what() << endl;
        return false;
    }
}

//Breaking down timestamp for IMU
bool Deserializing_IMU(shared_ptr<SerializedBagMessage> bag_message, ofstream& csv_file, double& first_timestamp) 
{
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

        //header timestamp extraction - convert to seconds
        uint64_t timestamp_sec = imu_msg->header.stamp.sec;
        uint64_t timestamp_nanosec = imu_msg->header.stamp.nanosec;
        double time_in_seconds = timestamp_sec + (timestamp_nanosec / 1000000000.0);
        
        //set first timestamp if not set (first message)
        if (first_timestamp < 0) {
            first_timestamp = time_in_seconds;
        }
    
        //calculate relative time
        double relative_time = time_in_seconds - first_timestamp;

        csv_file << fixed << setprecision(9);
        csv_file << relative_time << ",";

        //self describing features
        csv_file << imu_msg->linear_acceleration.x << ","; 
        csv_file << imu_msg->linear_acceleration.y << ",";
        csv_file << imu_msg->linear_acceleration.z << ",";

        csv_file << imu_msg->angular_velocity.x << ",";
        csv_file << imu_msg->angular_velocity.y << ",";
        csv_file << imu_msg->angular_velocity.z;

        csv_file << endl;
        return true;
    }

    catch (const exception& e) {
        cout << "Error deserializing IMU message: " << e.what() << endl;
        return false;
    }
}

//Breaking down timestamp for joint states
bool Deserializing_JS(shared_ptr<SerializedBagMessage> bag_message, ofstream& csv_file, double& first_timestamp) {
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

        //header timestamp extraction - convert to seconds
        uint64_t timestamp_sec = js_msg->header.stamp.sec;
        uint64_t timestamp_ns = js_msg->header.stamp.nanosec;
        double time_in_seconds = timestamp_sec + (timestamp_ns / 1000000000.0);
        
        //set first timestamp if not set (first message)
        if (first_timestamp < 0) {
            first_timestamp = time_in_seconds;
        }
        
        //calculate relative time
        double relative_time = time_in_seconds - first_timestamp;

        csv_file << fixed << setprecision(9);
        csv_file << relative_time << ",";

        if (js_msg->velocity.size() < 2 || js_msg->position.size() < 2) {
            cout << "Error: insufficient joint_state data" << endl;
            return false;
        }
        else {
            csv_file << js_msg->velocity[1] << ","; //Right wheel velocity
            csv_file << js_msg->velocity[0] << ","; //Left wheel velocity
            csv_file << js_msg->position[1] << ","; //Right wheel encoder
            csv_file << js_msg->position[0];        //Left wheel encoder
            csv_file << endl;
        
            return true;
        }
    }

    catch (const exception& e) {
        cout << "Error deserializing Joint States message: " << e.what() << endl;
        return false;
    }
}

//Breaking down timestamp for error
bool Deserializing_Error(shared_ptr<SerializedBagMessage> bag_message, ofstream& csv_file, double& first_timestamp) 
{
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

        //header timestamp extractio`n - convert to seconds
        uint64_t timestamp_ns = bag_message->time_stamp;
        double time_in_seconds = timestamp_ns / 1000000000.0;
        
        //set first timestamp if not set (first message)
        if (first_timestamp < 0) {
            first_timestamp = time_in_seconds;
        }
        
        //calculate relative time
        double relative_time = time_in_seconds - first_timestamp;

        csv_file << fixed << setprecision(9);
        csv_file << relative_time << ",";
        
        if (err_msg->data.size() >= 3) {
            csv_file << err_msg->data[0] << ",";  // tracking_error
            csv_file << err_msg->data[1] << ",";  // error_x
            csv_file << err_msg->data[2];         // error_y
        } else {
            csv_file << "0,0,0";
        }

        csv_file << endl;
        return true;
    }

    catch (const exception& e) {
        cout << "Error deserializing Tracking Error message: " << e.what() << endl;
        return false;
    }
}

//Finding Battery state values
bool Deserializing_Bat(shared_ptr<SerializedBagMessage> bag_message, ofstream& csv_file, double& first_timestamp) {
    // Deserialize the message
    Serialization<sensor_msgs::msg::BatteryState> deserializer;
    //binary to text
    try {

        //getting binary data
        rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);

        //empty joint state message
        auto bat_msg = make_shared<sensor_msgs::msg::BatteryState>();

        //adding bag data
        deserializer.deserialize_message(&serialized_msg, bat_msg.get());

        //header timestamp extraction - convert to seconds
        uint64_t timestamp_sec = bat_msg->header.stamp.sec;
        uint64_t timestamp_ns = bat_msg->header.stamp.nanosec;
        double time_in_seconds = timestamp_sec + (timestamp_ns / 1000000000.0);
        
        //set first timestamp if not set (first message)
        if (first_timestamp < 0) {
            first_timestamp = time_in_seconds;
        }
        
        //calculate relative time
        double relative_time = time_in_seconds - first_timestamp;

        csv_file << fixed << setprecision(9);
        csv_file << relative_time << ",";
        //self describing features
        csv_file << bat_msg->voltage << ","; 
        csv_file << bat_msg->current << ","; 
        csv_file << bat_msg->percentage << ",";
        csv_file << bat_msg->temperature;

        csv_file << endl;
        return true;
    }

    catch (const exception& e) {
        cout << "Error deserializing Battery State message: " << e.what() << endl;
        return false;
    }
}

//Finding Sensor State values
bool Deserializing_SS(shared_ptr<SerializedBagMessage> bag_message, ofstream& csv_file, double& first_timestamp) {
    // Deserialize the message
    Serialization<turtlebot3_msgs::msg::SensorState> deserializer;
    //binary to text
    try {

        //getting binary data
        rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);

        //empty joint state message
        auto ss_msg = make_shared<turtlebot3_msgs::msg::SensorState>();

        //adding bag data
        deserializer.deserialize_message(&serialized_msg, ss_msg.get());

        //header timestamp extraction - convert to seconds
        uint64_t timestamp_sec = ss_msg->header.stamp.sec;
        uint64_t timestamp_ns = ss_msg->header.stamp.nanosec;
        double time_in_seconds = timestamp_sec + (timestamp_ns / 1000000000.0);
        
        //set first timestamp if not set (first message)
        if (first_timestamp < 0) {
            first_timestamp = time_in_seconds;
        }
        
        //calculate relative time
        double relative_time = time_in_seconds - first_timestamp;

        csv_file << fixed << setprecision(9);
        csv_file << relative_time << ",";
        //self describing features
        csv_file << ss_msg-> right_encoder << ",";
        csv_file << ss_msg-> left_encoder << ",";
        csv_file << ss_msg-> torque;

        csv_file << endl;
        return true;
    }

    catch (const exception& e) {
        cout << "Error deserializing Sensor States message: " << e.what() << endl;
        return false;
    }
}

int BagtoCSV (const string& bag_path, 
    const string& odom_file, 
    const string& cmd_file,
    const string& imu_file,
    const string& js_file,
    const string& err_file,
    const string& bat_file,
    const string& ss_file)
{
    int successful_inputs = 0;
    int total_messages = 0;

    cout << "\n<=== ROS BAG TO CSV CONVERSION ===>" << endl;
    cout << "ROS Bag: " << bag_path << endl;

    //to measure latency (need a better method but that comes later)
    auto start_time = std::chrono::high_resolution_clock::now();
    unique_ptr<Reader> reader;

    //attempt to open bag
    if (!openBag(bag_path, reader)) {
        cout << "Bag was not able to open" << endl;
        return 0;
    }

    //get bag values
    showBag(reader.get());


    ofstream odom_csv, cmd_csv, imu_csv, js_csv, err_csv, bat_csv, ss_csv;
    if (!createOdomCSV(odom_file, odom_csv) ||
        !create_CmdCSV(cmd_file, cmd_csv) ||
        !create_imuCSV(imu_file, imu_csv) ||
        !create_JSCSV(js_file, js_csv) ||
        !create_ErrorCSV(err_file, err_csv) ||
        !create_batCSV(bat_file, bat_csv) ||
        !create_SSCSV(ss_file, ss_csv)) {
        cout << "Could not create at least one CSV file" << endl;
        return 0;
    }

    int odom_messages = 0;
    int odom_in = 0;
    int cmd_vel_messages = 0;
    int cmd_in = 0;
    int imu_messages = 0;
    int imu_in = 0;
    int js_messages = 0;
    int js_in = 0;
    int err_messages = 0;
    int err_in = 0;
    int bat_messages = 0;
    int bat_in = 0;
    int ss_messages = 0;
    int ss_in = 0;
    
    double first_timestamp = -1.0; 
    
    while (reader->has_next())
    {
        auto bag_message = reader->read_next();
        total_messages++;

        const string& t = bag_message->topic_name;

        if (OdomId(t)) {
            odom_messages++;
            if (Deserializing_Odom(bag_message, odom_csv, first_timestamp)) { odom_in++; successful_inputs++; }
        }
        else if (CmdId(t)) {
            cmd_vel_messages++;
            if (Deserializing_Cmd(bag_message, cmd_csv, first_timestamp)) { cmd_in++; successful_inputs++; }
        }
        else if (IMUId(t)) {
            imu_messages++;
            if (Deserializing_IMU(bag_message, imu_csv, first_timestamp)) { imu_in++; successful_inputs++; }
        }
        else if (JSId(t)) {
            js_messages++;
            if (Deserializing_JS(bag_message, js_csv, first_timestamp)) { js_in++; successful_inputs++; }
        }
        else if (ErrorId(t)) {
            err_messages++;
            if (Deserializing_Error(bag_message, err_csv, first_timestamp)) { err_in++; successful_inputs++; }
        }
        else if (BatId(t)) {
            bat_messages++;
            if (Deserializing_Bat(bag_message, bat_csv, first_timestamp)) { bat_in++; successful_inputs++; }
        }
        else if (SSId(t)) {
            ss_messages++;
            if (Deserializing_SS(bag_message, ss_csv, first_timestamp)) { ss_in++; successful_inputs++; }
        }
    }

    // Close all files
    odom_csv.close();
    cmd_csv.close();
    imu_csv.close();
    js_csv.close();
    err_csv.close();
    bat_csv.close();
    ss_csv.close();

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Print summary
    cout << "\n<--- Odometry Conversion Summary --->" << endl;
    cout << "Odometry messages found: " << odom_messages << endl;
    cout << "Messages converted to CSV: " << odom_in << endl;
    cout << "\n<--- Command Velocity Conversion Summary --->" << endl;
    cout << "Command velocity messages found: " << cmd_vel_messages << endl;
    cout << "Messages converted to CSV: " << cmd_in << endl;
    cout << "\n<--- IMU Conversion Summary --->" << endl;
    cout << "IMU messages found: " << imu_messages << endl;
    cout << "Messages converted to CSV: " << imu_in << endl;
    cout << "\n<--- Joint State Conversion Summary --->" << endl;
    cout << "Joint State messages found: " << js_messages << endl;
    cout << "Messages converted to CSV: " << js_in << endl;
    cout << "\n<--- Tracking Error Conversion Summary --->" << endl;
    cout << "Tracking Error messages found: " << err_messages << endl;
    cout << "Messages converted to CSV: " << err_in << endl;
    cout << "\n<--- Battery State Conversion Summary --->" << endl;
    cout << "Battery State messages found: " << bat_messages << endl;
    cout << "Messages converted to CSV: " << bat_in << endl;
    cout << "\n<--- Sensor State Conversion Summary --->" << endl;
    cout << "Sensor State messages found: " << ss_messages << endl;
    cout << "Messages converted to CSV: " << ss_in << endl;

    return successful_inputs;
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