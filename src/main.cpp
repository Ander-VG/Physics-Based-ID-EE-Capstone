#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>
#include <filesystem>

#include "data_transition.h"
#include "physics_check.h"

using namespace std;
using namespace std::chrono;
namespace fs = std::filesystem;

void printUsage(const char* program_name) {
    cout << "╔════════════════════════════════════════════════════════════════╗" << endl;
    cout << "║         TurtleBot3 Physics Based IDS - Unified Pipeline        ║" << endl;
    cout << "╚════════════════════════════════════════════════════════════════╝" << endl;
    cout << "\nUsage:" << endl;
    cout << "  " << program_name << " <bag_file> <output_dir> [experiment_name]" << endl;
    cout << "\nArguments:" << endl;
    cout << "  bag_file         : Path to ROS2 bag file (.db3)" << endl;
    cout << "  output_dir       : Directory for CSV outputs" << endl;
    cout << "  experiment_name  : Prefix for output files (optional, default: 'experiment')" << endl;
    cout << "\nExample:" << endl;
    cout << "  " << program_name << " ~/data/benign/rosbag1/rosbag2_0.db3 ./results benign_exp1" << endl;
    cout << "\nOutput:" << endl;
    cout << "  - CSV files: odom, cmd_vel, imu, js, error, battery, sensor" << endl;
    cout << "  - Intrusion detection report" << endl;
    cout << "════════════════════════════════════════════════════════════════\n" << endl;
}

int main(int argc, char** argv) {
    // Start total timer
    auto total_start = high_resolution_clock::now();
    
    // Parse command line arguments
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }
    
    string bag_path = argv[1];
    string output_dir = argv[2];
    string experiment_name = (argc >= 4) ? argv[3] : "experiment";
    
    // Verify bag file exists
    if (!fs::exists(bag_path)) {
        cerr << "Error: Bag file not found: " << bag_path << endl;
        return 1;
    }
    
    // Create output directory if it doesn't exist
    try {
        fs::create_directories(output_dir);
    } catch (const exception& e) {
        cerr << "Error: Could not create output directory: " << e.what() << endl;
        return 1;
    }
    
    // Define output file paths
    string odom_csv = output_dir + "/" + experiment_name + "_odom.csv";
    string cmd_csv = output_dir + "/" + experiment_name + "_cmd_vel.csv";
    string imu_csv = output_dir + "/" + experiment_name + "_imu.csv";
    string js_csv = output_dir + "/" + experiment_name + "_js.csv";
    string error_csv = output_dir + "/" + experiment_name + "_error.csv";
    string battery_csv = output_dir + "/" + experiment_name + "_battery.csv";
    string sensor_csv = output_dir + "/" + experiment_name + "_sensor.csv";
    
    cout << "\n╔════════════════════════════════════════════════════════════════╗" << endl;
    cout << "║        TurtleBot3 Intrusion Detection Pipeline                 ║" << endl;
    cout << "╚════════════════════════════════════════════════════════════════╝" << endl;
    cout << "\n📦 Input:  " << bag_path << endl;
    cout << "📁 Output: " << output_dir << endl;
    cout << "🏷️  Name:   " << experiment_name << endl;
    
    // ========================================================================
    // STEP 1: Convert ROS Bag to CSV Files
    // ========================================================================
    cout << "\n" << string(64, '=') << endl;
    cout << "STEP 1: Converting ROS Bag to CSV Files" << endl;
    cout << string(64, '=') << endl;
    
    auto step1_start = high_resolution_clock::now();
    
    // Initialize ROS2
    initializeROS(argc, argv);
    
    // Convert bag to CSV
    int messages_converted = BagtoCSV(
        bag_path,
        odom_csv,
        cmd_csv,
        imu_csv,
        js_csv,
        error_csv,
        battery_csv,
        sensor_csv
    );
    
    // Shutdown ROS2
    shutdownROS();
    
    auto step1_end = high_resolution_clock::now();
    auto step1_duration = duration_cast<milliseconds>(step1_end - step1_start);
    
    if (messages_converted < 0) {
        cerr << "\n STEP 1 FAILED: Bag conversion error" << endl;
        return 1;
    }
    
    cout << "\n STEP 1 COMPLETE" << endl;
    cout << "   Messages converted: " << messages_converted << endl;
    cout << "   Time elapsed: " << step1_duration.count() / 1000.0 << " seconds" << endl;
    
    // ========================================================================
    // STEP 2: Run Intrusion Detection Analysis
    // ========================================================================
    cout << "\n" << string(64, '=') << endl;
    cout << "STEP 2: Running Physics-Based Intrusion Detection" << endl;
    cout << string(64, '=') << endl;
    
    auto step2_start = high_resolution_clock::now();
    
    int detection_result = runIntrusionDetection(
        odom_csv,
        cmd_csv,
        imu_csv,
        js_csv,
        error_csv,
        battery_csv
    );
    
    auto step2_end = high_resolution_clock::now();
    auto step2_duration = duration_cast<milliseconds>(step2_end - step2_start);
    
    if (detection_result != 0) {
        cerr << "\n STEP 2 FAILED: Intrusion detection error" << endl;
        return 1;
    }
    
    cout << "\n STEP 2 COMPLETE" << endl;
    cout << "   Time elapsed: " << step2_duration.count() / 1000.0 << " seconds" << endl;
    
    // ========================================================================
    // FINAL SUMMARY
    // ========================================================================
    auto total_end = high_resolution_clock::now();
    auto total_duration = duration_cast<milliseconds>(total_end - total_start);
    
    cout << "\n" << string(64, '=') << endl;
    cout << " PIPELINE COMPLETE!" << endl;
    cout << string(64, '=') << endl;
    cout << " Summary:" << endl;
    cout << "   Step 1 (Conversion):  " << fixed << setprecision(2) 
         << step1_duration.count() / 1000.0 << " seconds" << endl;
    cout << "   Step 2 (Detection):   " << fixed << setprecision(2)
         << step2_duration.count() / 1000.0 << " seconds" << endl;
    cout << "   ─────────────────────────────────" << endl;
    cout << "   Total Runtime:        " << fixed << setprecision(2)
         << total_duration.count() / 1000.0 << " seconds" << endl;
    cout << "\n All outputs saved to: " << output_dir << endl;
    cout << string(64, '=') << "\n" << endl;
    
    return 0;
}