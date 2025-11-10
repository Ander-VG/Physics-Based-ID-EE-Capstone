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
  
int main(int argc, char** argv) {
    // Start total timer

    cout << "╔════════════════════════════════════════════════════════════════╗" << endl;
    cout << "║                  TurtleBot3 Physics Based IDS                  ║" << endl;
    cout << "╚════════════════════════════════════════════════════════════════╝" << endl;

    // Prevent segfault if launched incorrectly
    if (argc < 3) {
        cerr << "ERROR: Requires <bag_path> <output_dir>" << endl;
        return 1;
    }
    
    auto total_start = high_resolution_clock::now();
    
    string bag_path = argv[1];
    string output_dir = argv[2];
    
    // Verify bag file exists
    if (!fs::exists(bag_path)) {
        cerr << "Error: Bag file not found: " << bag_path << endl;
        return 1;
    }
    
    // Create output directory if it doesn't exist
    try {
        fs::create_directories(output_dir);
    } 
    catch (const exception& e) {
        cerr << "Error: Could not create output directory: " << e.what() << endl;
        return 1;
    }
    
    // Define output file paths
    string ROSdata_csv = output_dir + "/ROSdata.csv";

    cout << "\nInput: " << bag_path << endl;
    cout << "Output: " << output_dir << endl;

    cout << "\nFeatures: " << endl;
    cout << "- Odom" << endl;
    cout << "- Command Velocities" << endl;
    cout << "- Inertial Measurement Unit" << endl;
    cout << "- Joint States" << endl;
    cout << "- Tracking Error" << endl;
    cout << "- Battery Status" << endl;
    cout << endl;
    
    // STEP 1: Convert ROS Bag to CSV Files

    cout << "\nSTEP 1: Converting ROS Bag to CSV Files" << endl;
    
    auto step1_start = high_resolution_clock::now();
    
    // Initialize ROS2
    initializeROS(argc, argv);
    
    // Convert bag to CSV
    int rows_written = BagtoCSV(bag_path, ROSdata_csv);
    
    // Shutdown ROS2
    shutdownROS();
    
    auto step1_end = high_resolution_clock::now();
    auto step1_duration = duration_cast<milliseconds>(step1_end - step1_start);
    
    if (rows_written < 0) {
        cerr << "\n STEP 1 FAILED: Bag conversion error" << endl;
        return 1;
    }
    
    cout << "\n STEP 1 COMPLETE" << endl;
    cout << "   Messages converted: " << rows_written << endl;
    cout << "   Time elapsed: " << step1_duration.count() / 1000.0 << " seconds" << endl;
    
    // STEP 2: Run Intrusion Detection Analysis
    cout << "STEP 2: Running Physics-Based Intrusion Detection" << endl;
    
    auto step2_start = high_resolution_clock::now();
    
    int detection_result = runIntrusionDetection(ROSdata_csv);
    
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
    
    cout << " Summary:" << endl;
    cout << "   Step 1 (Conversion):  " << fixed << setprecision(2) 
         << step1_duration.count() / 1000.0 << " seconds" << endl;
    cout << "   Step 2 (Detection):   " << fixed << setprecision(2)
         << step2_duration.count() / 1000.0 << " seconds" << endl;
    cout << "   ─────────────────────────────────" << endl;
    cout << "   Total Runtime:        " << fixed << setprecision(2)
         << total_duration.count() / 1000.0 << " seconds" << endl;
    
    return 0;
    
}