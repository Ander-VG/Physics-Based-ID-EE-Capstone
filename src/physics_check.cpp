#include "physics_check.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <iomanip>
#include <limits>
#include <chrono>
#include <algorithm>

#define M_PI 3.14159265358979323846

// Using declarations for all std functions
using std::string;
using std::vector;
using std::map;
using std::cout;
using std::cerr;
using std::endl;
using std::ifstream;
using std::stringstream;
using std::getline;
using std::sqrt;
using std::stod;
using std::to_string;
using std::min;
using std::max;
using std::setprecision;
using std::fixed;
using std::numeric_limits;
using std::exception;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::microseconds;

//To represent the sections that will need to change once CSVs stop being used and instead the program connects directly 
//to the Kubernetes cluster, there will be 2 classifications: WNM (will need modification) and OK (will work as is)

//ok
//Odometry data class
struct OdomData {
    double timestamp;
    double x, y, theta;
    
    OdomData() : timestamp(0), x(0), y(0), theta(0) {}
};

//ok
//Command Velocity data structure
struct CmdVelData {
    double timestamp;
    double linear_x, angular_z;
    
    CmdVelData() : timestamp(0), linear_x(0), angular_z(0) {}
};

//ok
//Inertial Measurement Unit data structure
struct ImuData {
    double timestamp;
    double accel_x, accel_y, accel_z;
    double gyro_x, gyro_y, gyro_z;
    
    ImuData() : timestamp(0), accel_x(0), accel_y(0), accel_z(0),
                gyro_x(0), gyro_y(0), gyro_z(0) {}
};

//ok
//Motors data structure
struct JointStatesData {
    double timestamp;
    double v_R, v_L;  // Right and left wheel velocities
    double right_encoder, left_encoder;
    
    JointStatesData() : timestamp(0), v_R(0), v_L(0), right_encoder(0), left_encoder(0) {}
};

//ok
//Tracking Error data structure
struct ErrorData {
    double timestamp;
    double tracking_error;
    double error_x, error_y;
    
    ErrorData() : timestamp(0), tracking_error(0), error_x(0), error_y(0) {}
};

//ok
//Battery data structure
struct BatteryData {
    double timestamp;
    double voltage, current, percentage;
    
    BatteryData() : timestamp(0), voltage(0), current(0), percentage(0) {}
};

//ok
//Sensor State data structure (might be where LiDAR + Sonar data goes eventually)
struct SensorData {
    double timestamp;
    double right_encoder, left_encoder, torque;
    
    SensorData() : timestamp(0), right_encoder(0), left_encoder(0), torque(0) {}
};

//ok
// self describing
struct Anomaly {
    double timestamp;
    string type;
    string description;
    double severity; // 0-1 scale
};

//ok
//Provides a general structure for physics validators
class PhysicsValidator {
public:
    virtual ~PhysicsValidator() {}
    virtual vector<Anomaly> validate() = 0;
    virtual string getName() const = 0;
};

//WNM: It iterates because it assumes data is in vectors, which will not be the case when connected to Kubernetes
//Checking command velocity with Odometry and Battery
class CmdVelOdomValidator : public PhysicsValidator {
private:
    vector<OdomData>& odom_data; //extracted odom data
    vector<CmdVelData>& cmd_data; //extracted cmd_vel data
    vector<BatteryData>& battery_data; //extracted battery data
    const double MAX_LINEAR_VEL = 0.22;  // m/s
    const double MAX_ANGULAR_VEL = 2.84; // rad/s
    const double TIME_TOLERANCE = 0.2;   // seconds
    const double VEL_TOLERANCE = 0.1;   // m/s tolerance
    double prev_cmd_angular = 0;
    double prev_cmd_linear = 0;
    int transient_skip_counter;
    const double CONTROL_LAG = 0.15;  // 150ms control lag
    int consecutive_angular_violations = 0;  // Track persistence
    int consecutive_linear_violations = 0;
    const int VIOLATION_THRESHOLD = 3;  // Must persist 3 samples
    //POSSIBLE UPGRADES = Make Violation Threshold adjustable and control lag time adjustable later


    //Find closest timestamps for cmd_vel & battery to match with odom
    CmdVelData* findClosestCmd(double timestamp) {
        CmdVelData* closest = nullptr;
        double min_diff = numeric_limits<double>::max();
        for (auto& cmd : cmd_data) {
            double diff = std::abs(cmd.timestamp - timestamp);
            if (diff < min_diff) {
                min_diff = diff;
                closest = &cmd;
            }
        }
        if (min_diff > TIME_TOLERANCE) {
            return nullptr;
        }
        return closest;
    }
  
    BatteryData* findClosestBattery(double timestamp) {
        if (battery_data.empty()) return nullptr;
        BatteryData* closest = nullptr;
        double min_diff = numeric_limits<double>::max();
        for (auto& battery : battery_data) {
            double diff = std::abs(battery.timestamp - timestamp);
            if (diff < min_diff) {
                min_diff = diff;
                closest = &battery;
            }
        }
        if (min_diff > TIME_TOLERANCE) {
            return nullptr;
        }
        return closest;
    }
  
//getAvgCmd is meant to provide the correct commanded velocity considering control lag
//It does this for angular and linear velocities separately

    struct WindowAvg {
        double avg;
        int count;
    };

    WindowAvg getAvgAngularCmd(double start_time, double end_time) {
        double sum = 0;
        int count = 0;
        
        // Account for control lag by shifting time window backward
        double lag_adjusted_start = start_time - CONTROL_LAG;
        double lag_adjusted_end = end_time - CONTROL_LAG;
        
        for (auto& cmd : cmd_data) {
            if (cmd.timestamp >= lag_adjusted_start && 
                cmd.timestamp <= lag_adjusted_end) {
                sum += cmd.angular_z; // Use absolute value
                count++;
            }
        }
        
        return {(count > 0) ? sum / count : 0.0, count};
    }
    
    WindowAvg getAvgLinearCmd(double start_time, double end_time) {
        double sum = 0;
        int count = 0;
        
        double lag_adjusted_start = start_time - CONTROL_LAG;
        double lag_adjusted_end = end_time - CONTROL_LAG;
        
        for (auto& cmd : cmd_data) {
            if (cmd.timestamp >= lag_adjusted_start && 
                cmd.timestamp <= lag_adjusted_end) {
                sum += cmd.linear_x;  // Use absolute value
                count++;
            }
        }
        
        return {(count > 0) ? sum / count : 0.0, count};
    }

  //determines the state of the battery to implement an appropriate angular tolerance
    double getAdaptiveAngularTolerance(double timestamp) {
        BatteryData* battery = findClosestBattery(timestamp);
        if (!battery) return 0.20;  // Default if no battery data
        
        // Voltage-based adaptive thresholds
        double voltage = battery->voltage;
        if (voltage > 12.0) {
            return 0.10;  // Full power - strict checking
        } else if (voltage > 11.5) {
            return 0.15;  // Slight degradation
        } else if (voltage > 11.0) {
            return 0.25;  // Moderate degradation
        } else if (voltage > 10.5) {
            return 0.35;  // Significant degradation
        } else {
            return 0.45;  // Critical battery - very relaxed
        }
    }

public:
    CmdVelOdomValidator(vector<OdomData>& odom, vector<CmdVelData>& cmd, vector<BatteryData>& battery)
        : odom_data(odom), cmd_data(cmd), battery_data(battery){}
    
    string getName() const override {
        return "CMD_VEL vs ODOM Validator";
    }

    vector<Anomaly> validate() override {
        vector<Anomaly> anomalies;
        const double WARMUP_TIME = 10.0;
        const double MIN_SIGNIFICANT_VEL = 0.05;  // FIX: Added definition
        int lb_warnings = 0;
        bool battery_context = false;
        WindowAvg prev_avg_cmd_angular = {0.0, 0};
        WindowAvg prev_avg_cmd_linear = {0.0, 0};
        int transient_skip_counter = 0;
        const int MAX_TRANSIENT_SAMPLES = 40;  // Max 2 seconds of transient
        const int MIN_TRANSIENT_SAMPLES = 10;  // Min 0.5 seconds
        const double CMD_CHANGE_THRESHOLD_ANG = 0.5;  // rad/s
        const double CMD_CHANGE_THRESHOLD_LIN = 0.1;  // m/s
        const double SETTLING_THRESHOLD_ANG = 0.3;    // rad/s - velocity must be within this to exit transient
        const double SETTLING_THRESHOLD_LIN = 0.05;   // m/s

        for (size_t i = 1; i < odom_data.size(); ++i) {
            OdomData& curr = odom_data[i];
            OdomData& prev = odom_data[i-1];
            
            if (curr.timestamp < WARMUP_TIME) continue;
            
            double dt = curr.timestamp - prev.timestamp;
            if (dt <= 0 || dt > 1.0) continue;
            
            // Calculate actual velocities from odometry and adjust for direction
            double dx = curr.x - prev.x;
            double dy = curr.y - prev.y;
        
            double dtheta = curr.theta - prev.theta;
            while (dtheta > M_PI) dtheta -= 2*M_PI;
            while (dtheta < -M_PI) dtheta += 2*M_PI;
            double actual_angular = dtheta / dt;
            double avg_theta = prev.theta + (dtheta / 2.0);

            double actual_linear = (dx * cos(avg_theta)) + (dy * sin(avg_theta))/dt;
            double actual_linear_magnitude = sqrt(dx*dx + dy*dy) / dt;
            double actual_linear_signed = (actual_linear >= 0.0) ? actual_linear_magnitude : -actual_linear_magnitude;
            
            BatteryData* battery = findClosestBattery(curr.timestamp);
            
            //Checks if it can provide battery context if it hasn't done so already
            if (battery && !battery_context){
                cout << " Voltage at " << battery->voltage << "V ("
                << battery->percentage << " %)" << endl;
                battery_context = true;
            }

            auto avg_cmd_angular = getAvgAngularCmd(prev.timestamp, curr.timestamp);
            auto avg_cmd_linear = getAvgLinearCmd(prev.timestamp, curr.timestamp);
            
            // Skip if no commands found in the time window
            if (avg_cmd_angular.count == 0 && avg_cmd_linear.count == 0) {
                continue;
            }

            //<========== TRANSIENT DETECTION WITH SETTLING ==========>
            // Detect large command changes indicating acceleration/deceleration
            double cmd_angular_change = std::abs(avg_cmd_angular.avg - prev_avg_cmd_angular.avg);
            double cmd_linear_change = std::abs(avg_cmd_linear.avg - prev_avg_cmd_linear.avg);
            
            // If command changed significantly, enter transient period
            if (cmd_angular_change > CMD_CHANGE_THRESHOLD_ANG || 
                cmd_linear_change > CMD_CHANGE_THRESHOLD_LIN) {
                transient_skip_counter = MAX_TRANSIENT_SAMPLES;
                consecutive_linear_violations = 0;
                consecutive_angular_violations = 0;
            }
            
            // During transient period, check if velocity has settled
            if (transient_skip_counter > 0) {
                transient_skip_counter--;
                
                // Calculate how close actual velocity is to commanded velocity
                double angular_error = std::abs(actual_angular - avg_cmd_angular.avg);
                double linear_error = std::abs(actual_linear - avg_cmd_linear.avg);
                
                // After minimum transient time, check if velocity has settled
                int samples_elapsed = MAX_TRANSIENT_SAMPLES - transient_skip_counter;
                
                if (samples_elapsed >= MIN_TRANSIENT_SAMPLES) {
                    // Check if both velocities are within settling threshold
                    bool angular_settled = (angular_error < SETTLING_THRESHOLD_ANG) || 
                                        (std::abs(avg_cmd_angular.avg) < 0.1);  // Or if command is small
                    bool linear_settled = (linear_error < SETTLING_THRESHOLD_LIN) || 
                                        (std::abs(avg_cmd_linear.avg) < 0.05);   // Or if command is small
                    
                    // Exit transient early if velocity has settled
                    if (angular_settled && linear_settled) {
                        transient_skip_counter = 0;  // Settled! Resume validation
                    }
                }
                
                // If still in transient, skip validation
                if (transient_skip_counter > 0) {
                    prev_avg_cmd_angular = avg_cmd_angular;
                    prev_avg_cmd_linear = avg_cmd_linear;
                    continue;
                }
            }
            
            // Update previous commands for next iteration
            prev_avg_cmd_angular = avg_cmd_angular;
            prev_avg_cmd_linear = avg_cmd_linear;
            // ======================================================
            
            // ========== LINEAR VELOCITY CHECK (STEADY-STATE ONLY) ==========
            double vel_diff = std::abs(actual_linear - avg_cmd_linear.avg);
            
            if (vel_diff > VEL_TOLERANCE) {
                consecutive_linear_violations++;
                
                if (consecutive_linear_violations >= VIOLATION_THRESHOLD) {
                    Anomaly a;
                    a.timestamp = curr.timestamp;
                    a.type = "VELOCITY_MISMATCH";
                    a.description = "Avg commanded vel: " + to_string(avg_cmd_linear.avg) +
                                " m/s, Actual vel: " + to_string(actual_linear) + 
                                " m/s (persisted " + to_string(consecutive_linear_violations) + " samples)";
                    a.severity = min(1.0, vel_diff / MAX_LINEAR_VEL);
                    
                    if (battery) {
                        a.description += " [Battery: " + to_string(battery->voltage) + "V]";
                    }
                    
                    anomalies.push_back(a);
                }
            } else {
                consecutive_linear_violations = 0;
            }
            
            // ========== ANGULAR VELOCITY CHECK (STEADY-STATE ONLY) ==========
            double ang_tolerance = getAdaptiveAngularTolerance(curr.timestamp);
            double ang_diff = std::abs(actual_angular - avg_cmd_angular.avg);
            
            if (ang_diff > ang_tolerance) {
                consecutive_angular_violations++;
                
                if (consecutive_angular_violations >= VIOLATION_THRESHOLD) {
                    Anomaly a;
                    a.timestamp = curr.timestamp;
                    a.type = "ANGULAR_MISMATCH";

                    string desc = "Avg commanded ang: " + to_string(avg_cmd_angular.avg) +
                                " rad/s, Actual ang: " + to_string(actual_angular) + 
                                " rad/s (persisted " + to_string(consecutive_angular_violations) + " samples)";
                    
                    if (battery) {
                        desc += " [Battery: " + to_string(battery->voltage) + "V (" +
                                to_string((int)battery->percentage) + "%)]";
                    }
                    
                    a.description = desc;
                    a.severity = min(1.0, ang_diff / MAX_ANGULAR_VEL);
                    anomalies.push_back(a);
                }
            } else {
                consecutive_angular_violations = 0;
            }
            
            if (battery && battery->voltage < 11.0 && lb_warnings < 3) {
                lb_warnings++;
                cout << "  LOW BATTERY at " << curr.timestamp << "s: "
                    << battery->voltage << "V (" << (int)battery->percentage
                    << "%) - Degraded performance expected" << endl;
            }

            // ========== SIGN MISMATCH DETECTION ==========
            CmdVelData* latest_cmd = findClosestCmd(curr.timestamp - CONTROL_LAG);
            
            if (latest_cmd) {
                //skip sign checks if robot is rotating in place
                bool is_rotating = std::abs(latest_cmd->angular_z) > 0.4 || std::abs(actual_angular) > 0.25; //95th percentile is .405

                // Linear sign check
                if (!is_rotating &&
                    std::abs(latest_cmd->linear_x) > MIN_SIGNIFICANT_VEL && 
                    std::abs(actual_linear) > MIN_SIGNIFICANT_VEL && 
                    latest_cmd->linear_x * actual_linear < 0) {
                    
                    Anomaly a;
                    a.timestamp = curr.timestamp;
                    a.type = "LINEAR_SIGN_MISMATCH";
                    a.description = "Direction mismatch - Commanded: " + 
                                  to_string(latest_cmd->linear_x) + " m/s, " +
                                  "Actual: " + to_string(actual_linear) + " m/s " +
                                  "(moving opposite direction)";
                    a.severity = 0.6;
                    
                    if (battery) {
                        a.description += " [Battery: " + to_string(battery->voltage) + "V]";
                    }
                    
                    anomalies.push_back(a);
                }

                // Angular sign check - FIX: Use latest_cmd->angular_z instead of avg_cmd_angular.avg
                if (std::abs(latest_cmd->angular_z) > MIN_SIGNIFICANT_VEL && 
                    std::abs(actual_angular) > MIN_SIGNIFICANT_VEL && 
                    latest_cmd->angular_z * actual_angular < 0) {
                    
                    Anomaly a;
                    a.timestamp = curr.timestamp;
                    a.type = "ANGULAR_SIGN_MISMATCH";
                    a.description = "Rotation direction mismatch - Commanded: " + 
                                  to_string(latest_cmd->angular_z) + " rad/s, " +
                                  "Actual: " + to_string(actual_angular) + " rad/s " +
                                  "(rotating opposite direction)";
                    a.severity = 0.85;
                    
                    if (battery) {
                        a.description += " [Battery: " + to_string(battery->voltage) + "V]";
                    }
                    
                    anomalies.push_back(a);
                }
            }
        }
        
        return anomalies;
    }
};


// Checking if IMU coordinates with Odometry
//WNM: It iterates because it assumes data is in vectors, which will not be the case when connected to Kubernetes
class ImuOdomValidator : public PhysicsValidator {
private:
    vector<OdomData>& odom_data;
    vector<ImuData>& imu_data;
    
    const double ACCEL_TOLERANCE = 2.1;  // m/s^2
    const double GYRO_TOLERANCE = 0.16;  // rad/s
    const double TIME_TOLERANCE = 0.2;   // seconds
    const double SENSOR_LAG = 0.05;      // 50ms typical IMU lag
    int consecutive_accel_violations = 0;
    int consecutive_gyro_violations = 0;
    const int VIOLATION_THRESHOLD = 3; 
    
    // ImuData* findClosestImu(double timestamp) {
    //     if (imu_data.empty()) return nullptr;
    //     ImuData* closest = nullptr;
    //     double min_diff = numeric_limits<double>::max();
        
    //     for (auto& imu : imu_data) {
    //         double diff = std::abs(imu.timestamp - timestamp);
    //         if (diff < min_diff) {
    //             min_diff = diff;
    //             closest = &imu;
    //         }
    //     }
        
    //     if (min_diff > TIME_TOLERANCE) {
    //         return nullptr;
    //     }
        
    //     return closest;
    // }

    //a struct similar to WindowAvg in CmdVelOdomValidator for
    //keeping IMU averages
    struct IMUAvg {
        double accel;
        double gyro;
        int count;
    };

    IMUAvg getAvgIMU (double start_t, double end_t) {
        double sum_accel = 0;
        double sum_gyro = 0;
        int count = 0;

        double lag_start = start_t - SENSOR_LAG;
        double lag_end = end_t - SENSOR_LAG;

        for (auto& imu : imu_data) {
            if (imu.timestamp >= lag_start && imu.timestamp <= lag_end) {
                double linear_accel = sqrt(imu.accel_x * imu.accel_x + imu.accel_y * imu.accel_y);
                sum_accel += linear_accel;
                sum_gyro += imu.gyro_z;
                count++;
            }
        }
        if (count == 0) {
            return {0.0, 0.0, 0};
        } 
        else {
            return {sum_accel / count, sum_gyro / count, count};
        }
    }
    
public:
    ImuOdomValidator(vector<OdomData>& odom, vector<ImuData>& imu)
        : odom_data(odom), imu_data(imu) {}
    
    string getName() const override {
        return "IMU vs ODOM Validator";
    }
    
    vector<Anomaly> validate() override {
        vector<Anomaly> anomalies;
        const double WARMUP_TIME = 10.0;
        const int MAX_TRANSIENT_SAMPLES = 20;  // 1 second transient
        const int MIN_TRANSIENT_SAMPLES = 5;   // 0.25 seconds minimum
        const double ACCEL_CHANGE_THRESHOLD = 1.0;  // m/s^2
        const double GYRO_CHANGE_THRESHOLD = 0.3;   // rad/s
        const double SETTLING_THRESHOLD_ACCEL = 0.5; // m/s^2
        const double SETTLING_THRESHOLD_GYRO = 0.1;  // rad/s

        double prev_odom_accel = 0.0;
        double prev_odom_gyro = 0.0;
        int transient_skip_counter = 0;

        //iterates through csv data to find anomalies
        for (size_t i = 2; i < odom_data.size(); ++i) {
            OdomData& curr = odom_data[i];
            OdomData& prev = odom_data[i-1];
            OdomData& prev2 = odom_data[i-2];
            
            //ignores if within warmup time
            if (curr.timestamp < WARMUP_TIME) continue;

            double dt1 = curr.timestamp - prev.timestamp;
            double dt2 = prev.timestamp - prev2.timestamp;
            
            if (dt1 <= 0 || dt1 > 1.0 || dt2 <= 0 || dt2 > 1.0) continue;
            
            // Calculate acceleration from odometry
            double dx2 = curr.x - prev.x;
            double dy2 = curr.y - prev.y;
            double v2 = sqrt(dx2*dx2 + dy2*dy2) / dt2;
            
            double dx1= prev.x - prev2.x;
            double dy1 = prev.y - prev2.y;
            double v1 = sqrt(dx1*dx1 + dy1*dy1) / dt1;
            
            double odom_accel = std::abs((v1 - v2))/dt1;
            

            // Calculate angular velocity from odometry with angle wrapping
            double dtheta = curr.theta - prev.theta;
            while (dtheta > M_PI) dtheta -= 2*M_PI;
            while (dtheta < -M_PI) dtheta += 2*M_PI;
            double odom_gyro = dtheta / dt2;

            auto avg_imu = getAvgIMU(prev.timestamp, curr.timestamp);

            if (avg_imu.count == 0) {
                continue;
            }

            //<========== TRANSIENT DETECTION WITH SETTLING ==========>
            double accel_change = std::abs(odom_accel - prev_odom_accel);
            double gyro_change = std::abs(odom_gyro - prev_odom_gyro);
            
            if (accel_change > ACCEL_CHANGE_THRESHOLD || 
                gyro_change > GYRO_CHANGE_THRESHOLD) {
                transient_skip_counter = MAX_TRANSIENT_SAMPLES;
                consecutive_accel_violations = 0;
                consecutive_gyro_violations = 0;
            }

            // Check if settled during transient
            if (transient_skip_counter > 0) {
                transient_skip_counter--;
                
                double accel_error = std::abs(odom_accel - avg_imu.accel);
                double gyro_error = std::abs(odom_gyro - avg_imu.gyro);
                
                int samples_elapsed = MAX_TRANSIENT_SAMPLES - transient_skip_counter;
                
                if (samples_elapsed >= MIN_TRANSIENT_SAMPLES) {
                    bool accel_settled = accel_error < SETTLING_THRESHOLD_ACCEL;
                    bool gyro_settled = gyro_error < SETTLING_THRESHOLD_GYRO;
                    
                    if (accel_settled && gyro_settled) {
                        transient_skip_counter = 0;
                    }
                }
                
                if (transient_skip_counter > 0) {
                    prev_odom_accel = odom_accel;
                    prev_odom_gyro = odom_gyro;
                    continue;
                }
            }
            
            prev_odom_accel = odom_accel;
            prev_odom_gyro = odom_gyro;
            // ==========================================
            
            // ========== ACCELERATION CHECK (STEADY-STATE) ==========
            double accel_diff = std::abs(odom_accel - avg_imu.accel);
            
            if (accel_diff > ACCEL_TOLERANCE) {
                consecutive_accel_violations++;
                
                if (consecutive_accel_violations >= VIOLATION_THRESHOLD) {
                    Anomaly a;
                    a.timestamp = curr.timestamp;
                    a.type = "IMU_ODOM_ACCEL_MISMATCH";
                    a.description = "IMU accel: " + to_string(avg_imu.accel) +
                                  " m/s^2, Odom accel: " + to_string(odom_accel) + 
                                  " m/s^2 (persisted " + to_string(consecutive_accel_violations) + " samples)";
                    a.severity = min(1.0, accel_diff / 2.0);
                    anomalies.push_back(a);
                }
            } else {
                consecutive_accel_violations = 0;
            }
            
            // ========== GYRO CHECK (STEADY-STATE) ==========
            double gyro_diff = std::abs(odom_gyro - avg_imu.gyro);
            
            if (gyro_diff > GYRO_TOLERANCE) {
                consecutive_gyro_violations++;
                
                if (consecutive_gyro_violations >= VIOLATION_THRESHOLD) {
                    Anomaly a;
                    a.timestamp = curr.timestamp;
                    a.type = "IMU_ODOM_GYRO_MISMATCH";
                    a.description = "IMU gyro: " + to_string(avg_imu.gyro) +
                                  " rad/s, Odom gyro: " + to_string(odom_gyro) + 
                                  " rad/s (persisted " + to_string(consecutive_gyro_violations) + " samples)";
                    a.severity = min(1.0, gyro_diff / 1.0);
                    anomalies.push_back(a);
                }
            } else {
                consecutive_gyro_violations = 0;
            }

            //ImuData* imu = findClosestImu(curr.timestamp);
            
            // if (imu) {
            //     // Check acceleration consistency
            //     double imu_linear_accel = sqrt(imu->accel_x * imu->accel_x + imu->accel_y * imu->accel_y);
            //     double accel_diff = std::abs(odom_accel - imu_linear_accel);
                
            //     if (accel_diff > ACCEL_TOLERANCE) {
            //         Anomaly a;
            //         a.timestamp = curr.timestamp;
            //         a.type = "IMU_ODOM_ACCEL_MISMATCH";
            //         a.description = "IMU accel: " + to_string(imu_linear_accel) + 
            //                       " m/s^2, Odom accel: " + to_string(odom_accel) + " m/s^2";
            //         a.severity = min(1.0, accel_diff / 2.0);
            //         anomalies.push_back(a);
            //     }
                
            //     // Check gyro consistency
            //     double gyro_diff = std::abs(odom_gyro - imu->gyro_z);
            //     if (gyro_diff > GYRO_TOLERANCE) {
            //         Anomaly a;
            //         a.timestamp = curr.timestamp;
            //         a.type = "IMU_ODOM_GYRO_MISMATCH";
            //         a.description = "IMU gyro: " + to_string(imu->gyro_z) + 
            //                       " rad/s, Odom gyro: " + to_string(odom_gyro) + " rad/s";
            //         a.severity = min(1.0, gyro_diff / 1.0);
            //         anomalies.push_back(a);
            //     }
            //}
        }
        
        return anomalies;
    }
};


//Checking if motor commands align with Odometry
//WNM: It iterates because it assumes data is in vectors, which will not be the case when connected to Kubernetes
class JointStatesValidator : public PhysicsValidator {
private:
    vector<JointStatesData>& joint_data;
    vector<OdomData>& odom_data;
    
    const double WHEEL_RADIUS = 0.033;  // TurtleBot3 wheel radius in meters
    const double WHEEL_BASE = 0.160;    // Distance between wheels in meters
    const double TIME_TOLERANCE = 0.2;
    const double VEL_TOLERANCE = .14;
    const double ANGULAR_TOLERANCE = 0.53; // rad/s tolerance
    
    OdomData* findClosestOdom(double timestamp) {
        OdomData* closest = nullptr;
        double min_diff = numeric_limits<double>::max();
        
        for (auto& odom : odom_data) {
            double diff = std::abs(odom.timestamp - timestamp);
            if (diff < min_diff) {
                min_diff = diff;
                closest = &odom;
            }
        }
        
        if (min_diff > TIME_TOLERANCE) {
            return nullptr;
        }
        
        return closest;
    }
    
public:
    JointStatesValidator(vector<JointStatesData>& joint, vector<OdomData>& odom)
        : joint_data(joint), odom_data(odom) {}
    
    string getName() const override {
        return "Joint States Validator";
    }
    
    vector<Anomaly> validate() override {
        vector<Anomaly> anomalies;
        const double WARMUP_TIME = 10.0;
        
        for (size_t i = 1; i < joint_data.size(); ++i) {
            JointStatesData& curr = joint_data[i];
            JointStatesData& prev = joint_data[i-1];
            
            if (curr.timestamp < WARMUP_TIME) continue;
            double dt = curr.timestamp - prev.timestamp;
            if (dt <= 0 || dt > 1.0) continue;
            
            // Calculate expected robot velocities from wheel velocities
            double linear_from_wheels = (curr.v_R + curr.v_L) / 2.0 * WHEEL_RADIUS;
            double angular_from_wheels = (curr.v_R - curr.v_L) / WHEEL_BASE * WHEEL_RADIUS;
            
            OdomData* odom_curr = findClosestOdom(curr.timestamp);
            OdomData* odom_prev = findClosestOdom(prev.timestamp);
            
            if (odom_curr && odom_prev) {
                int odom_dt = odom_curr->timestamp - odom_prev->timestamp;
                if (odom_dt > 0 && odom_dt < 1.0) {
                    double dx = odom_curr->x - odom_prev->x;
                    double dy = odom_curr->y - odom_prev->y;
                    double linear_from_odom = sqrt(dx*dx + dy*dy) / odom_dt;
                    double dtheta = odom_curr->theta - odom_prev->theta;
                    while (dtheta > M_PI) dtheta -= 2*M_PI;
                    while (dtheta < -M_PI) dtheta += 2*M_PI;
                    double angular_from_odom = dtheta / odom_dt;
                    
                    // Check linear velocity consistency
                    double linear_diff = std::abs(linear_from_wheels - linear_from_odom);
                    if (linear_diff > VEL_TOLERANCE) {
                        Anomaly a;
                        a.timestamp = curr.timestamp;
                        a.type = "JOINT_ODOM_LINEAR_MISMATCH";
                        a.description = "Wheel-derived vel: " + to_string(linear_from_wheels) + 
                                      " m/s, Odom vel: " + to_string(linear_from_odom) + " m/s";
                        a.severity = min(1.0, linear_diff / 0.22);
                        anomalies.push_back(a);
                    }
                    
                    // Check angular velocity consistency
                    double angular_diff = std::abs(angular_from_wheels - angular_from_odom);
                    if (angular_diff > ANGULAR_TOLERANCE) {
                        Anomaly a;
                        a.timestamp = curr.timestamp;
                        a.type = "JOINT_ODOM_ANGULAR_MISMATCH";
                        a.description = "Wheel-derived ang: " + to_string(angular_from_wheels) + 
                                      " rad/s, Odom ang: " + to_string(angular_from_odom) + " rad/s";
                        a.severity = min(1.0, angular_diff / 2.84);
                        anomalies.push_back(a);
                    }
                }
            }
        }
        
        return anomalies;
    }
};


//Comparing with official parameters from TurtleBot3
class PhysicalLimitsValidator : public PhysicsValidator {
private:
    vector<OdomData>& odom_data;
    vector<CmdVelData>& cmd_data;
    vector<ImuData>& imu_data;
    
    const double MAX_LINEAR_VEL = 0.22;      // m/s
    const double MAX_ANGULAR_VEL = 2.84;     // rad/s
    const double MAX_ACCEL = 1.0;            // m/s^2
    const double MAX_IMU_ACCEL = 2.0;       // m/s^2 
    const double MAX_IMU_GYRO = 2.84;         // rad/s
    
public:
    PhysicalLimitsValidator(vector<OdomData>& odom, vector<CmdVelData>& cmd, vector<ImuData>& imu)
        : odom_data(odom), cmd_data(cmd), imu_data(imu) {}
    
    string getName() const override {
        return "Physical Limits Validator";
    }
    
    vector<Anomaly> validate() override {
        vector<Anomaly> anomalies;
        
        // Check cmd_vel commands
        for (const auto& cmd : cmd_data) {
            if (std::abs(cmd.linear_x) > MAX_LINEAR_VEL) {
                Anomaly a;
                a.timestamp = cmd.timestamp;
                a.type = "SPEC_VIOLATION_CMD_LINEAR";
                a.description = "Commanded linear velocity " + to_string(cmd.linear_x) + 
                              " m/s exceeds TurtleBot3 Burger specification (" + 
                              to_string(MAX_LINEAR_VEL) + " m/s)";
                a.severity = 1.0;
                anomalies.push_back(a);
            }
            
            if (std::abs(cmd.angular_z) > MAX_ANGULAR_VEL) {
                Anomaly a;
                a.timestamp = cmd.timestamp;
                a.type = "SPEC_VIOLATION_CMD_ANGULAR";
                a.description = "Commanded angular velocity " + to_string(cmd.angular_z) + 
                              " rad/s exceeds TurtleBot3 Burger specification (" + 
                              to_string(MAX_ANGULAR_VEL) + " rad/s)";
                a.severity = 1.0;
                anomalies.push_back(a);
            }
        }
        
        // Check odom velocities
        for (size_t i = 1; i < odom_data.size(); ++i) {
            double dt = odom_data[i].timestamp - odom_data[i-1].timestamp;
            if (dt > 0 && dt < 1.0) {
                double dx = odom_data[i].x - odom_data[i-1].x;
                double dy = odom_data[i].y - odom_data[i-1].y;
                double linear_vel = sqrt(dx*dx + dy*dy) / dt;
                double dtheta = odom_data[i].theta - odom_data[i-1].theta;
                while (dtheta > M_PI) dtheta -= 2*M_PI;
                while (dtheta < -M_PI) dtheta += 2*M_PI;
                double angular_vel = std::abs(dtheta / dt);
                
                if (linear_vel > MAX_LINEAR_VEL) {
                    Anomaly a;
                    a.timestamp = odom_data[i].timestamp;
                    a.type = "SPEC_VIOLATION_ODOM_LINEAR";
                    a.description = "Odom linear velocity " + to_string(linear_vel) + 
                                  " m/s exceeds physical limits";
                    a.severity = 1.0;
                    anomalies.push_back(a);
                }
                
                if (angular_vel > MAX_ANGULAR_VEL) {
                    Anomaly a;
                    a.timestamp = odom_data[i].timestamp;
                    a.type = "SPEC_VIOLATION_ODOM_ANGULAR";
                    a.description = "Odom angular velocity " + to_string(angular_vel) + 
                                  " rad/s exceeds physical limits";
                    a.severity = 1.0;
                    anomalies.push_back(a);
                }
            }
        }
        
        // Check IMU limits
        for (const auto& imu : imu_data) {
            double horizontal_accel = sqrt(imu.accel_x*imu.accel_x + 
                               imu.accel_y*imu.accel_y);

            if (horizontal_accel > MAX_IMU_ACCEL) {
                Anomaly a;
                a.timestamp = imu.timestamp;
                a.type = "IMU_ACCEL_EXCEEDED";
                a.description = "IMU acceleration " + to_string(horizontal_accel) + 
                              " m/s^2 exceeds reasonable limits";
                a.severity = 0.9;
                anomalies.push_back(a);
            }
            
            if (std::abs(imu.gyro_z) > MAX_IMU_GYRO) {
                Anomaly a;
                a.timestamp = imu.timestamp;
                a.type = "IMU_GYRO_EXCEEDED";
                a.description = "IMU gyro " + to_string(imu.gyro_z) + 
                              " rad/s exceeds reasonable limits";
                a.severity = 0.9;
                anomalies.push_back(a);
            }
        }
        
        return anomalies;
    }
};


//implementing error into calculations
class TrackingErrorValidator : public PhysicsValidator {
private:
    vector<ErrorData>& error_data;
    const double MAX_ERROR_RATE = 1.89;      // Was 2.3, tracking error rate
    const double MAX_POS_ERROR_RATE = 1.48;  // 
    const double WARMUP_TIME = 10.0;
    const double COOLDOWN_TIME = 2.0;
    const double MIN_DT = 0.0016;  // Based on 5th percentile × 0.5
    double MAX_DT = 0.0135;        // Based on 95th percentile × 2.0
    const double SPIKE_FILTER = 50.0;
    const int VIOLATION_THRESHOLD = 3;
  
    int consec_tr_vlts = 0;
    int consec_pos_vlts = 0;

public:
    TrackingErrorValidator(vector<ErrorData>& error)
        : error_data(error) {
        calculateMaxDT();
    }

    string getName() const override {  
        return "Tracking Error Validator";
    }

    void calculateMaxDT() {
        vector<double> dts;
        int zero_dt = 0;
        int tiny_dt = 0;
        
        for (size_t i = 1; i < error_data.size(); ++i) {
            double dt = error_data[i].timestamp - error_data[i-1].timestamp;
            
            if (dt == 0) zero_dt++;
            if (dt > 0 && dt < 0.001) tiny_dt++;
            
            if (dt > 0 && dt < 100.0) {
                dts.push_back(dt);
            }
        }
        
        if (dts.empty()) {
            MAX_DT = 5.0;
            return;
        }
        
        sort(dts.begin(), dts.end());
        double median_dt = dts[dts.size() / 2];
        double min_dt = dts.front();
        double max_dt = dts.back();
        
        MAX_DT = median_dt * 10.0;
        
        cout << "Tracking Error Validator:" << endl;
        cout << "  Zero dt samples: " << zero_dt << endl;
        cout << "  Tiny dt (<0.001s) samples: " << tiny_dt << endl;
        cout << "  Min dt: " << min_dt << " s" << endl;
        cout << "  Median dt: " << median_dt << " s" << endl;
        cout << "  Max dt: " << max_dt << " s" << endl;
        cout << "  Adaptive MAX_DT: " << MAX_DT << " s" << endl;
        cout << "  Current MIN_DT filter: " << MIN_DT << " s" << endl;
        cout << endl;
    }
    
    vector<Anomaly> validate() override {
        vector<Anomaly> anomalies;
        if (error_data.size() < 2) return anomalies;
        
        double max_timestamp = error_data.back().timestamp;
  
        for (size_t i = 1; i < error_data.size(); ++i) {
            ErrorData& curr = error_data[i];
            ErrorData& prev = error_data[i-1];
            
            //Skip warmup and cooldown periods
            if (curr.timestamp < WARMUP_TIME) continue;
            if (curr.timestamp > max_timestamp - COOLDOWN_TIME) continue;
  
            double dt = curr.timestamp - prev.timestamp;
  
            if (dt <= MIN_DT || dt > MAX_DT) {
                consec_pos_vlts = 0;
                consec_tr_vlts = 0;
                continue;
            }
            
            // ========== TRACKING ERROR RATE CHECK ==========
            double error_rate = std::abs(curr.tracking_error - prev.tracking_error) / dt;
            
            if (error_rate > SPIKE_FILTER) {
                consec_tr_vlts = 0;
                continue;  // Skip this sample
            }
  
            if (error_rate > MAX_ERROR_RATE) {
                consec_tr_vlts++;
  
                if (consec_tr_vlts >= VIOLATION_THRESHOLD) {                
                    Anomaly a;
                    a.timestamp = curr.timestamp;
                    a.type = "RAPID_ERROR_INCREASE";
                    a.description = "Tracking error increasing at " + to_string(error_rate) +
                                  " m/s (persisted " + to_string(consec_tr_vlts) +
                                  " samples - may indicate localization attack)";
                    a.severity = min(1.0, error_rate / (MAX_ERROR_RATE * 2));
                    anomalies.push_back(a);
                }
            } else {
                consec_tr_vlts = 0;  // Reset counter if below threshold
            }
            
            // ========== POSITION ERROR RATE CHECK ==========
            double prev_pos_error = sqrt(prev.error_x*prev.error_x + prev.error_y*prev.error_y);
            double curr_pos_error = sqrt(curr.error_x*curr.error_x + curr.error_y*curr.error_y);
            double pos_error_rate = std::abs(curr_pos_error - prev_pos_error) / dt;
            
            //filter out spikes
            if (pos_error_rate > SPIKE_FILTER) {
                consec_pos_vlts = 0;  // FIX: Reset pos counter, not tr counter
                continue;
            }
  
            if (pos_error_rate > MAX_POS_ERROR_RATE) {
                consec_pos_vlts++;
                
                if (consec_pos_vlts >= VIOLATION_THRESHOLD) {
                    Anomaly a;
                    a.timestamp = curr.timestamp;
                    a.type = "RAPID_POSITION_ERROR_CHANGE";
                    a.description = "Position error changing at " + to_string(pos_error_rate) +
                                  " m/s (persisted " + to_string(consec_pos_vlts) +
                                  " samples - may indicate GPS/localization spoofing)";
                    a.severity = min(1.0, pos_error_rate / (MAX_POS_ERROR_RATE * 2));
                    anomalies.push_back(a);
                }
            } else {
                consec_pos_vlts = 0;  // Reset counter if below threshold
            }
        }  
        
        return anomalies;
    }
};

// Main Intrusion Detector
class IntrusionDetector {
private:
    vector<OdomData> odom_data;
    vector<CmdVelData> cmd_data;
    vector<ImuData> imu_data;
    vector<JointStatesData> joint_data;
    vector<ErrorData> error_data;
    vector<BatteryData> battery_data;
    vector<PhysicsValidator*> validators;
    vector<Anomaly> all_anomalies;
    
    vector<string> parseCSVLine(const string& line) {
        vector<string> result;
        stringstream ss(line);
        string cell;
        
        while (getline(ss, cell, ',')) {
            // Trim whitespace
            cell.erase(0, cell.find_first_not_of(" \t\r\n"));
            cell.erase(cell.find_last_not_of(" \t\r\n") + 1);
            result.push_back(cell);
        }
        return result;
    }
    
public:
    IntrusionDetector() {}
    
    ~IntrusionDetector() {
        for (auto validator : validators) {
            delete validator;
        }
    }
    
    bool loadOdomData(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Failed to open " << filename << endl;
            return false;
        }
        
        string line;
        bool first_line = true;
        
        while (getline(file, line)) {
            if (first_line) {
                first_line = false;
                continue; // Skip header
            }
            
            if (line.empty()) continue;
            
            auto fields = parseCSVLine(line);
            if (fields.size() >= 4) {
                OdomData data;
                try {
                    data.timestamp = stod(fields[0]);
                    data.x = stod(fields[1]);
                    data.y = stod(fields[2]);
                    data.theta = stod(fields[3]);
                    odom_data.push_back(data);
                } catch (const exception& e) {
                    cerr << "Error parsing odom line: " << e.what() << endl;
                }
            }
        }
        
        cout << "Loaded " << odom_data.size() << " odom samples" << endl;
        return !odom_data.empty();
    }
    
    bool loadBatteryData(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Failed to open " << filename << endl;
            return false;
        }
    
        string line;
        getline(file, line); 
    
        while (getline(file, line)) {
            stringstream ss(line);
            BatteryData battery;
        
            string token;
            getline(ss, token, ','); battery.timestamp = stod(token);
            getline(ss, token, ','); battery.voltage = stod(token);
            getline(ss, token, ','); battery.current = stod(token);
            getline(ss, token, ','); battery.percentage = stod(token);
        
            battery_data.push_back(battery);
        }
    
        cout << "Loaded " << battery_data.size() << " battery samples" << endl;
        return !battery_data.empty();
    }
    
    bool loadCmdVelData(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Failed to open " << filename << endl;
            return false;
        }
        
        string line;
        bool first_line = true;
        
        while (getline(file, line)) {
            if (first_line) {
                first_line = false;
                continue;
            }
            
            if (line.empty()) continue;
            
            auto fields = parseCSVLine(line);
            if (fields.size() >= 3) {
                CmdVelData data;
                try {
                    data.timestamp = stod(fields[0]);
                    data.linear_x = stod(fields[1]);
                    data.angular_z = stod(fields[2]);
                    cmd_data.push_back(data);
                } catch (const exception& e) {
                    cerr << "Error parsing cmd_vel line: " << e.what() << endl;
                }
            }
        }
        
        cout << "Loaded " << cmd_data.size() << " cmd_vel samples" << endl;
        return !cmd_data.empty();
    }
    
    bool loadImuData(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Warning: Could not open " << filename << " (optional)" << endl;
            return false;
        }
        
        string line;
        bool first_line = true;
        
        while (getline(file, line)) {
            if (first_line) {
                first_line = false;
                continue;
            }
            
            if (line.empty()) continue;
            
            auto fields = parseCSVLine(line);
            if (fields.size() >= 7) {
                ImuData data;
                try {
                    data.timestamp = stod(fields[0]);
                    data.accel_x = stod(fields[1]);
                    data.accel_y = stod(fields[2]);
                    data.accel_z = stod(fields[3]);
                    data.gyro_x = stod(fields[4]);
                    data.gyro_y = stod(fields[5]);
                    data.gyro_z = stod(fields[6]);
                    imu_data.push_back(data);
                } catch (const exception& e) {
                    cerr << "Error parsing imu line: " << e.what() << endl;
                }
            }
        }
        
        cout << "Loaded " << imu_data.size() << " imu samples" << endl;
        return !imu_data.empty();
    }
    
    bool loadJointStatesData(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Warning: Could not open " << filename << " (optional)" << endl;
            return false;
        }
        
        string line;
        bool first_line = true;
        
        while (getline(file, line)) {
            if (first_line) {
                first_line = false;
                continue;
            }
            
            if (line.empty()) continue;
            
            auto fields = parseCSVLine(line);
            if (fields.size() >= 3) {
                JointStatesData data;
                try {
                    data.timestamp = stod(fields[0]);
                    data.v_R = stod(fields[1]);
                    data.v_L = stod(fields[2]);
                    joint_data.push_back(data);
                } catch (const exception& e) {
                    cerr << "Error parsing joint_states line: " << e.what() << endl;
                }
            }
        }
        
        cout << "Loaded " << joint_data.size() << " joint_states samples" << endl;
        return !joint_data.empty();
    }
    
    bool loadErrorData(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Warning: Could not open " << filename << " (optional)" << endl;
            return false;
        }
        
        string line;
        bool first_line = true;
        
        while (getline(file, line)) {
            if (first_line) {
                first_line = false;
                continue;
            }
            
            if (line.empty()) continue;
            
            auto fields = parseCSVLine(line);
            if (fields.size() >= 4) {
                ErrorData data;
                try {
                    data.timestamp = stod(fields[0]);
                    data.tracking_error = stod(fields[1]);
                    data.error_x = stod(fields[2]);
                    data.error_y = stod(fields[3]);
                    error_data.push_back(data);
                } catch (const exception& e) {
                    cerr << "Error parsing error line: " << e.what() << endl;
                }
            }
        }
        
        cout << "Loaded " << error_data.size() << " error samples" << endl;
        return !error_data.empty();
    }
    
    void initializeValidators() {
        validators.push_back(new CmdVelOdomValidator(odom_data, cmd_data, battery_data));
        validators.push_back(new PhysicalLimitsValidator(odom_data, cmd_data, imu_data));
        
        if (!imu_data.empty()) {
            validators.push_back(new ImuOdomValidator(odom_data, imu_data));
        }
        
        if (!joint_data.empty()) {
            validators.push_back(new JointStatesValidator(joint_data, odom_data));
        }
        
        if (!error_data.empty()) {
            validators.push_back(new TrackingErrorValidator(error_data));
        }
    }
    
    void runDetection() {
        cout << "\n=== Running Intrusion Detection ===" << endl;
        
        for (auto validator : validators) {
            cout << "\nRunning: " << validator->getName() << endl;
            auto anomalies = validator->validate();
            
            cout << "  Found " << anomalies.size() << " anomalies" << endl;
            
            all_anomalies.insert(all_anomalies.end(), anomalies.begin(), anomalies.end());
        }
    }
    
    void generateReport() {
        cout << "\n\n========================================" << endl;
        cout << "    INTRUSION DETECTION REPORT" << endl;
        cout << "========================================\n" << endl;
        
        cout << "Total Anomalies Detected: " << all_anomalies.size() << "\n" << endl;
        
        if (all_anomalies.empty()) {
            cout << "✓ NO INTRUSIONS DETECTED - System appears secure" << endl;
            return;
        }
        
        // Group by type
        map<string, int> anomaly_counts;
        for (const auto& a : all_anomalies) {
            anomaly_counts[a.type]++;
        }
        
        cout << "Anomaly Breakdown:" << endl;
        for (const auto& pair : anomaly_counts) {
            cout << "  " << pair.first << ": " << pair.second << endl;
        }
        
        cout << "\nDetailed Anomalies (Top 20):" << endl;
        cout << "----------------------------------------" << endl;
        
        int count = 0;
        for (const auto& a : all_anomalies) {
            if (count++ >= 20) break;
            
            cout << fixed << setprecision(3);
            cout << "[" << a.timestamp << "s] " 
                 << a.type << " (severity: " << a.severity << ")" << endl;
            cout << "  " << a.description << endl;
        }
        
        // Overall assessment
        double avg_severity = 0;
        for (const auto& a : all_anomalies) {
            avg_severity += a.severity;
        }
        avg_severity /= all_anomalies.size();
        
        cout << "\n========================================" << endl;
        cout << "Average Anomaly Severity: " << fixed 
             << setprecision(2) << avg_severity << endl;
        
        if (avg_severity > 0.7) {
            cout << "⚠ HIGH RISK - Possible intrusion detected!" << endl;
        } else if (avg_severity > 0.4) {
            cout << "⚠ MEDIUM RISK - Investigate anomalies" << endl;
        } else {
            cout << "✓ LOW RISK - Minor anomalies detected" << endl;
        }
        cout << "========================================\n" << endl;
    }
};

int runIntrusionDetection(
    const std::string& odom_file,
    const std::string& cmd_file,
    const std::string& imu_file,
    const std::string& joint_file,
    const std::string& error_file,
    const std::string& battery_file)
{
    try {
        auto start_total = std::chrono::high_resolution_clock::now();
        
        std::cout << "\nTurtleBot3 Rule Based Algorithm\n" << std::endl;
        
        IntrusionDetector detector;
        
        // Load data with timing
        auto start_load = std::chrono::high_resolution_clock::now();
        
        if (!detector.loadOdomData(odom_file)) {
            std::cerr << "Failed to load odometry data" << std::endl;
            return 1;
        }
        
        if (!detector.loadCmdVelData(cmd_file)) {
            std::cerr << "Failed to load cmd_vel data" << std::endl;
            return 1;
        }
        
        detector.loadImuData(imu_file);  
        detector.loadJointStatesData(joint_file);  
        detector.loadErrorData(error_file);  
        detector.loadBatteryData(battery_file);
        
        auto end_load = std::chrono::high_resolution_clock::now();
        auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_load - start_load);
        
        // Initialize validators
        auto start_init = std::chrono::high_resolution_clock::now();
        detector.initializeValidators();
        auto end_init = std::chrono::high_resolution_clock::now();
        auto init_time = std::chrono::duration_cast<std::chrono::microseconds>(end_init - start_init);
        
        // Run detection with timing
        auto start_detect = std::chrono::high_resolution_clock::now();
        detector.runDetection();
        auto end_detect = std::chrono::high_resolution_clock::now();
        auto detect_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_detect - start_detect);
        
        // Generate report
        auto start_report = std::chrono::high_resolution_clock::now();
        detector.generateReport();
        auto end_report = std::chrono::high_resolution_clock::now();
        auto report_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_report - start_report);
        
        // End total timer
        auto end_total = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total);
        
        // Print timing summary
        std::cout << "\n========================================" << std::endl;
        std::cout << "         PERFORMANCE TIMING" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Data Loading:        " << load_time.count() << " ms" << std::endl;
        std::cout << "Validator Init:      " << init_time.count() << " μs" << std::endl;
        std::cout << "Detection Analysis:  " << detect_time.count() << " ms" << std::endl;
        std::cout << "Report Generation:   " << report_time.count() << " ms" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "TOTAL RUNTIME:       " << total_time.count() << " ms";
        std::cout << " (" << std::fixed << std::setprecision(3) << total_time.count() / 1000.0 << " seconds)" << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in intrusion detection: " << e.what() << std::endl;
        return 1;
    }
}
