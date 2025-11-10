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
#include <set>

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
using std::set;
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
//All physics values
struct botData {
    double timestamp;
    double x, y, theta; // Odom
    double linear_x, angular_z; //CmdVel
    double accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z; //Imu
    double v_R, v_L, right_encoder, left_encoder; //Joint States
    double tracking_error, error_x, error_y; //tracking error
    double voltage, current, percentage; //battery info

    botData() : timestamp(0), x(0), y(0), theta(0), linear_x(0), angular_z(0), accel_x(0),
                accel_y(0), accel_z(0), gyro_x(0), gyro_y(0), gyro_z(0), v_R(0), v_L(0),
                right_encoder(0), left_encoder(0), tracking_error(0), error_x(0),
                error_y(0), voltage(0), current(0), percentage(0) {}
};

//ok
// self describing
struct Anomaly {
    double timestamp;
    string type;
    string description;
    double severity; // 0-1 scale
    int consecutive_count;
    double duration;

    Anomaly() : timestamp(0), severity(0), consecutive_count(1), duration(0) {}
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
    vector<botData>& data;
    const double MAX_LINEAR_VEL = 0.22;
    const double MAX_ANGULAR_VEL = 2.84;
    const double MIN_SIGNIFICANT_VEL = 0.05;
    
    int consecutive_angular_violations = 0;
    int consecutive_linear_violations = 0;
    const int VIOLATION_THRESHOLD = 3;
    const int LAG_SAMPLES = 3;
    
public:
    CmdVelOdomValidator(vector<botData>& bot_data) : data(bot_data) {}
    
    string getName() const override {
        return "CMD_VEL vs ODOM Validator";
    }

    vector<Anomaly> validate() override {
        vector<Anomaly> anomalies;
        const double WARMUP_TIME = 10.0;
        bool battery_context = false;
        int lb_warnings = 0;
        
        double prev_cmd_angular = 0;
        double prev_cmd_linear = 0;
        int transient_skip_counter = 0;
        const int MAX_TRANSIENT_SAMPLES = 40;
        const int MIN_TRANSIENT_SAMPLES = 10;
        const double CMD_CHANGE_THRESHOLD_ANG = 0.5;
        const double CMD_CHANGE_THRESHOLD_LIN = 0.1;
        const double SETTLING_THRESHOLD_ANG = 0.3;
        const double SETTLING_THRESHOLD_LIN = 0.05;

        for (size_t i = LAG_SAMPLES + 1; i < data.size(); ++i) {
            botData& curr = data[i];
            botData& prev = data[i-1];
            botData& lagged_cmd = data[i - LAG_SAMPLES];
            
            if (curr.timestamp < WARMUP_TIME) continue;
            
            double dt = curr.timestamp - prev.timestamp;
            if (dt <= 0 || dt > 1.0) continue;
            
            // Battery context
            if (!battery_context && curr.voltage > 0) {
                cout << " Voltage at " << curr.voltage << "V ("
                     << curr.percentage << "%)" << endl;
                battery_context = true;
            }
            
            // CALCULATE ACTUAL VELOCITIES (MOVING AVERAGE)
            const int VELOCITY_WINDOW = 7;
            double sum_dx = 0, sum_dy = 0, sum_dtheta = 0, sum_dt = 0;
            int valid_samples = 0;
            
            for (int j = 0; j < VELOCITY_WINDOW && i >= (size_t)(j + 1); j++) {
                botData& c = data[i - j];
                botData& p = data[i - j - 1];
                double local_dt = c.timestamp - p.timestamp;
                
                if (local_dt > 0 && local_dt < 1.0) {
                    sum_dx += c.x - p.x;
                    sum_dy += c.y - p.y;
                    
                    double local_dtheta = c.theta - p.theta;
                    while (local_dtheta > M_PI) local_dtheta -= 2*M_PI;
                    while (local_dtheta < -M_PI) local_dtheta += 2*M_PI;
                    sum_dtheta += local_dtheta;
                    
                    sum_dt += local_dt;
                    valid_samples++;
                }
            }
            
            // Skip if insufficient valid samples
            if (valid_samples < 2) {
                consecutive_linear_violations = 0;
                consecutive_angular_violations = 0;
                continue;
            }
            
            double actual_angular = sum_dtheta / sum_dt;
            double avg_theta = curr.theta;  // Use current theta for direction
            double actual_linear = (sum_dx * cos(avg_theta) + sum_dy * sin(avg_theta)) / sum_dt;
            double actual_linear_magnitude = sqrt(sum_dx*sum_dx + sum_dy*sum_dy) / sum_dt;
            double actual_linear_signed = (actual_linear >= 0.0) ? 
                                         actual_linear_magnitude : -actual_linear_magnitude;
            
            // TRANSIENT DETECTION
            double cmd_angular_change = std::abs(lagged_cmd.angular_z - prev_cmd_angular);
            double cmd_linear_change = std::abs(lagged_cmd.linear_x - prev_cmd_linear);
            
            if (cmd_angular_change > CMD_CHANGE_THRESHOLD_ANG || 
                cmd_linear_change > CMD_CHANGE_THRESHOLD_LIN) {
                transient_skip_counter = MAX_TRANSIENT_SAMPLES;
                consecutive_linear_violations = 0;
                consecutive_angular_violations = 0;
            }
            
            if (transient_skip_counter > 0) {
                transient_skip_counter--;
                
                double angular_error = std::abs(actual_angular - lagged_cmd.angular_z);
                double linear_error = std::abs(actual_linear_signed - lagged_cmd.linear_x);
                
                int samples_elapsed = MAX_TRANSIENT_SAMPLES - transient_skip_counter;
                
                if (samples_elapsed >= MIN_TRANSIENT_SAMPLES) {
                    bool angular_settled = (angular_error < SETTLING_THRESHOLD_ANG) || 
                                         (std::abs(lagged_cmd.angular_z) < 0.1);
                    bool linear_settled = (linear_error < SETTLING_THRESHOLD_LIN) || 
                                        (std::abs(lagged_cmd.linear_x) < 0.05);
                    
                    if (angular_settled && linear_settled) {
                        transient_skip_counter = 0;
                    }
                }
                
                if (transient_skip_counter > 0) {
                    prev_cmd_angular = lagged_cmd.angular_z;
                    prev_cmd_linear = lagged_cmd.linear_x;
                    continue;
                }
            }
            
            prev_cmd_angular = lagged_cmd.angular_z;
            prev_cmd_linear = lagged_cmd.linear_x;
            
            double linear_tolerance = getAdaptiveLinearTolerance(curr.voltage);  
            double angular_tolerance = getAdaptiveAngularTolerance(curr.voltage);
            
            // Increase tolerance during commanded circular motion
            if (std::abs(lagged_cmd.angular_z) > 0.2) {
                angular_tolerance *= 4;  
                linear_tolerance *= 1.5;   
            }             
            //CHECK 1: LINEAR VELOCITY MISMATCH
            double vel_diff = std::abs(actual_linear_signed - lagged_cmd.linear_x);
            
            if (vel_diff > linear_tolerance) {  
                consecutive_linear_violations++;
                
                if (consecutive_linear_violations >= VIOLATION_THRESHOLD) {
                    Anomaly a;
                    a.timestamp = curr.timestamp;
                    a.type = "VELOCITY_MISMATCH";
                    a.description = "Commanded: " + to_string(lagged_cmd.linear_x) +
                                " m/s, Actual: " + to_string(actual_linear_signed) + 
                                " m/s (persisted " + to_string(consecutive_linear_violations) + " samples)";
                    a.severity = min(1.0, vel_diff / MAX_LINEAR_VEL);
                    
                    if (curr.voltage > 0) {
                        a.description += " [Battery: " + to_string(curr.voltage) + 
                                       "V, tolerance: " + to_string(linear_tolerance) + " m/s]";  
                    }
                    
                    anomalies.push_back(a);
                }
            } else {
                consecutive_linear_violations = 0;
            }
            
            //CHECK 2: ANGULAR VELOCITY MISMATCH 
            double ang_diff = std::abs(actual_angular - lagged_cmd.angular_z);
            
            if (ang_diff > angular_tolerance) {  
                consecutive_angular_violations++;
                
                if (consecutive_angular_violations >= VIOLATION_THRESHOLD) {
                    Anomaly a;
                    a.timestamp = curr.timestamp;
                    a.type = "ANGULAR_MISMATCH";
                    a.description = "Commanded: " + to_string(lagged_cmd.angular_z) +
                                " rad/s, Actual: " + to_string(actual_angular) + 
                                " rad/s (persisted " + to_string(consecutive_angular_violations) + " samples)";
                    a.severity = min(1.0, ang_diff / MAX_ANGULAR_VEL);
                    
                    if (curr.voltage > 0) {
                        a.description += " [Battery: " + to_string(curr.voltage) + 
                                       "V (" + to_string((int)curr.percentage) + "%), " +
                                       "tolerance: " + to_string(angular_tolerance) + " rad/s]";  
                    }
                    
                    anomalies.push_back(a);
                }
            } else {
                consecutive_angular_violations = 0;
            }
            
            // CHECK 3: LINEAR DIRECTION REVERSAL 
            bool battery_healthy_for_direction = (curr.voltage > 11.0);  
            
            bool is_rotating = std::abs(lagged_cmd.angular_z) > 0.4 || std::abs(actual_angular) > 0.25;
            
            if (battery_healthy_for_direction &&  
                !is_rotating &&
                std::abs(lagged_cmd.linear_x) > MIN_SIGNIFICANT_VEL &&
                std::abs(actual_linear_signed) > MIN_SIGNIFICANT_VEL && 
                lagged_cmd.linear_x * actual_linear_signed < 0) {
                
                Anomaly a;
                a.timestamp = curr.timestamp;
                a.type = "LINEAR_SIGN_MISMATCH";
                a.description = "Direction reversal - Commanded: " + 
                              to_string(lagged_cmd.linear_x) + " m/s, Actual: " + 
                              to_string(actual_linear_signed) + " m/s";
                a.severity = 0.6;
                
                if (curr.voltage > 0) {
                    a.description += " [Battery: " + to_string(curr.voltage) + "V]";
                }
                
                anomalies.push_back(a);
            }
            
            // CHECK 4: ANGULAR DIRECTION REVERSAL 
            if (battery_healthy_for_direction && 
                std::abs(lagged_cmd.angular_z) > MIN_SIGNIFICANT_VEL &&
                std::abs(actual_angular) > MIN_SIGNIFICANT_VEL && 
                lagged_cmd.angular_z * actual_angular < 0) {
                
                Anomaly a;
                a.timestamp = curr.timestamp;
                a.type = "ANGULAR_SIGN_MISMATCH";
                a.description = "Rotation reversal - Commanded: " + 
                              to_string(lagged_cmd.angular_z) + " rad/s, Actual: " + 
                              to_string(actual_angular) + " rad/s";
                a.severity = 0.85;
                
                if (curr.voltage > 0) {
                    a.description += " [Battery: " + to_string(curr.voltage) + "V]";
                }
                
                anomalies.push_back(a);
            }
            
            // Low battery warning
            if (curr.voltage > 0 && curr.voltage < 11.0 && lb_warnings < 3) {
                lb_warnings++;
                cout << "  LOW BATTERY at " << curr.timestamp << "s: "
                     << curr.voltage << "V (" << (int)curr.percentage
                     << "%) - Degraded performance expected" << endl;
            }
        }
        
        return anomalies;
    }
    
private:
    double getAdaptiveLinearTolerance(double voltage) {
        if (voltage <= 0) return 0.12;      // No battery data - slightly relaxed
        if (voltage > 12.0) return 0.08;    // Full power - strict
        else if (voltage > 11.5) return 0.10;  // Slight degradation
        else if (voltage > 11.0) return 0.16;  // Moderate degradation
        else if (voltage > 10.5) return 0.20;  // Significant degradation
        else return 0.25;                    // Critical battery - very relaxed
    }
    
    double getAdaptiveAngularTolerance(double voltage) {
        if (voltage <= 0) return 0.20;
        if (voltage > 12.0) return 0.10;
        else if (voltage > 11.5) return 0.15;
        else if (voltage > 11.0) return 0.27;
        else if (voltage > 10.5) return 0.35;
        else return 0.45;
    }
};


// Checking if IMU coordinates with Odometry
//WNM It iterates because it assumes data is in vectors, which will not be the case when connected to Kubernetes
class ImuOdomValidator : public PhysicsValidator {
private:
    vector<botData>& data;
    
    const double ACCEL_TOLERANCE = 2.1;  // m/s^2
    const double GYRO_TOLERANCE = 0.16;  // rad/s
    const int LAG_SAMPLES = 1;  // ~50ms lag at 50Hz = 1 sample
    
    int consecutive_accel_violations = 0;
    int consecutive_gyro_violations = 0;
    const int VIOLATION_THRESHOLD = 3;
    
public:
    ImuOdomValidator(vector<botData>& bot_data) : data(bot_data) {}
    
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

        for (size_t i = max(2, LAG_SAMPLES + 1); i < data.size(); ++i) {
            botData& curr = data[i];
            botData& prev = data[i-1];
            botData& prev2 = data[i-2];
            botData& lagged_imu = data[i - LAG_SAMPLES];  //IMU from ~50ms ago
            
            if (curr.timestamp < WARMUP_TIME) continue;

            double dt1 = curr.timestamp - prev.timestamp;
            double dt2 = prev.timestamp - prev2.timestamp;
            
            if (dt1 <= 0 || dt1 > 1.0 || dt2 <= 0 || dt2 > 1.0) continue;
            
            //CALCULATE ACCELERATION FROM ODOMETRY
            double dx2 = curr.x - prev.x;
            double dy2 = curr.y - prev.y;
            double v2 = sqrt(dx2*dx2 + dy2*dy2) / dt2;
            
            double dx1 = prev.x - prev2.x;
            double dy1 = prev.y - prev2.y;
            double v1 = sqrt(dx1*dx1 + dy1*dy1) / dt1;
            
            double odom_accel = std::abs((v2 - v1)) / dt1; 
            
            //Calculate angular velocity from odometry
            double dtheta = curr.theta - prev.theta;
            while (dtheta > M_PI) dtheta -= 2*M_PI;
            while (dtheta < -M_PI) dtheta += 2*M_PI;
            double odom_gyro = dtheta / dt1; 
            
            //Get Imu values
            double imu_accel = sqrt(lagged_imu.accel_x * lagged_imu.accel_x + 
                                   lagged_imu.accel_y * lagged_imu.accel_y);
            double imu_gyro = lagged_imu.gyro_z;
            
            // Skip if IMU data is zero (might be missing/invalid)
            if (imu_accel == 0 && imu_gyro == 0) {
                continue;
            }
            
            // TRANSIENT DETECTION WITH SETTLING
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
                
                double accel_error = std::abs(odom_accel - imu_accel);
                double gyro_error = std::abs(odom_gyro - imu_gyro);
                
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
            
            // CHECK 1: ACCELERATION MISMATCH (STEADY-STATE)
            double accel_diff = std::abs(odom_accel - imu_accel);
            
            if (accel_diff > ACCEL_TOLERANCE) {
                consecutive_accel_violations++;
                
                if (consecutive_accel_violations >= VIOLATION_THRESHOLD) {
                    Anomaly a;
                    a.timestamp = curr.timestamp;
                    a.type = "IMU_ODOM_ACCEL_MISMATCH";
                    a.description = "IMU accel: " + to_string(imu_accel) +
                                  " m/s^2, Odom accel: " + to_string(odom_accel) + 
                                  " m/s^2 (persisted " + to_string(consecutive_accel_violations) + " samples)";
                    a.severity = min(1.0, accel_diff / 2.0);

                    anomalies.push_back(a);
                }
            } else {
                consecutive_accel_violations = 0;
            }
            
            // CHECK 2: GYRO MISMATCH (STEADY-STATE) 
            double gyro_diff = std::abs(odom_gyro - imu_gyro);
            
            if (gyro_diff > GYRO_TOLERANCE) {
                consecutive_gyro_violations++;
                
                if (consecutive_gyro_violations >= VIOLATION_THRESHOLD) {
                    Anomaly a;
                    a.timestamp = curr.timestamp;
                    a.type = "IMU_ODOM_GYRO_MISMATCH";
                    a.description = "IMU gyro: " + to_string(imu_gyro) +
                                  " rad/s, Odom gyro: " + to_string(odom_gyro) + 
                                  " rad/s (persisted " + to_string(consecutive_gyro_violations) + " samples)";
                    a.severity = min(1.0, gyro_diff / 1.0);

                    anomalies.push_back(a);
                }
            } else {
                consecutive_gyro_violations = 0;
            }
        }
        
        return anomalies;
    }
};


//Checking if motor commands align with Odometry
//WNM: It iterates because it assumes data is in vectors, which will not be the case when connected to Kubernetes
class JointStatesValidator : public PhysicsValidator {
private:
    vector<botData>& data;
    
    const double WHEEL_RADIUS = 0.033;  // TurtleBot3 wheel radius in meters
    const double WHEEL_BASE = 0.160;    // Distance between wheels in meters
    const double VEL_TOLERANCE = 0.14;
    const double ANGULAR_TOLERANCE = 0.53; // rad/s tolerance
    
    int consecutive_linear_violations = 0;
    int consecutive_angular_violations = 0;
    const int VIOLATION_THRESHOLD = 3;
    
public:
    JointStatesValidator(vector<botData>& bot_data) : data(bot_data) {}
    
    string getName() const override {
        return "Joint States Validator";
    }
    
    vector<Anomaly> validate() override {
        vector<Anomaly> anomalies;
        const double WARMUP_TIME = 10.0;
        
        for (size_t i = 1; i < data.size(); ++i) {
            botData& curr = data[i];
            botData& prev = data[i-1];
            
            if (curr.timestamp < WARMUP_TIME) continue;
            
            double dt = curr.timestamp - prev.timestamp;
            if (dt <= 0 || dt > 1.0) continue;
            
            // CALCULATE VELOCITIES FROM WHEEL ENCODERS 
            double linear_from_wheels = (curr.v_R + curr.v_L) / 2.0 * WHEEL_RADIUS;
            double angular_from_wheels = (curr.v_R - curr.v_L) / WHEEL_BASE * WHEEL_RADIUS;
            
            // CALCULATE VELOCITIES FROM ODOMETRY 
            double dx = curr.x - prev.x;
            double dy = curr.y - prev.y;
            double linear_from_odom = sqrt(dx*dx + dy*dy) / dt;
            
            double dtheta = curr.theta - prev.theta;
            while (dtheta > M_PI) dtheta -= 2*M_PI;
            while (dtheta < -M_PI) dtheta += 2*M_PI;
            double angular_from_odom = dtheta / dt;
            
            // Skip if wheel data is zero (might be missing/invalid)
            if (curr.v_R == 0 && curr.v_L == 0 && linear_from_odom > 0.01) {
                continue;  // Wheels report zero but robot is moving - data issue
            }
            
            // CHECK 1: LINEAR VELOCITY CONSISTENCY 
            double linear_diff = std::abs(linear_from_wheels - linear_from_odom);
            
            if (linear_diff > VEL_TOLERANCE) {
                consecutive_linear_violations++;
                
                if (consecutive_linear_violations >= VIOLATION_THRESHOLD) {
                    Anomaly a;
                    a.timestamp = curr.timestamp;
                    a.type = "JOINT_ODOM_LINEAR_MISMATCH";
                    a.description = "Wheel-derived vel: " + to_string(linear_from_wheels) + 
                                  " m/s, Odom vel: " + to_string(linear_from_odom) + 
                                  " m/s (persisted " + to_string(consecutive_linear_violations) + " samples)";
                    a.severity = min(1.0, linear_diff / 0.22);

                    anomalies.push_back(a);
                }
            } else {
                consecutive_linear_violations = 0;
            }
            
            // ========== CHECK 2: ANGULAR VELOCITY CONSISTENCY ==========
            double angular_diff = std::abs(angular_from_wheels - angular_from_odom);
            
            if (angular_diff > ANGULAR_TOLERANCE) {
                consecutive_angular_violations++;
                
                if (consecutive_angular_violations >= VIOLATION_THRESHOLD) {
                    Anomaly a;
                    a.timestamp = curr.timestamp;
                    a.type = "JOINT_ODOM_ANGULAR_MISMATCH";
                    a.description = "Wheel-derived ang: " + to_string(angular_from_wheels) + 
                                  " rad/s, Odom ang: " + to_string(angular_from_odom) + 
                                  " rad/s (persisted " + to_string(consecutive_angular_violations) + " samples)";
                    a.severity = min(1.0, angular_diff / 2.84);

                    anomalies.push_back(a);
                }
            } else {
                consecutive_angular_violations = 0;
            }
        }
        
        return anomalies;
    }
};


//Comparing with official parameters from TurtleBot3
class PhysicalLimitsValidator : public PhysicsValidator {
private:
    vector<botData>& data;
    
    const double MAX_LINEAR_VEL = 0.3;      // m/s
    const double MAX_ANGULAR_VEL = 2.84;     // rad/s
    const double MAX_IMU_ACCEL = 2.0;        // m/s^2 
    const double MAX_IMU_GYRO = 2.84;        // rad/s
    
public:
    PhysicalLimitsValidator(vector<botData>& bot_data) : data(bot_data) {}
    
    string getName() const override {
        return "Physical Limits Validator";
    }
    
    vector<Anomaly> validate() override {
        vector<Anomaly> anomalies;
        const double WARMUP_TIME = 10.0;
        
        for (size_t i = 1; i < data.size(); ++i) {
            botData& curr = data[i];
            botData& prev = data[i-1];
            
            if (curr.timestamp < WARMUP_TIME) continue;
            
            double dt = curr.timestamp - prev.timestamp;
            if (dt <= 0 || dt > 1.0) continue;
            
            // CHECK 1: COMMANDED LINEAR VELOCITY 
            if (std::abs(curr.linear_x) > MAX_LINEAR_VEL) {
                Anomaly a;
                a.timestamp = curr.timestamp;
                a.type = "SPEC_VIOLATION_CMD_LINEAR";
                a.description = "Commanded linear velocity " + to_string(curr.linear_x) + 
                              " m/s exceeds TurtleBot3 Burger specification (" + 
                              to_string(MAX_LINEAR_VEL) + " m/s)";
                a.severity = 1.0;
                anomalies.push_back(a);
            }
            
            // CHECK 2: COMMANDED ANGULAR VELOCITY 
            if (std::abs(curr.angular_z) > MAX_ANGULAR_VEL) {
                Anomaly a;
                a.timestamp = curr.timestamp;
                a.type = "SPEC_VIOLATION_CMD_ANGULAR";
                a.description = "Commanded angular velocity " + to_string(curr.angular_z) + 
                              " rad/s exceeds TurtleBot3 Burger specification (" + 
                              to_string(MAX_ANGULAR_VEL) + " rad/s)";
                a.severity = 1.0;
                anomalies.push_back(a);
            }
            
            // CHECK 3: ACTUAL LINEAR VELOCITY FROM ODOM (MOVING AVERAGE)
            // Use 5-sample window to filter noise while preserving attack signatures
            const int VELOCITY_WINDOW = 7;
            double sum_dx = 0, sum_dy = 0, sum_dt = 0;
            int valid_samples = 0;
            
            for (int j = 0; j < VELOCITY_WINDOW && i >= (size_t)(j + 1); j++) {
                botData& c = data[i - j];
                botData& p = data[i - j - 1];
                double local_dt = c.timestamp - p.timestamp;
                
                // Include all reasonable dt values (not just perfect ones)
                if (local_dt > 0 && local_dt < 1.0) {
                    sum_dx += c.x - p.x;
                    sum_dy += c.y - p.y;
                    sum_dt += local_dt;
                    valid_samples++;
                }
            }
            
            // Only check if we have enough valid samples
            if (valid_samples < 2) continue;
            
            double linear_vel = sqrt(sum_dx*sum_dx + sum_dy*sum_dy) / sum_dt;
            
            if (linear_vel > MAX_LINEAR_VEL) {
                Anomaly a;
                a.timestamp = curr.timestamp;
                a.type = "SPEC_VIOLATION_ODOM_LINEAR";
                a.description = "Smoothed odom linear velocity " + to_string(linear_vel) + 
                              " m/s exceeds physical limits (" + 
                              to_string(MAX_LINEAR_VEL) + " m/s) [" + 
                              to_string(valid_samples) + "-sample average]";
                a.severity = 1.0;
                anomalies.push_back(a);
            }
            
            // CHECK 4: ACTUAL ANGULAR VELOCITY FROM ODOM (MOVING AVERAGE)
            // Use 5-sample window to filter noise while preserving attack signatures
            double sum_dtheta = 0;
            sum_dt = 0;
            valid_samples = 0;
            
            for (int j = 0; j < VELOCITY_WINDOW && i >= (size_t)(j + 1); j++) {
                botData& c = data[i - j];
                botData& p = data[i - j - 1];
                double local_dt = c.timestamp - p.timestamp;
                
                if (local_dt > 0 && local_dt < 1.0) {
                    double dtheta = c.theta - p.theta;
                    // Handle angle wrapping
                    while (dtheta > M_PI) dtheta -= 2*M_PI;
                    while (dtheta < -M_PI) dtheta += 2*M_PI;
                    sum_dtheta += dtheta;
                    sum_dt += local_dt;
                    valid_samples++;
                }
            }
            
            if (valid_samples < 2) continue;
            
            double angular_vel = std::abs(sum_dtheta / sum_dt);
            
            if (angular_vel > MAX_ANGULAR_VEL) {
                Anomaly a;
                a.timestamp = curr.timestamp;
                a.type = "SPEC_VIOLATION_ODOM_ANGULAR";
                a.description = "Smoothed odom angular velocity " + to_string(angular_vel) + 
                              " rad/s exceeds physical limits (" + 
                              to_string(MAX_ANGULAR_VEL) + " rad/s) [" + 
                              to_string(valid_samples) + "-sample average]";
                a.severity = 1.0;
                anomalies.push_back(a);
            }
            
            // CHECK 5: IMU ACCELERATION 
            double horizontal_accel = sqrt(curr.accel_x * curr.accel_x + 
                                          curr.accel_y * curr.accel_y);

            if (horizontal_accel > MAX_IMU_ACCEL) {
                Anomaly a;
                a.timestamp = curr.timestamp;
                a.type = "IMU_ACCEL_EXCEEDED";
                a.description = "IMU acceleration " + to_string(horizontal_accel) + 
                              " m/s^2 exceeds reasonable limits (" + 
                              to_string(MAX_IMU_ACCEL) + " m/s^2)";
                a.severity = 0.9;
                anomalies.push_back(a);
            }
            
            // CHECK 6: IMU GYRO 
            if (std::abs(curr.gyro_z) > MAX_IMU_GYRO) {
                Anomaly a;
                a.timestamp = curr.timestamp;
                a.type = "IMU_GYRO_EXCEEDED";
                a.description = "IMU gyro " + to_string(curr.gyro_z) + 
                              " rad/s exceeds reasonable limits (" + 
                              to_string(MAX_IMU_GYRO) + " rad/s)";
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
    vector<botData>& data;
    const double MAX_ERROR_RATE = 1.89;      // m/s - tracking error rate
    const double MAX_POS_ERROR_RATE = 1.48;  // m/s - position error rate
    const double WARMUP_TIME = 10.0;
    const double COOLDOWN_TIME = 2.0;
    const double MIN_DT = 0.0016;  // Based on 5th percentile × 0.5
    double MAX_DT = 0.0135;        // Based on 95th percentile × 2.0
    const double SPIKE_FILTER = 50.0;
    const int VIOLATION_THRESHOLD = 3;
    
    int consec_tr_vlts = 0;
    int consec_pos_vlts = 0;

public:
    TrackingErrorValidator(vector<botData>& bot_data) : data(bot_data) {
        calculateMaxDT();
    }

    string getName() const override {  
        return "Tracking Error Validator";
    }

    void calculateMaxDT() {
        vector<double> dts;
        int zero_dt = 0;
        int tiny_dt = 0;
        
        for (size_t i = 1; i < data.size(); ++i) {
            double dt = data[i].timestamp - data[i-1].timestamp;
            
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
        
        MAX_DT = median_dt * 10.0;
    }
    
    vector<Anomaly> validate() override {
        vector<Anomaly> anomalies;
        if (data.size() < 2) return anomalies;
        
        double max_timestamp = data.back().timestamp;
  
        for (size_t i = 1; i < data.size(); ++i) {
            botData& curr = data[i];
            botData& prev = data[i-1];
            
            // Skip warmup and cooldown periods
            if (curr.timestamp < WARMUP_TIME) continue;
            if (curr.timestamp > max_timestamp - COOLDOWN_TIME) continue;
  
            double dt = curr.timestamp - prev.timestamp;
  
            if (dt <= MIN_DT || dt > MAX_DT) {
                consec_pos_vlts = 0;
                consec_tr_vlts = 0;
                continue;
            }
            
            // CHECK 1: TRACKING ERROR RATE 
            double error_rate = std::abs(curr.tracking_error - prev.tracking_error) / dt;
            
            // Filter out spikes
            if (error_rate > SPIKE_FILTER) {
                consec_tr_vlts = 0;
                continue;
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
                consec_tr_vlts = 0;
            }
            
            // CHECK 2: POSITION ERROR RATE
            double prev_pos_error = sqrt(prev.error_x * prev.error_x + 
                                        prev.error_y * prev.error_y);
            double curr_pos_error = sqrt(curr.error_x * curr.error_x + 
                                        curr.error_y * curr.error_y);
            double pos_error_rate = std::abs(curr_pos_error - prev_pos_error) / dt;
            
            // Filter out spikes
            if (pos_error_rate > SPIKE_FILTER) {
                consec_pos_vlts = 0;
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
                consec_pos_vlts = 0;
            }
        }  
        
        return anomalies;
    }
};

// Main Intrusion Detector
class IntrusionDetector {
private:
    vector<botData> data;  
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
    
    int countUniqueTimestamps() {
        set<double> unique_timestamps;
        for (const auto& a : all_anomalies) {
            unique_timestamps.insert(a.timestamp);
        }
        return unique_timestamps.size();
    }
    
public:
    IntrusionDetector() {}
    
    ~IntrusionDetector() {
        for (auto validator : validators) {
            delete validator;
        }
    }
    
    // SINGLE UNIFIED CSV LOADER 
    bool loadUnifiedCSV(const string& filename) {
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
            
            if (fields.size() >= 22) {  // Adjust based on your CSV
                botData bot;
                try {
                    int idx = 0;
                    bot.timestamp = stod(fields[idx++]);
                    
                    // Odometry
                    bot.x = stod(fields[idx++]);
                    bot.y = stod(fields[idx++]);
                    bot.theta = stod(fields[idx++]);
                    
                    // Command Velocity
                    bot.linear_x = stod(fields[idx++]);
                    bot.angular_z = stod(fields[idx++]);
                    
                    // IMU
                    bot.accel_x = stod(fields[idx++]);
                    bot.accel_y = stod(fields[idx++]);
                    bot.accel_z = stod(fields[idx++]);
                    bot.gyro_x = stod(fields[idx++]);
                    bot.gyro_y = stod(fields[idx++]);
                    bot.gyro_z = stod(fields[idx++]);
                    
                    // Joint States
                    bot.v_R = stod(fields[idx++]);
                    bot.v_L = stod(fields[idx++]);
                    bot.right_encoder = stod(fields[idx++]);
                    bot.left_encoder = stod(fields[idx++]);
                    
                    // Tracking Error
                    bot.tracking_error = stod(fields[idx++]);
                    bot.error_x = stod(fields[idx++]);
                    bot.error_y = stod(fields[idx++]);

                    // Battery
                    bot.voltage = stod(fields[idx++]);
                    bot.current = stod(fields[idx++]);
                    bot.percentage = stod(fields[idx++]);

                    idx++;
                    
                    data.push_back(bot);
                } catch (const exception& e) {
                    cerr << "Error parsing line: " << e.what() << endl;
                }
            }
        }
        
        cout << "Loaded " << data.size() << " unified samples" << endl;
        return !data.empty();
    }
    
    void initializeValidators() {
        // All validators now use the same unified data
        validators.push_back(new CmdVelOdomValidator(data));
        validators.push_back(new PhysicalLimitsValidator(data));
        validators.push_back(new ImuOdomValidator(data));
        validators.push_back(new JointStatesValidator(data));
        validators.push_back(new TrackingErrorValidator(data));
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
            cout << "✓ No anomalies detected - System appears secure" << endl;  // ✅ Fixed extra space
            return;
        }
        
        // Group by type
        map<string, int> anomaly_counts;
        map<string, double> type_severities;
        
        for (const auto& a : all_anomalies) {
            anomaly_counts[a.type]++;
            type_severities[a.type] += a.severity;
        }
        
        cout << "Anomaly Breakdown:" << endl;
        for (const auto& pair : anomaly_counts) {
            double avg_severity = type_severities[pair.first] / pair.second;
            cout << "  " << pair.first << ": " << pair.second 
                 << " (avg severity: " << fixed << setprecision(2) << avg_severity << ")" << endl;
        }
        
        cout << "\nDetailed Anomalies (Top 20 by Severity):" << endl;
        cout << "----------------------------------------" << endl;
        
        // Sort by severity
        vector<Anomaly> sorted_anomalies = all_anomalies;
        sort(sorted_anomalies.begin(), sorted_anomalies.end(), 
             [](const Anomaly& a, const Anomaly& b) {
                 return a.severity > b.severity;
             });
        
        int count = 0;
        for (const auto& a : sorted_anomalies) {
            if (count++ >= 20) break;
            
            cout << fixed << setprecision(3);
            cout << "[" << a.timestamp << "s] " 
                 << a.type << " (severity: " << a.severity << ")" << endl;
            cout << "  " << a.description << endl;
        }
        
        // Calculate statistics
        double avg_severity = 0;
        double max_severity = 0;
        int high_severity_count = 0;
        
        for (const auto& a : all_anomalies) {
            avg_severity += a.severity;
            max_severity = max(max_severity, a.severity);
            if (a.severity >= 0.7) high_severity_count++;
        }
        avg_severity /= all_anomalies.size();
        
        // Count critical types
        int sign_mismatches = anomaly_counts["LINEAR_SIGN_MISMATCH"] + 
                             anomaly_counts["ANGULAR_SIGN_MISMATCH"];
        int spec_violations = anomaly_counts["SPEC_VIOLATION_CMD_LINEAR"] +
                             anomaly_counts["SPEC_VIOLATION_CMD_ANGULAR"] +
                             anomaly_counts["SPEC_VIOLATION_ODOM_LINEAR"] +
                             anomaly_counts["SPEC_VIOLATION_ODOM_ANGULAR"];
        
        cout << "\n========================================" << endl;
        cout << "         DETECTION STATISTICS" << endl;
        cout << "========================================" << endl;
        cout << "Total Timestamps Evaluated:    " << data.size() << endl;
        cout << "Timestamps with Anomalies:     " << countUniqueTimestamps() << endl;
        cout << "Total Anomaly Instances:       " << all_anomalies.size() << endl;
        cout << "Average Severity:              " << fixed << setprecision(2) << avg_severity << endl;
        cout << "Maximum Severity:              " << fixed << setprecision(2) << max_severity << endl;
        cout << "High Severity Instances:       " << high_severity_count << " (≥0.70)" << endl;
        cout << "Direction Reversals:           " << sign_mismatches << endl;
        cout << "Specification Violations:      " << spec_violations << endl;
        cout << "========================================\n" << endl;
    }
}; 

int runIntrusionDetection(const std::string& unified_csv_file)
{
    try {
        auto start_total = std::chrono::high_resolution_clock::now();
        
        IntrusionDetector detector;
        
        // Load unified CSV with timing
        auto start_load = std::chrono::high_resolution_clock::now();
        
        if (!detector.loadUnifiedCSV(unified_csv_file)) {
            std::cerr << "Failed to load data from " << unified_csv_file << std::endl;
            return 1;
        }
        
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