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
#include <tuple>
#include <cstring>


#define M_PI 3.14159265358979323846

using std::string;
using std::pair;
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
using std::ofstream;
using std::get;
using std::tuple;
using std::make_tuple;
using std::get;
using std::memcpy;


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
    
    int consecutive_angular_violations = 0;
    int consecutive_linear_violations = 0;
    const int VIOLATION_THRESHOLD = 3;
    const int LAG_SAMPLES = 1;
    
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
                angular_tolerance *= 5;  
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
                std::abs(lagged_cmd.linear_x) > 0 &&
                std::abs(actual_linear_signed) > 0 && 
                lagged_cmd.linear_x * actual_linear_signed < 0) {
                
                Anomaly a;
                a.timestamp = curr.timestamp;
                a.type = "LINEAR_SIGN_MISMATCH";
                a.description = "Direction reversal - Commanded: " + 
                              to_string(lagged_cmd.linear_x) + " m/s, Actual: " + 
                              to_string(actual_linear_signed) + " m/s";
                a.severity = min(1.0, std::abs(actual_linear_signed)/lagged_cmd.linear_x);
                
                if (curr.voltage > 0) {
                    a.description += " [Battery: " + to_string(curr.voltage) + "V]";
                }
                
                anomalies.push_back(a);
            }
            
            // CHECK 4: ANGULAR DIRECTION REVERSAL 
            bool is_mostly_straight = std::abs(lagged_cmd.linear_x) > 0.08 &&  std::abs(lagged_cmd.angular_z) < 0.2; 

            if (battery_healthy_for_direction && is_mostly_straight &&
                std::abs(lagged_cmd.angular_z) > 0 &&
                std::abs(actual_angular) > 0 && 
                lagged_cmd.angular_z * actual_angular < 0) {
                
                Anomaly a;
                a.timestamp = curr.timestamp;
                a.type = "ANGULAR_SIGN_MISMATCH";
                a.description = "Rotation reversal - Commanded: " + 
                              to_string(lagged_cmd.angular_z) + " rad/s, Actual: " + 
                              to_string(actual_angular) + " rad/s";
                a.severity = min(1.0, std::abs(actual_angular)/lagged_cmd.angular_z);
                
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
        else if (voltage > 11.0) return 0.5;
        else if (voltage > 10.5) return 0.55;
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
        const double WARMUP_TIME = 15.0;
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
    const double VEL_TOLERANCE = 0.08;
    const double ANGULAR_TOLERANCE = 0.53; // rad/s tolerance
    
    int consecutive_linear_violations = 0;
    int consecutive_angular_violations = 0;
    const int VIOLATION_THRESHOLD = 3;

    //Cumulative position tracking
    double cumulative_wheel_x = 0.0;
    double cumulative_wheel_y = 0.0;
    double cumulative_wheel_theta = 0.0;
    double last_reset_time = 0.0;
    bool position_initialized = false;
    
    const double RESET_INTERVAL = 20.0;  // Reset every 20 seconds to prevent drift accumulation
    const double MAX_POSITION_DRIFT = 0.25;  // 25cm threshold
    int consecutive_drift_violations = 0;

    double last_drift_report = 0.0;  // Add this
    const double DRIFT_REPORT_COOLDOWN = 5.0;  // Report every 5 seconds
    
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

            //initialize and reset of cumulative position
            if (!position_initialized || (curr.timestamp - last_reset_time) >= RESET_INTERVAL) {
                cumulative_wheel_x = curr.x;
                cumulative_wheel_y = curr.y;
                cumulative_wheel_theta = curr.theta;
                last_reset_time = curr.timestamp;
                position_initialized = true;
                consecutive_drift_violations = 0;  // Reset violation counter
            }
            
            // CALCULATE VELOCITIES FROM WHEEL ENCODERS 
            double linear_from_wheels = (curr.v_R + curr.v_L) / 2.0 * WHEEL_RADIUS;
            double angular_from_wheels = (curr.v_R - curr.v_L) / WHEEL_BASE * WHEEL_RADIUS;

            cumulative_wheel_theta += angular_from_wheels * dt;
            
            // Normalize theta to [-pi, pi]
            while (cumulative_wheel_theta > M_PI) cumulative_wheel_theta -= 2*M_PI;
            while (cumulative_wheel_theta < -M_PI) cumulative_wheel_theta += 2*M_PI;
            
            // Update x, y based on wheel velocities and current heading
            cumulative_wheel_x += linear_from_wheels * cos(cumulative_wheel_theta) * dt;
            cumulative_wheel_y += linear_from_wheels * sin(cumulative_wheel_theta) * dt;
            
            // CALCULATE POSITION DRIFT (wheel integration vs reported odom)
            double position_drift = sqrt(
                pow(curr.x - cumulative_wheel_x, 2) + 
                pow(curr.y - cumulative_wheel_y, 2)
            );
            
            // CHECK 1: CUMULATIVE POSITION DRIFT
            if (position_drift > MAX_POSITION_DRIFT) {
                consecutive_drift_violations++;
                
                if (consecutive_drift_violations >= VIOLATION_THRESHOLD &&
                (curr.timestamp - last_drift_report) > DRIFT_REPORT_COOLDOWN) {
                    Anomaly a;
                    a.timestamp = curr.timestamp;
                    a.type = "WHEEL_ODOM_POSITION_DRIFT";
                    a.description = "Position drift " + to_string(position_drift) + 
                                  " m between wheel-integrated position and reported odometry " +
                                  "(persisted " + to_string(consecutive_drift_violations) + 
                                  " samples) - indicates odometry spoofing";
                    a.severity = min(1.0, position_drift / 0.5);  // Scale to 50cm
                    
                    anomalies.push_back(a);
                    last_drift_report = curr.timestamp;
                }
            } else {
                consecutive_drift_violations = 0;
            }
            
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
            
            // CHECK 2: LINEAR VELOCITY CONSISTENCY 
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
            
            // CHECK 3: ANGULAR VELOCITY CONSISTENCY 
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
                a.severity = min(1.0, curr.linear_x/MAX_LINEAR_VEL);
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
                a.severity = min(1.0, curr.angular_z/MAX_ANGULAR_VEL);
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
                a.severity = min(1.0, linear_vel/MAX_LINEAR_VEL);
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
                a.severity = min(1.0, angular_vel/MAX_ANGULAR_VEL);
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
                a.severity = min(1.0, horizontal_accel/MAX_IMU_ACCEL);
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
                a.severity = min(1.0, curr.gyro_z/MAX_IMU_GYRO);
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
    const double MAX_ABSOLUTE_ERROR = 0.5;
    bool absolute_error_flagged = false;  
    double last_absolute_error_report = 0.0;  
    const double ERROR_REPORT_COOLDOWN = 5.0;  // Report every 5 seconds
    
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

            // CHECK: ABSOLUTE TRACKING ERROR (with cooldown)
            if (curr.tracking_error > MAX_ABSOLUTE_ERROR) {
                if (!absolute_error_flagged || 
                    (curr.timestamp - last_absolute_error_report) > ERROR_REPORT_COOLDOWN) {
                    
                    Anomaly a;
                    a.timestamp = curr.timestamp;
                    a.type = "ABSOLUTE_TRACKING_ERROR_EXCEEDED";
                    a.description = "Cumulative tracking error " + to_string(curr.tracking_error) + 
                                " m exceeds threshold (" + to_string(MAX_ABSOLUTE_ERROR) + 
                                " m) - indicates sustained position spoofing";
                    a.severity = min(1.0, curr.tracking_error / 1.0);
                    anomalies.push_back(a);
                    
                    absolute_error_flagged = true;
                    last_absolute_error_report = curr.timestamp;
                }
            }

            double dt = curr.timestamp - prev.timestamp;

            if (dt <= MIN_DT || dt > MAX_DT) {
                consec_pos_vlts = 0;
                consec_tr_vlts = 0;
                continue;
            }
            
            // CHECK 1: TRACKING ERROR RATE 
            double error_rate = std::abs(curr.tracking_error - prev.tracking_error) / dt;
            
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

            // REMOVED DUPLICATE CHECK HERE!
            
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


//Adding 2 Validators for Replay Attack detection
//Hash Validator works with less information, works in O(1) and perfectly catches unmodified replay attacks
class HashReplayValidator : public PhysicsValidator {
private:
    vector<botData>& data;
    const int HASH_WINDOW = 25;      // 0.126 seconds at 100Hz
    const int MIN_SEPARATION = 395;  // 2 seconds minimum between occurrences
    const double WARMUP_TIME = 10.0;
    
    map<size_t, vector<size_t>> hash_to_indices;  // hash -> sample indices

public:
    HashReplayValidator(vector<botData>& bot_data) : data(bot_data) {}
    
    string getName() const override {
        return "Hash-Based Replay Validator";
    }
    
    vector<Anomaly> validate() override {
        vector<Anomaly> anomalies;
        set<size_t> reported_hashes;  // Avoid duplicate reports
        int total_sequences = 0;
        int collisions_checked = 0;
        
        cout << "  Building hash database..." << endl;
        
        // First pass: Build hash database
        for (size_t i = HASH_WINDOW; i < data.size(); ++i) {
            if (data[i].timestamp < WARMUP_TIME) continue;
            
            size_t hash = computeWindowHash(i - HASH_WINDOW, i);
            hash_to_indices[hash].push_back(i);
            total_sequences++;
        }
        
        cout << "  Checking " << total_sequences << " sequences against " 
             << hash_to_indices.size() << " unique hashes..." << endl;
        
        // Second pass: Detect replays
        for (const auto& hash_entry : hash_to_indices) {
            size_t hash = hash_entry.first;
            const vector<size_t>& indices = hash_entry.second;
            
            // Skip if this hash only appears once
            if (indices.size() < 2) continue;
            
            // Check all pairs of occurrences
            for (size_t i = 0; i < indices.size(); ++i) {
                for (size_t j = i + 1; j < indices.size(); ++j) {
                    size_t idx1 = indices[i];
                    size_t idx2 = indices[j];
                    
                    // Must be separated by at least MIN_SEPARATION samples
                    if (idx2 - idx1 < MIN_SEPARATION) continue;
                    
                    collisions_checked++;
                    
                    // Verify exact match (not just hash collision)
                    if (verifyExactMatch(idx1 - HASH_WINDOW, idx1, 
                                        idx2 - HASH_WINDOW, idx2)) {
                        
                        // Report only once per unique hash
                        if (reported_hashes.find(hash) == reported_hashes.end()) {
                            Anomaly a;
                            a.timestamp = data[idx2].timestamp;
                            a.type = "EXACT_REPLAY_DETECTED";
                            a.description = "Exact " + to_string(HASH_WINDOW) + 
                                          "-sample sequence replay detected. " +
                                          "Original at t=" + to_string(data[idx1].timestamp) + 
                                          "s, Replayed at t=" + to_string(data[idx2].timestamp) + 
                                          "s (" + to_string((idx2 - idx1) / 200.0) + 
                                          "s separation) - byte-for-byte match across 7 sensor streams";
                            a.severity = 1.0;  // Maximum severity - definitive attack
                            anomalies.push_back(a);
                            
                            reported_hashes.insert(hash);
                        }
                    }
                }
            }
        }
        
        cout << "  Verified " << collisions_checked << " potential collisions" << endl;
        
        return anomalies;
    }
    
private:
    // Compute hash using boost-style hash_combine for window
    size_t computeWindowHash(size_t start, size_t end) {
        size_t hash = 0;
        
        for (size_t i = start; i < end && i < data.size(); ++i) {
            // Hash multiple critical fields
            // Using boost::hash_combine algorithm
            hash ^= hashDouble(data[i].x) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= hashDouble(data[i].y) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= hashDouble(data[i].theta) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= hashDouble(data[i].linear_x) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= hashDouble(data[i].angular_z) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= hashDouble(data[i].v_R) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= hashDouble(data[i].v_L) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        
        return hash;
    }
    
    // Convert double to hash value
    size_t hashDouble(double val) {
        uint64_t bits;
        memcpy(&bits, &val, sizeof(double));
        return std::hash<uint64_t>{}(bits);
    }
    
    // Verify that two windows are exactly identical
    bool verifyExactMatch(size_t start1, size_t end1, 
                         size_t start2, size_t end2) {
        const double EPSILON = 1e-9;  // Tight tolerance for exact match
        
        for (size_t offset = 0; offset < (end1 - start1); ++offset) {
            const botData& d1 = data[start1 + offset];
            const botData& d2 = data[start2 + offset];
            
            // Check all critical fields
            if (fabs(d1.x - d2.x) > EPSILON ||
                fabs(d1.y - d2.y) > EPSILON ||
                fabs(d1.theta - d2.theta) > EPSILON ||
                fabs(d1.linear_x - d2.linear_x) > EPSILON ||
                fabs(d1.angular_z - d2.angular_z) > EPSILON ||
                fabs(d1.v_R - d2.v_R) > EPSILON ||
                fabs(d1.v_L - d2.v_L) > EPSILON) {
                return false;
            }
        }
        
        return true;
    }
};


//The Fourier Validator is capable of detecting slightly modified Replay Attacks
class FFTReplayValidator : public PhysicsValidator {
private:
    vector<botData>& data;
    
    // FFT parameters
    const int WINDOW_SIZE = 128;      // Power of 2 for efficient FFT (0.64s at 200Hz)
    const int STRIDE = 64;            // 50% overlap (0.32s at 200Hz)
    const double SIMILARITY_THRESHOLD = 0.995;  // Cosine similarity threshold
    const int MIN_MATCH_COUNT = 3;    // Minimum matching historical windows
    const double MIN_SEPARATION = 2.0; // Seconds between original and replay
    const double WARMUP_TIME = 10.0;
    const double REPORT_COOLDOWN = 1.0; // Avoid duplicate reports within 1 second
    
    // Spectral fingerprint
    struct Fingerprint {
        vector<double> power_spectrum;
        double timestamp;
        size_t sample_index;
        
        Fingerprint() : timestamp(0), sample_index(0) {}
    };
    
    vector<Fingerprint> history;
    const int MAX_HISTORY = 50;  // Keep last ~25 seconds at 50% overlap
    set<double> reported_timestamps; // Track reported attacks to avoid duplicates
    
public:
    FFTReplayValidator(vector<botData>& bot_data) : data(bot_data) {}
    
    string getName() const override {
        return "FFT-Based Replay Validator";
    }
    
    vector<Anomaly> validate() override {
        vector<Anomaly> anomalies;
        
        if (data.size() < WINDOW_SIZE) {
            cout << "  Insufficient data for FFT (need " << WINDOW_SIZE 
                 << " samples, have " << data.size() << ")" << endl;
            return anomalies;
        }
        
        int window_count = 0;
        int replay_detections = 0;
        
        // Slide window through data
        for (size_t i = 0; i + WINDOW_SIZE <= data.size(); i += STRIDE) {
            double window_time = data[i].timestamp;
            
            if (window_time < WARMUP_TIME) continue;
            
            // Extract multi-sensor composite signal
            vector<double> signal = extractCompositeSignal(i, WINDOW_SIZE);
            
            // Compute power spectrum
            Fingerprint current_fp;
            current_fp.timestamp = window_time;
            current_fp.sample_index = i;
            current_fp.power_spectrum = computePowerSpectrum(signal);
            
            // Compare with historical fingerprints
            int match_count = 0;
            double best_similarity = 0.0;
            double match_timestamp = 0.0;
            
            for (const auto& hist_fp : history) {
                // Enforce minimum time separation
                if (window_time - hist_fp.timestamp < MIN_SEPARATION) continue;
                
                double similarity = computeSpectralSimilarity(
                    current_fp.power_spectrum, 
                    hist_fp.power_spectrum
                );
                
                if (similarity > best_similarity) {
                    best_similarity = similarity;
                    match_timestamp = hist_fp.timestamp;
                }
                
                if (similarity >= SIMILARITY_THRESHOLD) {
                    match_count++;
                }
            }
            
            // Flag suspicious pattern repetition (avoid duplicate reports)
            if (match_count >= MIN_MATCH_COUNT && best_similarity >= SIMILARITY_THRESHOLD) {
                // Check if we've recently reported near this timestamp
                bool already_reported = false;
                for (double reported_time : reported_timestamps) {
                    if (fabs(window_time - reported_time) < REPORT_COOLDOWN) {
                        already_reported = true;
                        break;
                    }
                }
                
                if (!already_reported) {
                    Anomaly a;
                    a.timestamp = window_time;
                    a.type = "FFT_REPLAY_DETECTED";
                    a.description = "Frequency pattern matches " + to_string(match_count) + 
                                  " previous windows (similarity: " + 
                                  to_string(best_similarity) + 
                                  "). Strongest match at t=" + 
                                  to_string(match_timestamp) + 
                                  "s - indicates replayed motion pattern (robust to offsets/noise)";
                    a.severity = best_similarity;
                    a.consecutive_count = match_count;
                    anomalies.push_back(a);
                    
                    reported_timestamps.insert(window_time);
                    replay_detections++;
                }
            }
            
            // Add to history (FIFO)
            history.push_back(current_fp);
            if (history.size() > MAX_HISTORY) {
                history.erase(history.begin());
            }
            
            window_count++;
        }
        
        cout << "  Analyzed " << window_count << " windows, detected " 
             << replay_detections << " replay patterns" << endl;
        
        return anomalies;
    }
    
private:
    // Extract composite signal from multiple sensors
    vector<double> extractCompositeSignal(size_t start_idx, int length) {
        vector<double> signal(length);
        
        for (int i = 0; i < length; ++i) {
            const botData& d = data[start_idx + i];
            
            // Weighted multi-sensor fusion
            // Higher weights on sensors that are harder to fake consistently
            signal[i] = 
                d.x * 3.0 +           // Position - primary attack target
                d.y * 3.0 + 
                d.theta * 2.0 +       // Orientation
                d.v_R * 1.5 +         // Wheel velocities - hardware dependent
                d.v_L * 1.5 +
                d.gyro_z * 1.0 +      // IMU - independent sensor
                d.linear_x * 0.5 +    // Commands (lower weight)
                d.angular_z * 0.5;
        }
        
        return signal;
    }
    
    // Compute normalized power spectrum using DFT
    vector<double> computePowerSpectrum(const vector<double>& signal) {
        int N = signal.size();
        vector<double> power(N / 2);  // Only positive frequencies needed
        
        // Remove DC component (mean) - makes it robust to constant offsets!
        double mean = 0.0;
        for (double val : signal) mean += val;
        mean /= N;
        
        // Compute DFT
        for (int k = 0; k < N / 2; ++k) {
            double real = 0.0;
            double imag = 0.0;
            
            for (int n = 0; n < N; ++n) {
                double angle = 2.0 * M_PI * k * n / N;
                double centered_sample = signal[n] - mean;  // Remove DC
                real += centered_sample * cos(angle);
                imag -= centered_sample * sin(angle);
            }
            
            // Power = magnitude squared, normalized
            power[k] = (real * real + imag * imag) / (N * N);
        }
        
        // Normalize to unit energy
        double total_power = 0.0;
        for (double p : power) total_power += p;
        
        if (total_power > 1e-10) {
            for (double& p : power) p /= total_power;
        }
        
        return power;
    }
    
    // Compute cosine similarity between two power spectra
    double computeSpectralSimilarity(const vector<double>& spec1, 
                                     const vector<double>& spec2) {
        if (spec1.size() != spec2.size()) return 0.0;
        
        double dot_product = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;
        
        // Focus on frequencies where robot motion occurs (0-10 Hz)
        int max_bin = min((int)spec1.size(), (int)(10.0 * WINDOW_SIZE / 200.0));
        
        for (int i = 1; i < max_bin; ++i) {  // Skip DC component (i=0)
            dot_product += spec1[i] * spec2[i];
            norm1 += spec1[i] * spec1[i];
            norm2 += spec2[i] * spec2[i];
        }
        
        double denominator = sqrt(norm1) * sqrt(norm2);
        return (denominator > 1e-10) ? (dot_product / denominator) : 0.0;
    }
};

// Main Intrusion Detector
class IntrusionDetector {
private:
    vector<botData> data;  
    vector<PhysicsValidator*> validators;
    vector<Anomaly> all_anomalies;

    struct SpikeState{
        bool in_spike;
        double spike_start_time;
        double spike_peak_severity;
        int spike_sample_count;
        
        SpikeState() : in_spike(false), spike_start_time(0), 
                       spike_peak_severity(0), spike_sample_count(0) {}
    };

    SpikeState spike_state;
    
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
            
            if (fields.size() >= 20) {  // Adjust based on your CSV
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
                    bot.percentage = stod(fields[idx++]);
                    
                    data.push_back(bot);
                } catch (const exception& e) {
                    cerr << "Error parsing line: " << e.what() << endl;
                }
            }
        }
        
        cout << "Loaded " << data.size() << " unified samples" << endl;
        return !data.empty();
    }

    void detectSustainedAnomalies(double timestamp, 
                                   double aggregate_severity,
                                   vector<Anomaly>& anomaly_output) {
        SpikeState& spike = spike_state;
        
        const double SPIKE_THRESHOLD = 1.5;      // Entry threshold
        const double NORMAL_THRESHOLD = 0.3;     // Exit threshold
        const double MIN_SPIKE_DURATION = 3.0;   // 3 seconds
        
        // SPIKE ONSET
        if (!spike.in_spike && aggregate_severity >= SPIKE_THRESHOLD) {
            spike.in_spike = true;
            spike.spike_start_time = timestamp;
            spike.spike_peak_severity = aggregate_severity;
            spike.spike_sample_count = 1;
        }
        
        // SPIKE CONTINUATION
        else if (spike.in_spike && aggregate_severity >= NORMAL_THRESHOLD) {
            spike.spike_sample_count++;
            spike.spike_peak_severity = max(spike.spike_peak_severity, 
                                            aggregate_severity);
        }
        
        // SPIKE OFFSET - Check if sustained
        else if (spike.in_spike && aggregate_severity < NORMAL_THRESHOLD) {
            double spike_duration = timestamp - spike.spike_start_time;
            
            // If spike lasted long enough, flag as sustained attack
            if (spike_duration >= MIN_SPIKE_DURATION) {
                Anomaly a;
                a.timestamp = spike.spike_start_time; // Report at onset
                a.type = "SUSTAINED_ANOMALY";
                a.description = "Sustained physics violations for " + 
                              to_string(spike_duration) + "s (peak severity: " + 
                              to_string(spike.spike_peak_severity) + ", samples: " + 
                              to_string(spike.spike_sample_count) + ")";
                a.severity = 1.0; // Always high severity
                a.duration = spike_duration;
                
                anomaly_output.push_back(a);
            }
            
            // Reset spike tracking
            spike.in_spike = false;
            spike.spike_start_time = 0;
            spike.spike_peak_severity = 0;
            spike.spike_sample_count = 0;
        }
    }
    
    void initializeValidators() {
        // Odom Spoofing Attacks
        validators.push_back(new CmdVelOdomValidator(data));
        validators.push_back(new PhysicalLimitsValidator(data));
        validators.push_back(new ImuOdomValidator(data));
        validators.push_back(new JointStatesValidator(data));
        validators.push_back(new TrackingErrorValidator(data));

        // Replay Attacks
        validators.push_back(new HashReplayValidator(data));      
        validators.push_back(new FFTReplayValidator(data));       
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
        
        // APPLY AGGREGATE SEVERITY THRESHOLD + SUSTAINED DETECTION
        map<double, vector<Anomaly>> anomalies_by_timestamp;
        for (const auto& a : all_anomalies) {
            anomalies_by_timestamp[a.timestamp].push_back(a);
        }
        
        vector<Anomaly> filtered_anomalies;
        map<double, double> timestamp_severity_sums;
        vector<Anomaly> sustained_anomalies;
        
        // Process in chronological order for spike detection
        vector<double> sorted_timestamps;
        for (const auto& pair : anomalies_by_timestamp) {
            sorted_timestamps.push_back(pair.first);
        }
        sort(sorted_timestamps.begin(), sorted_timestamps.end());
        
        for (double timestamp : sorted_timestamps) {
            const auto& timestamp_anomalies = anomalies_by_timestamp[timestamp];
            
            // Calculate aggregate severity
            double total_severity = 0.0;
            for (const auto& anomaly : timestamp_anomalies) {
                total_severity += anomaly.severity;
            }
            timestamp_severity_sums[timestamp] = total_severity;
            
            // Check for sustained spikes
            detectSustainedAnomalies(timestamp, total_severity, sustained_anomalies);
            
            // Apply threshold filter
            if (total_severity > 0.6) {
                filtered_anomalies.insert(filtered_anomalies.end(), 
                                        timestamp_anomalies.begin(), 
                                        timestamp_anomalies.end());
            }
        }
        
        // Add sustained anomalies to the filtered set
        filtered_anomalies.insert(filtered_anomalies.end(),
                                sustained_anomalies.begin(),
                                sustained_anomalies.end());
        
        // Add sustained anomalies to the severity map so they appear in CSV
        for (const auto& sa : sustained_anomalies) {
            timestamp_severity_sums[sa.timestamp] = sa.severity;
        }
        
        all_anomalies = filtered_anomalies;
        
        cout << "Total Anomalies Detected: " << all_anomalies.size() << "\n" << endl;
        
        if (all_anomalies.empty()) {
            cout << "✓ No anomalies detected - System appears secure" << endl;
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
        int sustained_attacks = anomaly_counts["SUSTAINED_ANOMALY"];
        
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
        cout << "Sustained Attacks Detected:    " << sustained_attacks << endl;
        cout << "========================================\n" << endl;

        // Create anomaly-annotated CSV
        cout << "\nCreating annotated CSV with anomaly severities..." << endl;
        
        ofstream out("./output/ROSdata_with_anomalies.csv");
        ifstream in("./output/ROSdata.csv");
        
        string header;
        getline(in, header);
        out << header << ",anomaly_severity,anomaly_type,aggregate_severity" << endl;
        
        // Create map of timestamp -> max severity, type, and aggregate
        map<double, tuple<double, string, double>> anomaly_map;
        for (const auto& a : all_anomalies) {
            double t = a.timestamp;
            double aggregate = timestamp_severity_sums[t];
            
            if (anomaly_map.find(t) == anomaly_map.end() || a.severity > get<0>(anomaly_map[t])) {
                anomaly_map[t] = make_tuple(a.severity, a.type, aggregate);
            }
        }
        
        map<double, tuple<double, string, double>> sustained_anomaly_map;
        for (const auto& sa : sustained_anomalies) {
            // Find the closest actual timestamp in the data
            double closest_timestamp = data[0].timestamp;
            double min_diff = std::abs(data[0].timestamp - sa.timestamp);
            
            for (const auto& d : data) {
                double diff = std::abs(d.timestamp - sa.timestamp);
                if (diff < min_diff) {
                    min_diff = diff;
                    closest_timestamp = d.timestamp;
                }
                if (diff > min_diff) break; // Data is sorted, so we can stop
            }
            
            // Store as a special marker with the sustained info
            sustained_anomaly_map[closest_timestamp] = make_tuple(
                sa.severity, 
                "SUSTAINED_ANOMALY", 
                sa.severity
            );
        }
        
        // Write data with anomaly column
        string line;
        while (getline(in, line)) {
            auto fields = this->parseCSVLine(line);
            if (fields.empty()) continue;
            
            double timestamp = stod(fields[0]);
            out << line << ",";
            
            // Check for sustained anomaly first (higher priority)
            if (sustained_anomaly_map.find(timestamp) != sustained_anomaly_map.end()) {
                out << get<0>(sustained_anomaly_map[timestamp]) << "," 
                    << get<1>(sustained_anomaly_map[timestamp]) << ","
                    << get<2>(sustained_anomaly_map[timestamp]);
            }
            else if (anomaly_map.find(timestamp) != anomaly_map.end()) {
                out << get<0>(anomaly_map[timestamp]) << "," 
                    << get<1>(anomaly_map[timestamp]) << ","
                    << get<2>(anomaly_map[timestamp]);
            } else {
                out << "0.0,NONE,0.0";
            }
            out << endl;
        }
        
        in.close();
        out.close();
        
        cout << "Annotated CSV saved to ./output/ROSdata_with_anomalies.csv" << endl;
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