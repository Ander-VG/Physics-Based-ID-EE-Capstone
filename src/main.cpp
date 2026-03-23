#include <iostream>
#include <vector>
#include <mutex>
#include <cmath>
#include <memory>

// ROS2
#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

#include "sensor_det.h"

using std::placeholders::_1;
using std::vector;

enum State { NORMAL, ALERT };

double yaw_transform(double x, double y, double z, double w)
{
    double siny_cosp = 2.0 * (w * z + x * y);
    double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
    return std::atan2(siny_cosp, cosy_cosp);
}

class DetectorNode : public rclcpp::Node
{
public:
    DetectorNode()
    : Node("detector_node"),
      curr_state_(NORMAL),
      good_samps_(0),
      clk_(250)
    {
        auto cb_group = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
        auto sub_opts = rclcpp::SubscriptionOptions();
        sub_opts.callback_group = cb_group;

        odometry_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "odom", 10, std::bind(&DetectorNode::odometry_callback, this, _1), sub_opts);

        velocity_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "cmd_vel", 10, std::bind(&DetectorNode::velocity_callback, this, _1), sub_opts);

        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "imu", 10, std::bind(&DetectorNode::imu_callback, this, _1), sub_opts);

        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "joint_states", 10, std::bind(&DetectorNode::js_callback, this, _1), sub_opts);

        error_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "tracking_error", 10, std::bind(&DetectorNode::error_callback, this, _1), sub_opts);

        RCLCPP_INFO(this->get_logger(), "Detector node started. Detection driven by cmd_vel.");
    }

private:

    // ── Cached sensor values (fill-forward) ──────────────────────────────────

    std::mutex data_mutex_;

    float odom_x_ = 0, odom_y_ = 0, odom_theta_ = 0;
    float accel_x_ = 0, accel_y_ = 0, accel_z_ = 0;
    float gyro_x_  = 0, gyro_y_  = 0, gyro_z_  = 0;
    float v_R_ = 0, v_L_ = 0;
    float error_x_ = 0, error_y_ = 0;

    // ── FSM state ────────────────────────────────────────────────────────────

    State curr_state_;
    int good_samps_;
    const int clk_;

    // ── Callbacks ────────────────────────────────────────────────────────────

    void odometry_callback(const nav_msgs::msg::Odometry & msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        odom_x_     = msg.pose.pose.position.x;
        odom_y_     = msg.pose.pose.position.y;
        odom_theta_ = yaw_transform(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w);
    }

    void velocity_callback(const geometry_msgs::msg::Twist & msg) {
        float lin_x = msg.linear.x;
        float ang_z = msg.angular.z;

        vector<float> sample(15);

        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            sample = {
                odom_x_, odom_y_, odom_theta_, lin_x, ang_z,
                accel_x_, accel_y_, accel_z_, gyro_x_, gyro_y_,
                gyro_z_, v_R_, v_L_,
                error_x_, error_y_
            };
        }

        welford_results result = detector(sample);

        if (!result.calibrated || !result.anomaly) {
            welford_calibration(sample);
        }

        if (!result.calibrated) return;

        switch (curr_state_) {
            case NORMAL:
                if (result.anomaly) {
                    RCLCPP_WARN(this->get_logger(),
                        "ALERT  - Sensor spoofing detected  | T2 = %.2f  (thr = %.2f)",
                        result.t2_score, result.threshold);
                    curr_state_ = ALERT;
                    good_samps_ = 0;
                }
                break;

            case ALERT:
                if (result.anomaly) {
                    good_samps_ = 0;
                } else {
                    good_samps_++;
                    if (good_samps_ >= clk_) {
                        RCLCPP_INFO(this->get_logger(),
                            "NORMAL - System returned to normal | T2 = %.2f  (thr = %.2f)",
                            result.t2_score, result.threshold);
                        curr_state_ = NORMAL;
                        good_samps_ = 0;
                    }
                }
                break;
        }
    }

    void imu_callback(const sensor_msgs::msg::Imu & msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        accel_x_ = msg.linear_acceleration.x;
        accel_y_ = msg.linear_acceleration.y;
        accel_z_ = msg.linear_acceleration.z;
        gyro_x_  = msg.angular_velocity.x;
        gyro_y_  = msg.angular_velocity.y;
        gyro_z_  = msg.angular_velocity.z;
    }

    void js_callback(const sensor_msgs::msg::JointState & msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        v_R_ = msg.velocity[1];
        v_L_ = msg.velocity[0];
    }

    void error_callback(const std_msgs::msg::Float32MultiArray & msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        if (msg.data.size() >= 8) {
            error_x_ = msg.data[5];
            error_y_ = msg.data[6];
        }
    }

    // ── Subscription handles ─────────────────────────────────────────────────

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr velocity_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr error_sub_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<DetectorNode>();

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}