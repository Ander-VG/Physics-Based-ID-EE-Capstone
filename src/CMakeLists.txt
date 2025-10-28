cmake_minimum_required(VERSION 3.8)
project(data_transition)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ROS2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(rosbag2_storage REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(turtlebot3_msgs REQUIRED)

# ============================================
# LIBRARY 1: data_transition (bag conversion)
# ============================================
add_library(data_transition_lib
  data_transition.cpp
)

target_include_directories(data_transition_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(data_transition_lib
  rclcpp
  rosbag2_cpp
  rosbag2_storage
  nav_msgs
  geometry_msgs
  sensor_msgs
  std_msgs
  turtlebot3_msgs
)

# ============================================
# LIBRARY 2: physics_check (intrusion detection)
# ============================================
add_library(physics_check_lib
  physics_check.cpp
)

target_include_directories(physics_check_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>
)

# ============================================
# UNIFIED EXECUTABLE: intrusion_detection
# ============================================
add_executable(intrusion_detection
  main.cpp
)

target_include_directories(intrusion_detection PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(intrusion_detection
  data_transition_lib
  physics_check_lib
)

# Link ROS dependencies to main executable
ament_target_dependencies(intrusion_detection
  rclcpp
)

# ============================================
# Install
# ============================================
install(TARGETS
  intrusion_detection
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()