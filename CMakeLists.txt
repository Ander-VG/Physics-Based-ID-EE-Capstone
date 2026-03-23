cmake_minimum_required(VERSION 3.8)
project(sensor_det)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# build executable
add_executable(sensor_det src/main.cpp src/sensor_det.cpp)
ament_target_dependencies(sensor_det
  rclcpp
  std_msgs
  nav_msgs
  geometry_msgs
  sensor_msgs
)

target_include_directories(sensor_det PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

install(TARGETS sensor_det
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()