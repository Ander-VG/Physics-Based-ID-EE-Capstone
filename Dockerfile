FROM ros:humble

SHELL ["/bin/bash", "-c"]

# Install build tools and CycloneDDS RMW
RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    build-essential \
    libcurl4-openssl-dev \
    ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

# Create workspace and copy package into src/
WORKDIR /ws
COPY . /ws/src/sensor_det

# Ensure ROS libraries are on the linker path
ENV LD_LIBRARY_PATH=/opt/ros/humble/lib

# Force CycloneDDS and matching ROS domain
ENV ROS_DOMAIN_ID=0
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Build the ROS 2 workspace
RUN source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install

# Default command
CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && source /ws/install/setup.bash && ros2 run sensor_det sensor_det"]