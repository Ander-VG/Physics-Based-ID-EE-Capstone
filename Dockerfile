FROM ros:humble
 
SHELL ["/bin/bash", "-c"]
 
# Install build tools
RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    build-essential \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*
 
# Create workspace and copy package into src/
WORKDIR /ws
COPY . /ws/src/sensor_det
 
# Ensure ROS libraries are on the linker path (fixes arm64 cross-build)
ENV LD_LIBRARY_PATH=/opt/ros/humble/lib
 
# Build the ROS 2 workspace
RUN source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install
 
# Default command
CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && source /ws/install/setup.bash && ros2 run sensor_det sensor_det"]
 