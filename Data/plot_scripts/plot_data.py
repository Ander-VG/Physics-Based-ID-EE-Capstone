import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read all CSV files
odom = pd.read_csv('odom.csv')
cmd_vel = pd.read_csv('cmd_vel.csv')
imu = pd.read_csv('imu.csv')
js = pd.read_csv('js.csv')
error = pd.read_csv('error.csv')
battery = pd.read_csv('battery.csv')

# Create figure with subplots (4 rows, 2 columns)
fig, axes = plt.subplots(4, 2, figsize=(12, 10))
fig.suptitle('Complete Robot Data Visualization', fontsize=18, fontweight='bold')

# 1. Robot Trajectory (X-Y plot)
axes[0, 0].plot(odom['X'].values, odom['Y'].values, 'b-', linewidth=0.8)
axes[0, 0].set_xlabel('X (m)', fontsize=10)
axes[0, 0].set_ylabel('Y (m)', fontsize=10)
axes[0, 0].set_title('Robot Trajectory', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axis('equal')

# 2. Theta over time
axes[0, 1].plot(odom['Time'].values, odom['Theta'].values, 'r-', linewidth=0.5)
axes[0, 1].set_xlabel('Time (s)', fontsize=10)
axes[0, 1].set_ylabel('Theta (rad)', fontsize=10)
axes[0, 1].set_title('Robot Orientation', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 3. Command Velocities
axes[1, 0].plot(cmd_vel['Time'].values, cmd_vel['linear_x'].values, 'g-', linewidth=0.5, label='Linear X')
axes[1, 0].plot(cmd_vel['Time'].values, cmd_vel['angular_z'].values, 'orange', linewidth=0.5, label='Angular Z')
axes[1, 0].set_xlabel('Time (s)', fontsize=10)
axes[1, 0].set_ylabel('Velocity', fontsize=10)
axes[1, 0].set_title('Command Velocities', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Wheel Velocities & Encoders
ax4 = axes[1, 1]
ax4_twin = ax4.twinx()
ax4.plot(js['Time'].values, js['v_R'].values, 'purple', linewidth=0.5, label='v_R')
ax4.plot(js['Time'].values, js['v_L'].values, 'brown', linewidth=0.5, label='v_L')
ax4_twin.plot(js['Time'].values, js['right_encoder'].values, 'purple', linewidth=0.5, linestyle='--', alpha=0.5, label='Right Encoder')
ax4_twin.plot(js['Time'].values, js['left_encoder'].values, 'brown', linewidth=0.5, linestyle='--', alpha=0.5, label='Left Encoder')
ax4.set_xlabel('Time (s)', fontsize=10)
ax4.set_ylabel('Velocity (m/s)', fontsize=10, color='purple')
ax4_twin.set_ylabel('Encoder Position (rad)', fontsize=10, color='brown')
ax4.set_title('Wheel Velocities & Encoder Positions', fontsize=12, fontweight='bold')
ax4.legend(loc='upper left')
ax4_twin.legend(loc='upper right')
ax4.grid(True, alpha=0.3)

# 5. IMU Accelerometer
axes[2, 0].plot(imu['Time'].values, imu['imu_accel_x'].values, 'r-', linewidth=0.5, label='Accel X')
axes[2, 0].plot(imu['Time'].values, imu['imu_accel_y'].values, 'g-', linewidth=0.5, label='Accel Y')
axes[2, 0].plot(imu['Time'].values, imu['imu_accel_z'].values, 'b-', linewidth=0.5, label='Accel Z')
axes[2, 0].set_xlabel('Time (s)', fontsize=10)
axes[2, 0].set_ylabel('Acceleration (m/s²)', fontsize=10)
axes[2, 0].set_title('IMU Accelerometer', fontsize=12, fontweight='bold')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# 6. IMU Gyroscope
axes[2, 1].plot(imu['Time'].values, imu['imu_gyro_x'].values, 'r-', linewidth=0.5, label='Gyro X')
axes[2, 1].plot(imu['Time'].values, imu['imu_gyro_y'].values, 'g-', linewidth=0.5, label='Gyro Y')
axes[2, 1].plot(imu['Time'].values, imu['imu_gyro_z'].values, 'b-', linewidth=0.5, label='Gyro Z')
axes[2, 1].set_xlabel('Time (s)', fontsize=10)
axes[2, 1].set_ylabel('Angular Velocity (rad/s)', fontsize=10)
axes[2, 1].set_title('IMU Gyroscope', fontsize=12, fontweight='bold')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

# 7. Tracking Errors
axes[3, 0].plot(error['Time'].values, error['error_x'].values, 'b-', linewidth=0.5, label='Error X')
axes[3, 0].plot(error['Time'].values, error['error_y'].values, 'g-', linewidth=0.5, label='Error Y')
axes[3, 0].set_xlabel('Time (s)', fontsize=10)
axes[3, 0].set_ylabel('Error (m)', fontsize=10)
axes[3, 0].set_title('Position Errors (X, Y)', fontsize=12, fontweight='bold')
axes[3, 0].legend()
axes[3, 0].grid(True, alpha=0.3)

# 8. Battery State
ax8 = axes[3, 1]
ax8_twin = ax8.twinx()
ax8.plot(battery['Time'].values, battery['battery_voltage'].values, 'b-', linewidth=1, label='Voltage')
ax8_twin.plot(battery['Time'].values, battery['battery_percentage'].values, 'g-', linewidth=1, label='Percentage')
ax8.set_xlabel('Time (s)', fontsize=10)
ax8.set_ylabel('Voltage (V)', fontsize=10, color='blue')
ax8_twin.set_ylabel('Percentage (%)', fontsize=10, color='green')
ax8.set_title('Battery State', fontsize=12, fontweight='bold')
ax8.legend(loc='upper left')
ax8_twin.legend(loc='upper right')
ax8.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('complete_robot_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved as 'complete_robot_analysis.png'")
print(f"✓ Total data points: Odom={len(odom)}, Cmd={len(cmd_vel)}, IMU={len(imu)}")
print(f"✓ Battery: {len(battery)} samples, JS: {len(js)} samples")
plt.show()