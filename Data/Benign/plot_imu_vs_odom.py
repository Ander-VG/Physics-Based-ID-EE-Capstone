import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

print("Loading data files...")

# Load your CSV files
odom_data = pd.read_csv('odom_output.csv')
imu_data = pd.read_csv('imu_output.csv')

print(f"Loaded {len(odom_data)} odom samples")
print(f"Loaded {len(imu_data)} imu samples")

# Calculate accelerations from odometry by differentiating velocity
# First, we need velocity from position
odom_data['dt'] = odom_data['Time'].diff()
odom_data['dx'] = odom_data['X'].diff()
odom_data['dy'] = odom_data['Y'].diff()

# Calculate velocities
odom_data['vel_x'] = odom_data['dx'] / odom_data['dt']
odom_data['vel_y'] = odom_data['dy'] / odom_data['dt']

# Calculate accelerations from velocities
odom_data['accel_x'] = odom_data['vel_x'].diff() / odom_data['dt']
odom_data['accel_y'] = odom_data['vel_y'].diff() / odom_data['dt']

# Remove NaN values from differentiation
odom_data = odom_data.dropna()

print("\nCalculating acceleration differences...")

# Interpolate IMU data to match ODOM timestamps
imu_interp_x = interp1d(imu_data['Time'], imu_data['imu_accel_x'], 
                        bounds_error=False, fill_value='extrapolate')
imu_interp_y = interp1d(imu_data['Time'], imu_data['imu_accel_y'], 
                        bounds_error=False, fill_value='extrapolate')

odom_data['imu_x_interp'] = imu_interp_x(odom_data['Time'])
odom_data['imu_y_interp'] = imu_interp_y(odom_data['Time'])

# Calculate acceleration difference magnitude
odom_data['accel_diff'] = np.sqrt((odom_data['accel_x'] - odom_data['imu_x_interp'])**2 + 
                                   (odom_data['accel_y'] - odom_data['imu_y_interp'])**2)

# Remove any infinite or extremely large values (likely errors)
odom_data = odom_data[np.isfinite(odom_data['accel_diff'])]

print("\nCreating plots...")

# Create the visualization
fig, axes = plt.subplots(4, 1, figsize=(14, 12))

# Plot 1: X-axis acceleration comparison
axes[0].plot(odom_data['Time'].values, odom_data['accel_x'].values, label='ODOM Accel X', alpha=0.7, linewidth=1)
axes[0].plot(imu_data['Time'].values, imu_data['imu_accel_x'].values, label='IMU Accel X', alpha=0.7, linewidth=1)
axes[0].set_ylabel('Acceleration X (m/s²)', fontsize=10)
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)
axes[0].set_title('Linear Acceleration X Comparison', fontsize=12, fontweight='bold')

# Plot 2: Y-axis acceleration comparison
axes[1].plot(odom_data['Time'].values, odom_data['accel_y'].values, label='ODOM Accel Y', alpha=0.7, linewidth=1)
axes[1].plot(imu_data['Time'].values, imu_data['imu_accel_y'].values, label='IMU Accel Y', alpha=0.7, linewidth=1)
axes[1].set_ylabel('Acceleration Y (m/s²)', fontsize=10)
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)
axes[1].set_title('Linear Acceleration Y Comparison', fontsize=12, fontweight='bold')

# Plot 3: Acceleration difference magnitude
axes[2].plot(odom_data['Time'].values, odom_data['accel_diff'].values, color='red', linewidth=1)
axes[2].axhline(y=1.0, color='orange', linestyle='--', linewidth=2, label='Current Threshold (1.0 m/s²)')
axes[2].axhline(y=36.0, color='purple', linestyle='--', linewidth=2, label='Suggested Threshold (36.0 m/s²)')
axes[2].set_ylabel('Accel Difference (m/s²)', fontsize=10)
axes[2].legend(loc='upper right')
axes[2].grid(True, alpha=0.3)
axes[2].set_title('IMU vs ODOM Acceleration Difference Magnitude', fontsize=12, fontweight='bold')

# Plot 4: Histogram of acceleration differences
axes[3].hist(odom_data['accel_diff'].values, bins=100, color='blue', alpha=0.7, edgecolor='black')
axes[3].axvline(x=1.0, color='orange', linestyle='--', linewidth=2, label='Current Threshold (1.0)')
axes[3].axvline(x=odom_data['accel_diff'].quantile(0.95), color='green', linestyle='--', 
                linewidth=2, label=f"95th percentile ({odom_data['accel_diff'].quantile(0.95):.2f})")
axes[3].set_xlabel('Acceleration Difference (m/s²)', fontsize=10)
axes[3].set_ylabel('Frequency', fontsize=10)
axes[3].legend(loc='upper right')
axes[3].grid(True, alpha=0.3, axis='y')
axes[3].set_title('Distribution of Acceleration Differences', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('imu_vs_odom_analysis.png', dpi=150, bbox_inches='tight')
print("\n✅ Plot saved as 'imu_vs_odom_analysis.png'")

# Print statistics
print("\n" + "="*50)
print("     ACCELERATION DIFFERENCE STATISTICS")
print("="*50)
print(f"Max acceleration difference:  {odom_data['accel_diff'].max():.2f} m/s²")
print(f"Mean acceleration difference: {odom_data['accel_diff'].mean():.2f} m/s²")
print(f"Median (50th percentile):     {odom_data['accel_diff'].quantile(0.50):.2f} m/s²")
print(f"95th percentile:              {odom_data['accel_diff'].quantile(0.95):.2f} m/s²")
print(f"99th percentile:              {odom_data['accel_diff'].quantile(0.99):.2f} m/s²")
print(f"\nSamples above 1.0 m/s²:       {(odom_data['accel_diff'] > 1.0).sum()} ({(odom_data['accel_diff'] > 1.0).sum() / len(odom_data) * 100:.1f}%)")
print(f"Samples above 5.0 m/s²:       {(odom_data['accel_diff'] > 5.0).sum()} ({(odom_data['accel_diff'] > 5.0).sum() / len(odom_data) * 100:.1f}%)")
print(f"Samples above 10.0 m/s²:      {(odom_data['accel_diff'] > 10.0).sum()} ({(odom_data['accel_diff'] > 10.0).sum() / len(odom_data) * 100:.1f}%)")
print("="*50)

# Recommended threshold based on statistics
recommended_threshold = odom_data['accel_diff'].quantile(0.99) * 1.1  # 99th percentile + 10% margin
print(f"\n💡 RECOMMENDED THRESHOLD: {recommended_threshold:.2f} m/s²")
print(f"   (Based on 99th percentile + 10% margin)")
print("="*50)

plt.show()
