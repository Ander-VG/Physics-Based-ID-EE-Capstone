import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROS_csv = "~/Capstone/output/ROSdata.csv"
df = pd.read_csv(ROS_csv)

matrix = df.corr()

sns.heatmap(matrix, annot=True, vmin = -1, vmax = 1, center = 0, cmap = 'vlag')

print(f"v_R mean: {df['v_R'].mean()}")
print(f"v_L mean: {df['v_L'].mean()}")
print(f"Correlation: {df['v_R'].corr(df['v_L'])}")

time = df['Time'].values
v_R = df['v_R'].values
v_L = df['v_L'].values

# Plot wheel velocities
plt.figure(figsize=(14, 8))

plt.subplot(3, 1, 1)
plt.plot(time, v_R, label='v_R (right wheel)', alpha=0.7, linewidth=0.5)
plt.plot(time, v_L, label='v_L (left wheel)', alpha=0.7, linewidth=0.5)
plt.ylabel('Wheel Velocity (m/s)')
plt.title('Wheel Velocities Over Time', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
plt.scatter(v_R, v_L, alpha=0.3, s=1)
plt.xlabel('v_R (m/s)')
plt.ylabel('v_L (m/s)')
plt.title(f'v_R vs v_L (correlation: {df["v_R"].corr(df["v_L"]):.3f})', fontweight='bold')
plt.grid(True, alpha=0.3)
# Add diagonal line (where v_R = v_L)
min_val = min(v_R.min(), v_L.min())
max_val = max(v_R.max(), v_L.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='v_R = v_L', alpha=0.5)
plt.legend()

plt.subplot(3, 1, 3)
wheel_diff = v_R - v_L
plt.plot(time, wheel_diff, alpha=0.7, linewidth=0.5, color='purple')
plt.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
plt.xlabel('Time (s)')
plt.ylabel('v_R - v_L (m/s)')
plt.title('Wheel Velocity Difference (v_R - v_L)', fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('~/Capstone/output/wheel_velocity_analysis.png'.replace('~', '/home/anguiz'), dpi=300)
print(f"\n📊 Saved plot: ~/Capstone/output/wheel_velocity_analysis.png")
plt.show()

# ============================================================================
# Analysis
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

print(f"\n📊 Statistics:")
print(f"   v_R mean: {df['v_R'].mean():.6f} m/s")
print(f"   v_L mean: {df['v_L'].mean():.6f} m/s")
print(f"   Difference: {df['v_R'].mean() - df['v_L'].mean():.6f} m/s")
print(f"   Ratio: v_R/v_L = {df['v_R'].mean() / df['v_L'].mean():.2f}")

print(f"\n🔍 Interpretation:")

# Check if negative correlation
if df['v_R'].corr(df['v_L']) < 0:
    print(f"   🚨 NEGATIVE correlation ({df['v_R'].corr(df['v_L']):.3f})")
    print(f"      → Wheels moving in OPPOSITE directions!")
    print(f"      → Robot was likely SPINNING IN PLACE")
    
    # Check if one wheel is negative
    if (df['v_R'] < 0).any() or (df['v_L'] < 0).any():
        print(f"      → Some negative velocities detected (backward motion)")
    else:
        print(f"      → Both wheels positive but asymmetric")
        print(f"      → Robot turning constantly (not driving straight)")

elif df['v_R'].corr(df['v_L']) < 0.5:
    print(f"   ⚠️  WEAK positive correlation ({df['v_R'].corr(df['v_L']):.3f})")
    print(f"      → Wheels not synchronized well")
    print(f"      → Robot doing lots of turning/maneuvering")
    
else:
    print(f"   ✓ Good correlation ({df['v_R'].corr(df['v_L']):.3f})")
    print(f"      → Wheels mostly synchronized")
    print(f"      → Robot driving relatively straight")

# Check asymmetry
avg_diff = abs(df['v_R'].mean() - df['v_L'].mean())
if avg_diff > 0.03:
    print(f"\n   ⚠️  Large average difference ({avg_diff:.6f} m/s)")
    print(f"      → Robot has asymmetric baseline")
    print(f"      → Possible:")
    print(f"         - Calibration issue")
    print(f"         - Constant turning")
    print(f"         - Mechanical problem")

print(f"\n" + "="*70)