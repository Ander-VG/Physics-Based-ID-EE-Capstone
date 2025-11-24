import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load the trained model and scaler
model = joblib.load('isolation_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load the new test dataset
test_csv = "~/Capstone/Data/Testing/ROSdata.csv"  # Your test file
df_test = pd.read_csv(test_csv)

# Use the same features as training
feats = ['X', 'Y', 'Theta',                    
    'linear_x', 'angular_z',              
    'v_R', 'v_L',                         
    'imu_accel_x', 'imu_accel_y', 'imu_accel_z',  
    'imu_gyro_x', 'imu_gyro_y', 'imu_gyro_z',     
    'battery_percentage']

# Extract features
matx_test = df_test[feats]
time_index_test = df_test['Time']

# CRITICAL: Use transform (not fit_transform) with the saved scaler
matx_test_scaled = scaler.transform(matx_test)

# Make predictions
predictions = model.predict(matx_test_scaled)
anomaly_scores = model.decision_function(matx_test_scaled)

# Analyze results
df_test['anomaly_score'] = anomaly_scores
df_test['anomaly'] = predictions

n_anomalies = (predictions == -1).sum()
n_normal = (predictions == 1).sum()

print("\nTest Results:")
print(f"   - Normal samples: {n_normal:,} ({n_normal/len(predictions)*100:.2f}%)")
print(f"   - Flagged as anomalies: {n_anomalies:,} ({n_anomalies/len(predictions)*100:.2f}%)")

print(f"\nAnomaly Score Stats:")
print(f"   • Mean:   {anomaly_scores.mean():+.4f}")
print(f"   • Median: {np.median(anomaly_scores):+.4f}")
print(f"   • Min:    {anomaly_scores.min():+.4f}")
print(f"   • Max:    {anomaly_scores.max():+.4f}")

# Visualization
plt.figure(figsize=(14, 6))
normal = df_test[df_test['anomaly'] == 1]
anomalies = df_test[df_test['anomaly'] == -1]

plt.scatter(normal['Time'], normal['anomaly_score'], 
           label='Normal', alpha=0.6, s=10)
plt.scatter(anomalies['Time'], anomalies['anomaly_score'], 
           label='Anomaly', alpha=0.8, s=20, color='red')

plt.xlabel("Time (s)")
plt.ylabel("Anomaly Score")
plt.title("Anomaly Detection on Test Data")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Show most anomalous samples
print(f"\nTop 10 Most Anomalous Samples:")
most_anomalous_idx = np.argsort(anomaly_scores)[:10]
for i, idx in enumerate(most_anomalous_idx, 1):
    t = time_index_test.iloc[idx]
    score = anomaly_scores[idx]
    print(f"{i}. Time: {t:.2f}s, Score: {score:+.4f}")
    print(f"   linear_x={matx_test.iloc[idx]['linear_x']:.3f}, "
          f"angular_z={matx_test.iloc[idx]['angular_z']:.3f}")
    print(f"   v_R={matx_test.iloc[idx]['v_R']:.3f}, "
          f"v_L={matx_test.iloc[idx]['v_L']:.3f}")