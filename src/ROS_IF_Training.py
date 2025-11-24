import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import time


# Load dataset and define features
ML_csv = "~/Capstone/Data/Training/ROSdata.csv"
df= pd.read_csv(ML_csv)
feats = ['X', 'Y', 'Theta',                    
    'linear_x', 'angular_z',              
    'v_R', 'v_L',                         
    'imu_accel_x', 'imu_accel_y', 'imu_accel_z',  
    'imu_gyro_x', 'imu_gyro_y', 'imu_gyro_z',     
    'battery_percentage']


# Extract feature matrix (X) and time index
matx = df[feats]
time_index = df['Time']


# Show feature ranges
for feat in feats:
    min_val = matx[feat].min()
    max_val = matx[feat].max()
    range_val = max_val - min_val


# Apply scaling
scaler = StandardScaler()
matx_scaled = scaler.fit_transform(matx)
matx_scaled_df = pd.DataFrame(matx_scaled, columns=feats)


# Isolation Forest Model Training
start_time = time.time()

model = IsolationForest(
    contamination=0.01,      # Expect 1% outliers in benign data
    n_estimators=100,        # 100 decision trees
    max_samples='auto',      # Use all samples
    random_state=42,         # Reproducible results
    n_jobs=-1,               # Use all CPU cores
    verbose=0
)

model.fit(matx_scaled_df)

training_duration = time.time() - start_time

# Add anomaly scores and predictions to dataframe

df_analysis = df.copy()
df_analysis['anomaly_score'] = model.decision_function(matx_scaled_df)
df_analysis['anomaly'] = model.predict(matx_scaled_df)

# Model Report
print (f"Isolation Forest model trained in {training_duration:.2f} seconds.")

# Visualization of the results
plt.figure(figsize=(14, 6))

# Plot normal instances
normal = df_analysis[df_analysis['anomaly'] == 1]
plt.scatter(normal['Time'], normal['anomaly_score'], label='Normal')

# Plot anomalies
anomalies = df_analysis[df_analysis['anomaly'] == -1]
plt.scatter(anomalies['Time'], anomalies['anomaly_score'], label='Anomaly')
plt.xlabel("Time (s)")
plt.ylabel("Anomaly Score")
plt.legend()
plt.show()

predictions = model.predict(matx_scaled_df)
anomaly_scores = model.score_samples(matx_scaled)

n_anomalies = (predictions == -1).sum()
n_normal = (predictions == 1).sum()

print ("Results: ")
print(f"   - Normal samples: {n_normal:,} ({n_normal/len(predictions)*100:.2f}%)")
print(f"   - Flagged as anomalies: {n_anomalies:,} ({n_anomalies/len(predictions)*100:.2f}%) \n")

print(f"Stats: ")
# Lower score = more anomalous 
print(f"   • Mean:   {anomaly_scores.mean():+.4f}")
print(f"   • Median: {np.median(anomaly_scores):+.4f}")
print(f"   • Std:    {anomaly_scores.std():.4f}")
print(f"   • Min:    {anomaly_scores.min():+.4f}")
print(f"   • Max:    {anomaly_scores.max():+.4f}")

joblib.dump(model, 'isolation_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and scaler saved successfully")



