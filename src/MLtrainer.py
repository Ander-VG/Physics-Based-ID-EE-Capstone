#!/usr/bin/env python3
"""
Train ML models on labeled ROS 2 network flow data
Handles severe class imbalance using class weights
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
import os
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data path - Should be output from label_data.py
DATA_PATH = "/home/anguiz/Capstone/output/labeled_flows.csv"

# Output directory for trained models
OUTPUT_DIR = "/home/anguiz/Capstone/src/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Will be calculated from actual data
CLASS_WEIGHTS = None  # Calculated automatically

print("="*70)
print("🚀 ML MODEL TRAINING PIPELINE FOR ROS 2 IDS")
print("="*70)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n📂 Loading labeled data...")
try:
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded {len(df):,} samples")
except FileNotFoundError:
    print(f"❌ Error: File not found: {DATA_PATH}")
    print("\n💡 Run label_data.py first to create labeled dataset")
    sys.exit(1)

# Check for Label column
if 'Label' not in df.columns:
    print("❌ Error: CSV must have 'Label' column")
    print("   Run label_data.py first to add labels")
    sys.exit(1)

# Display class distribution
print(f"\n📊 Class Distribution:")
print(df['Label'].value_counts())
benign_count = (df['Label'] == 0).sum()
attack_count = (df['Label'] == 1).sum()

if attack_count == 0:
    print("❌ Error: No attack samples found!")
    sys.exit(1)

ratio = benign_count / attack_count
print(f"   Imbalance ratio: {ratio:.1f}:1 (benign:attack)")

# Calculate class weights automatically
CLASS_WEIGHTS = {0: 1, 1: int(ratio)}
print(f"   Using class weights: {CLASS_WEIGHTS}")

# Separate features and labels
X = df.drop(['Label'], axis=1, errors='ignore')

# Also drop Attack_Type if it exists (metadata, not a feature)
if 'Attack_Type' in X.columns:
    X = X.drop('Attack_Type', axis=1)

y = df['Label']

# Save feature names
feature_names = X.columns.tolist()
print(f"✅ Found {len(feature_names)} features")

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================

print("\n🔀 Splitting data (80% train, 20% test, stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,
    random_state=42
)

print(f"✅ Train set: {len(X_train):,} samples")
print(f"   - Benign: {(y_train == 0).sum():,}")
print(f"   - Attack: {(y_train == 1).sum():,}")
print(f"✅ Test set:  {len(X_test):,} samples")
print(f"   - Benign: {(y_test == 0).sum():,}")
print(f"   - Attack: {(y_test == 1).sum():,}")

# ============================================================================
# FEATURE SCALING
# ============================================================================

print("\n⚙️  Scaling features with StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save shared scaler
scaler_path = os.path.join(OUTPUT_DIR, 'scaler.joblib')
dump(scaler, scaler_path)
print(f"✅ Saved shared scaler: {scaler_path}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def save_model(model, model_name, output_dir, feature_names, scaler):
    """Save model, scaler, and feature list"""
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    dump(model, os.path.join(model_dir, f'{model_name}_model.joblib'))
    dump(scaler, os.path.join(model_dir, f'{model_name}_scaler.joblib'))
    
    with open(os.path.join(model_dir, f'{model_name}_used_features.txt'), 'w') as f:
        for feat in feature_names:
            f.write(feat + '\n')
    
    print(f"✅ Saved to: {model_dir}/")

def evaluate_model(y_true, y_pred, model_name):
    """Print evaluation metrics"""
    print(f"\n📊 {model_name} Performance:")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, 
                                target_names=['Normal', 'Attack'],
                                digits=4))
    
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Normal  Attack")
    print(f"Actual Normal   {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"       Attack   {cm[1][0]:6d}  {cm[1][1]:6d}")
    
    tn, fp, fn, tp = cm.ravel()
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"\n🎯 Key Metrics:")
    print(f"   Attack Detection Rate: {detection_rate*100:.2f}%")
    print(f"   False Positive Rate:   {fpr*100:.2f}%")
    
    return detection_rate, fpr

# ============================================================================
# TRAIN MODELS
# ============================================================================

results = {}

# Random Forest
print("\n" + "="*70)
print("🌲 TRAINING RANDOM FOREST")
print("="*70)
rf = RandomForestClassifier(
    n_estimators=100, max_depth=20, min_samples_split=10,
    min_samples_leaf=5, class_weight=CLASS_WEIGHTS,
    random_state=42, n_jobs=-1, verbose=1
)
rf.fit(X_train_scaled, y_train)
rf_preds = rf.predict(X_test_scaled)
results['rf'] = evaluate_model(y_test, rf_preds, "Random Forest")
save_model(rf, 'rf', OUTPUT_DIR, feature_names, scaler)

# Decision Tree
print("\n" + "="*70)
print("🌳 TRAINING DECISION TREE")
print("="*70)
dt = DecisionTreeClassifier(
    max_depth=15, min_samples_split=20, min_samples_leaf=10,
    class_weight=CLASS_WEIGHTS, random_state=42
)
dt.fit(X_train_scaled, y_train)
dt_preds = dt.predict(X_test_scaled)
results['dt'] = evaluate_model(y_test, dt_preds, "Decision Tree")
save_model(dt, 'dt', OUTPUT_DIR, feature_names, scaler)

# KNN
print("\n" + "="*70)
print("📍 TRAINING K-NEAREST NEIGHBORS")
print("="*70)
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
knn.fit(X_train_scaled, y_train)
knn_preds = knn.predict(X_test_scaled)
results['knn'] = evaluate_model(y_test, knn_preds, "KNN")
save_model(knn, 'knn', OUTPUT_DIR, feature_names, scaler)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print(f"\n📁 All models saved to: {OUTPUT_DIR}/")
print("\nModel Performance Comparison:")
print(f"{'Model':<20} {'Detection Rate':<20} {'FP Rate':<20}")
print("-" * 60)
for name, (dr, fpr) in results.items():
    print(f"{name.upper():<20} {dr*100:>6.2f}%{'':<13} {fpr*100:>6.2f}%")

print("\n🚀 Your ROS 2 IDS scripts are ready to use!")
print("="*70)

if __name__ == '__main__':
    pass