#!/usr/bin/env python3
"""
Train RF, DT, KNN models for network-based IDS
Matches CICFlowMeter output with 76 features (columns 8-83)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = "/home/anguiz/Capstone/ML_output/combined_all.csv"
MODEL_DIR = "/home/anguiz/Capstone/ML_output/models"

# Feature columns (CICFlowMeter features: columns 8-83, 0-indexed: 7-82)
FEATURE_COLS = list(range(7, 83))  # 76 features
LABEL_COL = 83  # Column 84 (0-indexed: 83)

def load_data():
    """Load and prepare the dataset."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    df = pd.read_csv(DATA_PATH)
    print(f"Total samples: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    # Get feature names
    feature_names = df.columns[FEATURE_COLS].tolist()
    print(f"Number of features: {len(feature_names)}")
    
    # Extract features and labels
    X = df.iloc[:, FEATURE_COLS].values
    y = df.iloc[:, LABEL_COL].values
    
    # Handle infinite and NaN values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for label, count in zip(unique, counts):
        pct = count / len(y) * 100
        label_name = "Benign" if label == 0 else "Attack"
        print(f"  {label_name} ({int(label)}): {count} ({pct:.1f}%)")
    
    return X, y, feature_names

def train_and_evaluate(X_train, X_test, y_train, y_test, model, model_name):
    """Train a model and print evaluation metrics."""
    print(f"\n{'=' * 60}")
    print(f"TRAINING: {model_name}")
    print("=" * 60)
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return model, accuracy

def main():
    # Create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Load data
    X, y, feature_names = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"\nScaler saved to: {scaler_path}")
    
    # Save feature names
    feature_path = os.path.join(MODEL_DIR, "feature_names.json")
    with open(feature_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"Feature names saved to: {feature_path}")
    
    # Define models
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'DecisionTree': DecisionTreeClassifier(
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            n_jobs=-1
        )
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        # KNN needs scaled data, others work fine without
        if name == 'KNN':
            trained_model, accuracy = train_and_evaluate(
                X_train_scaled, X_test_scaled, y_train, y_test, model, name
            )
        else:
            trained_model, accuracy = train_and_evaluate(
                X_train, X_test, y_train, y_test, model, name
            )
        
        results[name] = accuracy
        
        # Save model
        model_path = os.path.join(MODEL_DIR, f"{name.lower()}_model.joblib")
        joblib.dump(trained_model, model_path)
        print(f"Model saved to: {model_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nModel Performance:")
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {acc:.4f}")
    
    best_model = max(results, key=results.get)
    print(f"\nBest Model: {best_model} ({results[best_model]:.4f})")
    
    print("\nAll models saved to:", MODEL_DIR)
    print("\nFiles created:")
    for f in os.listdir(MODEL_DIR):
        print(f"  - {f}")

if __name__ == "__main__":
    main()