#!/usr/bin/env python3
"""
train_cnn_lstm_dl.py - Trains CNN-LSTM Deep Learning IDS pipeline

This training script produces models compatible with your cnn_lstm.py inference node.

Architecture (matching your cnn_lstm.py):
1. StandardScaler: Normalize features
2. CNN+LSTM: Deep learning feature extraction (TensorFlow)
   - Input shape: (n_samples, 1, n_features) - timesteps=1 for real-time
3. PCA: Reduce CNN-LSTM output dimensions  
4. KNN + Random Forest: Get probability predictions on PCA features
5. Fusion: Concatenate [KNN probs + RF probs]  (NOT including PCA!)
6. Logistic Regression: Final classification

Output files (matching cnn_lstm.py expectations):
- feature_extractor.keras  (CNN-LSTM TensorFlow model)
- scaler.joblib
- pca.joblib
- knn_classifier.joblib
- rf_classifier.joblib
- lr_classifier.joblib
- features.txt
- class_mapping.json
- pipeline_info.json

Usage:
    python3 train_cnn_lstm_dl.py
    python3 train_cnn_lstm_dl.py --epochs 50 --batch-size 64
"""

import pandas as pd
import numpy as np
import joblib
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Cannot train CNN-LSTM model.")

# Default paths
DEFAULT_INPUT = "/home/anguiz/Capstone/output/labeled_flows.csv"
DEFAULT_OUTPUT = "/home/anguiz/Capstone/src/models/cnn_lstm_dl"

# Columns to exclude
EXCLUDE_COLUMNS = [
    'Label', 'Attack_Type', 'Timestamp', 'Timestamp_Normalized',
    'Flow ID', 'Src IP', 'Dst IP', 'timestamp', 'flow_id',
    'src_ip', 'dst_ip', 'label', 'attack_type'
]


def find_feature_columns(df: pd.DataFrame) -> list:
    """Find valid numeric feature columns."""
    exclude_lower = {c.lower() for c in EXCLUDE_COLUMNS}
    
    feature_cols = []
    for col in df.columns:
        if col.lower() in exclude_lower:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        feature_cols.append(col)
    
    return feature_cols


def preprocess_data(df: pd.DataFrame, feature_cols: list) -> tuple:
    """Preprocess data for training."""
    X = df[feature_cols].copy()
    y = df['Label'].values
    
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    return X.values, y, feature_cols


def calculate_class_weights(y: np.ndarray) -> dict:
    """Calculate class weights for imbalanced data."""
    n_positive = np.sum(y)
    n_negative = len(y) - n_positive
    
    if n_positive == 0:
        return {0: 1.0, 1: 1.0}
    
    ratio = n_negative / n_positive
    return {0: 1.0, 1: min(ratio, 50.0)}


def build_cnn_lstm_feature_extractor(n_features: int, n_extracted: int = 64):
    """
    Build CNN-LSTM feature extractor model.
    
    CRITICAL: Input shape must be (timesteps, features) = (1, n_features)
    This matches your cnn_lstm.py inference code:
        X_reshaped = X_scaled.reshape(n_samples, timesteps, n_features)
        timesteps = 1  # Single timestep for real-time data
    
    Returns:
        feature_extractor model (outputs extracted features, not classifications)
    """
    # Input: (timesteps=1, features=n_features)
    inputs = layers.Input(shape=(1, n_features), name='input')
    
    x = layers.Conv1D(64, kernel_size=1, padding='same', activation='relu', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Dropout(0.2, name='dropout1')(x)
    
    x = layers.Conv1D(128, kernel_size=1, padding='same', activation='relu', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Dropout(0.2, name='dropout2')(x)
    
    # LSTM layer
    x = layers.LSTM(64, return_sequences=False, name='lstm')(x)
    x = layers.Dropout(0.3, name='dropout3')(x)
    
    # Feature extraction layer
    features = layers.Dense(n_extracted, activation='relu', name='feature_layer')(x)
    
    feature_extractor = Model(inputs=inputs, outputs=features, name='cnn_lstm_extractor')
    
    return feature_extractor


def build_full_classifier(n_features: int, n_extracted: int = 64):
    """Build full CNN-LSTM classifier for training."""
    inputs = layers.Input(shape=(1, n_features), name='input')
    
    x = layers.Conv1D(64, kernel_size=1, padding='same', activation='relu', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Dropout(0.2, name='dropout1')(x)
    
    x = layers.Conv1D(128, kernel_size=1, padding='same', activation='relu', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Dropout(0.2, name='dropout2')(x)
    
    x = layers.LSTM(64, return_sequences=False, name='lstm')(x)
    x = layers.Dropout(0.3, name='dropout3')(x)
    
    features = layers.Dense(n_extracted, activation='relu', name='feature_layer')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(features)
    
    classifier = Model(inputs=inputs, outputs=outputs, name='cnn_lstm_classifier')
    
    return classifier


def train_cnn_lstm(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   n_features: int, n_extracted: int,
                   class_weights: dict, epochs: int, batch_size: int):
    """Train the CNN-LSTM model and extract feature extractor."""
    
    print(f"\n  Building CNN-LSTM model...")
    print(f"    Input features: {n_features}")
    print(f"    Extracted features: {n_extracted}")
    print(f"    Input shape for LSTM: (batch, 1, {n_features})")
    
    classifier = build_full_classifier(n_features, n_extracted)
    
    classifier.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n  Model Architecture:")
    classifier.summary(print_fn=lambda x: print(f"    {x}"))
    
    # Reshape data for LSTM: (n_samples, timesteps=1, n_features)
    X_train_lstm = X_train.reshape(-1, 1, n_features)
    X_val_lstm = X_val.reshape(-1, 1, n_features)
    
    print(f"\n  Reshaped training data: {X_train_lstm.shape}")
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    print(f"\n  Training for up to {epochs} epochs (batch_size={batch_size})...")
    
    history = classifier.fit(
        X_train_lstm, y_train,
        validation_data=(X_val_lstm, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Extract feature extractor
    feature_extractor = build_cnn_lstm_feature_extractor(n_features, n_extracted)
    
    for layer in feature_extractor.layers:
        if layer.name in [l.name for l in classifier.layers]:
            classifier_layer = classifier.get_layer(layer.name)
            layer.set_weights(classifier_layer.get_weights())
    
    return classifier, feature_extractor, history


class CNNLSTMFusionPipeline:
    """CNN-LSTM Deep Learning Fusion Pipeline matching cnn_lstm.py."""
    
    def __init__(self, n_extracted: int = 64, n_pca: int = 16):
        self.n_extracted = n_extracted
        self.n_pca = n_pca
        self.scaler = StandardScaler()
        self.feature_extractor = None
        self.classifier = None
        self.pca = PCA(n_components=n_pca)
        self.knn = None
        self.rf = None
        self.lr = None
        self.n_features = None
        self.training_history = None
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            class_weights: dict, epochs: int = 50, batch_size: int = 64):
        """Fit the entire pipeline."""
        
        self.n_features = X_train.shape[1]
        
        print("\n  Step 1: Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        print("\n  Step 2: Training CNN-LSTM feature extractor...")
        self.classifier, self.feature_extractor, self.training_history = train_cnn_lstm(
            X_train_scaled, y_train,
            X_val_scaled, y_val,
            self.n_features, self.n_extracted,
            class_weights, epochs, batch_size
        )
        
        print("\n  Step 3: Extracting deep features...")
        X_train_lstm = X_train_scaled.reshape(-1, 1, self.n_features)
        X_train_deep = self.feature_extractor.predict(X_train_lstm, verbose=0)
        print(f"    Extracted feature shape: {X_train_deep.shape}")
        
        print(f"\n  Step 4: PCA reduction ({self.n_extracted} → {self.n_pca})...")
        X_train_pca = self.pca.fit_transform(X_train_deep)
        explained_var = np.sum(self.pca.explained_variance_ratio_)
        print(f"    Explained variance: {explained_var:.2%}")
        
        print("\n  Step 5: Training KNN on PCA features...")
        self.knn = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
        self.knn.fit(X_train_pca, y_train)
        
        print("\n  Step 6: Training Random Forest on PCA features...")
        self.rf = RandomForestClassifier(
            n_estimators=100, max_depth=15,
            class_weight=class_weights, random_state=42, n_jobs=-1
        )
        self.rf.fit(X_train_pca, y_train)
        
        # Fusion: ONLY probabilities (matches your cnn_lstm.py)
        print("\n  Step 7: Training Logistic Regression on fused probabilities...")
        knn_proba = self.knn.predict_proba(X_train_pca)
        rf_proba = self.rf.predict_proba(X_train_pca)
        X_fused = np.hstack([knn_proba, rf_proba])
        print(f"    Fused feature shape: {X_fused.shape}")
        
        self.lr = LogisticRegression(class_weight=class_weights, max_iter=1000, random_state=42)
        self.lr.fit(X_fused, y_train)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        X_lstm = X_scaled.reshape(-1, 1, self.n_features)
        X_deep = self.feature_extractor.predict(X_lstm, verbose=0)
        X_pca = self.pca.transform(X_deep)
        
        knn_proba = self.knn.predict_proba(X_pca)
        rf_proba = self.rf.predict_proba(X_pca)
        X_fused = np.hstack([knn_proba, rf_proba])
        
        return self.lr.predict(X_fused)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        X_lstm = X_scaled.reshape(-1, 1, self.n_features)
        X_deep = self.feature_extractor.predict(X_lstm, verbose=0)
        X_pca = self.pca.transform(X_deep)
        
        knn_proba = self.knn.predict_proba(X_pca)
        rf_proba = self.rf.predict_proba(X_pca)
        X_fused = np.hstack([knn_proba, rf_proba])
        
        return self.lr.predict_proba(X_fused)
    
    def get_all_predictions(self, X: np.ndarray) -> dict:
        X_scaled = self.scaler.transform(X)
        X_lstm = X_scaled.reshape(-1, 1, self.n_features)
        
        cnn_lstm_proba = self.classifier.predict(X_lstm, verbose=0).flatten()
        cnn_lstm_pred = (cnn_lstm_proba > 0.5).astype(int)
        
        X_deep = self.feature_extractor.predict(X_lstm, verbose=0)
        X_pca = self.pca.transform(X_deep)
        
        return {
            'cnn_lstm_pred': cnn_lstm_pred,
            'cnn_lstm_proba': cnn_lstm_proba,
            'knn_pred': self.knn.predict(X_pca),
            'knn_proba': self.knn.predict_proba(X_pca)[:, 1],
            'rf_pred': self.rf.predict(X_pca),
            'rf_proba': self.rf.predict_proba(X_pca)[:, 1],
            'final_pred': self.predict(X),
            'final_proba': self.predict_proba(X)[:, 1]
        }
    
    def save(self, output_dir: Path, feature_names: list, metrics: dict):
        """Save all pipeline components matching cnn_lstm.py expectations."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_extractor.save(output_dir / "feature_extractor.keras")
        
        joblib.dump(self.scaler, output_dir / "scaler.joblib")
        joblib.dump(self.pca, output_dir / "pca.joblib")
        joblib.dump(self.knn, output_dir / "knn_classifier.joblib")
        joblib.dump(self.rf, output_dir / "rf_classifier.joblib")
        joblib.dump(self.lr, output_dir / "lr_classifier.joblib")
        
        with open(output_dir / "features.txt", 'w') as f:
            for feat in feature_names:
                f.write(f"{feat}\n")
        
        # CRITICAL: class_mapping.json required by cnn_lstm.py
        class_mapping = {"Normal": 0, "Attack": 1}
        with open(output_dir / "class_mapping.json", 'w') as f:
            json.dump(class_mapping, f, indent=2)
        
        pipeline_info = {
            'architecture': 'Scale → CNN-LSTM(1,n) → PCA → KNN+RF → Fuse(proba) → LR',
            'n_input_features': len(feature_names),
            'lstm_input_shape': [1, len(feature_names)],
            'n_extracted_features': self.n_extracted,
            'n_pca_components': self.n_pca,
            'n_fused_features': 4,
            'pca_explained_variance': float(np.sum(self.pca.explained_variance_ratio_)),
            'cnn_lstm_epochs': len(self.training_history.history['loss']),
            'trained_at': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        with open(output_dir / "pipeline_info.json", 'w') as f:
            json.dump(pipeline_info, f, indent=2)
        
        print(f"\n  Saved pipeline to: {output_dir}")
        for f in ["feature_extractor.keras", "scaler.joblib", "pca.joblib",
                  "knn_classifier.joblib", "rf_classifier.joblib", "lr_classifier.joblib",
                  "features.txt", "class_mapping.json", "pipeline_info.json"]:
            print(f"    - {f}")


def evaluate_pipeline(pipeline: CNNLSTMFusionPipeline, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    print("\n  Evaluating CNN-LSTM fusion pipeline...")
    
    preds = pipeline.get_all_predictions(X_test)
    
    print("\n  Component-wise evaluation:")
    for name, pred in [('CNN-LSTM', preds['cnn_lstm_pred']), ('KNN', preds['knn_pred']),
                       ('RF', preds['rf_pred']), ('Fusion', preds['final_pred'])]:
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        print(f"    {name:<12} Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
    
    y_pred = preds['final_pred']
    y_proba = preds['final_proba']
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'cnn_lstm_accuracy': accuracy_score(y_test, preds['cnn_lstm_pred']),
    }
    
    try:
        metrics['auc_roc'] = roc_auc_score(y_test, y_proba)
        metrics['cnn_lstm_auc_roc'] = roc_auc_score(y_test, preds['cnn_lstm_proba'])
    except ValueError:
        metrics['auc_roc'] = 0.0
        metrics['cnn_lstm_auc_roc'] = 0.0
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Final Confusion Matrix:")
    print(f"    TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
    print(f"    FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train CNN-LSTM Deep Learning IDS pipeline")
    parser.add_argument('--input', '-i', default=DEFAULT_INPUT)
    parser.add_argument('--output', '-o', default=DEFAULT_OUTPUT)
    parser.add_argument('--epochs', '-e', type=int, default=50)
    parser.add_argument('--batch-size', '-b', type=int, default=64)
    parser.add_argument('--n-extracted', type=int, default=64)
    parser.add_argument('--n-pca', type=int, default=16)
    parser.add_argument('--test-size', type=float, default=0.2)
    
    args = parser.parse_args()
    
    if not TF_AVAILABLE:
        print("ERROR: TensorFlow required. Install with: pip install tensorflow")
        return 1
    
    print("="*60)
    print("CNN-LSTM DEEP LEARNING PIPELINE TRAINING")
    print("="*60)
    print(f"Input:          {args.input}")
    print(f"Output:         {args.output}")
    print(f"TensorFlow:     {tf.__version__}")
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {args.input}")
        return 1
    
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} samples")
    
    if 'Label' not in df.columns:
        print("ERROR: 'Label' column not found.")
        return 1
    
    feature_cols = find_feature_columns(df)
    X, y, feature_names = preprocess_data(df, feature_cols)
    print(f"Feature matrix shape: {X.shape}")
    
    class_weights = calculate_class_weights(y)
    
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.15, random_state=42, stratify=y_trainval
    )
    
    print(f"Training:   {len(X_train)} ({np.sum(y_train)} attacks)")
    print(f"Validation: {len(X_val)} ({np.sum(y_val)} attacks)")
    print(f"Test:       {len(X_test)} ({np.sum(y_test)} attacks)")
    
    pipeline = CNNLSTMFusionPipeline(n_extracted=args.n_extracted, n_pca=args.n_pca)
    pipeline.fit(X_train, y_train, X_val, y_val, class_weights, args.epochs, args.batch_size)
    
    metrics = evaluate_pipeline(pipeline, X_test, y_test)
    
    output_dir = Path(args.output)
    pipeline.save(output_dir, feature_names, metrics)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"CNN-LSTM Accuracy: {metrics['cnn_lstm_accuracy']:.4f}")
    print(f"Fusion Accuracy:   {metrics['accuracy']:.4f}")
    print(f"Fusion F1:         {metrics['f1']:.4f}")
    print("\n✓ Models ready for cnn_lstm.py node!")
    
    return 0


if __name__ == "__main__":
    exit(main())