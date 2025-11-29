import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')

# TensorFlow setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, LSTM, Dense, 
                                      Dropout, BatchNormalization, Input, Flatten)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Configuration
DATA_PATH = "/home/anguiz/Capstone/ML_output/NavBot25_train.csv"
MODEL_DIR = "/home/anguiz/Capstone/ML_output/models"

FEATURE_COLS = list(range(7, 83))  # 76 features
LABEL_COL = 83  # "Label" column

# Model hyperparameters
EPOCHS = 50
BATCH_SIZE = 128  # Larger batch for bigger dataset
LEARNING_RATE = 0.001

def load_data():
    """Load and prepare the dataset."""
    print("=" * 60)
    print("LOADING NAVBOT25 TRAINING DATA")
    print("=" * 60)
    
    df = pd.read_csv(DATA_PATH)
    print(f"Total samples: {len(df)}")
    
    X = df.iloc[:, FEATURE_COLS].values
    
    # Convert labels to binary: Normal=0, Attack=1
    labels_raw = df.iloc[:, LABEL_COL].values
    y = np.array([0 if label == 'Normal' else 1 for label in labels_raw])
    
    # Handle infinite and NaN values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for label, count in zip(unique, counts):
        pct = count / len(y) * 100
        label_name = "Benign" if label == 0 else "Attack"
        print(f"  {label_name} ({int(label)}): {count} ({pct:.1f}%)")
    
    return X, y

def build_cnn_lstm_model(input_shape, num_classes=2):
    """Build CNN-LSTM model."""
    model = Sequential([
        Input(shape=input_shape),
        
        # CNN layers to extract local patterns
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # LSTM layer for sequential patterns
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        
        # Dense layers for classification
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        Dropout(0.2),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Load data
    X, y = load_data()
    
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
    
    # Reshape for CNN: (samples, features, 1)
    X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    print(f"\nInput shape: {X_train_cnn.shape[1:]}")
    
    # Calculate class weights to handle imbalance
    n_samples = len(y_train)
    n_classes = 2
    class_counts = np.bincount(y_train.astype(int))
    class_weights = {i: n_samples / (n_classes * count) for i, count in enumerate(class_counts)}
    print(f"\nClass weights: {class_weights}")
    
    # Convert labels to categorical
    y_train_cat = to_categorical(y_train, num_classes=2)
    y_test_cat = to_categorical(y_test, num_classes=2)
    
    # Build model
    print("\n" + "=" * 60)
    print("BUILDING CNN-LSTM MODEL")
    print("=" * 60)
    
    model = build_cnn_lstm_model(input_shape=(X_train_cnn.shape[1], 1))
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Callbacks
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
    
    # Train
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    history = model.fit(
        X_train_cnn, y_train_cat,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    # Get predictions
    y_pred_proba = model.predict(X_test_cnn, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Standard threshold (0.5)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy (threshold=0.5): {accuracy:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"\nAttack Detection:")
    print(f"  Recall (Detection Rate): {recall:.2%}")
    print(f"  Precision: {precision:.2%}")
    print(f"  False Positive Rate: {fp_rate:.2%}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))
    
    # Threshold tuning for better recall
    print("\n" + "=" * 60)
    print("THRESHOLD TUNING")
    print("=" * 60)
    
    attack_proba = y_pred_proba[:, 1]
    
    print(f"\n{'Threshold':<12} {'Recall':<10} {'Precision':<12} {'FP Rate':<10}")
    print("-" * 50)
    
    best_thresh = 0.5
    best_f1 = 0
    best_metrics = None
    
    for thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        y_pred_thresh = (attack_proba >= thresh).astype(int)
        
        tp = np.sum((y_pred_thresh == 1) & (y_test == 1))
        fp = np.sum((y_pred_thresh == 1) & (y_test == 0))
        fn = np.sum((y_pred_thresh == 0) & (y_test == 1))
        tn = np.sum((y_pred_thresh == 0) & (y_test == 0))
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        print(f"{thresh:<12.2f} {recall:<10.2%} {precision:<12.2%} {fp_rate:<10.2%}")
        
        if recall >= 0.80 and f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_metrics = {'recall': recall, 'precision': precision, 'fp_rate': fp_rate, 'f1': f1}
    
    if best_metrics:
        print(f"\n*** OPTIMAL THRESHOLD: {best_thresh:.2f} ***")
        print(f"  Recall: {best_metrics['recall']:.1%}")
        print(f"  Precision: {best_metrics['precision']:.1%}")
        print(f"  FP Rate: {best_metrics['fp_rate']:.1%}")
    
    # Save model
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    
    model_path = os.path.join(MODEL_DIR, "cnn_lstm_navbot.keras")
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    scaler_path = os.path.join(MODEL_DIR, "cnn_lstm_navbot_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")
    
    # Save config
    config = {
        'input_shape': list(X_train_cnn.shape[1:]),
        'n_features': X_train_scaled.shape[1],
        'optimal_threshold': float(best_thresh),
        'accuracy': float(accuracy),
        'best_metrics': {k: float(v) for k, v in best_metrics.items()} if best_metrics else None
    }
    config_path = os.path.join(MODEL_DIR, "cnn_lstm_navbot_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to: {config_path}")

if __name__ == "__main__":
    main()