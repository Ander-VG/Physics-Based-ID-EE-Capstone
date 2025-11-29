import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
DATA_PATH = "/home/anguiz/Capstone/ML_output/NavBot25_train.csv"
MODEL_DIR = "/home/anguiz/Capstone/ML_output/models"

# Feature columns (CICFlowMeter features: columns 8-83, 0-indexed: 7-82)
FEATURE_COLS = list(range(7, 83))  # 76 features
LABEL_COL = 83  # "Label" column

def load_data():
    """Load and prepare the dataset."""
    print("=" * 60)
    print("LOADING NAVBOT25 TRAINING DATA")
    print("=" * 60)
    
    df = pd.read_csv(DATA_PATH)
    print(f"Total samples: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    # Get feature names
    feature_names = df.columns[FEATURE_COLS].tolist()
    print(f"Number of features: {len(feature_names)}")
    
    # Extract features
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
    
    return X, y, feature_names

def train_and_evaluate(X_train, X_val, y_train, y_val, model, model_name):
    """Train a model and print evaluation metrics."""
    print(f"\n{'=' * 60}")
    print(f"TRAINING: {model_name}")
    print("=" * 60)
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_val)
    
    # Metrics
    accuracy = accuracy_score(y_val, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_val, y_pred)
    print(cm)
    
    # Calculate key metrics
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"\nAttack Detection:")
    print(f"  Recall (Detection Rate): {recall:.2%}")
    print(f"  Precision: {precision:.2%}")
    print(f"  False Positive Rate: {fp_rate:.2%}")
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['Benign', 'Attack']))
    
    return model, {'accuracy': accuracy, 'recall': recall, 'precision': precision, 'fp_rate': fp_rate}

def main():
    # Create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Load data
    X, y, feature_names = load_data()
    
    # Split for validation only (not creating demo set - already done)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Save scaler
    scaler_path = os.path.join(MODEL_DIR, "scaler_navbot.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"\nScaler saved to: {scaler_path}")
    
    # Save feature names
    feature_path = os.path.join(MODEL_DIR, "feature_names_navbot.json")
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
        # KNN needs scaled data
        if name == 'KNN':
            trained_model, metrics = train_and_evaluate(
                X_train_scaled, X_val_scaled, y_train, y_val, model, name
            )
        else:
            trained_model, metrics = train_and_evaluate(
                X_train, X_val, y_train, y_val, model, name
            )
        
        results[name] = metrics
        
        # Save model
        model_path = os.path.join(MODEL_DIR, f"{name.lower()}_navbot.joblib")
        joblib.dump(trained_model, model_path)
        print(f"Model saved to: {model_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nModel Performance (on validation set):")
    print(f"{'Model':<15} {'Accuracy':<10} {'Recall':<10} {'Precision':<10} {'FP Rate':<10}")
    print("-" * 55)
    for name, metrics in sorted(results.items(), key=lambda x: x[1]['recall'], reverse=True):
        print(f"{name:<15} {metrics['accuracy']:<10.2%} {metrics['recall']:<10.2%} {metrics['precision']:<10.2%} {metrics['fp_rate']:<10.2%}")
    
    best_model = max(results, key=lambda x: results[x]['recall'])
    print(f"\nBest Model (by recall): {best_model}")
    
    print("\nAll models saved to:", MODEL_DIR)

if __name__ == "__main__":
    main()