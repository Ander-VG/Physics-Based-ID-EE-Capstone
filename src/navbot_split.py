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
DATA_PATH = "/home/anguiz/Capstone/ML_output/NavBot25.csv"
MODEL_DIR = "/home/anguiz/Capstone/src/models"
OUTPUT_DIR = "/home/anguiz/Capstone/ML_output"

FEATURE_COLS = list(range(7, 83))  # 76 features
LABEL_COL = 83  # Label column

def split_dataset():
    """Split NavBot25 into train and demo sets."""
    print("=" * 60)
    print("SPLITTING NAVBOT25 DATASET")
    print("=" * 60)
    
    df = pd.read_csv(DATA_PATH)
    print(f"Total samples: {len(df)}")
    
    # Stratified split by Label column
    train_df, demo_df = train_test_split(
        df, 
        test_size=0.2, 
        stratify=df.iloc[:, LABEL_COL], 
        random_state=42
    )
    
    print(f"\nTrain set: {len(train_df)} samples (80%)")
    print(f"Demo set:  {len(demo_df)} samples (20%)")
    
    # Save splits
    train_path = os.path.join(OUTPUT_DIR, "NavBot25_train.csv")
    demo_path = os.path.join(OUTPUT_DIR, "NavBot25_demo.csv")
    
    train_df.to_csv(train_path, index=False)
    demo_df.to_csv(demo_path, index=False)
    
    print(f"\nSaved: {train_path}")
    print(f"Saved: {demo_path}")
    
    # Show distribution in both sets
    print("\n" + "-" * 40)
    print("TRAIN SET distribution:")
    train_counts = train_df.iloc[:, LABEL_COL].value_counts()
    for label, count in train_counts.items():
        pct = count / len(train_df) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    print("\nDEMO SET distribution:")
    demo_counts = demo_df.iloc[:, LABEL_COL].value_counts()
    for label, count in demo_counts.items():
        pct = count / len(demo_df) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    return train_df, demo_df

def load_and_prepare(df):
    """Prepare features and labels from dataframe."""
    X = df.iloc[:, FEATURE_COLS].values
    
    # Convert labels to binary: Normal=0, Attack=1
    labels_raw = df.iloc[:, LABEL_COL].values
    y = np.array([0 if label == 'Normal' else 1 for label in labels_raw])
    
    # Handle infinite and NaN values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X, y

def train_and_evaluate(X_train, X_test, y_train, y_test, model, model_name):
    """Train a model and print evaluation metrics."""
    print(f"\n{'=' * 60}")
    print(f"TRAINING: {model_name}")
    print("=" * 60)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
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
    
    return model, {'accuracy': accuracy, 'recall': recall, 'precision': precision, 'fp_rate': fp_rate}

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Split dataset
    train_df, demo_df = split_dataset()
    
    # Prepare training data
    print("\n" + "=" * 60)
    print("PREPARING TRAINING DATA")
    print("=" * 60)
    
    X_train_full, y_train_full = load_and_prepare(train_df)
    
    # Further split training set for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, 
        test_size=0.2, 
        stratify=y_train_full, 
        random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Binary distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print("\nBinary class distribution (train):")
    for label, count in zip(unique, counts):
        pct = count / len(y_train) * 100
        label_name = "Benign" if label == 0 else "Attack"
        print(f"  {label_name}: {count} ({pct:.1f}%)")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Save scaler
    scaler_path = os.path.join(MODEL_DIR, "scaler_navbot.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"\nScaler saved to: {scaler_path}")
    
    # Save feature names
    feature_names = train_df.columns[FEATURE_COLS].tolist()
    feature_path = os.path.join(MODEL_DIR, "feature_names_navbot.json")
    with open(feature_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    
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
    
    # Train and evaluate
    results = {}
    for name, model in models.items():
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
    
    print("\n" + "=" * 60)
    print("FILES CREATED")
    print("=" * 60)
    print(f"  Training data: {OUTPUT_DIR}/NavBot25_train.csv")
    print(f"  Demo data:     {OUTPUT_DIR}/NavBot25_demo.csv (KEEP HIDDEN FOR DEMO)")
    print(f"  Models:        {MODEL_DIR}/*_navbot.joblib")

if __name__ == "__main__":
    main()