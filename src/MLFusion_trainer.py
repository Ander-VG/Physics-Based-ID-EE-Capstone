import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = "/home/anguiz/Capstone/ML_output/NavBot25_train.csv"
MODEL_DIR = "/home/anguiz/Capstone/src/models"

# Feature columns (CICFlowMeter features: columns 8-83, 0-indexed: 7-82)
FEATURE_COLS = list(range(7, 83))  # 76 features
LABEL_COL = 83  # Column 84 (0-indexed: 83) - "Label" column

# Fusion parameters
PCA_COMPONENTS = 20  # Reduce to 20 principal components
KNN_NEIGHBORS = 5
RF_ESTIMATORS = 50

def load_data():
    """Load and prepare the dataset."""
    print("=" * 60)
    print("LOADING NAVBOT25 TRAINING DATA")
    print("=" * 60)
    
    df = pd.read_csv(DATA_PATH)
    print(f"Total samples: {len(df)}")
    
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

def train_fusion_model(X_train, X_test, y_train, y_test):
    """Train the ML Fusion pipeline."""
    print(f"\n{'=' * 60}")
    print("TRAINING ML FUSION PIPELINE")
    print("=" * 60)
    
    # Step 1: Scale features
    print("\n[Step 1] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 2: PCA dimensionality reduction
    print(f"[Step 2] PCA reduction to {PCA_COMPONENTS} components...")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    explained_var = sum(pca.explained_variance_ratio_) * 100
    print(f"  Explained variance: {explained_var:.1f}%")
    
    # Step 3: Train base classifiers
    print(f"[Step 3] Training base classifiers...")
    
    # KNN
    print(f"  Training KNN (k={KNN_NEIGHBORS})...")
    knn = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS, weights='distance', n_jobs=-1)
    knn.fit(X_train_pca, y_train)
    knn_proba = knn.predict_proba(X_train_pca)
    knn_proba_test = knn.predict_proba(X_test_pca)
    
    # Random Forest
    print(f"  Training Random Forest (n={RF_ESTIMATORS})...")
    rf = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        max_depth=15,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_pca, y_train)
    rf_proba = rf.predict_proba(X_train_pca)
    rf_proba_test = rf.predict_proba(X_test_pca)
    
    # Step 4: Create meta-features
    print("[Step 4] Creating meta-features...")
    X_meta_train = np.hstack([knn_proba, rf_proba])
    X_meta_test = np.hstack([knn_proba_test, rf_proba_test])
    print(f"  Meta-feature shape: {X_meta_train.shape}")
    
    # Step 5: Train meta-classifier (Logistic Regression)
    print("[Step 5] Training Logistic Regression meta-classifier...")
    meta_clf = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    meta_clf.fit(X_meta_train, y_train)
    
    # Evaluate
    y_pred = meta_clf.predict(X_meta_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'=' * 60}")
    print("FUSION MODEL RESULTS")
    print("=" * 60)
    print(f"\nAccuracy: {accuracy:.4f}")
    
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
    
    # Return all components
    return {
        'scaler': scaler,
        'pca': pca,
        'knn': knn,
        'rf': rf,
        'meta_clf': meta_clf,
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'fp_rate': fp_rate
    }

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
    
    # Train fusion model
    components = train_fusion_model(X_train, X_test, y_train, y_test)
    
    # Save all components
    print(f"\n{'=' * 60}")
    print("SAVING MODEL COMPONENTS")
    print("=" * 60)
    
    # Save as a single pipeline object
    fusion_pipeline = {
        'scaler': components['scaler'],
        'pca': components['pca'],
        'knn': components['knn'],
        'rf': components['rf'],
        'meta_clf': components['meta_clf']
    }
    
    pipeline_path = os.path.join(MODEL_DIR, "ml_fusion_navbot.joblib")
    joblib.dump(fusion_pipeline, pipeline_path)
    print(f"Fusion pipeline saved to: {pipeline_path}")
    
    # Save config
    config = {
        'pca_components': PCA_COMPONENTS,
        'knn_neighbors': KNN_NEIGHBORS,
        'rf_estimators': RF_ESTIMATORS,
        'n_features': len(feature_names),
        'accuracy': components['accuracy'],
        'recall': components['recall'],
        'precision': components['precision'],
        'fp_rate': components['fp_rate']
    }
    config_path = os.path.join(MODEL_DIR, "ml_fusion_navbot_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to: {config_path}")
    
    print(f"\nFusion model accuracy: {components['accuracy']:.4f}")
    print(f"Fusion model recall: {components['recall']:.2%}")

if __name__ == "__main__":
    main()