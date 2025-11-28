import pandas as pd
import numpy as np
import joblib
import json
import argparse
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)

# Default paths
DEFAULT_INPUT = "/home/anguiz/Capstone/output/labeled_flows.csv"
MODEL_BASE_DIR = "/home/anguiz/Capstone/src/models"

# CICFlowMeter 78 features (Title Case with spaces)
# These are the expected feature names from CICFlowMeter output
CICFLOWMETER_FEATURES = [
    'Src Port', 'Dst Port', 'Protocol', 'Flow Duration',
    'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
    'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
    'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std',
    'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std',
    'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot', 'Fwd IAT Mean',
    'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Tot',
    'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
    'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s',
    'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std',
    'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt',
    'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt', 'CWE Flag Cnt',
    'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg',
    'Bwd Seg Size Avg', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg',
    'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts',
    'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts', 'Init Fwd Win Byts',
    'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean',
    'Active Std', 'Active Max', 'Active Min', 'Idle Mean',
    'Idle Std', 'Idle Max', 'Idle Min'
]

# Columns to exclude from features
EXCLUDE_COLUMNS = [
    'Label', 'Attack_Type', 'Timestamp', 'Timestamp_Normalized',
    'Flow ID', 'Src IP', 'Dst IP', 'timestamp', 'flow_id',
    'src_ip', 'dst_ip', 'label', 'attack_type'
]


def find_feature_columns(df: pd.DataFrame) -> list:
    """
    Find valid feature columns in the dataframe.
    Handles both Title Case and snake_case naming conventions.
    """
    available_cols = set(df.columns)
    exclude_lower = {c.lower() for c in EXCLUDE_COLUMNS}
    
    # Filter out excluded columns and non-numeric columns
    feature_cols = []
    for col in df.columns:
        if col.lower() in exclude_lower:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        feature_cols.append(col)
    
    return feature_cols


def preprocess_data(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    Preprocess data for training.
    
    Returns:
        (X, y, feature_names)
    """
    # Extract features
    X = df[feature_cols].copy()
    y = df['Label'].values
    
    # Handle missing values
    X = X.fillna(0)
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], 0)
    
    return X.values, y, feature_cols


def calculate_class_weights(y: np.ndarray) -> dict:
    """Calculate class weights based on imbalance ratio."""
    n_samples = len(y)
    n_positive = np.sum(y)
    n_negative = n_samples - n_positive
    
    if n_positive == 0:
        print("WARNING: No positive samples found!")
        return {0: 1, 1: 1}
    
    # Weight inversely proportional to class frequency
    weight_positive = n_samples / (2 * n_positive)
    weight_negative = n_samples / (2 * n_negative)
    
    # Normalize so negative class weight = 1
    ratio = weight_positive / weight_negative
    
    print(f"  Class distribution: {n_negative} benign, {n_positive} attack")
    print(f"  Imbalance ratio: {n_negative/n_positive:.1f}:1")
    print(f"  Class weights: {{0: 1, 1: {ratio:.1f}}}")
    
    return {0: 1, 1: ratio}


def train_random_forest(X_train, y_train, class_weights: dict) -> RandomForestClassifier:
    """Train Random Forest classifier."""
    print("\n  Training Random Forest...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train, class_weights: dict) -> DecisionTreeClassifier:
    """Train Decision Tree classifier."""
    print("\n  Training Decision Tree...")
    
    model = DecisionTreeClassifier(
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight=class_weights,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_knn(X_train, y_train, scaler: StandardScaler) -> KNeighborsClassifier:
    """
    Train K-Nearest Neighbors classifier.
    Note: KNN doesn't support class_weight, so we scale features instead.
    """
    print("\n  Training K-Nearest Neighbors...")
    
    # KNN needs scaled features
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',  # Weight by distance to handle imbalance somewhat
        metric='euclidean',
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name: str, scaler=None) -> dict:
    """Evaluate model and return metrics."""
    print(f"\n  Evaluating {model_name}...")
    
    # Scale if needed (for KNN)
    if scaler is not None:
        X_test_eval = scaler.transform(X_test)
    else:
        X_test_eval = X_test
    
    # Predictions
    y_pred = model.predict(X_test_eval)
    y_proba = model.predict_proba(X_test_eval)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }
    
    if y_proba is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_test, y_proba)
        except ValueError:
            metrics['auc_roc'] = 0.0
    
    # Print results
    print(f"\n  {model_name} Results:")
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1 Score:  {metrics['f1']:.4f}")
    if 'auc_roc' in metrics:
        print(f"    AUC-ROC:   {metrics['auc_roc']:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n    Confusion Matrix:")
    print(f"    TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
    print(f"    FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")
    
    return metrics


def save_model(model, output_dir: Path, model_name: str, 
               scaler: StandardScaler, feature_names: list, metrics: dict):
    """
    Save model, scaler, and metadata.
    
    Output structure matches your existing IDS nodes:
    - /models/{model_name}_model.joblib
    - /models/{model_name}_scaler.joblib  
    - /models/{model_name}_used_features.txt
    
    Also saves to subdirectory for ml_fusion compatibility:
    - /models/{model_name}/model.joblib (etc.)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # === PRIMARY: Flat structure for rf_ids_node.py, dt_ids_node.py, knn_ids_node.py ===
    # These expect: rf_model.joblib, rf_scaler.joblib, rf_used_features.txt
    model_path = output_dir / f"{model_name}_model.joblib"
    scaler_path = output_dir / f"{model_name}_scaler.joblib"
    features_path = output_dir / f"{model_name}_used_features.txt"
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    with open(features_path, 'w') as f:
        for feat in feature_names:
            f.write(f"{feat}\n")
    
    print(f"  Saved: {model_path.name}, {scaler_path.name}, {features_path.name}")
    
    # === SECONDARY: Subdirectory structure for organization ===
    model_subdir = output_dir / model_name
    model_subdir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, model_subdir / f"{model_name}_model.joblib")
    joblib.dump(scaler, model_subdir / f"{model_name}_scaler.joblib")
    
    with open(model_subdir / f"{model_name}_used_features.txt", 'w') as f:
        for feat in feature_names:
            f.write(f"{feat}\n")
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'n_features': len(feature_names),
        'trained_at': datetime.now().isoformat(),
        'metrics': metrics
    }
    with open(model_subdir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Also saved to: {model_subdir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Train standalone ML models for IDS"
    )
    parser.add_argument(
        '--input', '-i',
        default=DEFAULT_INPUT,
        help=f"Input labeled CSV file (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        '--output-dir', '-o',
        default=MODEL_BASE_DIR,
        help=f"Base directory for model output (default: {MODEL_BASE_DIR})"
    )
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        choices=['rf', 'dt', 'knn', 'all'],
        default=['all'],
        help="Models to train (default: all)"
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help="Test set proportion (default: 0.2)"
    )
    
    args = parser.parse_args()
    
    # Determine which models to train
    if 'all' in args.models:
        models_to_train = ['rf', 'dt', 'knn']
    else:
        models_to_train = args.models
    
    print("="*60)
    print("STANDALONE MODEL TRAINING SCRIPT")
    print("="*60)
    print(f"Input:      {args.input}")
    print(f"Output dir: {args.output_dir}")
    print(f"Models:     {', '.join(models_to_train)}")
    print(f"Test size:  {args.test_size}")
    
    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {args.input}")
        print("Run label_data.py first to create labeled data.")
        return 1
    
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} samples")
    
    if 'Label' not in df.columns:
        print("ERROR: 'Label' column not found. Run label_data.py first.")
        return 1
    
    # Find feature columns
    feature_cols = find_feature_columns(df)
    print(f"Found {len(feature_cols)} feature columns")
    
    # Preprocess
    X, y, feature_names = preprocess_data(df, feature_cols)
    print(f"Feature matrix shape: {X.shape}")
    
    # Calculate class weights
    class_weights = calculate_class_weights(y)
    
    # Train/test split
    print("\n" + "="*60)
    print("SPLITTING DATA")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}")
    print(f"Training attack samples: {np.sum(y_train)}")
    print(f"Test attack samples:     {np.sum(y_test)}")
    
    # Initialize scaler (used by all models for consistency)
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    output_dir = Path(args.output_dir)
    results = {}
    
    # Train Random Forest
    if 'rf' in models_to_train:
        print("\n" + "="*60)
        print("RANDOM FOREST")
        print("="*60)
        
        rf_model = train_random_forest(X_train, y_train, class_weights)
        rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
        save_model(rf_model, output_dir, "rf", scaler, feature_names, rf_metrics)
        results['rf'] = rf_metrics
    
    # Train Decision Tree
    if 'dt' in models_to_train:
        print("\n" + "="*60)
        print("DECISION TREE")
        print("="*60)
        
        dt_model = train_decision_tree(X_train, y_train, class_weights)
        dt_metrics = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
        save_model(dt_model, output_dir, "dt", scaler, feature_names, dt_metrics)
        results['dt'] = dt_metrics
    
    # Train KNN
    if 'knn' in models_to_train:
        print("\n" + "="*60)
        print("K-NEAREST NEIGHBORS")
        print("="*60)
        
        knn_scaler = StandardScaler()
        knn_model = train_knn(X_train, y_train, knn_scaler)
        knn_metrics = evaluate_model(knn_model, X_test, y_test, "KNN", knn_scaler)
        save_model(knn_model, output_dir, "knn", knn_scaler, feature_names, knn_metrics)
        results['knn'] = knn_metrics
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    print(f"\n{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 55)
    for model_name, metrics in results.items():
        print(f"{model_name.upper():<15} "
              f"{metrics['accuracy']:<10.4f} "
              f"{metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} "
              f"{metrics['f1']:<10.4f}")
    
    print("\n✓ Training complete!")
    print(f"  Models saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())