import pandas as pd
import numpy as np
import joblib
import json
import argparse
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)

# Default paths
DEFAULT_INPUT = "/home/anguiz/Capstone/output/labeled_flows.csv"
DEFAULT_OUTPUT = "/home/anguiz/Capstone/src/models/ml_fusion"

# Columns to exclude from features
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
    
    # Handle missing and infinite values
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    return X.values, y, feature_cols


def calculate_class_weights(y: np.ndarray) -> dict:
    """Calculate class weights for imbalanced data."""
    n_samples = len(y)
    n_positive = np.sum(y)
    n_negative = n_samples - n_positive
    
    if n_positive == 0:
        return {0: 1, 1: 1}
    
    ratio = n_negative / n_positive
    return {0: 1, 1: min(ratio, 50)}  # Cap at 50 to prevent extreme weights


class MLFusionPipeline:
    """
    ML Fusion Pipeline for IDS.
    
    Pipeline architecture:
    Raw Features → Scale → PCA(n_components)
                         ↓
                    ┌────┴────┐
                    │         │
                   KNN       RF
                    │         │
                    └────┬────┘
                         ↓
              Fuse: [PCA features, KNN_proba, RF_proba]
                         ↓
                        LR
                         ↓
                   Final Prediction
    """
    
    def __init__(self, n_pca_components: int = 16):
        self.n_pca_components = n_pca_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_pca_components)
        self.knn = None
        self.rf = None
        self.lr = None
        self.feature_names = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, class_weights: dict):
        """Fit the entire fusion pipeline."""
        print("\n  Step 1: Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"  Step 2: PCA dimensionality reduction ({X.shape[1]} → {self.n_pca_components})...")
        X_pca = self.pca.fit_transform(X_scaled)
        explained_var = np.sum(self.pca.explained_variance_ratio_)
        print(f"    Explained variance: {explained_var:.2%}")
        
        print("  Step 3: Training KNN on PCA features...")
        self.knn = KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='euclidean',
            n_jobs=-1
        )
        self.knn.fit(X_pca, y)
        
        print("  Step 4: Training Random Forest on PCA features...")
        self.rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1
        )
        self.rf.fit(X_pca, y)
        
        print("  Step 5: Creating fusion features...")
        # Get FULL probability arrays (shape: N x n_classes)
        # For binary: shape is (N, 2) with [prob_class_0, prob_class_1]
        knn_proba = self.knn.predict_proba(X_pca)  # (N, 2) for binary
        rf_proba = self.rf.predict_proba(X_pca)    # (N, 2) for binary
        
        # Fuse: [PCA features, KNN full proba, RF full proba]
        # This matches your mlfusion_ids_node.py: np.concatenate([X_pca, knn_proba, rf_proba], axis=1)
        X_fused = np.concatenate([X_pca, knn_proba, rf_proba], axis=1)
        print(f"    Fusion feature shape: {X_fused.shape}")
        print(f"    (PCA: {X_pca.shape[1]} + KNN_proba: {knn_proba.shape[1]} + RF_proba: {rf_proba.shape[1]})")
        
        print("  Step 6: Training Logistic Regression on fused features...")
        self.lr = LogisticRegression(
            class_weight=class_weights,
            max_iter=1000,
            random_state=42
        )
        self.lr.fit(X_fused, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fusion pipeline."""
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        
        # Full probability arrays
        knn_proba = self.knn.predict_proba(X_pca)
        rf_proba = self.rf.predict_proba(X_pca)
        
        X_fused = np.concatenate([X_pca, knn_proba, rf_proba], axis=1)
        
        return self.lr.predict(X_fused)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using the fusion pipeline."""
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        
        knn_proba = self.knn.predict_proba(X_pca)
        rf_proba = self.rf.predict_proba(X_pca)
        
        X_fused = np.concatenate([X_pca, knn_proba, rf_proba], axis=1)
        
        return self.lr.predict_proba(X_fused)
    
    def get_intermediate_predictions(self, X: np.ndarray) -> dict:
        """Get predictions from each stage for analysis."""
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        
        knn_proba = self.knn.predict_proba(X_pca)
        rf_proba = self.rf.predict_proba(X_pca)
        
        return {
            'knn_pred': self.knn.predict(X_pca),
            'knn_proba': knn_proba[:, 1],  # Attack probability
            'rf_pred': self.rf.predict(X_pca),
            'rf_proba': rf_proba[:, 1],    # Attack probability
            'final_pred': self.predict(X),
            'final_proba': self.predict_proba(X)[:, 1]
        }
    
    def save(self, output_dir: Path, feature_names: list, metrics: dict):
        """
        Save all pipeline components.
        
        Output matches mlfusion_ids_node.py expectations:
        - scaler.joblib
        - pca.joblib
        - knn_classifier.joblib  (not knn_model.joblib!)
        - rf_classifier.joblib   (not rf_model.joblib!)
        - lr_classifier.joblib   (not lr_model.joblib!)
        - features.txt
        - pipeline_info.json
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual components with EXACT names your IDS node expects
        joblib.dump(self.scaler, output_dir / "scaler.joblib")
        joblib.dump(self.pca, output_dir / "pca.joblib")
        joblib.dump(self.knn, output_dir / "knn_classifier.joblib")  # Note: _classifier suffix
        joblib.dump(self.rf, output_dir / "rf_classifier.joblib")    # Note: _classifier suffix
        joblib.dump(self.lr, output_dir / "lr_classifier.joblib")    # Note: _classifier suffix
        
        # Save feature names
        with open(output_dir / "features.txt", 'w') as f:
            for feat in feature_names:
                f.write(f"{feat}\n")
        
        # Save pipeline info (matches your node's expected format)
        pipeline_info = {
            'architecture': 'Scale → PCA → KNN+RF → Fuse → LR',
            'n_input_features': len(feature_names),
            'n_pca_components': self.n_pca_components,
            'n_fused_features': self.n_pca_components + 4,  # PCA + KNN_proba(2) + RF_proba(2) for binary
            'pca_explained_variance': float(np.sum(self.pca.explained_variance_ratio_)),
            'trained_at': datetime.now().isoformat(),
            'metrics': metrics,
            # Attack classes for format_prediction_message()
            'attack_classes': {
                "0": "Normal",
                "1": "Attack"
            }
        }
        
        with open(output_dir / "pipeline_info.json", 'w') as f:
            json.dump(pipeline_info, f, indent=2)
        
        print(f"\n  Saved pipeline to: {output_dir}")
        print(f"    - scaler.joblib")
        print(f"    - pca.joblib")
        print(f"    - knn_classifier.joblib")
        print(f"    - rf_classifier.joblib")
        print(f"    - lr_classifier.joblib")
        print(f"    - features.txt")
        print(f"    - pipeline_info.json")


def evaluate_pipeline(pipeline: MLFusionPipeline, X_test: np.ndarray, 
                     y_test: np.ndarray) -> dict:
    """Evaluate the fusion pipeline."""
    print("\n  Evaluating fusion pipeline...")
    
    # Get predictions at each stage
    intermediate = pipeline.get_intermediate_predictions(X_test)
    
    # Evaluate each component
    print("\n  Component-wise evaluation:")
    
    for name, pred in [('KNN', intermediate['knn_pred']), 
                       ('RF', intermediate['rf_pred']),
                       ('Fusion (LR)', intermediate['final_pred'])]:
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        print(f"    {name:<15} Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
    
    # Final metrics
    y_pred = intermediate['final_pred']
    y_proba = intermediate['final_proba']
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }
    
    try:
        metrics['auc_roc'] = roc_auc_score(y_test, y_proba)
    except ValueError:
        metrics['auc_roc'] = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Final Confusion Matrix:")
    print(f"    TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
    print(f"    FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")
    
    # Component comparison
    print("\n  Individual component metrics:")
    for name, proba_key in [('KNN', 'knn_proba'), ('RF', 'rf_proba')]:
        try:
            auc = roc_auc_score(y_test, intermediate[proba_key])
            print(f"    {name} AUC-ROC: {auc:.4f}")
        except ValueError:
            pass
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train ML Fusion IDS pipeline"
    )
    parser.add_argument(
        '--input', '-i',
        default=DEFAULT_INPUT,
        help=f"Input labeled CSV file (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        '--output', '-o',
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        '--pca-components', '-p',
        type=int,
        default=16,
        help="Number of PCA components (default: 16)"
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help="Test set proportion (default: 0.2)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("ML FUSION PIPELINE TRAINING")
    print("="*60)
    print(f"Input:          {args.input}")
    print(f"Output:         {args.output}")
    print(f"PCA components: {args.pca_components}")
    print(f"Test size:      {args.test_size}")
    
    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {args.input}")
        return 1
    
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} samples")
    
    if 'Label' not in df.columns:
        print("ERROR: 'Label' column not found.")
        return 1
    
    # Find and preprocess features
    feature_cols = find_feature_columns(df)
    print(f"Found {len(feature_cols)} feature columns")
    
    X, y, feature_names = preprocess_data(df, feature_cols)
    print(f"Feature matrix shape: {X.shape}")
    
    # Adjust PCA components if needed
    if args.pca_components > X.shape[1]:
        print(f"Warning: Reducing PCA components from {args.pca_components} to {X.shape[1]}")
        args.pca_components = X.shape[1]
    
    # Class weights
    class_weights = calculate_class_weights(y)
    
    # Train/test split
    print("\n" + "="*60)
    print("SPLITTING DATA")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"Training: {len(X_train)} samples ({np.sum(y_train)} attacks)")
    print(f"Test:     {len(X_test)} samples ({np.sum(y_test)} attacks)")
    
    # Train pipeline
    print("\n" + "="*60)
    print("TRAINING FUSION PIPELINE")
    print("="*60)
    print(f"\nPipeline: Scale → PCA({args.pca_components}) → KNN+RF → Fuse → LR")
    
    pipeline = MLFusionPipeline(n_pca_components=args.pca_components)
    pipeline.fit(X_train, y_train, class_weights)
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    metrics = evaluate_pipeline(pipeline, X_test, y_test)
    
    # Save
    print("\n" + "="*60)
    print("SAVING PIPELINE")
    print("="*60)
    
    output_dir = Path(args.output)
    pipeline.save(output_dir, feature_names, metrics)
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"\nFinal Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {metrics.get('auc_roc', 'N/A'):.4f}")
    
    print("\n✓ ML Fusion training complete!")
    
    return 0


if __name__ == "__main__":
    exit(main())