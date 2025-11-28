#!/usr/bin/env python3
"""
Correlation Matrix Analysis for TurtleBot3 Intrusion Detection
Analyzes ML-ready CSV to determine optimal features for Isolation Forest
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def load_data(csv_path):
    """Load the ML-ready CSV file"""
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print('='*70)
    
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded: {csv_path}")
    print(f"✓ Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"✓ Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df

def explore_data(df):
    """Basic data exploration"""
    print(f"\n{'='*70}")
    print("DATA EXPLORATION")
    print('='*70)
    
    print("\nColumn Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\nData Types:")
    print(df.dtypes)
    
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  ✓ No missing values!")
    else:
        print(missing[missing > 0])
    
    print(f"\nBasic Statistics:")
    print(df.describe())
    
    # Check for constant columns
    print(f"\nChecking for constant/near-constant columns...")
    constant_cols = []
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            if df[col].std() < 1e-10:
                constant_cols.append(col)
                print(f"  ⚠ {col}: constant (std={df[col].std():.2e})")
    
    if not constant_cols:
        print("  ✓ No constant columns found")
    
    return constant_cols

def compute_correlation(df, method='pearson'):
    """Compute correlation matrix"""
    print(f"\n{'='*70}")
    print(f"COMPUTING CORRELATION MATRIX ({method.upper()})")
    print('='*70)
    
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"✓ Analyzing {len(numeric_cols)} numeric features")
    
    # Compute correlation
    corr_matrix = df[numeric_cols].corr(method=method)
    
    return corr_matrix, numeric_cols

def find_high_correlations(corr_matrix, threshold=0.95):
    """Find highly correlated feature pairs"""
    print(f"\n{'='*70}")
    print(f"HIGH CORRELATIONS (|r| > {threshold})")
    print('='*70)
    
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > threshold:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    if high_corr_pairs:
        print(f"\n⚠ Found {len(high_corr_pairs)} highly correlated pairs:")
        for pair in sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True):
            print(f"  {pair['feature1']:20s} ↔ {pair['feature2']:20s} : {pair['correlation']:+.3f}")
    else:
        print(f"✓ No feature pairs with |correlation| > {threshold}")
    
    return high_corr_pairs

def recommend_features_to_drop(high_corr_pairs, corr_matrix):
    """Recommend which features to drop from highly correlated pairs"""
    print(f"\n{'='*70}")
    print("FEATURE SELECTION RECOMMENDATIONS")
    print('='*70)
    
    features_to_drop = set()
    
    for pair in high_corr_pairs:
        feat1 = pair['feature1']
        feat2 = pair['feature2']
        
        # Keep the feature with higher average correlation to other features
        avg_corr1 = corr_matrix[feat1].abs().mean()
        avg_corr2 = corr_matrix[feat2].abs().mean()
        
        if avg_corr1 > avg_corr2:
            features_to_drop.add(feat1)
            print(f"  DROP: {feat1:20s} (avg |corr|={avg_corr1:.3f})")
            print(f"  KEEP: {feat2:20s} (avg |corr|={avg_corr2:.3f})")
        else:
            features_to_drop.add(feat2)
            print(f"  KEEP: {feat1:20s} (avg |corr|={avg_corr1:.3f})")
            print(f"  DROP: {feat2:20s} (avg |corr|={avg_corr2:.3f})")
        print()
    
    return list(features_to_drop)

def plot_correlation_heatmap(corr_matrix, output_path, figsize=(16, 14)):
    """Create and save correlation heatmap"""
    print(f"\n{'='*70}")
    print("GENERATING CORRELATION HEATMAP")
    print('='*70)
    
    plt.figure(figsize=figsize)
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # Create heatmap
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        vmin=-1,
        vmax=1
    )
    
    plt.title('Feature Correlation Matrix\n(Lower Triangle Only)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved heatmap: {output_path}")
    
    plt.close()

def plot_correlation_distribution(corr_matrix, output_path):
    """Plot distribution of correlation values"""
    print(f"\n{'='*70}")
    print("GENERATING CORRELATION DISTRIBUTION")
    print('='*70)
    
    # Get upper triangle correlations (excluding diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    correlations = corr_matrix.where(mask).stack().values
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(correlations, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero correlation')
    axes[0].axvline(0.95, color='orange', linestyle='--', linewidth=1.5, label='High correlation (0.95)')
    axes[0].axvline(-0.95, color='orange', linestyle='--', linewidth=1.5)
    axes[0].set_xlabel('Correlation Coefficient', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Feature Correlations', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot([correlations], vert=True, widths=0.5)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=2, label='Zero correlation')
    axes[1].axhline(0.95, color='orange', linestyle='--', linewidth=1.5, label='High correlation')
    axes[1].axhline(-0.95, color='orange', linestyle='--', linewidth=1.5)
    axes[1].set_ylabel('Correlation Coefficient', fontsize=12)
    axes[1].set_title('Correlation Box Plot', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved distribution plot: {output_path}")
    
    plt.close()

def analyze_feature_importance(df, corr_matrix):
    """Analyze which features might be most important for anomaly detection"""
    print(f"\n{'='*70}")
    print("FEATURE IMPORTANCE ANALYSIS")
    print('='*70)
    
    # Features that should have strong physical relationships
    expected_relationships = {
        'linear_x vs X,Y': ['linear_x', 'X', 'Y'],
        'angular_z vs Theta': ['angular_z', 'Theta'],
        'imu_gyro_z vs angular_z': ['imu_gyro_z', 'angular_z'],
        'v_R,v_L vs linear_x': ['v_R', 'v_L', 'linear_x'],
    }
    
    print("\nExpected Physical Relationships:")
    for relationship, features in expected_relationships.items():
        available_features = [f for f in features if f in corr_matrix.columns]
        if len(available_features) >= 2:
            print(f"\n  {relationship}:")
            for i, feat1 in enumerate(available_features):
                for feat2 in available_features[i+1:]:
                    corr_val = corr_matrix.loc[feat1, feat2]
                    status = "✓" if abs(corr_val) > 0.3 else "⚠"
                    print(f"    {status} {feat1:20s} ↔ {feat2:20s}: {corr_val:+.3f}")

def save_recommended_features(corr_matrix, features_to_drop, constant_cols, output_path):
    """Save recommended feature list for ML model"""
    print(f"\n{'='*70}")
    print("SAVING RECOMMENDED FEATURES")
    print('='*70)
    
    all_features = corr_matrix.columns.tolist()
    
    # Remove Time column (not a feature)
    if 'Time' in all_features:
        all_features.remove('Time')
    
    # Remove constant columns
    all_features = [f for f in all_features if f not in constant_cols]
    
    # Remove highly correlated features
    recommended_features = [f for f in all_features if f not in features_to_drop]
    
    print(f"\n  Total features: {len(corr_matrix.columns)}")
    print(f"  - Time column: 1")
    print(f"  - Constant features: {len(constant_cols)}")
    print(f"  - Highly correlated: {len(features_to_drop)}")
    print(f"  ─────────────────────────")
    print(f"  ✓ Recommended features: {len(recommended_features)}")
    
    # Save to file
    with open(output_path, 'w') as f:
        f.write("# Recommended Features for Isolation Forest\n")
        f.write(f"# Generated from correlation analysis\n")
        f.write(f"# Total features: {len(recommended_features)}\n\n")
        for feat in recommended_features:
            f.write(f"{feat}\n")
    
    print(f"\n✓ Saved to: {output_path}")
    
    return recommended_features

def main():
    parser = argparse.ArgumentParser(
        description='Correlation analysis for TurtleBot3 ML data'
    )
    parser.add_argument(
        'csv_file',
        help='Path to ML-ready CSV file'
    )
    parser.add_argument(
        '--output-dir',
        default='./correlation_analysis',
        help='Output directory for plots and reports (default: ./correlation_analysis)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.95,
        help='Correlation threshold for flagging features (default: 0.95)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("TURTLEBOT3 CORRELATION ANALYSIS FOR ISOLATION FOREST")
    print("="*70)
    
    # Load data
    df = load_data(args.csv_file)
    
    # Explore data
    constant_cols = explore_data(df)
    
    # Compute correlation
    corr_matrix, numeric_cols = compute_correlation(df)
    
    # Find high correlations
    high_corr_pairs = find_high_correlations(corr_matrix, threshold=args.threshold)
    
    # Recommend features to drop
    features_to_drop = recommend_features_to_drop(high_corr_pairs, corr_matrix)
    
    # Analyze feature importance
    analyze_feature_importance(df, corr_matrix)
    
    # Generate plots
    plot_correlation_heatmap(
        corr_matrix,
        output_dir / 'correlation_heatmap.png'
    )
    
    plot_correlation_distribution(
        corr_matrix,
        output_dir / 'correlation_distribution.png'
    )
    
    # Save recommended features
    recommended_features = save_recommended_features(
        corr_matrix,
        features_to_drop,
        constant_cols,
        output_dir / 'recommended_features.txt'
    )
    
    # Save correlation matrix to CSV
    corr_csv_path = output_dir / 'correlation_matrix.csv'
    corr_matrix.to_csv(corr_csv_path)
    print(f"\n✓ Saved correlation matrix: {corr_csv_path}")
    
    # Final summary
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print('='*70)
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"  • correlation_heatmap.png")
    print(f"  • correlation_distribution.png")
    print(f"  • correlation_matrix.csv")
    print(f"  • recommended_features.txt")
    print(f"\n✓ Ready for Isolation Forest model training!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
