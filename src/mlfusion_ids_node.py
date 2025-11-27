#!/usr/bin/env python3
"""
ML Fusion IDS Node for ROS 2 Humble
====================================
Architecture: PCA + KNN+RF Ensemble + Logistic Regression Fusion

This implements a hybrid machine learning approach:
1. StandardScaler: Normalize features
2. PCA: Reduce dimensionality (78 → 16)
3. KNN + Random Forest: Get probability predictions
4. Fusion: Concatenate [PCA features + KNN probs + RF probs]
5. Logistic Regression: Final classification

Features:
- Incremental processing (only new rows)
- File monitoring (checks for changes every 0.5 seconds)
- Continuous monitoring loop
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from joblib import load
import pandas as pd
import numpy as np
import json
import os
import sys
import time

# ============================================================================
# CONFIGURATION - YOUR PATHS
# ============================================================================

MODEL_BASE_PATH = "/home/anguiz/Capstone/src/models/ml_fusion"

SCALER_PATH = os.path.join(MODEL_BASE_PATH, "scaler.joblib")
PCA_PATH = os.path.join(MODEL_BASE_PATH, "pca.joblib")
KNN_PATH = os.path.join(MODEL_BASE_PATH, "knn_classifier.joblib")  
RF_PATH = os.path.join(MODEL_BASE_PATH, "rf_classifier.joblib")
LR_PATH = os.path.join(MODEL_BASE_PATH, "lr_classifier.joblib")
FEATURES_PATH = os.path.join(MODEL_BASE_PATH, "features.txt")
PIPELINE_INFO_PATH = os.path.join(MODEL_BASE_PATH, "pipeline_info.json")

CSV_FILE_PATH = "/home/anguiz/Capstone/ML_output/PCAP.csv"
OUTPUT_LOG_PATH = "/home/anguiz/Capstone/ML_output/ml_fusion_ids_output.log"


def load_complete_pipeline(logger):
    """Load the complete ML fusion pipeline"""
    try:
        logger.info("Loading ML fusion pipeline components \n")
        
        required_files = [SCALER_PATH, PCA_PATH, KNN_PATH, RF_PATH, LR_PATH, FEATURES_PATH, PIPELINE_INFO_PATH]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            raise FileNotFoundError(f"Missing files: {missing_files}")
        
        scaler = load(SCALER_PATH)
        pca = load(PCA_PATH) 
        knn_classifier = load(KNN_PATH)
        rf_classifier = load(RF_PATH)
        lr_classifier = load(LR_PATH)
        
        with open(FEATURES_PATH, 'r') as f:
            features = [line.strip() for line in f.readlines()]
        
        with open(PIPELINE_INFO_PATH, 'r') as f:
            pipeline_info = json.load(f)
            
        logger.info("All pipeline components loaded successfully")
        logger.info(f"   - Features: {len(features)}")
        logger.info(f"   - PCA components: {pca.n_components_}")
        logger.info("   - Architecture: Scale → PCA → KNN+RF → Fusion → LR")
        
        return scaler, pca, knn_classifier, rf_classifier, lr_classifier, features, pipeline_info
        
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        raise

def apply_fusion_pipeline(raw_data, scaler, pca, knn, rf, lr, logger):
    """Apply the complete ML fusion transformation pipeline"""
    try:
        # Step 1: Scale features
        if hasattr(raw_data, 'values'):
            X_scaled = scaler.transform(raw_data)
        else:
            X_scaled = scaler.transform(raw_data)
        logger.debug(f"After scaling: {X_scaled.shape}")
        
        # Step 2: Apply PCA (78 → 16)
        X_pca = pca.transform(X_scaled)
        logger.debug(f"After PCA: {X_pca.shape}")
        
        # Step 3: Get classifier probabilities on PCA features
        knn_proba = knn.predict_proba(X_pca)
        rf_proba = rf.predict_proba(X_pca)
        logger.debug(f"KNN probabilities: {knn_proba.shape}")
        logger.debug(f"RF probabilities: {rf_proba.shape}")
        
        # Step 4: Fuse features [PCA + KNN probs + RF probs]
        X_fused = np.concatenate([X_pca, knn_proba, rf_proba], axis=1)
        logger.debug(f"After fusion: {X_fused.shape}")
        
        # Step 5: Final prediction using Logistic Regression
        final_predictions = lr.predict(X_fused)
        logger.debug(f"Final predictions: {final_predictions.shape}")
        
        return final_predictions
        
    except Exception as e:
        logger.error(f"Pipeline transformation failed: {e}")
        raise

def prepare_features(df, expected_features, logger):
    """Prepare and align features from CSV data"""
    try:
        # Column name mapping (78 features)
        column_mapping = {
            'src_port': 'Src Port', 'dst_port': 'Dst Port', 
            'flow_duration': 'Flow Duration',
            'tot_fwd_pkts': 'Tot Fwd Pkts', 'tot_bwd_pkts': 'Tot Bwd Pkts',
            'totlen_fwd_pkts': 'TotLen Fwd Pkts', 'totlen_bwd_pkts': 'TotLen Bwd Pkts',
            'fwd_pkt_len_max': 'Fwd Pkt Len Max', 'fwd_pkt_len_min': 'Fwd Pkt Len Min',
            'fwd_pkt_len_mean': 'Fwd Pkt Len Mean', 'fwd_pkt_len_std': 'Fwd Pkt Len Std',
            'bwd_pkt_len_max': 'Bwd Pkt Len Max', 'bwd_pkt_len_min': 'Bwd Pkt Len Min',
            'bwd_pkt_len_mean': 'Bwd Pkt Len Mean', 'bwd_pkt_len_std': 'Bwd Pkt Len Std',
            'flow_byts_s': 'Flow Byts/s', 'flow_pkts_s': 'Flow Pkts/s',
            'flow_iat_mean': 'Flow IAT Mean', 'flow_iat_std': 'Flow IAT Std',
            'flow_iat_max': 'Flow IAT Max', 'flow_iat_min': 'Flow IAT Min',
            'fwd_iat_tot': 'Fwd IAT Tot', 'fwd_iat_mean': 'Fwd IAT Mean',
            'fwd_iat_std': 'Fwd IAT Std', 'fwd_iat_max': 'Fwd IAT Max', 'fwd_iat_min': 'Fwd IAT Min',
            'bwd_iat_tot': 'Bwd IAT Tot', 'bwd_iat_mean': 'Bwd IAT Mean', 
            'bwd_iat_std': 'Bwd IAT Std', 'bwd_iat_max': 'Bwd IAT Max', 'bwd_iat_min': 'Bwd IAT Min',
            'fwd_psh_flags': 'Fwd PSH Flags', 'bwd_psh_flags': 'Bwd PSH Flags',
            'fwd_urg_flags': 'Fwd URG Flags', 'bwd_urg_flags': 'Bwd URG Flags',
            'fwd_header_len': 'Fwd Header Len', 'bwd_header_len': 'Bwd Header Len',
            'fwd_pkts_s': 'Fwd Pkts/s', 'bwd_pkts_s': 'Bwd Pkts/s',
            'pkt_len_min': 'Pkt Len Min', 'pkt_len_max': 'Pkt Len Max',
            'pkt_len_mean': 'Pkt Len Mean', 'pkt_len_std': 'Pkt Len Std', 'pkt_len_var': 'Pkt Len Var',
            'fin_flag_cnt': 'FIN Flag Cnt', 'syn_flag_cnt': 'SYN Flag Cnt', 'rst_flag_cnt': 'RST Flag Cnt',
            'psh_flag_cnt': 'PSH Flag Cnt', 'ack_flag_cnt': 'ACK Flag Cnt', 'urg_flag_cnt': 'URG Flag Cnt',
            'cwe_flag_count': 'CWE Flag Count', 'ece_flag_cnt': 'ECE Flag Cnt',
            'down_up_ratio': 'Down/Up Ratio', 'pkt_size_avg': 'Pkt Size Avg',
            'fwd_seg_size_avg': 'Fwd Seg Size Avg', 'bwd_seg_size_avg': 'Bwd Seg Size Avg',
            'fwd_byts_b_avg': 'Fwd Byts/b Avg', 'fwd_pkts_b_avg': 'Fwd Pkts/b Avg',
            'fwd_blk_rate_avg': 'Fwd Blk Rate Avg', 'bwd_byts_b_avg': 'Bwd Byts/b Avg',
            'bwd_pkts_b_avg': 'Bwd Pkts/b Avg', 'bwd_blk_rate_avg': 'Bwd Blk Rate Avg',
            'subflow_fwd_pkts': 'Subflow Fwd Pkts', 'subflow_fwd_byts': 'Subflow Fwd Byts',
            'subflow_bwd_pkts': 'Subflow Bwd Pkts', 'subflow_bwd_byts': 'Subflow Bwd Byts',
            'init_fwd_win_byts': 'Init Fwd Win Byts', 'init_bwd_win_byts': 'Init Bwd Win Byts',
            'fwd_act_data_pkts': 'Fwd Act Data Pkts', 'fwd_seg_size_min': 'Fwd Seg Size Min',
            'active_mean': 'Active Mean', 'active_std': 'Active Std',
            'active_max': 'Active Max', 'active_min': 'Active Min',
            'idle_mean': 'Idle Mean', 'idle_std': 'Idle Std', 
            'idle_max': 'Idle Max', 'idle_min': 'Idle Min'
        }
        
        logger.info(f"Input CSV has {len(df.columns)} columns")
        
        df_aligned = pd.DataFrame()
        mapped_count = 0
        missing_count = 0
        
        for expected_feat in expected_features:
            found = False
            
            if expected_feat in df.columns:
                df_aligned[expected_feat] = df[expected_feat]
                found = True
                mapped_count += 1
            else:
                for csv_col, model_feat in column_mapping.items():
                    if model_feat == expected_feat and csv_col in df.columns:
                        df_aligned[expected_feat] = df[csv_col]
                        found = True
                        mapped_count += 1
                        break
            
            if not found:
                df_aligned[expected_feat] = 0.0
                missing_count += 1
                logger.warn(f" Missing feature '{expected_feat}' filled with 0")
        
        logger.info(f"Mapped {mapped_count} features, {missing_count} missing")
        
        df_aligned = df_aligned.apply(pd.to_numeric, errors='coerce')
        df_aligned = df_aligned.fillna(0)
        df_aligned = df_aligned.replace([np.inf, -np.inf], 0)
        
        return df_aligned
        
    except Exception as e:
        logger.error(f"Feature preparation failed: {e}")
        raise

def format_prediction_message(predictions, pipeline_info, clock):
    """Format prediction results into messages"""
    try:
        attack_classes_raw = pipeline_info.get('attack_classes', {
            "0": "Normal", "1": "Attack"
        })
        
        class_names = {}
        for key, value in attack_classes_raw.items():
            try:
                class_names[int(key)] = value
            except (ValueError, TypeError):
                pass
        
        messages = []
        for pred in predictions:
            pred_int = int(pred) if hasattr(pred, '__int__') else pred
            class_name = class_names.get(pred_int, f"Unknown({pred_int})")
            timestamp = clock.now().seconds_nanoseconds()[0] / 1e9
            
            if pred_int == 0:
                msg = f"[{timestamp:.2f}] Normal traffic detected"
            else:
                msg = f"[{timestamp:.2f}] ATTACK DETECTED: {class_name} (Class {pred_int})"
                
            messages.append(msg)
            
        return messages
        
    except Exception as e:
        return [f"Prediction: {pred}" for pred in predictions]



class MLFusionIDSNode(Node):
    """ML Fusion Intrusion Detection System ROS 2 Node"""
    
    def __init__(self):
        super().__init__('ml_fusion_ids_node')
        
        self.get_logger().info("🚀 Initializing ML Fusion IDS Node...")
        
        # Load pipeline
        try:
            self.scaler, self.pca, self.knn, self.rf, self.lr, self.features, self.pipeline_info = load_complete_pipeline(self.get_logger())
            self.get_logger().info("Pipeline loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize pipeline: {e}")
            raise
            
        # ROS Publisher
        self.alert_pub = self.create_publisher(String, '/ids/alerts', 10)
        
        # Processing parameters
        self.batch_size = 100
        self.log_file = OUTPUT_LOG_PATH
        self.last_processed_line = 0
        
        # Create timer (2 Hz = check every 0.5 seconds)
        self.timer = self.create_timer(0.5, self.process_new_data)
        
        self.get_logger().info("ML Fusion IDS Node initialized successfully")
        self.get_logger().info(f"Monitoring: {CSV_FILE_PATH}")

    def process_new_data(self):
        """Process only new lines added to CSV"""
        try:
            if not os.path.exists(CSV_FILE_PATH):
                return
                
            df = pd.read_csv(CSV_FILE_PATH)
            total_lines = len(df)
            
            if total_lines <= self.last_processed_line:
                return
                
            new_data = df.iloc[self.last_processed_line:]
            self.get_logger().info(f"🔄 Processing {len(new_data)} new rows")
            
            # Prepare features
            X_prepared = prepare_features(new_data, self.features, self.get_logger())
            
            # Apply pipeline
            predictions = apply_fusion_pipeline(
                X_prepared, self.scaler, self.pca, 
                self.knn, self.rf, self.lr, self.get_logger()
            )
            
            # Format and publish
            messages = format_prediction_message(predictions, self.pipeline_info, self.get_clock())
            
            for msg in messages:
                self.alert_pub.publish(String(data=msg))
                
                with open(self.log_file, 'a') as f:
                    f.write(msg + "\n")
                
                if "ATTACK" in msg:
                    self.get_logger().warn(msg)
                else:
                    self.get_logger().info(msg)
            
            self.last_processed_line = total_lines
            
        except FileNotFoundError:
            self.get_logger().debug(f'CSV file not found: {CSV_FILE_PATH}')
        except Exception as e:
            self.get_logger().error(f'Processing error: {e}')


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = MLFusionIDSNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Keyboard interrupt")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()