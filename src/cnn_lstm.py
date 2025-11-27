#!/usr/bin/env python3
"""
CNN+LSTM Deep Learning IDS Node for ROS 2 Humble
=================================================
Real-time Intrusion Detection using CNN+LSTM neural network with KNN+RF fusion

Architecture: CNN+LSTM (TensorFlow) → PCA → KNN+RF → Logistic Regression Fusion

This is the ACTUAL deep learning model using TensorFlow/Keras.
Unlike ml_fusion_ids_node.py which uses traditional ML, this uses a
trained neural network (feature_extractor.keras) for feature extraction.

Pipeline:
1. StandardScaler: Normalize features
2. CNN+LSTM: Deep learning feature extraction (TensorFlow)
3. PCA: Reduce CNN-LSTM output dimensions
4. KNN + Random Forest: Get probability predictions
5. Fusion: Concatenate [KNN probs + RF probs]
6. Logistic Regression: Final classification
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sys
import os
import json
import time
from datetime import datetime

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Scientific computing
import numpy as np
import pandas as pd

# Machine Learning
try:
    from joblib import load
    print("Joblib imported successfully")
except ImportError as e:
    print(f"Joblib import failed: {e}")
    print("Install with: pip3 install joblib")
    sys.exit(1)

try:
    import sklearn
    print(f"scikit-learn imported successfully (version: {sklearn.__version__})")
except ImportError as e:
    print(f"scikit-learn import failed: {e}")
    print("Install with: pip3 install scikit-learn")
    sys.exit(1)

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    print(f"TensorFlow imported successfully (version: {tf.__version__})")
except ImportError as e:
    print(f"TensorFlow import failed: {e}")
    print("Install with: pip3 install tensorflow")
    sys.exit(1)

# ============================================================================
# CONFIGURATION - YOUR PATHS
# ============================================================================

MODEL_BASE_PATH = "/home/anguiz/Capstone/src/models/cnn_lstm_dl"

SCALER_PATH = os.path.join(MODEL_BASE_PATH, "scaler.joblib")
FEATURE_EXTRACTOR_PATH = os.path.join(MODEL_BASE_PATH, "feature_extractor.keras")
PCA_PATH = os.path.join(MODEL_BASE_PATH, "pca.joblib")
KNN_PATH = os.path.join(MODEL_BASE_PATH, "knn_classifier.joblib")
RF_PATH = os.path.join(MODEL_BASE_PATH, "rf_classifier.joblib")
LR_PATH = os.path.join(MODEL_BASE_PATH, "lr_classifier.joblib")
FEATURES_PATH = os.path.join(MODEL_BASE_PATH, "features.txt")
CLASS_MAPPING_PATH = os.path.join(MODEL_BASE_PATH, "class_mapping.json")
PIPELINE_INFO_PATH = os.path.join(MODEL_BASE_PATH, "pipeline_info.json")

CSV_FILE_PATH = "/home/anguiz/Capstone/ML_output/PCAP.csv"
OUTPUT_LOG_PATH = "/home/anguiz/Capstone/ML_output/CNN_LSTM_prediction.log"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_cnn_lstm_pipeline(logger):
    """Load the complete CNN+LSTM deep learning pipeline"""
    try:
        logger.info("Loading CNN+LSTM deep learning pipeline...")
        
        # Check if all files exist
        required_files = [
            SCALER_PATH, FEATURE_EXTRACTOR_PATH, PCA_PATH, 
            KNN_PATH, RF_PATH, LR_PATH, 
            FEATURES_PATH, CLASS_MAPPING_PATH, PIPELINE_INFO_PATH
        ]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            raise FileNotFoundError(f"Missing files: {missing_files}")
        
        # Load scaler
        scaler = load(SCALER_PATH)
        logger.info("Loaded scaler.joblib")
        
        # Load feature extractor (CNN+LSTM) - TensorFlow model
        feature_extractor = load_model(FEATURE_EXTRACTOR_PATH, compile=False)
        feature_extractor.compile(optimizer='adam', loss='categorical_crossentropy')
        logger.info("Loaded feature_extractor.keras (TensorFlow CNN+LSTM)")
        
        # Load PCA
        pca = load(PCA_PATH)
        logger.info("Loaded pca.joblib")
        
        # Load classifiers
        knn_classifier = load(KNN_PATH)
        logger.info("Loaded knn_classifier.joblib")
        
        rf_classifier = load(RF_PATH)
        logger.info("Loaded rf_classifier.joblib")
        
        lr_classifier = load(LR_PATH)
        logger.info("Loaded lr_classifier.joblib")
        
        # Load feature names
        with open(FEATURES_PATH, 'r') as f:
            features = [line.strip() for line in f.readlines()]
        logger.info(f"Loaded {len(features)} feature names")
        
        # Load class mapping
        with open(CLASS_MAPPING_PATH, 'r') as f:
            class_mapping = json.load(f)
        
        # Convert to list indexed by class ID
        class_names = [''] * len(class_mapping)
        for name, idx in class_mapping.items():
            class_names[idx] = name
        logger.info(f"Loaded class mappings: {len(class_names)} classes")
        
        # Load pipeline info
        with open(PIPELINE_INFO_PATH, 'r') as f:
            pipeline_info = json.load(f)
        logger.info("Loaded pipeline_info.json")
        
        logger.info("Architecture: CNN+LSTM (TensorFlow) → PCA → KNN+RF → LR")
        
        return (scaler, feature_extractor, pca, knn_classifier, 
                rf_classifier, lr_classifier, features, class_names, pipeline_info)
        
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        raise

def apply_cnn_lstm_pipeline(raw_data, scaler, feature_extractor, pca, knn, rf, lr, logger):
    """Apply the complete CNN+LSTM deep learning pipeline"""
    try:
        # Step 1: Scale features
        X_array = raw_data.values if hasattr(raw_data, 'values') else raw_data
        X_scaled = scaler.transform(X_array)
        logger.debug(f"After scaling: {X_scaled.shape}")
        
        # Step 2: Reshape for CNN+LSTM (add time dimension for LSTM)
        n_samples, n_features = X_scaled.shape
        timesteps = 1  # Single timestep for real-time data
        X_reshaped = X_scaled.reshape(n_samples, timesteps, n_features)
        logger.debug(f"After reshaping for LSTM: {X_reshaped.shape}")
        
        # Step 3: Extract deep features using CNN+LSTM (TensorFlow)
        extracted_features = feature_extractor.predict(X_reshaped, verbose=0)
        logger.debug(f"After CNN+LSTM extraction: {extracted_features.shape}")
        
        # Step 4: Apply PCA for dimensionality reduction
        pca_features = pca.transform(extracted_features)
        logger.debug(f"After PCA: {pca_features.shape}")
        
        # Step 5: Get probability predictions from KNN and RF
        knn_proba = knn.predict_proba(pca_features)
        rf_proba = rf.predict_proba(pca_features)
        logger.debug(f"KNN probabilities: {knn_proba.shape}")
        logger.debug(f"RF probabilities: {rf_proba.shape}")
        
        # Step 6: Fuse KNN and RF probabilities
        fused_features = np.hstack([knn_proba, rf_proba])
        logger.debug(f"After fusion: {fused_features.shape}")
        
        # Step 7: Final prediction using Logistic Regression
        final_predictions = lr.predict(fused_features)
        final_probabilities = lr.predict_proba(fused_features)
        logger.debug(f"Final predictions: {final_predictions.shape}")
        
        return final_predictions, final_probabilities
        
    except Exception as e:
        logger.error(f"Pipeline transformation failed: {e}")
        raise

def prepare_features(df, expected_features, logger):
    """Prepare and align features from CSV data"""
    try:
        # Column name mapping (same 78 features as ml_fusion)
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
                logger.warn(f"Missing feature '{expected_feat}' filled with 0")
        
        logger.info(f"Mapped {mapped_count} features, {missing_count} missing")
        
        df_aligned = df_aligned.apply(pd.to_numeric, errors='coerce')
        df_aligned = df_aligned.fillna(0)
        df_aligned = df_aligned.replace([np.inf, -np.inf], 0)
        
        return df_aligned
        
    except Exception as e:
        logger.error(f"Feature preparation failed: {e}")
        raise

def format_prediction_message(predictions, probabilities, class_names, clock):
    """Format prediction results into messages"""
    try:
        messages = []
        for pred, proba in zip(predictions, probabilities):
            pred_int = int(pred)
            attack_class = class_names[pred_int] if pred_int < len(class_names) else f"Unknown({pred_int})"
            confidence = np.max(proba)
            timestamp = clock.now().seconds_nanoseconds()[0] / 1e9
            
            alert_data = {
                'timestamp': timestamp,
                'prediction': pred_int,
                'attack_class': attack_class,
                'confidence': float(confidence),
                'is_attack': bool(pred_int != 0),
                'probabilities': [float(p) for p in proba]
            }
            
            msg_json = json.dumps(alert_data)
            messages.append((msg_json, pred_int != 0))
            
        return messages
        
    except Exception as e:
        return [(f"Prediction: {pred}", False) for pred in predictions]

# ============================================================================
# ROS 2 NODE CLASS
# ============================================================================

class CNNLSTMDeepLearningIDSNode(Node):
    """CNN+LSTM Deep Learning Intrusion Detection System ROS 2 Node"""
    
    def __init__(self):
        super().__init__('cnn_lstm_dl_ids_node')
        
        print("\n" + "="*60)
        print("CNN+LSTM DEEP LEARNING IDS")
        print("="*60)
        print(f"TensorFlow: {tf.__version__}")
        print(f"🐍 Python: {sys.version.split()[0]}")
        print("="*60)
        
        self.get_logger().info("🚀 Initializing CNN+LSTM Deep Learning IDS Node...")
        
        # Load pipeline
        try:
            (self.scaler, self.feature_extractor, self.pca, 
             self.knn, self.rf, self.lr, self.features, 
             self.class_names, self.pipeline_info) = load_cnn_lstm_pipeline(self.get_logger())
            self.get_logger().info("Deep learning pipeline loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize pipeline: {e}")
            raise
            
        # ROS Publisher
        self.alert_pub = self.create_publisher(String, '/ids/alerts', 10)
        
        # Processing parameters
        self.batch_size = 100
        self.log_file = OUTPUT_LOG_PATH
        self.last_processed_line = 0
        
        # Statistics
        self.total_predictions = 0
        self.attack_detections = 0
        self.start_time = time.time()
        
        # Create timer (2 Hz = check every 0.5 seconds)
        self.timer = self.create_timer(0.5, self.process_new_data)
        
        self.get_logger().info("CNN+LSTM Deep Learning IDS Node initialized")
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
            self.get_logger().info(f"Processing {len(new_data)} new rows")
            
            # Prepare features
            X_prepared = prepare_features(new_data, self.features, self.get_logger())
            
            # Apply CNN+LSTM deep learning pipeline
            predictions, probabilities = apply_cnn_lstm_pipeline(
                X_prepared, self.scaler, self.feature_extractor, self.pca,
                self.knn, self.rf, self.lr, self.get_logger()
            )
            
            # Format and publish
            messages = format_prediction_message(
                predictions, probabilities, self.class_names, self.get_clock()
            )
            
            for msg_json, is_attack in messages:
                self.total_predictions += 1
                
                if is_attack:
                    self.attack_detections += 1
                    self.get_logger().warn(f"ATTACK: {msg_json}")
                else:
                    self.get_logger().info(f"Normal: {msg_json}")
                
                # Publish
                self.alert_pub.publish(String(data=msg_json))
                
                # Log to file
                with open(self.log_file, 'a') as f:
                    f.write(msg_json + "\n")
            
            self.last_processed_line = total_lines
            
            # Print statistics periodically
            if self.total_predictions % 100 == 0:
                uptime = time.time() - self.start_time
                attack_rate = (self.attack_detections / self.total_predictions * 100) if self.total_predictions > 0 else 0
                self.get_logger().info(f"Stats: {self.total_predictions} predictions, {self.attack_detections} attacks ({attack_rate:.2f}%), uptime: {uptime:.1f}s")
            
        except FileNotFoundError:
            self.get_logger().debug(f'CSV file not found: {CSV_FILE_PATH}')
        except Exception as e:
            self.get_logger().error(f'Processing error: {e}')


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = CNNLSTMDeepLearningIDSNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Keyboard interrupt")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()