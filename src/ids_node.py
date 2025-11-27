import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from joblib import load
import pandas as pd
import json
  

class ids_node(Node):
    # Initilize the node
    def __init__(self):
        super().__init__('ids_node')
    # Assign Paths
        self.csv_path = "/home/anguiz/Capstone/output/ML_data.csv"
        self.model_path = "/home/anguiz/Capstone/src/models/model.joblib"
        self.scaler_path = "/home/anguiz/Capstone/src/models/scaler.joblib"
        self.features_path = "/home/anguiz/Capstone/src/models/used_features.txt"
        self.output_log_path = "/home/anguiz/Capstone/output/ids_output.log" 
    # Load Pipeline
        self.get_logger().info('Loading ML Pipeline\n')
        try:
            self.model = load(self.model_path)
            self.scaler = load(self.scaler_path)
            with open(self.features_path) as f:
                self.used_features = [line.strip() for line in f.readlines()]
            self.get_logger().info(f'Loaded model, scaler, and {len(self.used_features)} features')
        except Exception as e:
            self.get_logger().error(f'Failed to load pipeline: {e}')
            raise
    # Publisher
        self.alert_pub = self.create_publisher(String, '/ids/alerts', 10)
    # Processing state
        self.seen_rows = 0
        
    # Create timer (2 Hz = check every 0.5 seconds)
        self.timer = self.create_timer(0.5, self.process_csv)
        
        self.get_logger().info('Generic IDS Node initialized successfully')
        self.get_logger().info(f'Monitoring: {self.csv_path}')
    
    def process_csv(self):
        """Process new rows in CSV file"""
        try:
        # Load CSV
            df = pd.read_csv(self.csv_path)
            
            if df.shape[0] > self.seen_rows:
        # Get only new rows
                new_df = df.iloc[self.seen_rows:].copy()
                
        # Ensure feature alignment
                for feat in self.used_features:
                    if feat not in new_df.columns:
                        new_df[feat] = 0
                
                new_df = new_df[self.used_features]
                new_df = new_df.apply(pd.to_numeric, errors="coerce").fillna(0)
                
                # Scale + predict
                X_scaled = self.scaler.transform(new_df)
                preds = self.model.predict(X_scaled)
                
                # Publish and log each prediction
                with open(self.output_log_path, "a") as log_file:
                    for p in preds:
                        is_attack = (p == 1)
                        alert_data = {
                            'timestamp': self.get_clock().now().seconds_nanoseconds()[0],
                            'prediction': int(p),
                            'is_attack': bool(is_attack),
                            'message': "ALERT: Attack detected!" if is_attack else "Normal traffic"
                        }
                        
                        msg_json = json.dumps(alert_data)
                        
                        if is_attack:
                            self.get_logger().warn(alert_data['message'])
                        else:
                            self.get_logger().info(alert_data['message'])
                        
                        msg = String()
                        msg.data = msg_json
                        self.alert_pub.publish(msg)
                        log_file.write(msg_json + "\n")
                
                self.seen_rows = df.shape[0]
                
        except FileNotFoundError:
            self.get_logger().debug(f'CSV file not found: {self.csv_path}')
        except Exception as e:
            self.get_logger().error(f'IDS error: {e}')


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = ids_node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()


