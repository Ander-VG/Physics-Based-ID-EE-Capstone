import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime

# Attack timing log - timestamps in seconds from simulation start
ATTACK_LOG = [
    {'type': 'odom_spoof', 'start': 20.23795, 'end': 24.23786},
    {'type': 'odom_spoof', 'start': 40.23792, 'end': 44.23786},
    {'type': 'odom_spoof', 'start': 63.23812, 'end': 67.23799},
    {'type': 'cpu_stress', 'start': 85.23796, 'end': 100.255},
    {'type': 'odom_spoof', 'start': 110.2379, 'end': 114.2379},
    {'type': 'odom_spoof', 'start': 140.2381, 'end': 144.238},
    {'type': 'cpu_stress', 'start': 165.2379, 'end': 180.2378},
    {'type': 'odom_spoof', 'start': 195.238, 'end': 199.2379},
    {'type': 'odom_spoof', 'start': 225.238, 'end': 229.238},
    {'type': 'odom_spoof', 'start': 255.2379, 'end': 259.236},
]

# Default paths
DEFAULT_INPUT = "/home/anguiz/Capstone/output/network_flows.csv"
DEFAULT_OUTPUT = "/home/anguiz/Capstone/output/labeled_flows.csv"


def get_timestamp_column(df: pd.DataFrame) -> str:
    """Find the timestamp column in the dataframe."""
    # CICFlowMeter common timestamp column names
    timestamp_candidates = [
        'Timestamp', 'timestamp', 'Flow Start', 'flow_start',
        'Start Time', 'start_time', 'Time', 'time'
    ]
    
    for col in timestamp_candidates:
        if col in df.columns:
            return col
    
    # Check for partial matches
    for col in df.columns:
        if 'time' in col.lower() or 'stamp' in col.lower():
            return col
    
    raise ValueError(f"No timestamp column found. Available columns: {list(df.columns)}")


def parse_timestamp(ts_value) -> float:
    """
    Parse timestamp to seconds from simulation start.
    
    CICFlowMeter can output timestamps in various formats:
    - Unix epoch (float/int)
    - ISO datetime string
    - Relative seconds
    """
    if pd.isna(ts_value):
        return np.nan
    
    # Already numeric (seconds or epoch)
    if isinstance(ts_value, (int, float, np.integer, np.floating)):
        return float(ts_value)
    
    # String timestamp
    ts_str = str(ts_value).strip()
    
    # Try parsing as datetime
    datetime_formats = [
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S',
        '%d/%m/%Y %H:%M:%S',
        '%m/%d/%Y %H:%M:%S',
        '%Y/%m/%d %H:%M:%S',
    ]
    
    for fmt in datetime_formats:
        try:
            dt = datetime.strptime(ts_str, fmt)
            return dt.timestamp()
        except ValueError:
            continue
    
    # Try direct float conversion
    try:
        return float(ts_str)
    except ValueError:
        pass
    
    raise ValueError(f"Cannot parse timestamp: {ts_value}")


def normalize_timestamps(timestamps: np.ndarray) -> np.ndarray:
    """
    Normalize timestamps to seconds from start.
    
    If timestamps are epoch values (large numbers), subtract the minimum
    to get relative time from simulation start.
    """
    min_ts = np.nanmin(timestamps)
    
    # If minimum is > 1e9, likely Unix epoch - normalize to relative time
    if min_ts > 1e9:
        print(f"  Detected Unix epoch timestamps (min={min_ts:.2f})")
        print(f"  Normalizing to relative time from start...")
        return timestamps - min_ts
    
    # If already small values, assume relative timestamps
    if min_ts < 1000:
        print(f"  Detected relative timestamps (min={min_ts:.2f})")
        return timestamps
    
    # Ambiguous - assume relative but warn
    print(f"  Warning: Ambiguous timestamp range (min={min_ts:.2f})")
    print(f"  Assuming relative timestamps...")
    return timestamps


def is_in_attack_window(timestamp: float, buffer: float = 0.5) -> tuple:
    """
    Check if a timestamp falls within any attack window.
    
    Args:
        timestamp: Time in seconds from simulation start
        buffer: Tolerance buffer in seconds around attack windows
    
    Returns:
        (is_attack: bool, attack_type: str or None)
    """
    if np.isnan(timestamp):
        return False, None
    
    for attack in ATTACK_LOG:
        start = attack['start'] - buffer
        end = attack['end'] + buffer
        
        if start <= timestamp <= end:
            return True, attack['type']
    
    return False, None


def label_flows(df: pd.DataFrame, timestamp_col: str, buffer: float = 0.5) -> pd.DataFrame:
    """
    Label each flow as benign (0) or attack (1) based on timestamp.
    
    Args:
        df: DataFrame with network flows
        timestamp_col: Name of timestamp column
        buffer: Tolerance buffer around attack windows
    
    Returns:
        DataFrame with Label and Attack_Type columns added
    """
    print(f"\nLabeling {len(df)} flows...")
    
    # Parse timestamps
    print("  Parsing timestamps...")
    timestamps = df[timestamp_col].apply(parse_timestamp).values
    
    # Normalize to relative time
    timestamps = normalize_timestamps(timestamps)
    
    # Store normalized timestamps for reference
    df['Timestamp_Normalized'] = timestamps
    
    # Label each flow
    print("  Applying attack windows...")
    labels = []
    attack_types = []
    
    for ts in timestamps:
        is_attack, attack_type = is_in_attack_window(ts, buffer)
        labels.append(1 if is_attack else 0)
        attack_types.append(attack_type if attack_type else 'benign')
    
    df['Label'] = labels
    df['Attack_Type'] = attack_types
    
    return df


def print_statistics(df: pd.DataFrame):
    """Print labeling statistics."""
    total = len(df)
    attack_count = df['Label'].sum()
    benign_count = total - attack_count
    
    print("\n" + "="*60)
    print("LABELING STATISTICS")
    print("="*60)
    print(f"Total flows:    {total:,}")
    print(f"Benign (0):     {benign_count:,} ({100*benign_count/total:.2f}%)")
    print(f"Attack (1):     {attack_count:,} ({100*attack_count/total:.2f}%)")
    print(f"Imbalance ratio: {benign_count/max(attack_count, 1):.1f}:1")
    
    print("\nAttack type breakdown:")
    attack_breakdown = df[df['Label'] == 1]['Attack_Type'].value_counts()
    for attack_type, count in attack_breakdown.items():
        print(f"  {attack_type}: {count:,}")
    
    print("\nTimestamp range:")
    print(f"  Min: {df['Timestamp_Normalized'].min():.2f}s")
    print(f"  Max: {df['Timestamp_Normalized'].max():.2f}s")
    print(f"  Duration: {df['Timestamp_Normalized'].max() - df['Timestamp_Normalized'].min():.2f}s")
    
    # Show attack windows coverage
    print("\nAttack windows defined:")
    for i, attack in enumerate(ATTACK_LOG):
        print(f"  {i+1}. {attack['type']}: {attack['start']:.2f}s - {attack['end']:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Label network flow data with attack timing windows"
    )
    parser.add_argument(
        '--input', '-i',
        default=DEFAULT_INPUT,
        help=f"Input CSV file (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        '--output', '-o',
        default=DEFAULT_OUTPUT,
        help=f"Output CSV file (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        '--buffer', '-b',
        type=float,
        default=0.5,
        help="Buffer time (seconds) around attack windows (default: 0.5)"
    )
    parser.add_argument(
        '--timestamp-col', '-t',
        default=None,
        help="Timestamp column name (auto-detected if not specified)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("NETWORK FLOW LABELING SCRIPT")
    print("="*60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Buffer: {args.buffer}s")
    
    # Load data
    print("\nLoading data...")
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {args.input}")
        print("Waiting for partner to deliver network_flows.csv...")
        return 1
    
    df = pd.read_csv(args.input)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Find timestamp column
    timestamp_col = args.timestamp_col
    if timestamp_col is None:
        timestamp_col = get_timestamp_column(df)
    print(f"  Using timestamp column: '{timestamp_col}'")
    
    # Label flows
    df = label_flows(df, timestamp_col, args.buffer)
    
    # Print statistics
    print_statistics(df)
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(args.output, index=False)
    print(f"\nSaved labeled data to: {args.output}")
    
    # Sanity check
    if df['Label'].sum() == 0:
        print("\n⚠️  WARNING: No attack labels assigned!")
        print("  Check that timestamp ranges overlap with ATTACK_LOG windows.")
        print("  You may need to adjust the ATTACK_LOG or timestamp parsing.")
    
    return 0


if __name__ == "__main__":
    exit(main())