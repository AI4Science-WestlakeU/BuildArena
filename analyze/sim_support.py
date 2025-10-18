import pandas as pd
from typing import Dict, Any, Optional

from spatial.build import Blocks
from .sim_common import (
    parse_simulation_paths,
    parse_task_metadata,
    load_block_data,
    calculate_total_mass,
    extract_round_suffix,
    save_analysis_result,
    robust_read_csv, 
)

# Constants for support analysis
HEIGHT_THRESHOLD = 4.0 # Height threshold for ballast support
HOLD_THRESHOLD = 2.0    # Time threshold for holding above height
INITIAL_HEIGHT_THRESHOLD = 20.0 # Initial height threshold for ballast support

def analyze_simulation_support(machine_dir: str, sim_round: Optional[int] = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Analyze a support simulation machine and compute ballast support metrics.
    
    Args:
        machine_dir: Path to the machine simulation directory
        sim_round: Specific simulation round to analyze (1, 2, 3, etc.). If None, auto-detect.
        verbose: Whether to print intermediate results
        
    Returns:
        Dictionary containing analysis results including pass/fail status,
        mass, ballast weight metrics, and trajectory statistics
        
    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If data parsing fails
    """
    try:
        all_blocks = Blocks()
        
        # Parse paths and identifiers
        paths = parse_simulation_paths(machine_dir, sim_round)
        metadata = parse_task_metadata(machine_dir)
        
        if verbose:
            print(f"Task: {metadata['task_name']}")
            print(f"Category: {metadata['category']}, Level: {metadata['level']}")
            print(f"Model: {metadata['model_name']}")
            print(f"Machine ID: {metadata['machine_id']}")
        
        # Load and process block data
        block_data = load_block_data(paths['block_json_path'])
        total_mass = calculate_total_mass(block_data, all_blocks)
        
        if verbose:
            print(f"Block data: {block_data}")
            print(f"Total mass: {total_mass}")
        
        # Analyze ballast support metrics
        ballast_stats = _analyze_ballast_support(
            paths['simulation_csv_path'], verbose
        )
        
        # Determine pass/fail based on ballast hold time
        passed = ballast_stats['hold_time_above_threshold'] >= HOLD_THRESHOLD and not ballast_stats['initial_height_above_threshold']
        
        # Extract round suffix from file paths
        round_suffix = extract_round_suffix(paths)
        
        # Compile results
        result = {
            "pass": passed,
            "category": metadata['category'],
            "level": metadata['level'],
            "model_name": metadata['model_name'],
            "machine_id": metadata['machine_id'],
            "round_suffix": round_suffix,
            "machine_bsg_path": paths['machine_bsg_path'],
            "machine_json_path": paths['machine_json_path'],
            "num_blocks": len(block_data),
            "total_mass": total_mass,
            "ballast_max_weight": ballast_stats['max_weight'],
            "ballast_max_weight_at_threshold": ballast_stats['max_weight_at_threshold'] if passed else 0.0,
            "ballast_min_height": ballast_stats['min_height'],
            "hold_time_above_threshold": ballast_stats['hold_time_above_threshold'],
            "time_below_threshold": ballast_stats['time_below_threshold'],
            "hacked": ballast_stats['hacked'],
            "hyperparameters": {
                "HEIGHT_THRESHOLD": HEIGHT_THRESHOLD,
                "HOLD_THRESHOLD": HOLD_THRESHOLD,
                "INITIAL_HEIGHT_THRESHOLD": INITIAL_HEIGHT_THRESHOLD
            }
        }
        
        if verbose:
            print(f"Analysis result: {result}")
            
        return result
        
    except Exception as e:
        raise ValueError(f"Failed to analyze simulation: {str(e)}") from e


# Common functions moved to sim_common.py


def _analyze_ballast_support(simulation_csv_path: str, verbose: bool = False) -> Dict[str, float]:
    """
    Analyze ballast support metrics from simulation data.
    
    Args:
        simulation_csv_path: Path to the simulation CSV file
        verbose: Whether to print intermediate results
        
    Returns:
        Dictionary containing ballast support metrics
    """
    try:
        # Load raw simulation data using unified CSV reader
        raw_df = robust_read_csv(
            simulation_csv_path,
            min_columns=2,  # Need at least 2 columns for ballast analysis
            validate_numeric_columns=None,  # No numeric validation needed initially
            skip_to_first_valid=False  # Include all valid rows
        )
        
        # Filter for ID_Ballast entries
        ballast_rows = raw_df[raw_df.iloc[:, 1] == 'ID_Ballast'].copy()
        
        if verbose:
            print(f"Total CSV rows loaded: {len(raw_df)}")
            print(f"Unique IDs in CSV: {raw_df.iloc[:, 1].unique()}")
            print(f"ID_Ballast rows found: {len(ballast_rows)}")
            if not ballast_rows.empty:
                print(f"Sample ballast rows:\n{ballast_rows.head()}")
        
        if ballast_rows.empty:
            if verbose:
                print("Warning: No ID_Ballast entries found")
            return _get_default_ballast_stats()
        
        # Separate position data (5 columns) from weight data (3 columns)
        position_data = []
        weight_data = []
        
        for _, row in ballast_rows.iterrows():
            # Check if this is position data (5 columns: time, id, x, z, y)
            if len(row) >= 5 and pd.notna(row.iloc[4]):
                position_data.append({
                    'time': float(row.iloc[0]),
                    'id': row.iloc[1],
                    'x': float(row.iloc[2]),
                    'z': float(row.iloc[3]),
                    'y': float(row.iloc[4])
                })
            # Check if this is weight data (3 columns: time, id, weight)
            elif len(row) >= 3 and pd.notna(row.iloc[2]) and (len(row) < 4 or pd.isna(row.iloc[3])):
                weight_data.append({
                    'time': float(row.iloc[0]),
                    'id': row.iloc[1],
                    'weight': float(row.iloc[2])
                })
        
        position_df = pd.DataFrame(position_data)
        weight_df = pd.DataFrame(weight_data)
        
        if verbose:
            print(f"Ballast position entries: {len(position_df)}")
            print(f"Ballast weight entries: {len(weight_df)}")
            if not position_df.empty:
                print(f"Position data sample:\n{position_df.head()}")
                print(f"Position time range: {position_df['time'].min():.3f} - {position_df['time'].max():.3f}")
                print(f"Position z range: {position_df['z'].min():.3f} - {position_df['z'].max():.3f}")
            if not weight_df.empty:
                print(f"Weight data sample:\n{weight_df.head()}")
                print(f"Weight range: {weight_df['weight'].min():.3f} - {weight_df['weight'].max():.3f}")
        
        # Calculate ballast metrics
        metrics = _calculate_ballast_metrics(position_df, weight_df, verbose)
        
        if verbose:
            print(f"Ballast Support Metrics:")
            print(f"  Max Weight: {metrics['max_weight']:.3f}")
            print(f"  Max Weight at Threshold: {metrics['max_weight_at_threshold']:.3f} if {metrics['initial_height_above_threshold']} else 0.0")
            print(f"  Min Height: {metrics['min_height']:.3f}")
            print(f"  Hold Time Above Threshold: {metrics['hold_time_above_threshold']:.3f}")
            print(f"  Time Below Threshold: {metrics['time_below_threshold']:.3f}")
        
        return metrics
        
    except Exception as e:
        if verbose:
            print(f"Warning: Failed to analyze ballast support: {e}")
        return _get_default_ballast_stats()


def _calculate_ballast_metrics(position_df: pd.DataFrame, weight_df: pd.DataFrame, 
                              verbose: bool = False) -> Dict[str, float]:
    """Calculate ballast support metrics."""
    if position_df.empty:
        return _get_default_ballast_stats()
    
    # Sort by time for proper analysis
    position_df = position_df.sort_values('time').reset_index(drop=True)
    weight_df = weight_df.sort_values('time').reset_index(drop=True)
    
    # Initialize hacked flag
    hacked = False
    
    # Determine if the initial height of ballast is above threshold
    initial_height = float(position_df['z'].iloc[0])
    initial_height_above_threshold = initial_height > INITIAL_HEIGHT_THRESHOLD
    
    # Basic height and weight metrics
    min_height = float(position_df['z'].min())
    max_weight = float(weight_df['weight'].max()) if not weight_df.empty else 0.0
    
    # Find maximum weight at the time point when ballast goes below height threshold
    max_weight_at_threshold = 0.0
    
    # Merge position and weight data by time for analysis
    if not weight_df.empty:
        # Find times when ballast goes below threshold
        below_threshold_times = position_df[position_df['z'] < HEIGHT_THRESHOLD]['time'].values
        
        if len(below_threshold_times) > 0:
            # Find the first time it goes below threshold
            first_below_time = below_threshold_times[0]
            
            # Find the weight at or just before this time
            weight_before = weight_df[weight_df['time'] <= first_below_time]
            if not weight_before.empty:
                max_weight_at_threshold = float(weight_before.iloc[-1]['weight'])
        else:
            # Hold above threshold all the time
            max_weight_at_threshold = max_weight
            # Check if this might be a hacked case (stays very high)
            if min_height > 5.0:
                hacked = True
    
    # Calculate hold time above threshold
    above_threshold_mask = position_df['z'] >= HEIGHT_THRESHOLD
    hold_time_above_threshold = 0.0
    time_below_threshold = 0.0
    
    if len(position_df) > 1:
        # Calculate time intervals
        time_diffs = position_df['time'].diff().iloc[1:]
        above_threshold_diffs = time_diffs[above_threshold_mask.iloc[1:]]
        below_threshold_diffs = time_diffs[~above_threshold_mask.iloc[1:]]
        
        hold_time_above_threshold = float(above_threshold_diffs.sum())
        time_below_threshold = float(below_threshold_diffs.sum())
    
    return {
        'max_weight': max_weight,
        'max_weight_at_threshold': max_weight_at_threshold,
        'min_height': min_height,
        'hold_time_above_threshold': hold_time_above_threshold,
        'time_below_threshold': time_below_threshold,
        'hacked': hacked,
        'initial_height_above_threshold': initial_height_above_threshold
    }


def _get_default_ballast_stats() -> Dict[str, float]:
    """Return default ballast statistics for edge cases."""
    return {
        'max_weight': 0.0,
        'max_weight_at_threshold': 0.0,
        'min_height': 0.0,
        'hold_time_above_threshold': 0.0,
        'time_below_threshold': 0.0,
        'hacked': False,
        'initial_height_above_threshold': False
    }

def main(machine_dir: str, sim_round: int | None = None, verbose: bool = False):
    """Main function for standalone execution."""
    result = analyze_simulation_support(machine_dir, sim_round=sim_round, verbose=verbose)
    
    # Save result using the reusable function
    save_path = save_analysis_result(result)
    print(f"Result saved to {save_path}")
    
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine_dir", type=str, required=True)
    parser.add_argument("--sim_round", type=int, default=None, required=False)
    parser.add_argument("--verbose", action="store_true", required=False, default=False)
    args = parser.parse_args()
    main(args.machine_dir, args.sim_round, args.verbose)
