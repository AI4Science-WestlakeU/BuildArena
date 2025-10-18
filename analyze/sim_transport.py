import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple

from spatial.build import Blocks
from .sim_common import (
    parse_simulation_paths,
    parse_task_metadata,
    load_block_data,
    calculate_total_mass,
    clean_simulation_data,
    analyze_motion_metrics,
    analyze_target_trajectory_quality,
    extract_round_suffix,
    save_analysis_result
)

# Constants for transport analysis
DISTANCE_THRESHOLD = 10.0  # Distance threshold for successful transport
START_POSITION = np.array([0.0, 0.0, 0.0])
TARGET_POSITION = np.array([10.0, 10.0, 0.0])
START_TIME_SPEED = 2.0
START_TIME_TRAJECTORY = 2.0
WHEEL_DISTANCE_THRESHOLD = 1.0  # Threshold for wheel distance consistency
INITIAL_HEIGHT_THRESHOLD = 20.0 # Initial height threshold to filter out hacked cases

def analyze_simulation_transport(machine_dir: str, sim_round: Optional[int] = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Analyze a transport simulation machine and compute transport metrics.
    
    Args:
        machine_dir: Path to the machine simulation directory
        sim_round: Specific simulation round to analyze (1, 2, 3, etc.). If None, auto-detect.
        verbose: Whether to print intermediate results
        
    Returns:
        Dictionary containing analysis results including pass/fail status,
        mass, transport distance, wheel consistency, and trajectory statistics
        
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
        
        # Clean simulation data (only keep 5-column position data)
        position_df = clean_simulation_data(paths['simulation_csv_path'])
        
        # Determine tracking target (Ballast if exists, otherwise ID_1)
        target_id, target_stats = _determine_tracking_target(position_df, verbose)
        
        # Analyze wheel consistency for "one_piece" detection
        wheel_stats = _analyze_wheel_consistency(
            position_df, block_data, verbose
        )
        
        # Analyze motion metrics
        motion_stats = analyze_motion_metrics(
            paths['simulation_csv_path'], START_TIME_SPEED, verbose
        )
        
        # Analyze trajectory quality for the tracking target
        trajectory_stats = analyze_target_trajectory_quality(
            paths['simulation_csv_path'], target_id, START_POSITION, TARGET_POSITION, 
            START_TIME_TRAJECTORY, verbose
        )
        
        # Calculate transport distance (maximum distance from start position)
        max_transport_distance = target_stats['max_distance_from_start']
        
        # Determine pass/fail based on transport distance threshold
        passed = max_transport_distance > DISTANCE_THRESHOLD
        if metadata['level'] == "soft":
            passed = passed and wheel_stats['wheel_count'] >= 4
        
        initial_max_height = target_stats['initial_max_height']
        passed = passed and initial_max_height < INITIAL_HEIGHT_THRESHOLD
                    
        if passed and wheel_stats['wheel_count'] == 0:
            hacked = True
        else:
            hacked = False
            
        if initial_max_height > INITIAL_HEIGHT_THRESHOLD:
            hacked = True
        
        # Extract round suffix from file paths
        round_suffix = extract_round_suffix(paths)
        
        # Compile results
        result = {
            "pass": passed,
            "hacked": hacked,
            "category": metadata['category'],
            "level": metadata['level'],
            "model_name": metadata['model_name'],
            "machine_id": metadata['machine_id'],
            "round_suffix": round_suffix,
            "machine_bsg_path": paths['machine_bsg_path'],
            "machine_json_path": paths['machine_json_path'],
            "num_blocks": len(block_data),
            "total_mass": total_mass,
            "tracking_target": target_id,
            "max_transport_distance": max_transport_distance,
            "final_position": target_stats['final_position'],
            "initial_max_height": target_stats['initial_max_height'],
            "one_piece": wheel_stats['one_piece'],
            "wheel_count": wheel_stats['wheel_count'],
            "max_wheel_distance_variation": wheel_stats['max_distance_variation'],
            "max_height": motion_stats['max_height'],
            "max_speed": motion_stats['max_speed'],
            "max_progress_ratio": trajectory_stats['max_progress_ratio'],
            "direction_consistency": trajectory_stats['direction_consistency'],
            "backtrack_penalty": trajectory_stats['backtrack_penalty'],
            "lateral_deviation": trajectory_stats['lateral_deviation'],
            "path_efficiency": trajectory_stats['path_efficiency'],
            "final_distance_ratio": trajectory_stats['final_distance_ratio'],
            "hyperparameters": {
                "DISTANCE_THRESHOLD": DISTANCE_THRESHOLD,
                "START_TIME_SPEED": START_TIME_SPEED,
                "START_TIME_TRAJECTORY": START_TIME_TRAJECTORY,
                "START_POSITION": START_POSITION.tolist(),
                "TARGET_POSITION": TARGET_POSITION.tolist(),
                "WHEEL_DISTANCE_THRESHOLD": WHEEL_DISTANCE_THRESHOLD,
                "INITIAL_HEIGHT_THRESHOLD": INITIAL_HEIGHT_THRESHOLD
            }
        }
        
        if verbose:
            print(f"Analysis result: {result}")
            
        return result
        
    except Exception as e:
        raise ValueError(f"Failed to analyze simulation: {str(e)}") from e


def _determine_tracking_target(position_df: pd.DataFrame, verbose: bool = False) -> Tuple[str, Dict[str, Any]]:
    """
    Determine the tracking target (Ballast if exists, otherwise ID_1) and calculate its transport stats.
    
    Args:
        position_df: Cleaned position data DataFrame
        verbose: Whether to print intermediate results
        
    Returns:
        Tuple of (target_id, target_stats)
    """
    # Check if ID_Ballast exists in the data
    unique_ids = position_df['id'].unique()
    has_ballast = 'ID_Ballast' in unique_ids
    
    target_id = 'ID_Ballast' if has_ballast else 'ID_1'
    
    if verbose:
        print(f"Available IDs: {sorted(unique_ids)}")
        print(f"Tracking target: {target_id}")
    
    # Filter data for the target
    target_df = position_df[position_df['id'] == target_id].copy()
    
    if target_df.empty:
        if verbose:
            print(f"Warning: No data found for target {target_id}")
        return target_id, {
            'max_distance_from_start': 0.0,
            'final_position': [0.0, 0.0, 0.0],
            'initial_max_height': 0.0
        }
    
    # Sort by time
    target_df = target_df.sort_values('time').reset_index(drop=True)
    
    # Calculate distances from start position
    positions = target_df[['x', 'y']].values
    distances_from_start = np.linalg.norm(positions - START_POSITION[:2], axis=1)
    
    max_distance_from_start = float(np.max(distances_from_start))
    final_position = positions[-1].tolist()
    
    # Identify the maximum height of target at the beginning 5 seconds
    initial_max_height = float(target_df['z'][:int(START_TIME_SPEED)].max())
    
    if verbose:
        print(f"Target {target_id} max transport distance: {max_distance_from_start:.3f}")
        print(f"Target {target_id} final position: {final_position}")
    
    return target_id, {
        'max_distance_from_start': max_distance_from_start,
        'final_position': final_position,
        'initial_max_height': initial_max_height
    }


def _analyze_wheel_consistency(position_df: pd.DataFrame, block_data: Dict[str, str], 
                              verbose: bool = False) -> Dict[str, Any]:
    """
    Analyze wheel distance consistency to determine if the machine stays in one piece.
    
    Args:
        position_df: Cleaned position data DataFrame
        block_data: Block ID to name mapping
        verbose: Whether to print intermediate results
        
    Returns:
        Dictionary containing wheel consistency metrics
    """
    # Find wheel IDs
    wheel_ids = [f"ID_{block_id}" for block_id, block_name in block_data.items() 
                 if "Wheel" in block_name]
    
    if verbose:
        print(f"Wheel IDs found: {wheel_ids}")
    
    if len(wheel_ids) < 2:
        # If there are fewer than 2 wheels, consider it as one piece
        return {
            'one_piece': True,
            'wheel_count': len(wheel_ids),
            'max_distance_variation': 0.0
        }
    
    # Filter position data for wheels only
    wheel_df = position_df[position_df['id'].isin(wheel_ids)].copy()
    
    if wheel_df.empty or len(wheel_df['id'].unique()) < 2:
        return {
            'one_piece': True,
            'wheel_count': len(wheel_ids),
            'max_distance_variation': 0.0
        }
    
    # Calculate pairwise distances between wheels at each time step
    max_distance_variation = 0.0
    
    # Group by time to analyze wheel positions at each timestep
    for time_val, time_group in wheel_df.groupby('time'):
        if len(time_group) < 2:
            continue
            
        # Get positions of all wheels at this timestep
        wheel_positions = {}
        for _, row in time_group.iterrows():
            wheel_positions[row['id']] = np.array([row['x'], row['y'], row['z']])
        
        # Calculate all pairwise distances
        wheel_list = list(wheel_positions.keys())
        distances = []
        for i in range(len(wheel_list)):
            for j in range(i + 1, len(wheel_list)):
                dist = np.linalg.norm(wheel_positions[wheel_list[i]] - wheel_positions[wheel_list[j]])
                distances.append(dist)
        
        if distances:
            # Calculate variation in distances (std deviation)
            distance_variation = float(np.std(distances))
            max_distance_variation = max(max_distance_variation, distance_variation)
    
    # Determine if machine is one piece based on distance variation threshold
    one_piece = max_distance_variation <= WHEEL_DISTANCE_THRESHOLD
    
    if verbose:
        print(f"Wheel consistency analysis:")
        print(f"  Wheel count: {len(wheel_ids)}")
        print(f"  Max distance variation: {max_distance_variation:.3f}")
        print(f"  One piece: {one_piece}")
    
    return {
        'one_piece': one_piece,
        'wheel_count': len(wheel_ids),
        'max_distance_variation': max_distance_variation
    }


def main(machine_dir: str, sim_round: int, verbose: bool):
    """Main function for standalone execution."""
    result = analyze_simulation_transport(machine_dir, sim_round=sim_round, verbose=verbose)
    
    # Save result using the reusable function
    save_path = save_analysis_result(result)
    print(f"Result saved to {save_path}")

    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine_dir", type=str, required=True)
    parser.add_argument("--sim_round", type=int, default=1, required=False)
    parser.add_argument("--verbose", action="store_true", required=False, default=False)
    args = parser.parse_args()
    main(args.machine_dir, args.sim_round, args.verbose)
