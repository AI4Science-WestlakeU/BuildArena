"""
Common functions shared between simulation analysis modules.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import re
import shutil

from spatial.build import Blocks


def parse_simulation_paths(machine_dir: str, sim_round: Optional[int] = None) -> Dict[str, str]:
    """
    Parse file paths for simulation data, handling various naming patterns.
    
    Args:
        machine_dir: Path to the machine simulation directory
        sim_round: Specific simulation round to use (1, 2, 3, etc.). If None, auto-detect.
        
    Returns:
        Dictionary containing paths to simulation files
        
    The function handles these patterns:
    - Standard: {machine_id}_sim.bsg, {machine_id}_sim.json, etc.
    - Numbered: {machine_id}_sim_1.bsg, {machine_id}_sim_2.bsg, etc.
    """
    machine_id = os.path.basename(machine_dir)
    
    def _build_paths(suffix: str) -> Dict[str, str]:
        """Build paths with given suffix."""
        return {
            'machine_bsg_path': os.path.join(machine_dir, f"{machine_id}_sim{suffix}.bsg"),
            'machine_json_path': os.path.join(machine_dir, f"{machine_id}_sim{suffix}.json"),
            'block_json_path': os.path.join(machine_dir, f"block_id_{machine_id}_sim{suffix}.json"),
            'simulation_csv_path': os.path.join(machine_dir, f"simulation_log_{machine_id}_sim{suffix}.csv"),
        }
    
    # If specific round is requested, use it directly
    if sim_round is not None:
        return _build_paths(f"_{sim_round}")
    
    # Try standard pattern first (no round number)
    standard_paths = _build_paths("")
    if all(os.path.exists(path) for path in standard_paths.values()):
        return standard_paths
    
    # Try to find any numbered simulation files (1, 2, 3, etc.)
    for round_num in range(1, 10):  # Check rounds 1-9
        numbered_paths = _build_paths(f"_{round_num}")
        if all(os.path.exists(path) for path in numbered_paths.values()):
            return numbered_paths
    
    # Return standard paths as fallback (will cause FileNotFoundError if files don't exist)
    return standard_paths


def parse_task_metadata(machine_dir: str) -> Dict[str, str]:
    """Parse task metadata from directory structure."""
    task_name = os.path.basename(os.path.dirname(os.path.dirname(machine_dir)))
    task_parts = task_name.split("_")
    
    return {
        'task_name': task_name,
        'category': task_parts[0],
        'level': task_parts[1],
        'model_name': "_".join(task_parts[2:-3]),
        'machine_id': os.path.basename(machine_dir),
    }


def load_block_data(block_json_path: str) -> Dict[str, str]:
    """Load and process block ID data."""
    try:
        with open(block_json_path, "r", encoding='utf-8') as f:
            block_id_json = json.load(f)
        return {k: v for d in block_id_json for k, v in d.items() if v != "Ballast"}
    except FileNotFoundError:
        raise FileNotFoundError(f"Block data file not found: {block_json_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in block data file: {e}")


def calculate_total_mass(block_data: Dict[str, str], all_blocks: Blocks) -> float:
    """Calculate total mass of all blocks."""
    try:
        return sum(all_blocks.blocks[block_name]["weight"] 
                  for block_name in block_data.values() if block_name != "Ballast")
    except KeyError as e:
        raise ValueError(f"Unknown block type in mass calculation: {e}")


def robust_read_csv(simulation_csv_path: str, 
                   min_columns: int = 5, 
                   validate_numeric_columns: Optional[list] = None,
                   skip_to_first_valid: bool = True) -> pd.DataFrame:
    """
    Robustly read CSV files with error handling for malformed data.
    
    This unified function handles all CSV reading needs across the analysis modules:
    1. Tries normal pandas reading first for performance
    2. Falls back to line-by-line parsing for malformed files
    3. Flexible column requirements and validation
    
    Args:
        simulation_csv_path: Path to the CSV file
        min_columns: Minimum number of columns required per row
        validate_numeric_columns: List of column indices (0-based) that must be numeric, 
                                 or None to skip numeric validation
        skip_to_first_valid: If True, skips rows until finding first valid row,
                           then continues filtering. If False, includes all valid rows.
        
    Returns:
        DataFrame with valid rows
    """
    try:
        # First try normal pandas reading for well-formed files
        df = pd.read_csv(simulation_csv_path, header=None)
        
        # If we got here, the file parsed successfully with pandas
        # Still apply filtering based on requirements
        if min_columns > 0:
            df = df.dropna(subset=df.columns[:min_columns])
            
        if validate_numeric_columns:
            for col_idx in validate_numeric_columns:
                if col_idx < len(df.columns):
                    # Keep only rows where this column can be converted to numeric
                    df = df[pd.to_numeric(df.iloc[:, col_idx], errors='coerce').notna()]
                    
        return df
        
    except pd.errors.ParserError:
        # Fallback: manual line-by-line parsing for malformed files
        rows = []
        found_valid_start = False
        
        with open(simulation_csv_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                parts = line.split(',')
                
                # Check minimum column requirement
                if len(parts) < min_columns:
                    continue
                
                # Validate numeric columns if specified
                if validate_numeric_columns:
                    valid_row = True
                    for col_idx in validate_numeric_columns:
                        if col_idx < len(parts):
                            try:
                                float(parts[col_idx])
                            except (ValueError, IndexError):
                                valid_row = False
                                break
                    if not valid_row:
                        continue
                
                # Handle skip_to_first_valid logic
                if skip_to_first_valid and not found_valid_start:
                    found_valid_start = True
                    rows.append(parts)
                elif not skip_to_first_valid or found_valid_start:
                    rows.append(parts)
        
        if not rows:
            raise ValueError(f"No valid rows found in {simulation_csv_path} with min_columns={min_columns}")
        
        # Create DataFrame from valid rows
        df = pd.DataFrame(rows)
        # df.columns = ['time', 'id', 'x', 'z', 'y']
        return df

def clean_simulation_data(simulation_csv_path: str) -> pd.DataFrame:
    """
    Clean simulation CSV data to keep only position data with 5 columns: time, id, x, z, y.
    
    Args:
        simulation_csv_path: Path to the simulation CSV file
        
    Returns:
        Cleaned DataFrame with columns [time, id, x, z, y]
    """
    try:
        # Use robust CSV reader that handles malformed files
        cleaned_df = robust_read_csv(simulation_csv_path)
        cleaned_df.columns = ['time', 'id', 'x', 'z', 'y']
        
        # Convert numeric columns
        cleaned_df['time'] = pd.to_numeric(cleaned_df['time'], errors='coerce')
        cleaned_df['x'] = pd.to_numeric(cleaned_df['x'], errors='coerce')
        cleaned_df['z'] = pd.to_numeric(cleaned_df['z'], errors='coerce')
        cleaned_df['y'] = pd.to_numeric(cleaned_df['y'], errors='coerce')
        
        # Remove rows with NaN values after conversion
        cleaned_df = cleaned_df.dropna()
        
        # Sort by time and id for consistent processing
        cleaned_df = cleaned_df.sort_values(['time', 'id']).reset_index(drop=True)
        
        metadata = parse_task_metadata(os.path.dirname(simulation_csv_path))
        machine_id = metadata['machine_id']
        task_name = metadata['task_name']
        clean_file_name = f"{task_name}_{machine_id}_clean.csv"
        clean_dir = "datacache/analysis_simulation"
        os.makedirs(clean_dir, exist_ok=True)
        clean_file_path = os.path.join(clean_dir, clean_file_name)
        cleaned_df.to_csv(clean_file_path, index=False)
        
        return cleaned_df
        
    except Exception as e:
        raise ValueError(f"Failed to clean simulation data: {e}")


def calculate_max_height(position_df: pd.DataFrame) -> float:
    """
    Calculate the maximum height (z-coordinate) reached by any block.
    
    Args:
        position_df: DataFrame with position data
        
    Returns:
        Maximum z-coordinate value
    """
    if position_df.empty:
        return 0.0
    
    return float(position_df['z'].max())


def calculate_max_speed(position_df: pd.DataFrame, start_time_speed: float = 5.0) -> float:
    """
    Calculate the maximum speed achieved by any block during simulation.
    Only considers data after start_time_speed seconds to exclude initial settling period.
    
    Args:
        position_df: DataFrame with position data
        start_time_speed: Time threshold to start speed calculation
        
    Returns:
        Maximum speed value across all blocks after the time threshold
    """
    if position_df.empty or len(position_df) < 2:
        return 0.0
    
    # Filter data to only include times after the threshold
    filtered_df = position_df[position_df['time'] > start_time_speed]
    
    if filtered_df.empty or len(filtered_df) < 2:
        return 0.0
    
    max_speed = 0.0
    
    # Group by block ID to calculate speed for each block individually
    for block_id, block_data in filtered_df.groupby('id'):
        if len(block_data) < 2:
            continue
            
        # Sort by time to ensure proper ordering
        block_data = block_data.sort_values('time').reset_index(drop=True)
        
        # Calculate position differences
        dt = block_data['time'].diff().iloc[1:]  # Time differences
        dx = block_data['x'].diff().iloc[1:]     # X position differences
        dy = block_data['y'].diff().iloc[1:]     # Y position differences  
        dz = block_data['z'].diff().iloc[1:]     # Z position differences
        
        # Filter out zero time differences to avoid division by zero
        valid_mask = dt > 0
        if not valid_mask.any():
            continue
            
        dt_valid = dt[valid_mask]
        dx_valid = dx[valid_mask]
        dy_valid = dy[valid_mask]
        dz_valid = dz[valid_mask]
        
        # Calculate 3D speed: sqrt(dx² + dy² + dz²) / dt
        distance_3d = np.sqrt(dx_valid**2 + dy_valid**2 + dz_valid**2)
        speeds = distance_3d / dt_valid
        
        # Update maximum speed if this block has a higher speed
        block_max_speed = speeds.max()
        if not np.isnan(block_max_speed) and block_max_speed > max_speed:
            max_speed = block_max_speed
    
    return float(max_speed)


def analyze_motion_metrics(simulation_csv_path: str, start_time_speed: float = 5.0, verbose: bool = False) -> Dict[str, float]:
    """
    Analyze motion metrics including maximum height and maximum speed.
    
    Args:
        simulation_csv_path: Path to the simulation CSV file
        start_time_speed: Time threshold to start speed calculation
        verbose: Whether to print intermediate results
        
    Returns:
        Dictionary containing max_height and max_speed
    """
    try:
        # Clean the simulation data
        position_df = clean_simulation_data(simulation_csv_path)
        
        if verbose:
            print(f"Cleaned position data: {len(position_df)} rows")
            print(f"Unique blocks: {position_df['id'].nunique()}")
            print(f"Time range: {position_df['time'].min():.2f} - {position_df['time'].max():.2f}")
            
            # Show filtering info for speed calculation
            after_threshold = position_df[position_df['time'] > start_time_speed]
            print(f"Data after {start_time_speed}s for speed calculation: {len(after_threshold)} rows")
        
        # Calculate maximum height
        max_height = calculate_max_height(position_df)
        
        # Calculate initial max height
        initial_max_height = float(position_df['z'][:int(start_time_speed)].max())
        
        # elevation change at the first 5 seconds after start_time_speed
        elevation_change_end_time = start_time_speed + 5.0
        elevation_data_slice_1 = position_df[(position_df['time'] > start_time_speed) & (position_df['time'] <= elevation_change_end_time)]
        elevation_data_slice_2 = position_df[(position_df['time'] > elevation_change_end_time) & (position_df['time'] <= elevation_change_end_time + 5.0)]
        if len(elevation_data_slice_1) > 0 and len(elevation_data_slice_2) > 0:
            elevation_change = float(elevation_data_slice_2['z'].max() - elevation_data_slice_1['z'].max())
        else:
            elevation_change = 0.0
        
        # Calculate maximum speed (filtered to after threshold)
        max_speed = calculate_max_speed(position_df, start_time_speed)
        
        # Calculate elevation change
        # elevation_change = max_height - initial_max_height
        
        if verbose:
            print(f"Maximum height (z): {max_height:.3f}")
            print(f"Initial maximum height: {initial_max_height:.3f}")
            print(f"Maximum speed: {max_speed:.3f}")
            print(f"Elevation change: {elevation_change:.3f}")
        return {
            'max_height': max_height,
            'initial_max_height': initial_max_height,
            'max_speed': max_speed,
            'elevation_change': elevation_change
        }
        
    except Exception as e:
        if verbose:
            print(f"Warning: Failed to analyze motion metrics: {e}")
        return {
            'max_height': 0.0,
            'initial_max_height': 0.0,
            'max_speed': 0.0,
            'elevation_change': 0.0
        }


def analyze_trajectory_quality(simulation_csv_path: str, start_pos: np.ndarray, target_pos: np.ndarray, 
                              start_time_trajectory: float = 0.0, verbose: bool = False) -> Dict[str, float]:
    """
    Analyze trajectory quality metrics including progress, deviation, and efficiency.
    
    Args:
        simulation_csv_path: Path to the simulation CSV file
        start_pos: Starting position as numpy array [x, y, z]
        target_pos: Target position as numpy array [x, y, z]
        start_time_trajectory: Time threshold to start trajectory analysis
        verbose: Whether to print intermediate results
        
    Returns:
        Dictionary containing trajectory quality metrics
    """
    try:
        # Clean the simulation data and filter to after start time
        position_df = clean_simulation_data(simulation_csv_path)
        filtered_df = position_df[position_df['time'] > start_time_trajectory]
        
        if filtered_df.empty or len(filtered_df) < 2:
            if verbose:
                print("Warning: Insufficient trajectory data for quality analysis")
            return _get_default_trajectory_stats()
        
        # Calculate center of mass trajectory
        com_trajectory = _calculate_center_of_mass_trajectory(filtered_df)
        
        if len(com_trajectory) < 2:
            return _get_default_trajectory_stats()
        
        # Calculate trajectory metrics
        metrics = _calculate_trajectory_metrics(
            com_trajectory, start_pos, target_pos, verbose
        )
        
        if verbose:
            print(f"Trajectory Quality Metrics:")
            print(f"  Max Progress Ratio: {metrics['max_progress_ratio']:.3f}")
            print(f"  Direction Consistency: {metrics['direction_consistency']:.3f}")
            print(f"  Backtrack Penalty: {metrics['backtrack_penalty']:.3f}")
            print(f"  Lateral Deviation: {metrics['lateral_deviation']:.3f}")
            print(f"  Path Efficiency: {metrics['path_efficiency']:.3f}")
            print(f"  Final Distance Ratio: {metrics['final_distance_ratio']:.3f}")
        
        return metrics
        
    except Exception as e:
        if verbose:
            print(f"Warning: Failed to analyze trajectory quality: {e}")
        return _get_default_trajectory_stats()


def analyze_target_trajectory_quality(simulation_csv_path: str, target_id: str, start_pos: np.ndarray, 
                                     target_pos: np.ndarray, start_time_trajectory: float = 0.0, 
                                     verbose: bool = False) -> Dict[str, float]:
    """
    Analyze trajectory quality metrics for a specific target block.
    
    Args:
        simulation_csv_path: Path to the simulation CSV file
        target_id: ID of the target block to track (e.g., 'ID_Ballast', 'ID_1')
        start_pos: Starting position as numpy array [x, y, z]
        target_pos: Target position as numpy array [x, y, z]
        start_time_trajectory: Time threshold to start trajectory analysis
        verbose: Whether to print intermediate results
        
    Returns:
        Dictionary containing trajectory quality metrics for the target block
    """
    try:
        # Clean the simulation data and filter to after start time
        position_df = clean_simulation_data(simulation_csv_path)
        filtered_df = position_df[position_df['time'] > start_time_trajectory]
        
        if filtered_df.empty or len(filtered_df) < 2:
            if verbose:
                print("Warning: Insufficient trajectory data for quality analysis")
            return _get_default_trajectory_stats()
        
        # Filter to target block only
        target_df = filtered_df[filtered_df['id'] == target_id]
        
        if target_df.empty or len(target_df) < 2:
            if verbose:
                print(f"Warning: No trajectory data found for target {target_id}")
            return _get_default_trajectory_stats()
        
        # Sort by time and reset index
        target_trajectory = target_df.sort_values('time').reset_index(drop=True)
        
        # Calculate trajectory metrics
        metrics = _calculate_trajectory_metrics(
            target_trajectory, start_pos, target_pos, verbose
        )
        
        if verbose:
            print(f"Target {target_id} Trajectory Quality Metrics:")
            print(f"  Max Progress Ratio: {metrics['max_progress_ratio']:.3f}")
            print(f"  Direction Consistency: {metrics['direction_consistency']:.3f}")
            print(f"  Backtrack Penalty: {metrics['backtrack_penalty']:.3f}")
            print(f"  Lateral Deviation: {metrics['lateral_deviation']:.3f}")
            print(f"  Path Efficiency: {metrics['path_efficiency']:.3f}")
            print(f"  Final Distance Ratio: {metrics['final_distance_ratio']:.3f}")
        
        return metrics
        
    except Exception as e:
        if verbose:
            print(f"Warning: Failed to analyze target trajectory quality: {e}")
        return _get_default_trajectory_stats()


def _calculate_center_of_mass_trajectory(position_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate center of mass trajectory from all blocks."""
    com = (
        position_df.groupby("time")[["x", "y", "z"]]
        .mean()
        .reset_index()
        .sort_values("time")
        .reset_index(drop=True)
    )
    return com


def _calculate_trajectory_metrics(
    trajectory_df: pd.DataFrame,
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    verbose: bool = False
) -> Dict[str, float]:
    """Calculate comprehensive trajectory quality metrics with time-weighting and robust handling."""
    eps = 1e-8

    # Basic setup
    ideal_direction = target_pos - start_pos
    ideal_distance = float(np.linalg.norm(ideal_direction))
    if ideal_distance < 1e-6:
        return _get_default_trajectory_stats()
    unit_direction = ideal_direction / ideal_distance

    # Arrays
    positions = trajectory_df[["x", "y", "z"]].to_numpy(dtype=float)
    times = trajectory_df["time"].to_numpy(dtype=float)

    if len(positions) < 2:
        return _get_default_trajectory_stats()

    # Time deltas & masks (robust to non-monotonic/duplicate timestamps)
    dt = np.diff(times)
    valid_steps = dt > 0.0
    if not np.any(valid_steps):
        return _get_default_trajectory_stats()

    dt = dt[valid_steps]                              # (n-1_valid,)
    dx = np.diff(positions, axis=0)[valid_steps, :]   # (n-1_valid, 3)
    T = float(times[-1] - times[0]) if times[-1] > times[0] else float(np.sum(dt))
    if T <= 0:
        return _get_default_trajectory_stats()

    # Velocities (only used for direction cosine; note not used for path length)
    velocities = dx / dt[:, None]                     # (n-1_valid, 3)
    speed = np.linalg.norm(velocities, axis=1)        # (n-1_valid,)

    # 1) Progress metrics
    # Along-track position a(t) = u^T (x - x0)
    relative_positions = positions - start_pos[None, :]
    along_track_positions = relative_positions @ unit_direction  # (n,)

    # Maximum progress ratio
    max_progress_ratio = float(
        min(1.0, max(0.0, float(np.max(along_track_positions)) / (ideal_distance + eps)))
    )

    # Direction consistency: c = (1/T) ∫ max(0, cosθ) dt, cosθ = (u^T v)/||v||
    # Avoid division by zero for zero-speed samples; time-weighted
    along_speed = (velocities @ unit_direction)                   # (n-1_valid,)
    cos_theta = np.divide(along_speed, np.maximum(speed, eps))    # (n-1_valid,)
    c_pos = np.maximum(0.0, cos_theta) * dt
    direction_consistency = float(np.sum(c_pos) / T)              # in [0,1]

    # Backtrack penalty: r_back = (backward progress distance) / D = (∫ max(0, -u^T v) dt) / D
    backtrack_distance = float(np.sum(np.maximum(0.0, -along_speed) * dt))
    backtrack_penalty = float(backtrack_distance / (ideal_distance + eps))

    # 2) Deviation metrics
    # Lateral error e_perp(t) = ||(I - u u^T)(x - x0)||
    P = np.eye(3) - np.outer(unit_direction, unit_direction)      # projection to lateral plane
    lateral_vec = relative_positions @ P.T                        # (n,3)
    e_perp = np.linalg.norm(lateral_vec, axis=1)                  # (n,)

    # Time-weighted lateral deviation: x_perp = (1/(D T)) ∫ e_perp(t) dt
    # Using piecewise constant approximation: interval [i, i+1] represented by e_perp[i] 
    # (could also use trapezoidal rule: 0.5*(e[i]+e[i+1]))
    e_seg = e_perp[:-1][valid_steps]                              # align with dt
    lateral_deviation = float(np.sum(e_seg * dt) / ((ideal_distance + eps) * T))

    # Path efficiency: total path length / effective progress - 1 (large if progress is small)
    total_path_length = float(np.sum(np.linalg.norm(dx, axis=1)))  # true path length (by displacement)
    effective_progress = float(max(np.max(along_track_positions), eps))
    path_efficiency = float(max(0.0, total_path_length / effective_progress - 1.0))

    # 3) Final remaining distance ratio
    final_along_track = float(along_track_positions[-1])
    final_distance_ratio = float(max(0.0, (ideal_distance - final_along_track) / (ideal_distance + eps)))

    return {
        "max_progress_ratio": max_progress_ratio,
        "direction_consistency": direction_consistency,
        "backtrack_penalty": backtrack_penalty,
        "lateral_deviation": lateral_deviation,
        "path_efficiency": path_efficiency,
        "final_distance_ratio": final_distance_ratio,
    }


def _get_default_trajectory_stats() -> Dict[str, float]:
    """Return default trajectory statistics for edge cases."""
    return {
        'max_progress_ratio': 0.0,
        'direction_consistency': 0.0,
        'backtrack_penalty': 0.0,
        'lateral_deviation': 0.0,
        'path_efficiency': 0.0,
        'final_distance_ratio': 1.0,  # Maximum penalty for not reaching target
    }


def extract_round_suffix(file_paths: Dict[str, str]) -> str:
    """
    Extract the round suffix from simulation file paths.
    
    Args:
        file_paths: Dictionary containing simulation file paths
        
    Returns:
        Round suffix string (e.g., "_sim_1", "_sim_2") or empty string if no round
    """
    # Use the BSG file to detect round suffix
    bsg_filename = os.path.basename(file_paths['machine_bsg_path'])
    
    if "_sim_" in bsg_filename:
        # Extract the round number from filename like "machine_id_sim_1.bsg"
        round_part = bsg_filename.split("_sim_")[-1].split(".")[0]
        if round_part.isdigit():
            return f"_{round_part}"
    
    return ""

def save_analysis_result(result: Dict[str, Any], base_dir: str = "datacache/analysis_simulation") -> str:
    """
    Save analysis result to JSON file with proper round suffix handling.
    
    Args:
        result: Analysis result dictionary containing category, level, model_name, machine_id, round_suffix
        base_dir: Base directory to save the file
        
    Returns:
        Path to the saved file
    """
    # Build filename with round suffix
    filename_base = f"sim_{result['category']}_{result['level']}_{result['model_name']}_{result['machine_id']}"
    if result.get('round_suffix'):
        filename_base += result['round_suffix']
    
    save_path = os.path.join(base_dir, f"{filename_base}.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return save_path


def parse_csv_filename(csv_path: str) -> Optional[int]:
    """
    Parse simulation round from CSV filename.
    
    Args:
        csv_path: Path to the simulation CSV file
        
    Returns:
        Simulation round number (1, 2, 3, etc.) or None if no round number found
        
    Examples:
        simulation_log_machine_id_sim.csv -> None
        simulation_log_machine_id_sim_1.csv -> 1
        simulation_log_machine_id_sim_2.csv -> 2
    """
    filename = os.path.basename(csv_path)
    
    # Pattern to match simulation_log_*_sim_#.csv or simulation_log_*_sim.csv
    match = re.search(r'_sim(?:_(\d+))?\.csv$', filename)
    
    if match:
        round_str = match.group(1)  # This will be None if no round number
        return int(round_str) if round_str else None
    
    return None


def parse_category_from_path(csv_path: str) -> str:
    """
    Parse category (lift, support, transport) from the CSV file path.
    
    Args:
        csv_path: Path to the simulation CSV file
        
    Returns:
        Category string ('lift', 'support', or 'transport')
        
    Raises:
        ValueError: If category cannot be determined from path
    """
    # Navigate up the directory structure to find the task directory
    # Expected structure: .../category_level_model_timestamp/simulation/machine_id/simulation_log_*.csv
    path_parts = os.path.normpath(csv_path).split(os.sep)
    
    # Look for directory names that start with known categories
    for part in reversed(path_parts):
        if part.startswith('lift_'):
            return 'lift'
        elif part.startswith('support_'):
            return 'support' 
        elif part.startswith('transport_'):
            return 'transport'
    
    raise ValueError(f"Cannot determine category from path: {csv_path}")


def route_simulation_analysis(csv_path: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Router function that analyzes a simulation CSV file by determining the category
    and routing to the appropriate analysis function.
    
    Args:
        csv_path: Path to the simulation CSV file
        verbose: Whether to print intermediate results
        
    Returns:
        Analysis result dictionary from the appropriate analysis function
        
    Raises:
        ValueError: If category cannot be determined or analysis fails
        FileNotFoundError: If required files are missing
    """
    try:
        # Parse simulation round from filename
        sim_round = parse_csv_filename(csv_path)
        
        # Get machine directory (parent directory of CSV file)
        machine_dir = os.path.dirname(csv_path)
        
        # Parse category from path
        category = parse_category_from_path(csv_path)
        
        if verbose:
            print(f"Router analysis:")
            print(f"  CSV path: {csv_path}")
            print(f"  Machine dir: {machine_dir}")
            print(f"  Category: {category}")
            print(f"  Simulation round: {sim_round}")
        
        # Import analysis modules dynamically to avoid circular imports
        if category == 'lift':
            from .sim_lift import main as analyze_simulation_lift_and_save
            result = analyze_simulation_lift_and_save(machine_dir, sim_round=sim_round, verbose=verbose)
        elif category == 'support':
            from .sim_support import main as analyze_simulation_support_and_save
            result = analyze_simulation_support_and_save(machine_dir, sim_round=sim_round, verbose=verbose)
        elif category == 'transport':
            from .sim_transport import main as analyze_simulation_transport_and_save
            result = analyze_simulation_transport_and_save(machine_dir, sim_round=sim_round, verbose=verbose)
        else:
            raise ValueError(f"Unknown category: {category}")
        
        copy_passed_bsg(result)
            
    except Exception as e:
        raise ValueError(f"Failed to route simulation analysis: {str(e)}") from e
    
def copy_passed_bsg(result: Dict[str, Any], base_dir: str = "datacache/passed_cases") -> Optional[str]:
    """
    Copy BSG file to passed cases directory if the simulation passed.
    
    Args:
        result: Analysis result dictionary containing pass status, category, level, model_name, and machine_bsg_path
        base_dir: Base directory for passed cases
        
    Returns:
        Path to the copied file if successful, None if not passed or failed to copy
    """
    # Only copy if the simulation passed
    if not result.get('pass', False):
        return None
    
    try:
        # Extract required fields
        category = result.get('category', '')
        level = result.get('level', '')
        model_name = result.get('model_name', '')
        machine_id = result.get('machine_id', '')
        machine_bsg_path = result.get('machine_bsg_path', '')
        round_suffix = result.get('round_suffix', '')
        
        # Validate required fields
        if not all([category, level, model_name, machine_id, machine_bsg_path]):
            raise ValueError("Missing required fields in result dictionary")
        
        if not os.path.exists(machine_bsg_path):
            raise FileNotFoundError(f"Source BSG file not found: {machine_bsg_path}")
        
        # Create target directory structure
        target_dir = os.path.join(base_dir, f"{category}_{level}")
        os.makedirs(target_dir, exist_ok=True)
        
        # Build target filename with model name as prefix
        target_filename = f"{model_name}_{machine_id}_sim{round_suffix}.bsg"
        target_path = os.path.join(target_dir, target_filename)
        
        # Copy the file
        shutil.copy2(machine_bsg_path, target_path)
        
        return target_path
        
    except Exception as e:
        print(f"Warning: Failed to copy BSG file: {e}")
        return None


def main(csv_path: str, verbose: bool = False) -> None:
    """Main function for standalone execution."""
    route_simulation_analysis(csv_path, verbose=verbose)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--verbose", action="store_true", required=False, default=False)
    args = parser.parse_args()
    main(args.csv_path, args.verbose)
