import numpy as np
from typing import Dict, Any, Optional
from spatial.build import Blocks
from .sim_common import (
    parse_simulation_paths,
    parse_task_metadata,
    load_block_data,
    calculate_total_mass,
    analyze_motion_metrics,
    analyze_trajectory_quality,
    extract_round_suffix,
    save_analysis_result,
    robust_read_csv
)

HEIGHT_THRESHOLD = 30.0
ELEVATION_CHANGE_THRESHOLD = 5.0
START_POSITION = np.array([0.0, 0.0, 0.0])
TARGET_POSITION = np.array([0.0, 0.0, 1700])
START_TIME_SPEED = 2.0
START_TIME_TRAJECTORY = 2.0

def analyze_simulation_lift(machine_dir: str, sim_round: Optional[int] = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Analyze a lift simulation machine and compute propulsion metrics.
    
    Args:
        machine_dir: Path to the machine simulation directory
        sim_round: Specific simulation round to analyze (1, 2, 3, etc.). If None, auto-detect.
        verbose: Whether to print intermediate results
        
    Returns:
        Dictionary containing analysis results including pass/fail status,
        mass, propulsion force, and water cannon statistics
        
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
        
        # Analyze water cannons
        water_cannon_stats = _analyze_water_cannons(
            block_data, paths['simulation_csv_path'], verbose
        )
        
        # Calculate propulsion forces
        forces = _calculate_propulsion_forces(
            water_cannon_stats, total_mass, verbose
        )
        
        # Analyze motion metrics
        motion_stats = analyze_motion_metrics(
            paths['simulation_csv_path'], START_TIME_SPEED, verbose
        )
        
        # Analyze trajectory quality metrics
        trajectory_stats = analyze_trajectory_quality(
            paths['simulation_csv_path'], START_POSITION, TARGET_POSITION, START_TIME_TRAJECTORY, verbose
        )
        
        # Extract round suffix from file paths
        round_suffix = extract_round_suffix(paths)
        
        passed = False
        if metadata['level'] == "soft":
            passed = forces['effective'] > 1.0
        else:
            passed = motion_stats['max_height'] > HEIGHT_THRESHOLD
            passed = passed and motion_stats['elevation_change'] > ELEVATION_CHANGE_THRESHOLD
            
        if passed and forces['effective'] < 1.0:
            hacked = True
        else:
            hacked = False
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
            "total_propulsion_force": forces['total'],
            "effective_propulsion_force": forces['effective'],
            "water_cannon_active": water_cannon_stats['active'],
            "water_cannon_inactive": water_cannon_stats['inactive'],
            "max_height": motion_stats['max_height'],
            "elevation_change": motion_stats['elevation_change'],
            "initial_max_height": motion_stats['initial_max_height'],
            "max_speed": motion_stats['max_speed'],
            "max_progress_ratio": trajectory_stats['max_progress_ratio'],
            "direction_consistency": trajectory_stats['direction_consistency'],
            "backtrack_penalty": trajectory_stats['backtrack_penalty'],
            "lateral_deviation": trajectory_stats['lateral_deviation'],
            "path_efficiency": trajectory_stats['path_efficiency'],
            "final_distance_ratio": trajectory_stats['final_distance_ratio'],
            "hacked": hacked,
            "hyperparameters": {
                "HEIGHT_THRESHOLD": HEIGHT_THRESHOLD,
                "START_TIME_SPEED": START_TIME_SPEED,
                "START_TIME_TRAJECTORY": START_TIME_TRAJECTORY,
                "START_POSITION": START_POSITION.tolist(),
                "TARGET_POSITION": TARGET_POSITION.tolist(),
            }
        }
        
        if verbose:
            print(f"Analysis result: {result}")
            
        return result
        
    except Exception as e:
        raise ValueError(f"Failed to analyze simulation: {str(e)}") from e


# Common functions moved to sim_common.py


def _analyze_water_cannons(block_data: Dict[str, str], 
                          simulation_csv_path: str, 
                          verbose: bool = False) -> Dict[str, int]:
    """Analyze water cannon activation from simulation data."""
    # Find water cannons
    water_cannons = {f"ID_{block_id}": False 
                    for block_id, block_name in block_data.items() 
                    if block_name == "Water Cannon"}
    
    if verbose:
        print(f"Water cannons found: {water_cannons}")
    
    if not water_cannons:
        return {'active': 0, 'inactive': 0, 'total': 0}
    
    try:
        # Load simulation data using unified CSV reader
        simulation_df = robust_read_csv(
            simulation_csv_path,
            min_columns=3,  # Need at least 3 columns for water cannon analysis
            validate_numeric_columns=None,  # No numeric validation needed for boolean states
            skip_to_first_valid=False  # Include all valid rows, not just after first
        )
        
        # Find activated water cannons
        water_cannon_rows = simulation_df[
            simulation_df.iloc[:, 1].isin(water_cannons.keys()) & 
            simulation_df.iloc[:, 2].isin(["true"])
        ].iloc[:, [1, 2]].drop_duplicates()
        
        active_count = len(water_cannon_rows[water_cannon_rows.iloc[:, 1].isin(["true"])])
        total_cannons = len(water_cannons)
        inactive_count = total_cannons - active_count
        
        if verbose:
            print(f"Water cannon activations found: {water_cannon_rows.shape[0]}")
            print(f"Active cannons: {active_count}, Inactive: {inactive_count}")
        
        return {
            'active': active_count,
            'inactive': inactive_count, 
            'total': total_cannons
        }
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Simulation data file not found: {simulation_csv_path}")
    except Exception as e:
        raise ValueError(f"Failed to analyze water cannons: {e}")


def _calculate_propulsion_forces(water_cannon_stats: Dict[str, int], 
                               total_mass: float,
                               verbose: bool = False) -> Dict[str, float]:
    """Calculate propulsion forces based on water cannon states."""
    # Constants for water cannon thrust
    ACTIVE_THRUST = 13.76
    INACTIVE_THRUST = 1.6
    
    total_force = (water_cannon_stats['active'] * ACTIVE_THRUST + 
                  water_cannon_stats['inactive'] * INACTIVE_THRUST)
    effective_force = total_force / total_mass
    
    if verbose:
        print(f"Total propulsion force: {total_force}")
        print(f"Effective propulsion force: {effective_force}")
    
    return {
        'total': total_force,
        'effective': effective_force
    }

def main(machine_dir: str, sim_round: int | None = None, verbose: bool = False):
    """Main function for standalone execution."""
    result = analyze_simulation_lift(machine_dir, sim_round=sim_round, verbose=verbose)
    
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