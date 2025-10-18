"""
Simulation for Lift category task
"""

import os
import shutil
import json
from typing import Literal
import random

from spatial.build import Assembly, Machine
from simulation.operations import run_simulation_sequence

from config import SavedMachines

def main(machine_json: str, level: Literal["soft", "medium", "hard"]):
    """
    Main function for lifting simulation
    """
    assert machine_json.endswith(".json"), "Machine JSON file must end with .json"
    assert level in ["soft", "medium", "hard"], "Level must be one of: soft, medium, hard"

    machine_id = os.path.basename(machine_json).split(".")[0]
    proj_dir = os.path.dirname(os.path.dirname(machine_json))
    task_dir = os.path.dirname(proj_dir)
    save_dir = os.path.join(task_dir, "simulation", machine_id)
    os.makedirs(save_dir, exist_ok=True)
    db_path = os.path.join(task_dir, "simulation_database.db")
    # Load machine
    try:
        machine = Assembly(name=f"{machine_id}_sim", save_dir=save_dir, db_path=db_path)
        machine.from_file(file_path=machine_json)
    except:
        machine = Machine(name=f"{machine_id}_sim", save_dir=save_dir, db_path=db_path)
        machine.from_file(file_path=machine_json)

    if level != "soft":
    # Initialize control config
        machine._init_control_config()
        # Press all keys for 30 seconds, as described in the Task Evaluation Protocol
        for block in machine.blocks.values():
            if block.name == "Water Cannon":
                machine.change_control_key(block_id=block.local_id, action="hold_to_fire", new_key="Alpha1")
        machine.add_control_sequence(time=2, key="Alpha1", hold_for=30)

    # Track all blocks except the starting block
    blocks = [block for block in machine.blocks.values() if block.name != "Starting Block"]
    if len(blocks) == 0:
        raise ValueError("No blocks found in the machine")
    if level == "soft" or len(blocks) <=4:
        for block in blocks:
            block.tracking = True
    else:
        # Randomly select 4 blocks to track
        selected_blocks = random.sample(blocks, 4)
        for block in selected_blocks:
            block.tracking = True
    
    # Save machine and put close to the ground
    machine.to_file(output_dir=save_dir)
    print(f"Machine saved to {save_dir}")

    # Copy machine to SavedMachines
    shutil.copy(os.path.join(save_dir, f"{machine_id}_sim.bsg"), os.path.join(SavedMachines, f"{machine_id}_sim.bsg"))
    print(f"Machine copied to {SavedMachines}")

    # Save machine block names - ID json
    with open(os.path.join(save_dir, f"block_id_{machine_id}_sim.json"), "w", encoding='utf-8') as f:
        json.dump([{block.local_id: block.name} for block in machine.blocks.values()], f)

    # Run simulation sequence
    csv_file = run_simulation_sequence(machine_name=f"{machine_id}_sim", output_dir=save_dir, duration=10 if level == "soft" else 30, set_ground=True)
    return csv_file

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run lift simulation')
    parser.add_argument('--machine_json', type=str, required=False, help='Path to the machine JSON file')
    parser.add_argument('--level', type=str, required=False, help='Level of the simulation')
    args = parser.parse_args()
    main(machine_json=args.machine_json, level=args.level)