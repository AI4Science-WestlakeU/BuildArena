"""
Simulation for Transport category task
"""

import os
import shutil
import json
from typing import Literal

from spatial.build import Assembly, Machine, Blocks, Block
from spatial.components import Vector
from simulation.operations import run_simulation_sequence

from config import SavedMachines

def main(machine_json: str, level: Literal["soft", "medium", "hard"], control_path: str | None = None):
    """
    Main function for transport simulation
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
        
    machine._init_control_config()

    # Load control path
    if control_path:
        gen = control_path.split("_")[-1].split(".")[0]
        machine.name = f"{machine_id}_sim_{gen}"
        with open(control_path, "r", encoding="utf-8") as f:
            control_data = json.load(f)
        control_sequence = control_data["control_sequence"]
        control_config = control_data["control_config"]
        for control in control_config:
            machine.change_control_key(block_id=control["block_id"], action=control["action"], new_key=control["key"])
        for control in control_sequence:
            machine.add_control_sequence(time=control["time"], key=control["key"], hold_for=control["hold_for"])
    else:
        raise ValueError("Control path is required")

    blocks = [block for block in machine.blocks.values() if block.name != "Starting Block"]
    if len(blocks) == 0:
        raise ValueError("No blocks found in the machine")
    
    if len(machine.keys) == 0:
        raise ValueError("No control keys assigned to the machine")

    if len(machine.control_sequence) == 0:
        raise ValueError("No control sequence found in the machine")
        
    x_positions = [block.center_pos.virtual[0] for block in machine.blocks.values()]
    y_positions = [block.center_pos.virtual[1] for block in machine.blocks.values()]
    z_positions = [block.center_pos.virtual[2] for block in machine.blocks.values()]
    top_y = max(y_positions)

    center_x = sum(x_positions) / len(x_positions)
    center_z = sum(z_positions) / len(z_positions)
    
    shift_virtual = [-center_x, 0.0, -center_z]

    if level != "soft":
        # Add a Ballast Load Block
        blocks_storage = Blocks()
        ballast: Block = blocks_storage.get(block_name="Ballast", local_id="Ballast")
        ballast.rotate(yaw = 0, pitch = 270, roll = 0)
        if level == "hard":
            ballast.geo.scale = Vector([4, 8, 1.5])
        ballast.geo.position = Vector([center_x, top_y + 3, center_z])
        ballast.tracking = True
        machine.do_collision = False
        machine._add_block(ballast)

    # Track all blocks except the starting block
    blocks = [block for block in machine.blocks.values() if block.name in ["Starting Block", "Ballast", "Powered Wheel"]]
    for block in blocks:
        block.tracking = True

    # Save machine and put close to the ground
    machine.to_file(output_dir=save_dir, shift_virtual=shift_virtual)
    print(f"Machine saved to {save_dir}")

    # Copy machine to SavedMachines
    shutil.copy(os.path.join(save_dir, f"{machine.name}.bsg"), os.path.join(SavedMachines, f"{machine.name}.bsg"))
    print(f"Machine copied to {SavedMachines}")

    # Save machine block names - ID json
    with open(os.path.join(save_dir, f"block_id_{machine.name}.json"), "w", encoding="utf-8") as f:
        json.dump([{block.local_id: block.name} for block in machine.blocks.values()], f, ensure_ascii=False, indent=2)

    # Run simulation sequence
    csv_file = run_simulation_sequence(machine_name=machine.name, output_dir=save_dir, duration=30, set_ground=True)
    return csv_file

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run transport simulation')
    parser.add_argument('--machine_json', type=str, required=False, help='Path to the machine JSON file')
    parser.add_argument('--level', type=str, required=False, help='Level of the simulation')
    args = parser.parse_args()
    main(machine_json=args.machine_json, level=args.level)