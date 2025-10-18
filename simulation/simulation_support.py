"""
Simulation for Support category task
"""

import os
import shutil
import json
from typing import Literal

from spatial.build import Assembly, Machine, Blocks, Block
from spatial.components import Orientation
from simulation.operations import run_simulation_sequence

from config import SavedMachines

def main(machine_json: str, level: Literal["soft", "medium", "hard"]):
    """
    Main function for support simulation
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

    blocks = [block for block in machine.blocks.values() if block.name != "Starting Block"]
    if len(blocks) == 0:
        raise ValueError("No blocks found in the machine")

    # Get position of all blocks
    x_positions = [block.center_pos.virtual[0] for block in machine.blocks.values()]
    y_positions = [block.center_pos.virtual[1] for block in machine.blocks.values()]
    z_positions = [block.center_pos.virtual[2] for block in machine.blocks.values()]

    # Compute the horizontal lengths
    length_x = max(x_positions) - min(x_positions)
    length_z = max(z_positions) - min(z_positions)
    print(f"Length x: {length_x}, Length z: {length_z}")

    # Compute the top and bottom of the machine
    top_y = max(y_positions)
    bottom_y = min(y_positions)

    # Compute the horizontal geometry center of the machine
    center_x = sum(x_positions) / len(x_positions)
    center_z = sum(z_positions) / len(z_positions)

    if level == "soft":
        gap_length = 5.0
    elif level == "medium":
        gap_length = 10.0
    elif level == "hard":
        gap_length = 20.0

    rotation = Orientation(rot=[1, 0, 0, 0])  # w,x,y,z
    shift_virtual = [-center_x, bottom_y - 1, -center_z]
    
    if length_z <= gap_length:
        if length_x >= length_z:
            rotation = Orientation(rot=[90, 0, 0])  # w,x,y,z
            shift_virtual = [center_z, bottom_y - 1, center_x]
    
    # Add a Ballast Load Block
    blocks_storage = Blocks()
    ballast: Block = blocks_storage.get(block_name="Ballast", local_id="Ballast")
    ballast.shift(shift_real = [center_x + 0.0, center_z + 0.0, top_y + 3])
    ballast.rotate(yaw = 0, pitch = 270, roll = 0)
    ballast.tracking = True
    ballast.change_mass = True
    machine.do_collision = False
    machine._add_block(ballast)
    
    # Save machine and put close to the terrain
    rotation = [rotation.quat.x, rotation.quat.y, rotation.quat.z, rotation.quat.w]
    print(f"Rotation: {rotation}")
    machine.to_file(output_dir=save_dir, shift_virtual=shift_virtual, rotation=rotation)
    print(f"Machine saved to {save_dir}")

    # Copy machine to SavedMachines
    shutil.copy(os.path.join(save_dir, f"{machine_id}_sim.bsg"), os.path.join(SavedMachines, f"{machine_id}_sim.bsg"))
    print(f"Machine copied to {SavedMachines}")

    # Save machine block names - ID json
    with open(os.path.join(save_dir, f"block_id_{machine_id}_sim.json"), "w", encoding='utf-8') as f:
        json.dump([{block.local_id: block.name} for block in machine.blocks.values()], f)

    # Run simulation sequence
    csv_file = run_simulation_sequence(machine_name=f"{machine_id}_sim", output_dir=save_dir, duration=17, set_ground=False)
    return csv_file

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run support simulation')
    parser.add_argument('--machine_json', type=str, required=False, help='Path to the machine JSON file')
    parser.add_argument('--level', type=str, required=False, help='Level of the simulation')
    args = parser.parse_args()
    main(machine_json=args.machine_json, level=args.level)