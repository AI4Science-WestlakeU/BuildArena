import os
import random
import string
import inspect
import functools
from typing import Literal

import numpy as np
import pandas as pd
import trimesh
from trimesh import Trimesh

import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from matplotlib.colors import Normalize
from matplotlib import colormaps

from .components import Vector, Face

def color_meshes(meshes, cmap='viridis'):  # Use 'viridis', 'plasma', 'inferno', etc.
    colormap = colormaps[cmap]  

    for i, mesh in enumerate(meshes):
        rgba = [int(c * 255) for c in colormap(i / len(meshes))]  # Normalize colormap
        mesh.visual.face_colors = rgba

def machine_preview(machine, cmap='viridis'):
    """For some reason, the 3D scene is mirrored front and back, tried to fix it but failed"""
    meshes = [block.collider for block in machine.blocks.values()]
    color_meshes(meshes, cmap=cmap)
    return trimesh.Scene([meshes])

def face_is_attached(face_a: Face, face_b: Face):
    is_parallel = np.isclose(np.linalg.norm(np.cross(face_a.normal.vec_abs.virtual, face_b.normal.vec_abs.virtual)), 0)
    same_pos = np.isclose(np.linalg.norm(face_a.center.virtual - face_b.center.virtual), 0)
    
    return is_parallel and same_pos

def random_id(digits=5):
    return ''.join(random.choices(string.ascii_lowercase, k=digits)) 

def rotation_matrix(yaw, pitch, roll):
    """
    Constructs a 3x3 rotation matrix using Z-Y-X (yaw-pitch-roll) angles.
    """
    yaw, pitch, roll = np.radians([-roll, yaw, pitch])
    
    cz = np.cos(yaw)
    sz = np.sin(yaw)

    cy = np.cos(pitch)
    sy = np.sin(pitch)

    cx = np.cos(roll)
    sx = np.sin(roll)

    # Rotation about Z (yaw)
    Rz = np.array([
        [cz, -sz, 0],
        [sz,  cz, 0],
        [0,   0,  1]
    ])

    # Rotation about Y (pitch)
    Ry = np.array([
        [cy,  0, sy],
        [0,   1, 0],
        [-sy, 0, cy]
    ])

    # Rotation about X (roll)
    Rx = np.array([
        [1, 0,   0],
        [0, cx, -sx],
        [0, sx,  cx]
    ])

    # Final rotation matrix: R = Rx * Ry * Rz
    R = Rx @ Ry @ Rz
    return R

def create_connector_mesh(point_a: np.ndarray, point_b: np.ndarray, radius: float = 0.05) -> Trimesh:
    """
    Create a thin stick-like trimesh (cylinder) from point A to point B.

    Parameters:
    - point_a (np.ndarray): 3D coordinates of the starting point (shape: (3,))
    - point_b (np.ndarray): 3D coordinates of the ending point (shape: (3,))
    - radius (float): Radius of the stick/cylinder.

    Returns:
    - trimesh.Trimesh: A trimesh object representing the stick.
    """
    vec = point_b - point_a
    length = np.linalg.norm(vec)
    if length == 0:
        raise ValueError("Points A and B must be different to form a stick.")

    direction = vec / length

    # Create a cylinder along the Z-axis
    cylinder = trimesh.creation.cylinder(radius=radius, height=length, sections=32)

    # Compute rotation matrix to align Z-axis to direction
    try:
        rotation_matrix = trimesh.geometry.align_vectors([0, 0, 1], direction)
    except Exception as e:
        raise RuntimeError(f"Failed to align vectors: {e}")

    cylinder.apply_transform(rotation_matrix)

    # Translate to midpoint
    midpoint = (point_a + point_b) / 2
    cylinder.apply_translation(midpoint)

    return cylinder

def operation(placeholder = None, log = True, group: Literal["default", "build_only", "build", "refine", "control"] = "default"):
    def decorator(func):
        """Decorator to automatically record operations in the history"""
        raw_doc = func.__doc__ or ""
        name = func.__name__

        if placeholder:
            new_doc = raw_doc.format(placeholder=placeholder)
        else:
            new_doc = raw_doc
            
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            op_name = func.__name__
            params = {}

            sig = inspect.signature(func)
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()

            for name, value in bound_args.arguments.items():
                if name == 'self':
                    continue
                else:
                    params[name] = value

            result = func(self, *args, **kwargs)
            
            # Record final operation history
            do_record = self.record_op and log and op_name != 'reset'
                
            if do_record:
                self.operation_history.append({
                    "op": op_name,
                    "params": params, 
                    "result": result
                })
                self.record_op = False
            
            # Record full operation history
            self.operation_history_full.append({
                    "op": op_name,
                    "params": params,
                    "result": result
                })
            
            # save full operation history
            self.save_full_op_history()

            return result
            
        if new_doc:
            wrapper.__doc__ = new_doc
        wrapper._is_operation = True  # Mark the operation function
        wrapper._group = group
        return wrapper
    return decorator

def xml_to_dict(elem):
    children = list(elem)
    if not children:
        return elem.text.strip() if elem.text else ""
    result = {}
    for child in children:
        result[child.tag] = xml_to_dict(child)
    return result


def format_float_array(arr, precision=2):
    """
    Convert a list or array of floats to a string with the given precision.
    Example: [1.234, 5.678] with precision=1 -> '1.2, 5.7'
    """
    coords = ', '.join(f"{x:.{precision}f}" for x in arr)
    return f"[{coords}]"

def int_to_char(num):
    """Convert an integer to a capital letter starting from A"""
    return chr(65 + num)