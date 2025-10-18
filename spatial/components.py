import os
from typing import List, Union, Any
import numpy as np
import quaternion

class Vector:
    def __init__(self, vector: Union[List[float], np.ndarray, "Vector"]):
        if isinstance(vector, Vector):
            # Share data
            self._data = vector._data
        else:
            # save as np.array and ensure data sharing and shape (3,)
            arr = np.array(vector, dtype=np.float64)
            self._data = arr.reshape(-1) if arr.size == 3 else arr
            
    @classmethod
    def from_real(cls, vector: List[float]):
        """Convert common sense coordinates [right, forward, top] to game setting coordinates [x, y, z]"""
        return cls([vector[0], vector[2], vector[1]])
    
    @property
    def virtual(self) -> np.ndarray:
        """Return game setting coordinates [x, y, z]"""
        return self._data
    
    @property
    def real(self) -> np.ndarray:
        """Return common sense coordinates  [right, forward, top]"""
        return self._data[[0, 2, 1]]  # Rearrange as [x, z, y]
    
    @property
    def norm(self):
        """Return vector norm"""
        norm_value = np.linalg.norm(self.virtual)
        if norm_value == 0:
            return Vector(self.virtual)  # Return itself if is zero
        
        norm_vector = Vector(self.virtual / norm_value)
        return norm_vector
    
    @property
    def caption(self):
        """Return the orientation caption for environment prompting (real)"""
        return describe_orientation(self)
    
    @property
    def coordinates(self):
        """Return the coordinates caption for environment prompting (real)"""
        return np.array2string(self.real, precision=2, separator=',', suppress_small=True)

class Orientation:
    def __init__(self, rot: Union[List[float], Any]=[1, 0, 0, 0], vec_base: Vector=Vector([0, 0, 1])):
        """Storage orientation related info:
            - quaternion relative to local vec_base (frozen), for .bsg file writing
            - absolute vector relative to global coordinates, for machine state prompting
            - absolute rotation matrix relative to global coordinates, for position calculation
        """
        self.vec_base = vec_base
        if isinstance(rot, quaternion.quaternion):
            self.quat = rot
        elif isinstance(rot, List) or isinstance(rot, np.ndarray):
            if len(rot) == 3:  # If the rot is 3D vector
                rot = Vector(rot).norm
                self.quat = self.rel_quat(rot)
        
            elif len(rot) == 4:  # If the rot is a quaternion
                self.quat = quaternion.from_float_array(rot)
        
        else:
            raise ValueError("Input array must be of shape (3,) for vector or (4,) for quaternion.")
            
        self.rot_mat = quaternion.as_rotation_matrix(self.quat)
    
    def rel_quat(self, rot: Vector) -> quaternion:
        """Return the relative local orientation quaternion"""
        if np.allclose(rot.virtual, 0):  # Handle zero rotation case
            return np.quaternion(1, 0, 0, 0)  # Identity quaternion
    
        dot_prod = np.dot(self.vec_base.virtual, rot.virtual)
        
        if np.isclose(dot_prod, -1.0):
            # Vectors are opposite; choose an arbitrary perpendicular axis
            if np.abs(self.vec_base.virtual[0]) < 0.1:
                perp_axis = np.array([1, 0, 0])
            else:
                perp_axis = np.array([0, 1, 0])
            
            perp_axis = perp_axis - np.dot(perp_axis, self.vec_base.virtual) * self.vec_base.virtual  # Ensure it's perpendicular
            perp_axis /= np.linalg.norm(perp_axis)  # Normalize it

            # Create a quaternion representing a 180-degree rotation around the perpendicular axis
            quat = np.quaternion(0, *perp_axis)
        else:
            cross_prod = np.cross(self.vec_base.virtual, rot.virtual)
            
            quat = np.quaternion(1 + dot_prod, *cross_prod)
        quat = quat.normalized()  # Normalize the quaternion
        return quat
    
    @property
    def vec_abs(self) -> Vector:
        """Return the absolute global orientation vector"""
        quat = self.quat.normalized()

        # Rotate the reference vector using the quaternion
        vec_abs = quaternion.as_rotation_matrix(quat) @ self.vec_base.virtual
        return Vector(vec_abs)
    
    @property
    def euler(self) -> Vector:
        """Return the euler angles. Do not use this for anything else but the .bsg file writing."""
        return Vector(np.degrees(quaternion.as_euler_angles(self.quat)) + [270, 0, 0])
        
    def rotate(self, R):
        """Rotate and update the current Orientation using a new rotation matrix R"""
        if R.shape != (3, 3):
            raise ValueError("Rotation matrix R must be of shape (3, 3)")
        
        # Apply the rotation matrix to the current rotation matrix
        new_rot_mat = R @ self.rot_mat  
        
        self.quat = quaternion.from_rotation_matrix(new_rot_mat)
        self.rot_mat = new_rot_mat
        
        return self
    
    @property
    def caption(self):
        """Return the orientation caption for environment prompting (real)"""
        return self.vec_abs.caption
        
class Geometry:
    def __init__(self, vec_base: List, shape: List, root: List, scale: List | None = None):
        self.position = Vector([0, 0, 0])
        self.rotation = Orientation(vec_base=Vector(vec_base))
        self.scale = Vector(np.array([1, 1, 1])) if scale is None else Vector(scale)
        self.shape = Vector(shape)
        self.root = Vector(root)
        
    @property
    def caption(self):
        """Return the position caption for environment prompting (real)"""
        return f"{np.array2string(self.position.real, precision=2, separator=',', suppress_small=True)}"
        
class Face:
    def __init__(self, color: str, rel_pos: List, face_normal: List, block_geo: Geometry, local_id: int, name: str):
        """Storage face info:
            - color, for machine state prompting
            - rel_pos, local face center position relative to block root, for position calculation
            - attachable flag, for machine state prompting
            - block geometry, for position calculation
            - center, absolute global center position of the face
            - normal, normal orientation of the face
        """
        self.color = color
        self.rel_pos = Vector(rel_pos)
        self.face_normal = Vector(face_normal)
        self.block_geo = block_geo
        self.sticky: bool = True
        self.att_to: Face = None
        # Attributes of parent block for machine state prompting
        self.local_id = local_id
        self.name = name
        
    def update_block_geo(self, block_geo: Geometry):
        """Update the block geometry"""
        self.block_geo = block_geo
    
    @property
    def caption(self):
        # return str(f'Face center position: {np.array2string(self.center.real, precision=3, separator=", ", suppress_small=True)}, Face normal orientation: {np.array2string(self.normal.vec_abs.real, precision=3, separator=", ", suppress_small=True)}, Face label: {self.color}')
        return str(f'Face label: {self.color}, Face center: {np.array2string(self.center.real, precision=3, separator=", ", suppress_small=True)}, Facing towards {self.normal.caption}')
    
    @property
    def center(self) -> Vector:
        center = Vector(self.block_geo.rotation.rot_mat @ (self.rel_pos.virtual * self.block_geo.shape.virtual) + self.block_geo.position.virtual)
        return center
    
    @property
    def normal(self) -> Orientation:
        normal = Orientation(rot = self.face_normal.virtual, vec_base = self.block_geo.rotation.vec_base).rotate(self.block_geo.rotation.rot_mat)
        return normal
        
def describe_orientation(vector: Vector) -> str:
    """
    Convert a 3D vector to a human-readable angular description
    
    Parameters:
        orientation (Vector): The orientation vector.

    Returns:
        str: A description in angular coordinates.
    """
    x, y, z = vector.norm.real
    
    #Handle cases where x and y are very small compared to z after normalization
    threshold = 1e-6
    if abs(x) < threshold and abs(y) < threshold:
        if z > 0:
            return "<straight up>"
        elif z < 0:
            return "<straight down>"

    # Compute horizontal (azimuth) angle in degrees
    azimuth_rad = np.arctan2(x, y)  # atan2(x, y) gives the angle from north
    azimuth_deg = np.degrees(azimuth_rad) % 360  # Convert to degrees and wrap to [0,360)

    # Compute pitch angle in degrees
    r = np.sqrt(x**2 + y**2)  # Horizontal component
    pitch_rad = np.arctan2(z, r)
    pitch_deg = np.degrees(pitch_rad)

    # Hide the azimuth degree if it's exactly aligned with a main direction
    if azimuth_deg % 90 <= 1e-6 or abs(azimuth_deg % 90 - 90) <= 1e-6:
        # Round to the nearest main direction
        azimuth_deg = round(azimuth_deg / 90) * 90
        # Define compass directions
        directions = [
            "North", "East", "South", "West"
        ]
        index = int(azimuth_deg // 90) % 4
        compass_direction = directions[index]
        direction_text = compass_direction
    else:
        # Define compass directions
        directions = [
            "N {deg:.1f}° E", "E {deg:.1f}° S", "S {deg:.1f}° W", "W {deg:.1f}° N"
        ]
        index = int(azimuth_deg // 90) % 4  # Nearest 45-degree sector
        azimuth_deg = azimuth_deg % 90  # Angle within the sector
        direction_text = directions[index].format(deg=azimuth_deg)

    return f"<{direction_text} with {pitch_deg:.1f}° pitch>"

def angle_between_vectors(a: Vector, b: Vector) -> float:
    """Compute the angle in degrees between two vectors a and b for environment prompting (real)."""
    a = a.real
    b = b.real
    # Compute the dot product
    dot_product = np.dot(a, b)
    # Compute the magnitudes (norms)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    # Compute the cosine of the angle
    cos_theta = dot_product / (norm_a * norm_b)
    # Ensure numerical stability
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    # Compute the angle in radians
    angle_rad = np.arccos(cos_theta)
    # Convert to degrees if needed
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def describe_spin(rotation_vector: Vector) -> str:
    """
    Converts a rotation vector [x, y, z] into an intuitive "Rolling towards" or "Spinning" description with tilt angle for environment prompting (real).
    
    Parameters:
        rotation (Vector): The spin direction vector determined by the right-hand rule (which is confusing but makes computing easier).

    Returns:
        str: A description of the rolling motion, spinning state, and tilt angle.
    """
    rotation_array = rotation_vector.real
    gravity = np.array([0, 0, -1], dtype=np.float64)

    # Compute tilt angle away from vertical
    alpha_deg = angle_between_vectors(rotation_array, gravity)
    tilt_angle_deg = 90.0 - alpha_deg

    # Compute rolling direction
    rolling_array = np.cross(rotation_array, gravity)
    # NOTE: Encapsulate the array into Vector for captioning would cause the y-z flip, so we do it twice to correct it
    rolling_vector = Vector(Vector(rolling_array).real)

    # Mostly vertical
    if abs(tilt_angle_deg) < 60:  
        return f"Rolling towards {rolling_vector.caption} with {tilt_angle_deg:.0f}° camber"
    # Mostly horizontal
    else: 
        direction = "counterclockwise" if rotation_array[2] > 0 else "clockwise"
        return f"Almost horizontal spinning {direction} around with the axis pointing at {rotation_vector.caption}"
        # return f"Almost horizontal spinning {direction} around with a {abs(90 - abs(tilt_angle_deg)):.1f}° tilt"
        
def main():
    rotation_vector = Vector([1, 0, 0])
    print(rotation_vector.real)
    des = describe_spin(rotation_vector)
    print(des)
    
if __name__ == "__main__":
    main()