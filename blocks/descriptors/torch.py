from spatial.build import Block
from spatial.utils import format_float_array

def descriptor(self: Block):
    geo = self.geo
    vec = geo.rotation.vec_abs.real
    flame_pos = vec + self.center_pos.real
    return f"Heating up a spherical area with radius 0.3 around {format_float_array(flame_pos)}"
