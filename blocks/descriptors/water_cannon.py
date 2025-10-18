from spatial.build import Block
from spatial.utils import format_float_array

def descriptor(self: Block):
    geo = self.geo
    pos = self.center_pos.real
    vec = geo.rotation.vec_abs.real
    outlet_pos = pos + vec * 1
    inlet_pos = pos - vec * 0.75
    return f"Inlet is at {format_float_array(inlet_pos)}, outlet is at {format_float_array(outlet_pos)}, sprays towards {geo.rotation.caption}"
