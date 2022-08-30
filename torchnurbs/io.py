import json
import numpy as np

from .surface import Surface


def load_json_file(path, device, detached_control_points=False):
    """Load parametric curves/surfaces from a JSON file formatted using on2json.exe"""
    def _parse_dict(obj, obj_type):
        if obj_type == "surface":
            control_points = np.array(obj["control_points"]["points"], dtype=np.float32)
            degree_u, degree_v = obj["degree_u"], obj["degree_v"]
            knot_vector_u, knot_vector_v = np.array(obj["knotvector_u"]), np.array(
                obj["knotvector_v"])
            size_u, size_v = obj["size_u"], obj["size_v"]

            assert len(control_points) == size_u * size_v
            assert len(knot_vector_u) == size_u + degree_u + 1
            assert len(knot_vector_v) == size_v + degree_v + 1

            return Surface(
                degree=(degree_u, degree_v),
                control_points=control_points.reshape(size_u, size_v, 3),
                knot_vector=(knot_vector_u, knot_vector_v),
                device=device,
                detached_control_points=detached_control_points
            )
        raise ValueError(f"Unsupported object type: {obj_type}")

    with open(path, "r") as f_handle:
        data = json.load(f_handle)

    data = data["shape"]
    obj_type = data["type"]
    loaded = [_parse_dict(entry, obj_type) for entry in data["data"]]
    return loaded
