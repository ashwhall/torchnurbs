import torchnurbs as tn
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

"""Demonstrates a nurbs surface in 3D space"""

d = {
    3: 'ndim',
    5: 'ctrl_u',
    10: 'ctrl_v',
    4: 'eval_u',
    6: 'eval_v',
    2: 'span'
}

def print_shape(n, t):
    if isinstance(t, list):
        shape = [len(i) for i in t]
    else:
        shape = t.shape
    print(n)
    print("\t\t(" + ", ".join([d.get(s, str(s)) for s in shape]) + ")")

degree = (1, 4)
control_points = np.array(
[[[0,0,0],[1,0,0],[2,0,0],[4,0,0],[8,0,0],[16,0,0],[20,0,0],[22,0,0],[25,0,0],[30,0,0]],[[0,-1,2],[1,-1,2],[2,-1,2],[4,-1,2],[8,-1,2],[16,-1,2],[20,-1,2],[22,-1,2],[25,-1,2],[30,-1,2]],[[0,-2,4],[1,-2,4],[2,-2,4],[4,-2,4],[8,-2,4],[16,-2,4],[20,-2,4],[22,-2,4],[25,-2,4],[30,-2,4]],[[0,-4,6],[1,-4,6],[2,-4,6],[4,-4,6],[8,-4,6],[16,-4,6],[20,-4,6],[22,-4,6],[25,-4,6],[30,-4,6]],[[0,-8,8],[1,-8,8],[2,-8,8],[4,-8,8],[8,-8,8],[16,-8,8],[20,-8,8],[22,-7,8],[25,-6,8],[30,-5,8]]]
, dtype=np.float32)

c = tn.Surface(
    degree=degree,
    control_points=control_points,
    # eval_delta=(.1, .12)
    eval_delta=(0.02, 0.01)
)
c.normalise()
c.denormalise()
# c.disable_multithreading()
pts = c()


print(c.to_dict())
1/0

def visualise(points):
    import open3d as o3d
    colours = np.eye(3)

    if not isinstance(points, (list, tuple)):
        points = [points]
    points = [pt.detach().numpy() if isinstance(pt, torch.Tensor) else pt for pt in points]
    pcds = []
    for i, (pts, col) in enumerate(zip(points, colours)):
        pcd = o3d.geometry.PointCloud()
        # print(pts)
        pcd.points = o3d.utility.Vector3dVector(pts)

        pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(pts) + col)
        if i == 1:
            colors = np.zeros_like(pts)
            colors[:, :2] += np.linspace(0, 1, len(pts))[..., None]
            pcd.colors = o3d.utility.Vector3dVector(colors)
        pcds.append(pcd)

    pcds.append(o3d.geometry.TriangleMesh.create_coordinate_frame())
    o3d.visualization.draw_geometries(pcds)

# visualise([control_points.reshape(-1, 3)])
visualise([pts, control_points.reshape(-1, 3)])



