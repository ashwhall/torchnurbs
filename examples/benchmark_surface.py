import torch
import numpy as np
import time
import torchnurbs as tn

"""Benchmark single/multi threaded surface calculation"""

degree = 3
control_points = np.array(
    [[[0, 0, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0], [20, 0, 0], [22, 0, 0], [25, 0, 0], [30, 0, 0]], [[0, -1, 2], [1, -1, 2], [2, -1, 2], [4, -1, 2], [8, -1, 2], [16, -1, 2], [20, -1, 2], [22, -1, 2], [25, -1, 2], [30, -1, 2]], [[0, -2, 4], [1, -2, 4], [2, -2, 4], [4, -2, 4], [8, -2, 4], [16, -2, 4], [20, -2, 4], [22, -2, 4], [25, -2, 4], [30, -2, 4]], [[0, -4, 6], [1, -4, 6], [2, -4, 6], [4, -4, 6], [8, -4, 6], [16, -4, 6], [20, -4, 6], [22, -4, 6], [25, -4, 6], [30, -4, 6]], [[0, -8, 8], [1, -8, 8], [2, -8, 8], [4, -8, 8], [8, -8, 8], [16, -8, 8], [20, -8, 8], [22, -7, 8], [25, -6, 8], [30, -5, 8]]], dtype=np.float32)

# control_points = np.random.rand(100, 100, 3) * 10

runs = 25
# eval_deltas = list(reversed(np.linspace(0.001, 0.1, 20)))
eval_deltas = list(reversed(np.logspace(np.log10(0.001), np.log10(0.1), 20)))
num_points = []
mt_delta_times = []
st_delta_times = []
for times_arr in (mt_delta_times, st_delta_times):
    for eval_delta in eval_deltas:
        c = tn.Surface(
            degree=degree,
            control_points=control_points,
            eval_delta=eval_delta,
            device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
        if times_arr is st_delta_times:
            c.disable_multithreading()
        point_count = np.prod([p.numel() for p in c.eval_parameters])
        num_points.append(point_count)

        start = time.time()
        for _ in range(runs):
            pts = c()
        mean_time = (time.time() - start) / runs
        times_arr.append(point_count / mean_time / 1000000)
        print("{}: {:.2f}".format(point_count, times_arr[-1]))
# print(eval_deltas, times_arr)
# exit(1)
num_points = num_points[:len(mt_delta_times)]
print(num_points)
print(mt_delta_times)
print(st_delta_times)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print('matplotlib not installed')
else:
    plt.plot(num_points, mt_delta_times, label='Multi-thread')
    plt.plot(num_points, st_delta_times, label='Single-thread')
    plt.title("Points Sampled per Second")
    plt.legend()
    plt.xlabel("Num points sampled")
    plt.ylabel("Million points per second")
    plt.show()


def visualise(points):
    try:
        import open3d as o3d
    except ImportError:
        print('open3d not installed')
        return
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
