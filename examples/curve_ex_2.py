import torchnurbs as tn
import torch

"""Demonstrates a nurbs curve in 3D space"""

control_points = torch.tensor([
    [1, 1, 0],
    [3, 7, 1],
    [5, 3, 2],
    [6, 7, 3],
    [8, 10, 4],
    [10, 8, 5]
], dtype=torch.float32)
print("========")
# control_points = torch.rand((5000, 2), dtype=torch.float32) * 10
degree = 2
# knot_vector = tn.utils.generate_knot_vector(degree, len(control_points))
c = tn.Curve(
    degree=degree,
    control_points=control_points,
    # knot_vector=knot_vector,
    eval_delta=0.1
)
print('Eval params:', c.eval_parameters[0].shape)
import time
start = time.time()
steps = 1
for _ in range(steps):
    points = c()
    break
print(steps / (time.time() - start))


points = points.T.detach().cpu().numpy()
print(points.shape)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(*control_points.T, 'g*')
ax.plot(*points, label='Hello!')
plt.legend()
plt.show()
