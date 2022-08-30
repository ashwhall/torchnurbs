import time
import torchnurbs as tn
import torch

"""Demonstrates a nurbs curve in 2D space"""

control_points = torch.tensor([
    [1, 1],
    [3, 7],
    [5, 3],
    [6, 7],
    [8, 10],
    [9, 5]
], dtype=torch.float32)
print('========')
degree = 2
knot_vector = torch.tensor([
    0, 0, 0, 3, 3, 5, 8, 8, 8
]).float()
c = tn.Curve(
    degree=degree,
    control_points=control_points,
    knot_vector=knot_vector,
    eval_delta=0.03
)
print(c.knot_vector)
print('Eval params:', c.eval_parameters[0].shape)
start = time.time()
steps = 1
for _ in range(steps):
    points = c()
    break
print(steps / (time.time() - start))


points = points.T.detach().cpu().numpy()
print(points.shape)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print('matplotlib not installed')
    exit(0)

plt.plot(*control_points.T, 'g*')
plt.plot(*points, label='Hello!')
plt.legend()
plt.show()
