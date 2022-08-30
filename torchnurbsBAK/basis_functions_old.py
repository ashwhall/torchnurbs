import numpy as np


def find_span_linear(knot_vector, num_ctrlpts, knot):
    span = 0
    while span < num_ctrlpts and knot_vector[span] <= knot:
        span += 1
    return span - 1


def gen_basis_function_a2_2(degree, knot_vector, span, knot):
    left = [0.0 for _ in range(degree + 1)]
    right = [0.0 for _ in range(degree + 1)]
    N = [1.0 for _ in range(degree + 1)]  # N[0] = 1.0 by definition

    for j in range(1, degree + 1):
        left[j] = knot - knot_vector[span + 1 - j]
        right[j] = knot_vector[span + j] - knot
        saved = 0.0
        for r in range(0, j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved

    return N

def gen_basis_function_a2_2_b(degree, knots, spans, knot_linspace):
    basis = []
    for span, knot in zip(spans, knot_linspace):
        basis.append(gen_basis_function_a2_2(degree, knots, span, knot))
    return basis


def find_spans_b(degree, knots, num_ctrlpts, t_vals):
    spans = []
    for knot in t_vals:
        spans.append(find_span_linear(knots, num_ctrlpts, knot))
    return spans

def evaluate_algo(control_points, knot_vector, degree, t_vals):
    spans = find_spans_b(degree, knot_vector, len(control_points), t_vals)
    N = gen_basis_function_a2_2_b(degree, knot_vector, spans, t_vals)

    points = []
    # Iterate over each time step
    for i in range(len(t_vals)):
        out_point = np.array([0, 0], dtype=np.float32)
        # Iterate over the points that influence this time step
        for d in range(degree + 1):
            out_point += control_points[spans[i] + d - degree] * N[i][d]
        points.append(out_point)
    return points

control_points = np.array([
    [1, 1],
    [3, 7],
    [5, 3],
    [6, 7],
    [8, 10],
    [10, 8]
])
degree = 0
def gen_knot_vector(degree, num_control_points):
    if num_control_points < degree + 1:
        raise ValueError(
            "Too few control points. Degree {} requires at least {} control points".format(
                degree,
                degree + 1))
    return np.concatenate((
        [0] * degree,
        np.linspace(0, 1, num_control_points - degree + 1),
        [1] * degree,
    ), 0)

knots = gen_knot_vector(degree, len(control_points))

eval_dom = np.linspace(0, 1, 100)
eval_path = np.array(evaluate_algo(control_points, knots, degree, eval_dom))


import matplotlib.pyplot as plt
control_points = control_points.T
eval_path = eval_path.T
plt.plot(*control_points, 'g*')
plt.plot(*eval_path, 'b-')
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.show()
