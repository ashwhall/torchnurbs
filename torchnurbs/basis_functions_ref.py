import time
from typing import List
import torch


def gen_basis_function(degree, knot_vector, span, knot):
    left = torch.zeros(degree + 1, dtype=torch.float32)
    right = torch.zeros(degree + 1, dtype=torch.float32)
    N = torch.ones(degree + 1, dtype=torch.float32)

    for j in torch.arange(1, degree + 1):
        left[j] = knot - knot_vector[span + 1 - j]
        right[j] = knot_vector[span + j] - knot
        saved = 0.0
        for r in torch.arange(0, j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved

    return N

def basis_function(degree, knots, spans, knot_linspace):
    basis = []
    for span, knot in zip(spans, knot_linspace):
        basis.append(gen_basis_function(degree, knots, span, knot))
    return basis


def gen_basis_function_torch(degree, knot_vector, span, knot):
    left = torch.zeros(degree + 1, dtype=torch.float32)
    right = torch.zeros(degree + 1, dtype=torch.float32)
    N = torch.ones(degree + 1, dtype=torch.float32)

    for j in torch.arange(1, degree + 1):
        left[j] = knot - knot_vector[span + 1 - j]
        right[j] = knot_vector[span + j] - knot
        saved = 0.0
        for r in torch.arange(0, j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved

    return N


@torch.jit.script
def basis_function_multithread(degree,
                               knots,
                               spans,
                               knot_linspace):
    futs: List[torch.jit.Future[torch.Tensor]] = []
    for span, knot in zip(spans, knot_linspace):
        futs.append(torch.jit.fork(gen_basis_function, degree, knots, span, knot))
    return [torch.jit.wait(fut) for fut in futs]

def do_gen_basis_function(degree, knots, spans, knot_linspace, multithread=False):
    if multithread:
        if isinstance(degree, (int, float)):
            degree = torch.tensor(degree, dtype=torch.int32, device=knots.device)
        basis_functions = basis_function_multithread(
            degree,
            knots,
            spans,
            knot_linspace,
        )
    else:
        basis_functions = basis_function(degree, knots, spans, knot_linspace)
    return torch.stack(basis_functions, 0)


def find_span_linear(knot_vector, num_ctrlpts, knot):
    span = 0
    while span < num_ctrlpts and knot_vector[span] <= knot:
        span += 1
    return span - 1


def find_spans_default(degree, knots, num_ctrlpts, t_vals):
    spans = []
    for knot in t_vals:
        spans.append(find_span_linear(knots, num_ctrlpts, knot))
    return spans


def find_span_linear_c(knot_vector, knot):
    span = 0
    while span < len(knot_vector) - 1 and knot > knot_vector[span]:
        span += 1
    return span - 1

def find_spans_torch(degree, knots, num_ctrlpts, t_vals):
    to_remove = len(knots) - num_ctrlpts
    slice_start = to_remove // 2
    slice_end = to_remove - slice_start
    last_possible = len(knots) - degree - 2
    knots = knots[slice_start:-slice_end - 1]
    dist = (knots[None] - t_vals[:, None])
    dist.sign_()
    dist[dist < 0] = 0
    vals = torch.sum(dist, 1, dtype=torch.int64)
    indices = torch.full((len(t_vals),), fill_value=len(knots) - 2, dtype=torch.int64)
    indices = indices - vals + 1 + slice_start
    indices = torch.min(indices, torch.full_like(indices, last_possible, dtype=torch.int64))
    return indices

def do_eval(N, spans, degree):
    spans = spans - degree
    spans = spans[..., None] + torch.arange(degree + 1)
    print(control_points.shape)
    print(spans.shape)
    selected = torch.stack(([torch.index_select(control_points, 0, s) for s in spans]), 0)
    print(selected.shape)
    points = selected * N[..., None]
    points = points.sum(1)
    return points

# USE_TORCH = False
def evaluate_algo(control_points, knot_vector, degree, t_vals):
    spans = find_spans_torch(degree, knot_vector, len(control_points), t_vals)
    start = time.time()
    N = do_gen_basis_function(degree,
                              knot_vector,
                              spans,
                              t_vals,
                              multithread=True)
    print("Gen basis functions: {:.3f}".format(time.time() - start))

    # Iterate over each time step
    # start = time.time()
    points = do_eval(N, spans, degree)
    # print("Evaluate: {:.3f}".format(time.time() - start))
    return points

control_points = torch.tensor([
    [1, 1],
    [3, 7],
    [5, 3],
    [6, 7],
    [8, 10],
    [10, 8]
], dtype=torch.float32)
control_points = torch.rand((6, 2), dtype=torch.float32) * 10
# degree = 2


def gen_knot_vector(degree, num_control_points, dtype=torch.float32, device=None):
    """TODO: What sort of knot vector? It's in the literature as some standard type"""
    if num_control_points < degree + 1:
        raise ValueError(
            "Too few control points. Degree {} requires at least {} control points".format(
                degree,
                degree + 1))
    return torch.cat((
        torch.zeros(degree, dtype=dtype, device=device),
        torch.linspace(0, 1, num_control_points - degree + 1, dtype=dtype, device=device),
        torch.ones(degree, dtype=dtype, device=device),
    ), 0)


import matplotlib.pyplot as plt
# control_points = control_points.T
plt.plot(*control_points.T, 'g*')

for degree in (1, 2, 3, 4, 5):
    # pass
# for degree in (3, ):
    # print("DEGREEEEEEE:", degree)
    knots = gen_knot_vector(degree, len(control_points))

    eval_dom = torch.linspace(0, 1, 8)
    start = time.time()
    eval_path = evaluate_algo(control_points, knots, degree, eval_dom)
    eval_path = eval_path.T
    print(degree, "{:.3f}".format(time.time() - start))

    plt.plot(*eval_path, label=degree)
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.legend()
plt.show()

# plt.plot(sizes, torch_times, label='Torch')
# plt.plot(sizes, non_torch_times, label='Python')
# plt.legend()
# plt.show()
