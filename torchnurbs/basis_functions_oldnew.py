import torch
from typing import List



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





def gen_basis_function_a2_c(degree, knot_vector, span, knot):
    # left = [0.0 for _ in range(degree + 1)]
    # right = [0.0 for _ in range(degree + 1)]
    # N = [1.0 for _ in range(degree + 1)]  # N[0] = 1.0 by definition
    left = torch.zeros(degree + 1, dtype=torch.float32)
    right = torch.zeros(degree + 1, dtype=torch.float32)
    N = torch.ones(degree + 1, dtype=torch.float32)
    NB = torch.ones(degree + 1, dtype=torch.float32)
    # N = torch.empty(degree + 1)
    for j in range(1, degree + 1):
        left[j] = knot - knot_vector[span + 1 - j]
        right[j] = knot_vector[span + j] - knot
        saved = 0
        temp1b = N[:j]
        temp2b = right[1:j+1]
        # print(temp1b.shape, temp2b.shape)
        temp3b = left[1:j+1].flip(0)
        # print("...")
        # print(temp3b)
        temp4b = temp2b + temp3b
        tempb = temp1b / temp4b

        temp5b = right[1:j+1]
        # print("...____...")
        # print('...', temp5b)
        # N[:j] = saved + temp5b + tempb
        # saved = temp3b * tempb
        temp6b = temp5b * tempb
        savedb = torch.zeros_like(temp5b)
        savedb = temp3b + tempb
        # savedb = torch.cat((savedb[:1], temp6b), 0)
        # print('RAW:', temp3b * tempb)
        NB[j] = (savedb + temp6b)[-1]
        # print("OUT NEW:", savedb[-1])
        savedgroup = torch.zeros((1,), dtype=torch.float32)
        savedgroup_part_2 = temp3b * tempb
        print(savedgroup, savedgroup_part_2)
        savedgroup = torch.cat((savedgroup, savedgroup_part_2), 0)
        savedgroup = savedgroup + temp6b
        # for temp3, tempp in zip(temp3b, tempb):
        #     savedgroup.append(temp3 * tempp)
        print(savedgroup)
        # print("NEW LOOP>>>>")
        saved_oldway = [0]
        for r, temp, temp5, temp3, temp6, savedc, savedg in zip(range(0, j), tempb, temp5b, temp3b, temp6b, savedb, savedgroup):
            # continue
            # temp1 = N[r]
            # temp2 = right[r + 1]
            # temp3 = left[j - r]
            # temp4 = temp2 + temp3
            # temp = temp1 / temp4
            # temp = N[r] / (right[r + 1] + left[j - r])
            # temp5 = right[r + 1]
            # print(temp5)
            N[r] = savedg + temp6
            # N[r] = saved + right[r + 1] * temp
            # temp6 = left[j - r]
            # saved = temp3 * temp
            # saved_oldway += [saved]
            # print('Saved:', saved)
            # print('SavedC:', savedc )
            # saved = left[j - r] * temp
        print(N.shape, savedgroup.shape)
        N[j] = savedgroup[-1]
        N[:j] = savedgroup
        print("GROUP: ", savedgroup)
        print("OLDWAY:", saved_oldway)
        # print('    OLD:', saved)
    print("__________________")
    print(N)
    print(NB)
    # import numpy as np
    # print("Exit...")
    # print(N)
    # print(np.array([n.numpy() for n in N]).shape)
    return N

def gen_basis_function_a2_2c(degree, knots, spans, knot_linspace):
    basis = []
    for span, knot in zip(spans, knot_linspace):
        basis.append(gen_basis_function_a2_c(degree, knots, span, knot))
    return basis



def gen_basis_function_a2_2_d(degree, knot_vector, span, knot):
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

"""
No Multithread
1 0.847
2 1.532
3 2.330
4 3.300
5 4.570

Multithread
1 0.392
2 0.536
3 0.716
4 0.991
5 2.187
"""
@torch.jit.script
def gen_basis_function_a2_2d_mt(
        degree,
        knots,
        spans,
        knot_linspace,
):
    futs = torch.jit.annotate(List[torch.jit.Future[torch.Tensor]], [])
    # futs = []
    for span, knot in zip(spans, knot_linspace):
        fut = torch.jit.fork(gen_basis_function_a2_2_d, degree, knots, span, knot)
        futs.append(fut)
        # basis.append(gen_basis_function_a2_2_d(degree, knots, span, knot))
    basis = [torch.jit.wait(fut) for fut in futs]
    return basis

def gen_basis_function_a2_2d(degree, knots, spans, knot_linspace):
    basis = []
    for span, knot in zip(spans, knot_linspace):
        basis.append(gen_basis_function_a2_2_d(degree, knots, span, knot))
    return basis

def find_span_linear(knot_vector, num_ctrlpts, knot):
    span = 0
    while span < num_ctrlpts and knot_vector[span] <= knot:
        span += 1
    return span - 1


def find_spans_b(degree, knots, num_ctrlpts, t_vals):
    spans = []
    for knot in t_vals:
        spans.append(find_span_linear(knots, num_ctrlpts, knot))
    return spans


def find_span_linear_c(knot_vector, knot):
    span = 0
    while span < len(knot_vector) - 1 and knot > knot_vector[span]:
        span += 1
    return span - 1

def find_spans_c(degree, knots, num_ctrlpts, t_vals):
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


def evaluate_algo(control_points, knot_vector, degree, t_vals):
    # spans = find_spans_b(degree, knot_vector, len(control_points), t_vals)
    # print('Old:', spans)
    spans = find_spans_c(degree, knot_vector, len(control_points), t_vals)
    # print('New:', spans)
    # N = gen_basis_function_a2_2_b(degree, knot_vector, spans, t_vals)
    # import numpy as np
    # print(np.array(N).shape)
    # print(N)
    # N = gen_basis_function_a2_2c(degree, knot_vector, spans, t_vals)
    N = gen_basis_function_a2_2d(
        torch.tensor(degree, dtype=torch.int32),
        knot_vector,
        spans,
        t_vals,
    )

    points = []
    # Iterate over each time step
    for i in range(len(t_vals)):
        out_point = torch.tensor([0, 0], dtype=torch.float32)
        # Iterate over the points that influence this time step

        for d in range(degree + 1):
            out_point += control_points[spans[i] + d - degree] * N[i][d]
        points.append(out_point)
    points = torch.stack(points, 0)
    return points

control_points = torch.tensor([
    [1, 1],
    [3, 7],
    [5, 3],
    [6, 7],
    [8, 10],
    [10, 8]
], dtype=torch.float32)
# control_points = torch.rand((200, 2), dtype=torch.float32) * 10
# degree = 2


def gen_knot_vector(degree, num_control_points, dtype=torch.float32, device=None):
    if num_control_points < degree + 1:
        raise ValueError(
            "Too few control points. Degree {} requires at least {} control points".format(
                degree,
                degree + 1))
    return torch.cat((
        torch.zeros(degree, dtype=torch.float32, device=device),
        torch.linspace(0, 1, num_control_points - degree + 1, dtype=torch.float32, device=device),
        torch.ones(degree, dtype=torch.float32, device=device),
    ), 0)


import matplotlib.pyplot as plt
# control_points = control_points.T
plt.plot(*control_points.T, 'g*')

import time
for degree in (1, 2, 3, 4, 5):
# for degree in (2, ):
    start = time.time()
    # print("DEGREEEEEEE:", degree)
    knots = gen_knot_vector(degree, len(control_points))

    eval_dom = torch.linspace(0, 1, 5000)
    eval_path = evaluate_algo(control_points, knots, degree, eval_dom)
    eval_path = eval_path.T
    print(degree, "{:.3f}".format(time.time() - start))



    plt.plot(*eval_path, label=degree)
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.legend()
plt.show()
