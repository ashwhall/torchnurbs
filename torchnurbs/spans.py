import torch
from typing import List


def find_span_linear(knot_vector, num_ctrlpts, knot):
    span = 0  # Knot span index starts from zero
    while span < num_ctrlpts and knot_vector[span] <= knot:
        span += 1

    return span - 1

def find_spans_traditional(degree, knots, num_ctrlpts, t_vals):
    # print("\n\n=======================================================================================")
    # print("=======================================================================================")
    # print("=======================================================================================")
    # print("degree", degree)
    # print("knots", len(knots), knots)
    # print("num_ctrlpts", num_ctrlpts)
    # print("t_vals", len(t_vals), t_vals)
    if torch.sum(knots[:degree + 1]) > 0:
        print("Not pinned")
    span_starts = []
    for knot in t_vals:
        span = find_span_linear(knots, num_ctrlpts, knot)
        span = max(span, degree)
        span_starts.append(span)
    span_starts = torch.tensor(span_starts, dtype=torch.long, device=knots.device)
    # print(span_starts)
    return span_starts
    to_remove = len(knots) - num_ctrlpts
    slice_start = to_remove // 2
    slice_end = to_remove - slice_start
    last_possible = len(knots) - degree - 2
    knots = knots[slice_start:-slice_end - 1]
    dist = (knots[None] - t_vals[:, None])
    dist.sign_()
    dist[dist < 0] = 0
    print(dist)
    vals = dist.sum(1).type(torch.int32)
    print(vals)
    print(len(knots) - 2 - vals + 1 + slice_start)
    indices = torch.full((len(t_vals),), fill_value=len(knots) - 2, dtype=torch.int64, device=knots.device)
    indices = indices - vals + 1 + slice_start
    indices = torch.min(indices, torch.full_like(indices, last_possible, dtype=torch.int64, device=knots.device))
    print(indices)
    print("=======================================================================================")
    print("=======================================================================================")
    print("=======================================================================================\n\n")
    return indices

def find_spans(degree, knots, num_ctrlpts, t_vals):
    # print("\n\n=======================================================================================")
    # print("=======================================================================================")
    # print("=======================================================================================")
    # print("degree", degree)
    # print("knots", knots)
    # print("num_ctrlpts", num_ctrlpts)
    # print("t_vals", t_vals)

    to_remove = len(knots) - num_ctrlpts
    slice_start = to_remove // 2
    slice_end = to_remove - slice_start
    last_possible = len(knots) - degree - 2
    knots = knots[slice_start:-slice_end - 1]
    dist = (knots[None] - t_vals[:, None])
    dist.sign_()
    dist[dist < 0] = 0
    # print(dist)
    vals = dist.sum(1).type(torch.int32)
    # print(vals)
    # print(len(knots) - 2 - vals + 1 + slice_start)
    indices = torch.full((len(t_vals),), fill_value=len(knots) - 2, dtype=torch.int64, device=knots.device)
    indices = indices - vals + 1 + slice_start
    indices = torch.min(indices, torch.full_like(indices, last_possible, dtype=torch.int64, device=knots.device))
    # print(indices)
    # print("=======================================================================================")
    # print("=======================================================================================")
    # print("=======================================================================================\n\n")
    return indices


d = {
    3: 'ndim',
    5: 'ctrl_u',
    10: 'ctrl_v',
    4: 'eval_u',
    6: 'eval_v',
    24: 'N',
    2: 'span',
    8: 'eval_u_spanned',
    12: 'eval_v_spanned'
}


def print_shape(n, t):
    if isinstance(t, list):
        shape = tuple(len(i) for i in t)
    else:
        shape = t.shape
    print("{:>18}:".format(n[:18]), "\t(" + ", ".join([d.get(s, str(s)) for s in shape]) + ")")


def select_points_using_spans(control_points, spans, degree, dim):
    spans = spans - degree
    if dim == 0:
        return torch.stack([control_points[s:s+degree+1] for s in spans], 0)
    out = torch.stack([control_points[:, s:s+degree+1] for s in spans], 0)
    return out


def select_points_using_spans_mt_inner(control_points, spans, degree, dim):
    if dim == 0:
        return torch.stack([control_points[s:s+degree+1] for s in spans], 0)
    return torch.stack([control_points[:, s:s+degree+1] for s in spans], 0)


@torch.jit.script
def select_points_using_spans_mt(
        control_points,
        spans,
        degree: torch.Tensor,
        dim: torch.Tensor,
        num_threads: torch.Tensor,
):
    futures: List[torch.jit.Future[torch.Tensor]] = []
    spans = spans - degree
    spans = torch.chunk(spans, num_threads)
    for spans_chunk in spans:
        futures.append(torch.jit.fork(select_points_using_spans_mt_inner, control_points, spans_chunk, degree, dim))

    out = torch.cat([torch.jit.wait(future) for future in futures], 0)
    return out


def select_mt(control_points, spans, degree, dim):
    return select_points_using_spans_mt(
        control_points,
        spans,
        degree,
        torch.tensor(dim, dtype=torch.int32, device=control_points.device),
        torch.tensor(torch.get_num_threads(), dtype=torch.int8, device=control_points.device)
    )
