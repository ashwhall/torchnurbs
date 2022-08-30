import torch
from .utils import combine_first_two_dims
from typing import List


def find_spans(degree, knots, num_ctrlpts, t_vals):
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
        torch.tensor(dim, dtype=torch.int32),
        torch.tensor(torch.get_num_threads(), dtype=torch.int8)
    )


def span_points_2d(spans, control_points, degree, multithread=False):
    spans_u = spans[0]
    degree_u = degree[0]
    select_func = select_mt if multithread else select_points_using_spans
    selected = select_func(control_points, spans_u, degree_u, 0)

    selected = combine_first_two_dims(selected)
    if selected.ndim == 2:
        # It's a curve, not a surface
        return selected

    spans_v = spans[1]
    degree_v = degree[1]
    selected = select_func(selected, spans_v, degree_v, 1)

    selected = selected.permute(0, 2, 1, 3)
    selected = combine_first_two_dims(selected)
    selected = selected.permute(1, 0, 2)
    return selected
