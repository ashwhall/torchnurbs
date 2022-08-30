from typing import List
import torch


def make_sparse_tensor(N, spans, degree, num_control_points):
    # print("____")
    # print(spans)
    # for n in N:
    #     print(["{:.2f}".format(n_.item()) for n_ in n])
    out_shape = (N.shape[0], num_control_points)
    # print("N", N.shape, "degree:", degree)
    # print("spans", spans)
    # print("spans", spans.shape, torch.arange(0, degree+1)[None].shape)
    spans = spans - degree + torch.arange(0, degree+1, device=N.device)[:, None]
    # print("spans", spans.T)
    spans = spans.T.reshape(-1)
    first_index = torch.arange(N.shape[0], device=N.device)
    # print("first_index", first_index.shape)
    first_index = first_index.repeat_interleave(N.shape[1])
    # print("first_index", first_index)
    indices = torch.stack((first_index, spans), 0)
    # print("indices", indices.shape)
    values = N.reshape(-1)
    # print(spans)
    # print("indices", indices)
    # print(indices.shape, values.shape, out_shape)
    # print(out_shape)
    sparse = torch.sparse_coo_tensor(indices, values, out_shape, dtype=N.dtype, device=N.device)
    # print(torch.max(sparse._values()), torch.max(sparse.to_dense()))
    # print(torch.max(values))
    # for n in sparse:
    #     print(["{:.2f}".format(n_.item()) for n_ in n])
    # print(sparse)
    # print("========================")
    return sparse.to_dense()


def _recurse_basis_function(degree, knot_vector, span, knot):
    left = torch.zeros(degree + 1, dtype=knot_vector.dtype, device=torch.device('cpu'))
    right = torch.zeros(degree + 1, dtype=knot_vector.dtype, device=torch.device('cpu'))
    N = torch.ones(degree + 1, dtype=knot_vector.dtype, device=torch.device('cpu'))

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


def _basis_function(degree, knots, spans, eval_parameters):
    basis = []
    for span, knot in zip(spans, eval_parameters):
        basis.append(_recurse_basis_function(degree, knots, span, knot))
    return basis


@torch.jit.script
def _basis_function_multithread(degree,
                                knots,
                                spans,
                                eval_parameters):
    futures: List[torch.jit.Future[torch.Tensor]] = []
    for span, knot in zip(spans, eval_parameters):
        futures.append(torch.jit.fork(_recurse_basis_function, degree, knots, span, knot))
    return [torch.jit.wait(future) for future in futures]


def generate_basis_functions(degree, knots, spans, eval_parameters, num_control_points, multithread=False):
    if multithread:
        basis_functions = _basis_function_multithread(
            torch.tensor(degree, dtype=torch.int32, device=torch.device('cpu')),
            knots,
            spans,
            eval_parameters,
        )
    else:
        basis_functions = _basis_function(degree, knots, spans, eval_parameters)

    basis_functions = torch.stack(basis_functions, 0)
    return make_sparse_tensor(basis_functions, spans, degree, num_control_points)
