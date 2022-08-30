from typing import List
import torch

def _recurse_basis_function(degree, knot_vector, span, knot):
    # print("degree:      ", degree)
    # print("knot_vector: ", knot_vector)
    # print("span:        ", span)
    # print("knot:        ", knot)
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


def generate_basis_functions(degree, knots, spans, eval_parameters, multithread=False):
    if multithread:
        basis_functions = _basis_function_multithread(
            torch.tensor(degree, dtype=torch.int32),
            knots,
            spans,
            eval_parameters,
        )
    else:
        basis_functions = _basis_function(degree, knots, spans, eval_parameters)
    return torch.stack(basis_functions, 0)
