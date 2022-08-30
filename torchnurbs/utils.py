import torch


def generate_knot_vector(degree, num_control_points, dtype=torch.float32, device=None):
    """Generates the knots appropriate for a uniform spline with clamped ends"""
    if num_control_points < degree + 1:
        raise ValueError(
            "Too few control points. Degree {} requires at least {} control points".format(
                degree,
                degree + 1))

    return torch.cat((
        torch.zeros(degree.item(), dtype=dtype, device=device),
        torch.linspace(0, 1, num_control_points - degree + 1, dtype=dtype, device=device),
        torch.ones(degree.item(), dtype=dtype, device=device),
    ), 0)


def normalise_knot_vector(knot_vector, dtype=None, device=None):
    """
    Normalises the given knot_vector so all values satisfy are in [0, 1], then validates the vector
    length and that all values are ascending.
    """
    knot_vector = to_tensor(knot_vector, dtype=dtype, device=device)
    knot_vector /= knot_vector[..., -1:] - knot_vector[..., :1]

    if torch.min(knot_vector[1:] - knot_vector[:-1]) < 0:
        raise ValueError("Knot vector values [x_0, x_1, ...] must satisfy x_{i-1} <= x_i")

    return knot_vector


def to_tensor(value, dtype=None, device=None, copy=False):
    """Returns `value` as a torch.Tensor - detaching and copying if instructed"""
    if isinstance(value, torch.Tensor):
        if copy:
            return value.detach().clone()
        return value
    return torch.tensor(value, dtype=dtype, device=device)
