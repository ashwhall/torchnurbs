from abc import ABCMeta, abstractmethod
from torch import nn
from methodtools import lru_cache
import torch
from .utils import generate_knot_vector, normalise_knot_vector, to_tensor
from .basis_functions import generate_basis_functions
from .spans import find_spans
from .evaluate import evaluate
from .loss import _compute_dist_mat


class AbstractNurbs(nn.Module, metaclass=ABCMeta):
    _DEFAULT_EVAL_DELTA = 0.05

    _dtype = torch.float32
    _device = torch.device('cpu')
    _multithread = True

    _ndim = None

    def __init__(self,
                 degree,
                 control_points,
                 knot_vector=None,
                 eval_delta=_DEFAULT_EVAL_DELTA,
                 device=torch.device('cpu'),
                 detached_control_points=False):
        super().__init__()
        self._device = device
        self._detached_control_points = detached_control_points
        self.set_degree(degree)
        self.set_control_points(control_points)
        self.set_knot_vector(knot_vector)
        self.set_eval_delta(eval_delta)
        self.normalised = False
        self.control_points_tensor = None  # If exists, is used instead of self.control_points

    def to_dict(self):
        return {
            "degree": self.degree.detach().cpu().numpy().tolist(),
            "control_points": self.control_points.detach().cpu().numpy().tolist(),
            "knot_vector": [v.detach().cpu().numpy().tolist() for v in self.knot_vector],
            "eval_delta": self.eval_delta
        }

    def evaluate(self, parameters):
        spans = self._get_spans(parameters)
        basis_functions = self._get_basis_functions(spans, parameters)
        return evaluate(
            basis_functions,
            self.get_control_points()
        )

    def get_control_points(self):
        return self.control_points if self.control_points_tensor is None else self.control_points_tensor

    def get_min_max(self):
        return self.control_points.min(), self.control_points.max()

    def normalise(self, dim_min, dim_max):
        if self.normalised:
            print("WARNING: normalise called on a normalised surface")
        else:
            self.set_control_points((self.control_points - dim_min) / (dim_max - dim_min))
            self.dim_min, self.dim_max = dim_min, dim_max
        self.normalised = True

    def denormalise(self, dim_min=None, dim_max=None):
        if not self.normalised:
            print("WARNING: denormalise called on an unnormalised surface")
        else:
            dim_min = dim_min or self.dim_min
            dim_max = dim_max or self.dim_max
            self.set_control_points((dim_max - dim_min) * self.control_points + dim_min)
        self.normalised = False

    def forward(self):
        return self.evaluate(self.eval_parameters)

    @abstractmethod
    def param_dimensions(self):
        pass

    def set_degree(self, degree):
        if isinstance(degree, int):
            degree = [degree] * self.param_dimensions()
        degree = to_tensor(degree, dtype=torch.int32, device=self._device)
        if any(d < 1 or d % 1 != 0 for d in degree):
            raise ValueError("Degree must be a natural number")
        self.degree = degree

    def reset_state(self):
        self.control_points_tensor = None

    def get_dtype(self):
        return self._dtype

    def to(self, device, **kwargs):
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        super().to(device, **kwargs)
        return self

    def set_knot_vector(self, knot_vector):
        if knot_vector is None:
            # Generate a knot vector for each
            knot_vector = tuple(
                generate_knot_vector(
                    self.degree[i], self.control_points.shape[i], dtype=self._dtype, device=self._device)
                for i in torch.arange(self.param_dimensions())
            )
        else:
            if not isinstance(knot_vector, tuple):
                knot_vector = (knot_vector,)
            knot_vector = tuple(
                to_tensor(dim_knot, dtype=self._dtype, device=self._device)
                for dim_knot in knot_vector
            )

        # Normalise and validate the knot vector(s)
        knot_vector = tuple(
            normalise_knot_vector(dim_knot, self._dtype, self._device)
            for dim_knot in knot_vector
        )

        if knot_vector is None:
            raise ValueError(f"Knot vector is invalid for degree={self.degree}")
        self.knot_vector = knot_vector

    def set_eval_delta(self, eval_delta):
        if isinstance(eval_delta, (float, int)):
            eval_delta = [eval_delta] * self.param_dimensions()
        if len(eval_delta) != self.param_dimensions():
            raise ValueError(
                f"Evaluation deltas must be a scalar or a list of length {self.param_dimensions()}"
            )
        if not all(0 < delta < 1 for delta in eval_delta):
            raise ValueError("Evaluation deltas must be between 0.0 and 1.0")
        self.eval_delta = eval_delta

        # Build a linspace in [0, 1] for each of the evaluation deltas
        eval_parameters = tuple(torch.linspace(0, 1, int(1 / delta) + 1,
                                dtype=self._dtype, device=self._device) for delta in eval_delta)
        self.set_eval_parameters(eval_parameters)

    def set_eval_parameters(self, eval_parameters):
        self.eval_parameters = tuple(
            to_tensor(t, dtype=self._dtype, device=self._device) for t in eval_parameters)

    def set_control_points(self, control_points):
        control_points = to_tensor(control_points, dtype=self._dtype, device=self._device)
        if not self._detached_control_points:
            control_points = nn.parameter.Parameter(control_points, requires_grad=True)
        ndim = control_points.shape[-1]
        self.control_points = control_points
        self.ndim = ndim

    def set_ndim(self, ndim):
        if not isinstance(ndim, int):
            raise ValueError("ndim must be a positive integer")
        if self.control_points is not None and self.control_points.shape[-1] != ndim:
            raise ValueError("ndim must match the shape of control points")
        self.ndim = ndim

    def get_control_surface_triangles(self):
        assert self.param_dimensions() == 2, "Only valid for surfaces"

        vertices = self.get_control_points().detach()
        grid_shape = vertices.shape[:-1]
        vertices = vertices.reshape(-1, 3)
        triangles = []

        for row in range(grid_shape[0] - 1):
            for col in range(grid_shape[1] - 1):
                top_left = row * grid_shape[1] + col
                top_right = top_left + 1
                bottom_left = (row + 1) * grid_shape[1] + col
                bottom_right = bottom_left + 1
                triangles.append((top_left, bottom_left, top_right))
                triangles.append((top_right, bottom_left, bottom_right))
        return vertices, torch.tensor(triangles, dtype=torch.long, device=self._device)

    def get_approx_area(self):
        """
        Computes the area of the control surface, using Heron's triangle area calculation on each
        of the triangles of which the control surface is composed.
        """
        assert self.param_dimensions() == 2, "Area only valid for surfaces"
        vertices, triangles = self.get_control_surface_triangles()
        corner_a = vertices[triangles[:, 0]]
        corner_b = vertices[triangles[:, 1]]
        corner_c = vertices[triangles[:, 2]]

        side_a = ((corner_b - corner_a) ** 2).sum(-1).sqrt()
        side_b = ((corner_c - corner_a) ** 2).sum(-1).sqrt()
        side_c = ((corner_c - corner_b) ** 2).sum(-1).sqrt()

        s = (side_a + side_b + side_c) / 2
        areas = (s * (s - side_a) * (s - side_b) * (s - side_c)).sqrt()
        return areas.sum()

    @lru_cache(maxsize=2)
    def _get_basis_functions_cached(self, degree, knot_vector, spans, eval_parameters, num_control_points):
        # This functions runs on the CPU as it's more efficient, and rarely called
        return generate_basis_functions(degree,
                                        knot_vector.to(torch.device('cpu')),
                                        spans.to(torch.device('cpu')),
                                        eval_parameters.to(torch.device('cpu')),
                                        num_control_points,
                                        multithread=self._multithread).to(knot_vector.device)

    def _get_basis_functions(self, spans, eval_parameters):
        return tuple(
            self._get_basis_functions_cached(
                int(degree),
                dim_knot,
                dim_spans,
                dim_eval_parameters,
                self.control_points.shape[idx]
            )
            for idx, (degree, dim_knot, dim_spans, dim_eval_parameters) in enumerate(zip(self.degree, self.knot_vector, spans, eval_parameters))
        )

    @lru_cache(maxsize=2)
    def _get_spans_cached(self, degree, knot_vector, control_points, eval_parameters, dim):
        return find_spans(degree[dim], knot_vector, control_points.shape[dim], eval_parameters)

    def _get_spans(self, eval_parameters):
        return tuple(
            self._get_spans_cached(self.degree, dim_knot, self.control_points,
                                   dim_eval_parameters, dim)
            for dim, (dim_knot, dim_eval_parameters) in enumerate(zip(self.knot_vector, eval_parameters))
        )

    def enable_multithreading(self):
        self._multithread = True

    def disable_multithreading(self):
        self._multithread = False
