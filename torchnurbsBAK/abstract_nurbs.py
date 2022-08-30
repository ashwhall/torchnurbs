from abc import ABCMeta, abstractmethod
from torch import nn
from functools import lru_cache
import torch
from .utils import generate_knot_vector, normalise_knot_vector, to_tensor
from .basis_functions import generate_basis_functions
from .spans import find_spans, span_points_2d
from .evaluate import do_eval


class Setter(object):
    def __init__(self, func):
        self.func = func
        self.var_name = func.__name__

    def __set__(self, obj, value):
        self.obj = obj
        return self.func(obj, value, self.set_var)

    def __get__(self, _, __):
        return self.obj.__dict__.get(self.var_name, None)

    def set_var(self, value):
        self.obj.__dict__[self.var_name] = value


class AbstractNurbs(nn.Module, metaclass=ABCMeta):
    _DEFAULT_EVAL_DELTA = 0.05

    _dtype = torch.float32
    _device = torch.device('cpu')
    _multithread = True

    _ndim = None
    def __init__(self, degree, control_points, knot_vector=None, eval_delta=_DEFAULT_EVAL_DELTA):
        super().__init__()
        self.degree = degree
        self.control_points = control_points
        self.knot_vector = knot_vector
        self.eval_delta = eval_delta

    def evaluate(self, parameters):
        spans = self._get_spans(parameters)
        basis_functions = self._get_basis_functions(spans, parameters)
        points_spans = self._get_point_spans(spans)
        return do_eval(
            basis_functions,
            points_spans,
            self.degree
        )



    def forward(self):
        return self.evaluate(self.eval_parameters)

    @abstractmethod
    def param_dimensions(self):
        pass

    @Setter
    def degree(self, degree, set_var):
        if isinstance(degree, int):
            degree = [degree] * self.param_dimensions()
        degree = to_tensor(degree, dtype=torch.int32, device=self._device)
        if any(d < 1 or d % 1 != 0 for d in degree):
            raise ValueError("Degree must be a natural number")
        set_var(degree)

    def get_dtype(self):
        return self._dtype

    @Setter
    def dtype(self, dtype, set_var):
        set_var(dtype)

    def to(self, device, **kwargs):
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        super().to(device, **kwargs)

    @Setter
    def knot_vector(self, knot_vector, set_var):
        if knot_vector is None:
            # Generate a knot vector for each
            knot_vector = tuple(
                generate_knot_vector(self.degree[i], self.control_points.shape[i])
                for i in torch.arange(self.param_dimensions())
            )
        else:
            knot_vector = tuple(
                to_tensor(dim_knot, dtype=self._dtype, device=self._device)
                for dim_knot in knot_vector
            )

        # Normalise and validate the knot vector(s)
        knot_vector = tuple(
            normalise_knot_vector(self.degree, dim_knot, self._dtype)
            for dim_knot in knot_vector
        )

        if knot_vector is None:
            raise ValueError(f"Knot vector is invalid for degree={self.degree}")
        set_var(knot_vector)

    @Setter
    def eval_delta(self, eval_delta, set_var):
        if isinstance(eval_delta, (float, int)):
            eval_delta = [eval_delta] * self.param_dimensions()
        if len(eval_delta) != self.param_dimensions():
            raise ValueError(
                f"Evaluation deltas must be a scalar or a list of length {self.param_dimensions()}"
            )
        if not all(0 < delta < 1 for delta in eval_delta):
            raise ValueError("Evaluation deltas must be between 0.0 and 1.0")
        set_var(eval_delta)

        # Build a linspace in [0, 1] for each of the evaluation deltas
        eval_parameters = tuple(torch.linspace(0, 1, int(1 / delta) + 1, dtype=self._dtype) for delta in eval_delta)
        self.eval_parameters = eval_parameters

    @Setter
    def eval_parameters(self, eval_parameters, set_var):
        eval_parameters = tuple(to_tensor(t, dtype=self._dtype, device=self._device) for t in eval_parameters)
        set_var(eval_parameters)

    @Setter
    def control_points(self, control_points, set_var):
        control_points = to_tensor(control_points, dtype=self._dtype, device=self._device)
        control_points = nn.Parameter(control_points, requires_grad=True)
        ndim = control_points.shape[-1]
        set_var(control_points)
        self.ndim = ndim

    @Setter
    def ndim(self, ndim, set_var):
        if not isinstance(ndim, int):
            raise ValueError("ndim must be a positive integer")
        if self.control_points is not None and self.control_points.shape[-1] != ndim:
            raise ValueError("ndim must match the shape of control points")
        set_var(ndim)

    @lru_cache(maxsize=2)
    def _get_basis_functions_cached(self, degree, knot_vector, spans, eval_parameters):
        return generate_basis_functions(degree,
                                        knot_vector,
                                        spans,
                                        eval_parameters,
                                        multithread=self._multithread)

    def _get_basis_functions(self, spans, eval_parameters):
        return tuple(
            self._get_basis_functions_cached(int(degree), dim_knot, dim_spans, dim_eval_parameters)
            for degree, dim_knot, dim_spans, dim_eval_parameters in zip(self.degree, self.knot_vector, spans, eval_parameters)
        )

    @lru_cache(maxsize=2)
    def _get_point_spans_cached(self, spans, control_points, degree):
        return span_points_2d(spans, control_points, degree, multithread=self._multithread)

    def _get_point_spans(self, spans):
        return self._get_point_spans_cached(spans, self.control_points, self.degree)

    @lru_cache(maxsize=2)
    def _get_spans_cached(self, degree, knot_vector, control_points, eval_parameters, dim):
        return find_spans(degree[dim], knot_vector, control_points.shape[dim], eval_parameters)

    def _get_spans(self, eval_parameters):
        return tuple(
            self._get_spans_cached(self.degree, dim_knot, self.control_points, dim_eval_parameters, dim)
            for dim, (dim_knot, dim_eval_parameters) in enumerate(zip(self.knot_vector, eval_parameters))
        )

    def enable_multithreading(self):
        self._multithread = True

    def disable_multithreading(self):
        self._multithread = False
