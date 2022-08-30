from . import utils, loss, io
from .curve import Curve
from .surface import Surface


def from_dict(the_dict, Class):
    return Class(**the_dict)
