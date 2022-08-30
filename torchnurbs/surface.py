from .abstract_nurbs import AbstractNurbs

class Surface(AbstractNurbs):
    def param_dimensions(self):
        return 2
