#
# Created by Homayoun Heidarzadeh
#
import numpy as np


# Efficient data lifting
class PhiIdentity:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return args[0]

    def __repr__(self):
        return "\"Phi(x1, x2 --> x1, x2)\""


class Phi1:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        X = args[0]
        return np.atleast_2d(X[:, 0] ** 2 + X[:, 1] ** 2).T

    def __repr__(self):
        return "\"Phi(x1, x2 --> x1^2 + x2^2)\""


class Phi2:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        X = args[0]
        return X ** 2

    def __repr__(self):
        return "\"Phi(x1, x2 --> x1^2, x2^2)\""


class Phi3:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        X = args[0]
        return np.c_[X ** 2, np.multiply(X[:, 0], X[:, 1])]

    def __repr__(self):
        return "\"Phi(x1, x2 --> x1^2, x2^2, x1.x2)\""
