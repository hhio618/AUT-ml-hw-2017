#
# Created by Homayoun Heidarzadeh
#
import numpy as np


# Efficient row-wise distances

class EuclideanRowWise:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        x = args[0]
        matrix = args[1]
        return np.sum(np.abs(x - matrix) ** 2, axis=-1) ** (1. / 2)

    def __repr__(self):
        return "euclidean"


class ManhattanRowWise:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        x = args[0]
        matrix = args[1]
        return np.sum(np.abs(x - matrix), axis=-1)

    def __repr__(self):
        return "manhattan"


class CosineRowWise:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        x = args[0]
        matrix = args[1]
        return 1 - np.sum(matrix * x, -1) / ((np.sum(matrix ** 2, axis=-1) ** (1. / 2)) * np.linalg.norm(x))

    def __repr__(self):
        return "Cosine"


class MinkowskiRowWise:
    def __init__(self, p):
        self.p = p

    def __call__(self, *args, **kwargs):
        x = args[0]
        matrix = args[1]
        return np.sum(np.abs(x - matrix) ** self.p, axis=-1) ** (1. / self.p)

    def __repr__(self):
        return "Minkowski(p=%1.1f)" % self.p
