import numpy as np


def load_data(fname, dtype=float, delm=None):
    return np.genfromtxt(fname, dtype=dtype, delimiter=delm)
