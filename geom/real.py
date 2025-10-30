import numpy as np


class Real:
    def __init__(self, precision: int = 32):
        if precision not in (16, 32, 64):
            raise ValueError("Precision must be one of 16, 32, or 64.")
        self.precision = precision
        self.dtypes = {
            16: np.float16,
            32: np.float32,
            64: np.float64,
        }

    def __call__(self, package=None):
        return self.dtypes[self.precision]
