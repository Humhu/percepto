"""Utility classes and functions
"""

import numpy as np

class Convergence(object):
    """Checks convergence over a moving window.
    """
    def __init__(self, min_n, tol, max_iter, use_delta=False):
        self.min_n = min_n
        self.tol = tol
        self._hist = []
        self.iter = 0
        self.max_iter = max_iter
        self.use_delta = use_delta
        self.last_converged = False

    def clear(self):
        self._hist = []
        self.iter = 0
        self.last_converged = False

    def check(self, f):
        self._hist.append(f)
        self.iter += 1

        if len(self._hist) < self.min_n:
            return False
        if self.iter > self.max_iter:
            return True
        self._hist = self._hist[-self.min_n:]

        if self.use_delta:
            hist = np.diff(self._hist)
        else:
            hist = np.array(self._hist)

        self.last_converged = np.all(np.abs(hist) < self.tol)
        return self.last_converged