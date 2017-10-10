"""Utility classes and functions
"""

import tensorflow as tf
import numpy as np


class Convergence(object):
    """Checks convergence over a moving window.
    """

    def __init__(self, min_n, max_iter, min_iter=0,
                 test_slope=True, max_slope=0.1,
                 test_bounds=True, max_bounds=1.0, bound_deltas=False,
                 test_stats=True, max_sd=3.0):
        self.min_n = min_n
        self._hist = []
        self.iter = 0
        self.min_iter = min_iter
        self.max_iter = max_iter

        self.test_slope = test_slope
        self.max_slope = max_slope
        self.test_bounds = test_bounds
        self.max_bounds = max_bounds
        self.bound_deltas = bound_deltas
        self.test_stats = test_stats
        self.max_sd = max_sd
        self.last_converged = False

    def _test_slope(self, hist):
        if not self.test_slope:
            return True

        y = hist - np.mean(hist)
        x = np.arange(len(y))
        slope = np.dot(x, y) / np.dot(x, x)
        return np.abs(slope) < self.max_slope

    def _test_bounds(self, hist):
        if self.bound_deltas:
            hist = np.diff(hist)
        return np.all(np.abs(hist) < self.max_bounds)

    def _test_stats(self, hist):
        dev = hist - np.mean(hist)
        sd = np.std(dev)
        return np.all(np.abs(dev) < self.max_sd * sd)

    def clear(self):
        self._hist = []
        self.iter = 0
        self.last_converged = False

    def check(self, f):
        self._hist.append(f)
        self.iter += 1

        if self.iter < self.min_iter:
            return False
        if len(self._hist) < self.min_n:
            return False
        if self.iter > self.max_iter:
            return True
        self._hist = self._hist[-self.min_n:]
        hist = np.array(self._hist)

        self.last_converged = self._test_slope(hist) \
            and self._test_bounds(hist) \
            and self._test_stats(hist)
        return self.last_converged

def optimizer_initializer(opt, var_list):
    """Creates a list of initializers for resetting a tensorflow
    optimizer
    """
    opt_vars = [opt.get_slot(var, name)
                for name in opt.get_slot_names()
                for var in var_list]
    if isinstance(opt, tf.train.AdamOptimizer):
        opt_vars.extend(list(opt._get_beta_accumulators()))
    return tf.variables_initializer(opt_vars)
