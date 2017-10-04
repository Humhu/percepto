"""Utility classes and functions
"""

import tensorflow as tf
import numpy as np

class Convergence(object):
    """Checks convergence over a moving window.
    """
    def __init__(self, min_n, tol, max_iter, use_delta=False, min_iter=0):
        self.min_n = min_n
        self.tol = tol
        self._hist = []
        self.iter = 0
        self.min_iter = min_iter
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

        if self.iter < self.min_iter:
            return False
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