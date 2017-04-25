"""Classes and functions for optimizing subsets of variables (partial optimization)
"""

import numpy as np
from reward_models import RewardModel

class PartialModelWrapper(RewardModel):
    """Wraps an objective function to allow optimization with partial input specification.

    Parameters
    ----------
    func : callable
        The objective function to wrap
    """
    def __init__(self, base_model):
        self.base = base_model
        self._base_input = None
        self._active_inds = None

    @property
    def active_inds(self):
        return self._active_inds

    @active_inds.setter
    def active_inds(self, inds):
        self._active_inds = np.array(inds, dtype=np.int32)

    @property
    def base_input(self):
        return self._base_input

    @base_input.setter
    def base_input(self, ival):
        self._base_input = ival

    def report_sample(self, x, reward):
        xfull = self.generate_full_input(x)
        self.base.report_sample(xfull, reward)

    def predict(self, x, return_std=False):
        xfull = self.generate_full_input(x)
        return self.base.predict(xfull, return_std=return_std)

    def clear(self):
        return self.base.clear()

    def generate_full_input(self, x):
        xfull = np.array(self._base_input)
        xfull[self._active_inds] = x
        return xfull
