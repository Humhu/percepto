"""Arm selection classes and methods.
"""
import numpy as np
import optim.reward_models as rewards
import cma

class ArmSelector(object):
    """Base class for all continuous-valued arm selectors.

    Arm selectors maximize acquisition functions over reward models.
    """
    def __init__(self, acq_func, optimizer):
        self.acq_func = acq_func
        self.optimizer = optimizer
        self.lower_bounds = -float('inf')
        self.upper_bounds = float('inf')

    @property
    def lower_bounds(self):
        return self.lower_bounds

    @lower_bounds.setter
    def lower_bounds(self, l):
        self.lower_bounds = l

    @property
    def upper_bounds(self):
        return self.upper_bounds

    @upper_bounds.setter
    def upper_bounds(self, u):
        self.upper_bounds = u

    def apply_bounds(self, a):
        """Applies bounds to a selected arm a.

        Parameters
        ----------
        a : iterable numeric
            The arm vector to apply bounds to

        Returns
        -------
        b : iterable numeric
            Input a with elements bounded
        """
        l_mask = a < self.lower_bounds
        u_mask = a > self.upper_bounds

        if np.iterable(self.lower_bounds):
            a[l_mask] = self.lower_bounds[l_mask]
        else:
            a[l_mask] = self.lower_bounds

        if np.iterable(self.upper_bounds):
            a[u_mask] = self.upper_bounds[u_mask]
        else:
            a[u_mask] = self.upper_bounds

    def pick_arm(self, x=None):
        """Pick an arm given the specified context.

        Parameters
        ----------
        x : varies depending on selector
            Input context if relevant, or None

        Returns
        -------
        arm : The selected arm, varies depending on selector
        """
        arm = self.optimizer.optimize(self.acq_func)

class UCBAcquisition(object):
    """Implements the UCB acquisition function.
    """
    def __init__(self, model):
        if not isinstance(model, rewards.RewardModel):
            raise ValueError('model must be a RewardModel')
        self.model = model
        self._exploration_rate = 1.0

    @property
    def exploration_rate(self):
        """Return the scale on the prediction uncertainty.
        """
        return self._exploration_rate

    @exploration_rate.setter
    def exploration_rate(self, b):
        if b < 0:
            raise RuntimeWarning('Negative value %f for exploration_rate' % b)
        self._exploration_rate = b

    def __call__(self, x):
        pred_mean, pred_sd = self.model.predict(x, return_std=True)
        return pred_mean + self.exploration_rate * pred_sd
