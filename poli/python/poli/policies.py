"""This module contains common stochastic policy classes.
"""
import abc
import math

class StochasticPolicy(object):
    """Base class for all stochastic policies.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, indim, outdim):
        self._input_dim = indim
        self._output_dim = outdim

    @abc.abstractmethod
    def sample_action(self, state):
        """Returns a randomly-selected action.

        Parameters:
        -----------
        state : TODO

        Returns:
        --------
        action : numpy array?
        """
        return None

    @abc.abstractmethod
    def gradient(self, state, action):
        """Computes the gradient of the action log-prob w.r.t. the parameters.
        """
        return None

    @abc.abstractmethod
    def logprob(self, state, action):
        """Computes the log-probability of an action given the state.
        """
        return None

    def prob(self, state, action):
        """Computes the probability of an action given the state.
        """
        return math.exp(self.logprob(state, action))

    def get_theta(self):
        """Return this policy's parameters. If it has no parameters, return None.
        """
        return None

    def set_theta(self, t):
        """Set this policy's parameters.
        """
        if t is not None:
            raise ValueError('This policy has no parameters!')

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim


