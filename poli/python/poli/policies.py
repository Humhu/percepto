"""This module contains common stochastic policy classes.
"""
import abc
import numpy as np
import math
from modprop.core.modules_core import *
from modprop.core.backprop import *
from modprop.modules.basic_modules import *
from modprop.modules.math_modules import *
from modprop.modules.reshape_modules import *
from modprop.modules.cost_modules import LogLikelihoodModule


class StochasticPolicy(object):
    """Base class for all stochastic policies.
    """
    __metaclass__ = abc.ABCMeta

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
        """Computes the log-prob of an action given the state.
        """
        return None

    def prob(self, state, action):
        return math.exp(self.action_logprob(state, action))

    def get_theta(self):
        """Return this policy's parameters. If it has no parameters, return None.
        """
        return None

    def set_theta(self, t):
        """Set this policy's parameters.
        """
        if t is not None:
            raise ValueError('This policy has no parameters!')

class LinearPolicy(StochasticPolicy):
    """A policy class that computes a mean and covariance from linear products.

    Parameters:
    -----------
    A : numpy ND-array mapping state to output mean
    B : numpy ND-array mapping state to log variances
    """

    def __init__(self, A, B, dim=None):
        if dim is None:
            if A is not None:
                dim = A.shape[0]
            elif B is not None:
                dim = B.shape[0]
            else:
                raise ValueError('Must specify A and B, or dim')

        if A is None:
            A = np.identity(dim)
        if B is None:
            B = np.identity(dim)

        if A.shape[0] != dim:
            raise ValueError('A matrix dimension mismatch.')
        if B.shape[0] != dim:
            raise ValueError('B matrix dimension mismatch.')

        self._Amod = ConstantModule(A)
        self._Bmod = ConstantModule(B)
        self._state = ConstantModule(None)

        self._Ax = MatrixProductModule()
        link_ports(in_port=self._Ax.left_port, out_port=self._Amod.out_port)
        link_ports(in_port=self._Ax.right_port, out_port=self._state.out_port)

        self._Bx = MatrixProductModule()
        link_ports(in_port=self._Bx.left_port, out_port=self._Bmod.out_port)
        link_ports(in_port=self._Bx.right_port, out_port=self._state.out_port)

        self._expBx = ExponentialModule()
        link_ports(in_port=self._expBx.in_port, out_port=self._Bx.out_port)

        self._expBxd = DiagonalReshapeModule()
        link_ports(in_port=self._expBxd.vec_in, out_port=self._expBx.out_port)

        self._act = ConstantModule(None)
        self._delAct = DifferenceModule()
        link_ports(in_port=self._delAct.left_port, out_port=self._act.out_port)
        link_ports(in_port=self._delAct.right_port, out_port=self._Ax.out_port)

        self._ll = LogLikelihoodModule()
        link_ports(in_port=self._ll.x_in, out_port=self._delAct.out_port)
        link_ports(in_port=self._ll.S_in, out_port=self._expBxd.diag_out)

        self._llSink = SinkModule()
        link_ports(in_port=self._llSink.in_port, out_port=self._ll.ll_out)

    def sample_action(self, state):
        self.__invalidate()
        self.__foreprop(state=state, action=None)
        return np.random.multivariate_normal(mean=self._Ax.out_port.value,
                                             cov=self._expBxd.diag_out.value)

    def logprob(self, state, action):
        self.__invalidate()
        self.__foreprop(state=state, action=action)
        return self._llSink.value

    def gradient(self, state, action):
        self.__invalidate()
        self.__foreprop(state=state, action=action)
        acc = AccumulatedBackprop(do_dx=np.identity(1))
        self._llSink.backprop_value = acc
        iterative_backprop(self._llSink)

        return np.hstack((self._Amod.backprop_value[0],
                          self._Bmod.backprop_value[0]))

    def __foreprop(self, state, action):
        self._act.value = np.atleast_1d(action)
        self._state.value = np.atleast_1d(state)
        iterative_foreprop(self._state)
        if action is not None:
            iterative_foreprop(self._act)
        iterative_foreprop(self._Amod)
        iterative_foreprop(self._Bmod)

    def __invalidate(self):
        iterative_invalidate(self._state)
        iterative_invalidate(self._act)
        iterative_invalidate(self._Amod)
        iterative_invalidate(self._Bmod)

    def get_theta(self):
        return np.hstack((self._Amod.value.flatten('F'),
                          self._Bmod.value.flatten('F')))

    def set_theta(self, t):
        n_A = len(self._Amod.value.flat)
        n_B = len(self._Bmod.value.flat)
        if n_A + n_B != len(t):
            raise ValueError('Parameter dimension mismatch!')
        self._Amod.value = np.reshape(
            t[:n_A], self._Amod.value.shape, order='F')
        self._Bmod.value = np.reshape(
            t[n_A:], self._Bmod.value.shape, order='F')

    @property
    def A(self):
        return self._Amod.value

    @property
    def B(self):
        return self._Bmod.value

    @property
    def mean(self):
        return self._Ax.out_port.value

    @property
    def cov(self):
        return self._expBxd.diag_out.value
