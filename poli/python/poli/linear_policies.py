"""This module contains various linear policies.
"""

from policies import StochasticPolicy
import numpy as np
import modprop


class LinearPolicy(StochasticPolicy):
    """A policy class that computes a mean and covariance from linear products.

    The output is Gaussian-distributed with mean = A*x and covariance = np.diag(B*x)
    for vector state x

    Parameters
    ----------
    input_dim  : integer
        The input vector dimension
    output_dim : integer
        The output vector dimension

    Properties
    ----------
    A : numpy 2D-array 
        Maps state to output mean
    B : numpy 2D-array 
        Maps state to log variances
    offset : float or numpy 2D-array
        Offset to add to the covariance
    """

    def __init__(self, input_dim, output_dim, offset=1E-9):
        super(LinearPolicy, self).__init__(input_dim, output_dim)

        A = np.zeros((output_dim, input_dim))
        B = np.zeros((output_dim, input_dim))

        self._Amod = modprop.ConstantModule(A)
        self._Bmod = modprop.ConstantModule(B)
        self._state = modprop.ConstantModule(None)
        offset = np.atleast_2d(offset)
        if offset.shape != (output_dim, output_dim):
            offset = offset[0, 0] * np.identity(output_dim)
        self._covOffset = modprop.ConstantModule(offset)

        self._Ax = modprop.MatrixProductModule()
        modprop.link_ports(in_port=self._Ax.left_port,
                           out_port=self._Amod.out_port)
        modprop.link_ports(in_port=self._Ax.right_port,
                           out_port=self._state.out_port)

        self._Bx = modprop.MatrixProductModule()
        modprop.link_ports(in_port=self._Bx.left_port,
                           out_port=self._Bmod.out_port)
        modprop.link_ports(in_port=self._Bx.right_port,
                           out_port=self._state.out_port)

        self._expBx = modprop.ExponentialModule()
        modprop.link_ports(in_port=self._expBx.in_port,
                           out_port=self._Bx.out_port)

        self._expBxd = modprop.DiagonalReshapeModule()
        modprop.link_ports(in_port=self._expBxd.vec_in,
                           out_port=self._expBx.out_port)

        self._expBxd_off = modprop.AdditionModule()
        modprop.link_ports(in_port=self._expBxd_off.left_port,
                           out_port=self._expBxd.diag_out)
        modprop.link_ports(in_port=self._expBxd_off.right_port,
                           out_port=self._covOffset.out_port)

        self._act = modprop.ConstantModule(None)
        self._delAct = modprop.DifferenceModule()
        modprop.link_ports(in_port=self._delAct.left_port,
                           out_port=self._act.out_port)
        modprop.link_ports(in_port=self._delAct.right_port,
                           out_port=self._Ax.out_port)

        self._ll = modprop.LogLikelihoodModule()
        modprop.link_ports(in_port=self._ll.x_in,
                           out_port=self._delAct.out_port)
        modprop.link_ports(in_port=self._ll.S_in,
                           out_port=self._expBxd_off.out_port)

        self._llSink = modprop.SinkModule()
        modprop.link_ports(in_port=self._llSink.in_port,
                           out_port=self._ll.ll_out)

    def sample_action(self, state):
        self.__invalidate()
        self.__foreprop(state=state, action=None)
        return np.random.multivariate_normal(mean=self.mean,
                                             cov=self.cov)

    def logprob(self, state, action):
        self.__invalidate()
        self.__foreprop(state=state, action=action)
        return self._llSink.value

    def gradient(self, state, action):
        self.__invalidate()
        self.__foreprop(state=state, action=action)
        acc = modprop.AccumulatedBackprop(do_dx=np.identity(1))
        self._llSink.backprop_value = acc
        modprop.iterative_backprop(self._llSink)

        return np.hstack((self._Amod.backprop_value[0],
                          self._Bmod.backprop_value[0]))

    def __foreprop(self, state, action):
        self._act.value = np.atleast_1d(action)
        self._state.value = np.atleast_1d(state)
        modprop.iterative_foreprop(self._state)
        if action is not None:
            modprop.iterative_foreprop(self._act)
        modprop.iterative_foreprop(self._Amod)
        modprop.iterative_foreprop(self._Bmod)
        modprop.iterative_foreprop(self._covOffset)

    def __invalidate(self):
        modprop.iterative_invalidate(self._state)
        modprop.iterative_invalidate(self._act)
        modprop.iterative_invalidate(self._Amod)
        modprop.iterative_invalidate(self._Bmod)
        modprop.iterative_invalidate(self._covOffset)

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

    @A.setter
    def A(self, a):
        self._Amod.value = a

    @property
    def B(self):
        return self._Bmod.value

    @B.setter
    def B(self, b):
        self._Bmod.value = b

    @property
    def mean(self):
        return self._Ax.out_port.value

    @property
    def cov(self):
        return self._expBxd_off.out_port.value


class FixedLinearPolicy(StochasticPolicy):
    """A policy class that computes a mean with fixed covariance from linear products.

    The output is Gaussian-distributed with mean = A*x and covariance = np.diag(b)
    for vector state x

    Parameters
    ----------
    input_dim  : integer
        The input vector dimension
    output_dim : integer
        The output vector dimension

    Properties
    ----------
    A : numpy 2D-array 
        Maps state to output mean
    b : numpy 1D-array 
        Constant log variances
    offset : float or numpy 2D-array
        Offset to add to the covariance
    """

    def __init__(self, input_dim, output_dim, offset=1E-9):
        super(FixedLinearPolicy, self).__init__(input_dim, output_dim)

        A = np.zeros((output_dim, input_dim))
        b = np.zeros(output_dim)

        self._Amod = modprop.ConstantModule(A)
        self._Bmod = modprop.ConstantModule(b)
        self._state = modprop.ConstantModule(None)
        offset = np.atleast_2d(offset)
        if offset.shape != (output_dim, output_dim):
            offset = offset[0, 0] * np.identity(output_dim)
        self._covOffset = modprop.ConstantModule(offset)

        self._Ax = modprop.MatrixProductModule()
        modprop.link_ports(in_port=self._Ax.left_port,
                           out_port=self._Amod.out_port)
        modprop.link_ports(in_port=self._Ax.right_port,
                           out_port=self._state.out_port)

        self._expBx = modprop.ExponentialModule()
        modprop.link_ports(in_port=self._expBx.in_port,
                           out_port=self._Bmod.out_port)

        self._expBxd = modprop.DiagonalReshapeModule()
        modprop.link_ports(in_port=self._expBxd.vec_in,
                           out_port=self._expBx.out_port)

        self._expBxd_off = modprop.AdditionModule()
        modprop.link_ports(in_port=self._expBxd_off.left_port,
                           out_port=self._expBxd.diag_out)
        modprop.link_ports(in_port=self._expBxd_off.right_port,
                           out_port=self._covOffset.out_port)

        self._act = modprop.ConstantModule(None)
        self._delAct = modprop.DifferenceModule()
        modprop.link_ports(in_port=self._delAct.left_port,
                           out_port=self._act.out_port)
        modprop.link_ports(in_port=self._delAct.right_port,
                           out_port=self._Ax.out_port)

        self._ll = modprop.LogLikelihoodModule()
        modprop.link_ports(in_port=self._ll.x_in,
                           out_port=self._delAct.out_port)
        modprop.link_ports(in_port=self._ll.S_in,
                           out_port=self._expBxd_off.out_port)

        self._llSink = modprop.SinkModule()
        modprop.link_ports(in_port=self._llSink.in_port,
                           out_port=self._ll.ll_out)

    def sample_action(self, state):
        self.__invalidate()
        self.__foreprop(state=state, action=None)
        return np.random.multivariate_normal(mean=self.mean,
                                             cov=self.cov)

    def logprob(self, state, action):
        self.__invalidate()
        self.__foreprop(state=state, action=action)
        return self._llSink.value

    def gradient(self, state, action):
        self.__invalidate()
        self.__foreprop(state=state, action=action)
        acc = modprop.AccumulatedBackprop(do_dx=np.identity(1))
        self._llSink.backprop_value = acc
        modprop.iterative_backprop(self._llSink)

        return np.hstack((self._Amod.backprop_value[0],
                          self._Bmod.backprop_value[0]))

    def __foreprop(self, state, action):
        self._act.value = np.atleast_1d(action)
        self._state.value = np.atleast_1d(state)
        modprop.iterative_foreprop(self._state)
        if action is not None:
            modprop.iterative_foreprop(self._act)
        modprop.iterative_foreprop(self._Amod)
        modprop.iterative_foreprop(self._Bmod)
        modprop.iterative_foreprop(self._covOffset)

    def __invalidate(self):
        modprop.iterative_invalidate(self._state)
        modprop.iterative_invalidate(self._act)
        modprop.iterative_invalidate(self._Amod)
        modprop.iterative_invalidate(self._Bmod)
        modprop.iterative_invalidate(self._covOffset)

    def get_theta(self):
        return np.hstack((self._Amod.value.flatten('F'),
                          self._Bmod.value))

    def set_theta(self, t):
        n_A = len(self._Amod.value.flat)
        n_b = len(self._Bmod.value)
        if n_A + n_b != len(t):
            raise ValueError('Parameter dimension mismatch!')
        self._Amod.value = np.reshape(
            t[:n_A], self._Amod.value.shape, order='F')
        self._Bmod.value = t[n_A:]

    @property
    def A(self):
        return self._Amod.value

    @A.setter
    def A(self, a):
        self._Amod.value = a

    @property
    def B(self):
        return self._Bmod.value

    @B.setter
    def B(self, bv):
        self._Bmod.value = bv

    @property
    def mean(self):
        return self._Ax.out_port.value

    @property
    def cov(self):
        return self._expBxd_off.out_port.value


class DeterministicLinearPolicy(StochasticPolicy):
    """A non-stochastic linear policy that outputs
    action = A * state.
    """
    def __init__(self, input_dim, output_dim):
        super(DeterministicLinearPolicy, self).__init__(indim=input_dim,
                                                        outdim=output_dim)
        self.A = np.zeros((output_dim, input_dim))

    def sample_action(self, state):
        return np.dot(self.A, state)

    def logprob(self, state, action):
        a = self.sample_action(state)
        if np.any(a != action):
            return float('-inf')
        else:
            return 0

    def gradient(self, state, action):
        return np.zeros(len(self.get_theta()))

    def get_theta(self):
        return self.A.flatten()

    def set_theta(self, a):
        self.A = np.reshape(a, self.A.shape)
