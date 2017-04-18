"""This module contains code for policy-gradient parameter exploration (PPGE)
"""

import numpy as np
import scipy.stats as sps
from policies import StochasticPolicy

class ParameterDistribution(StochasticPolicy):
    def __init__(self, input_dim, output_dim, sds=None):
        super(ParameterDistribution, self).__init__(indim=0, outdim=output_dim)
        self._mean = np.zeros(output_dim)
        self._logvars = np.zeros(output_dim)

        if sds is not None:
            if not np.iterable(sds):
                sds = np.full(len(self._logvars), float(sds))
            self.sds = sds

    def sample_action(self, state=None):
        return np.random.normal(loc=self.mean, scale=self.sds)

    def gradient(self, state, action):
        delta = action - self.mean
        inv_var = np.exp(-self._logvars)
        du = delta * inv_var
        dlv = -0.5 * (1.0 - delta * delta * inv_var)
        return np.hstack((du, dlv))

    def logprob(self, state, action):
        return np.sum(sps.norm.logpdf(x=action, loc=self.mean, scale=self.sds))

    def get_theta(self):
        return np.hstack((self.mean, self._logvars))

    def set_theta(self, th):
        N = len(self.mean) + len(self._logvars)
        if len(th) != N:
            raise ValueError('Got %d parameters but expected %d' % (len(th), N))
        self.mean = th[:len(self.mean)]
        self._logvars = th[len(self.mean):]

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, u):
        if len(u) != len(self._mean):
            raise ValueError('Incorrect mean shape')
        self._mean = u

    @property
    def sds(self):
        return np.exp(self._logvars * 0.5)

    @sds.setter
    def sds(self, s):
        if len(s) != len(self._logvars):
            raise ValueError('Incorrect SDs shape')
        self._logvars = 2.0 * np.log(s)
