"""This module contains experimental locally-linear policies.
"""

from policies import StochasticPolicy
import numpy as np


class MixturePolicy(StochasticPolicy):
    def __init__(self, input_dim, output_dim, base_type, default_theta,
                 min_weight=0.1, weight_cov=None):
        super(MixturePolicy, self).__init__(input_dim, output_dim)
        self._base_type = base_type
        self._default_theta = def_theta
        self.min_weight = min_weight

        if weight_cov is None:
            self._weight_cov = 3*np.identity(input_dim)
        else:
            self._weight_cov = weight_cov

        self.weight_funcs = []
        self.components = []

    def initialize_component(self, center, theta=None):
        if theta is None:
            theta = self._default_theta

        component = self._base_type(input_dim=self.input_dim,
                                    output_dim=self.output_dim)
        component.set_theta(theta)

        def wfunc(x):
            d = x - center
            return math.exp(-0.5 * np.dot(d, np.dot(self._weight_cov, d)))

        self.weight_funcs.append(wfunc)
        self.components.append(component)

    def sample_action(self, state):
        weights = np.array([f(state) for f in self.weight_funcs])
        if not np.any(weights > self.min_weight):
            self.initialize_component(state)
            return self.sample_action(state)
        
        acc = 0
        z = 0
        for 

    def gradient(self, state, action):

    def logprob(self, state, action):

    def get_theta(self):

    def set_theta(self, t):

    @property
    def num_components(self):
        return len(self.weight_funcs)
