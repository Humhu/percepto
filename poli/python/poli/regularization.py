"""This module contains classes for policy regularization.
"""

import numpy as np


def parse_regularizer(policy, spec):
    if 'type' not in spec:
        raise ValueError('Regularizer spec must specify type!')

    regularizer_type = spec.pop('type')
    lookup = {'L2': ParameterL2Regularizer}
    if regularizer_type not in lookup:
        raise ValueError('Regularizer type %s not valid type: %s' %
                         (regularizer_type, str(lookup.keys())))
    return lookup[regularizer_type](policy, **spec)


class ParameterL2Regularizer(object):
    """Computes a weighted L2 cost on the policy parameters.
    """

    def __init__(self, policy, scale):
        self.policy = policy
        scale = float(scale)
        if scale < 0:
            raise ValueError('L2 scale should be non-negative.')
        self.scale = scale

    def compute_objective(self):
        return self.__objective()

    def compute_gradient(self):
        return self.__gradient()

    def compute_objective_and_gradient(self):
        return self.__objective(), self.__gradient()

    def __objective(self):
        theta = self.policy.get_theta()
        return -0.5 * np.dot(theta, theta) * self.scale

    def __gradient(self):
        theta = self.policy.get_theta()
        return -theta * self.scale
