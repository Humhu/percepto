"""Utilities for parameter scaling
"""

import numpy as np


def parse_normalizer(info):
    rounders = []
    if 'valid_values' in info:
        valids = info['valid_values']
        rounders.append(ValueRounder(values=valids))
        lower_lim = min(valids)
        upper_lim = max(valids)
    else:
        lower_lim = info['lower_limit']
        upper_lim = info['upper_limit']

    if 'integer_valued' in info and info['integer_valued']:
        rounders.append(IntegerRounder())

    return ParameterNormalizer(min_val=lower_lim,
                               max_val=upper_lim,
                               rounders=rounders)


class IntegerRounder(object):
    """Rounds continuous variables to integer values
    """

    def __init__(self):
        pass

    def project(self, x):
        return int(round(x))


class ValueRounder(object):
    """Rounds continuous variables to discrete values
    """

    def __init__(self, values):
        self.values = values

    def project(self, x):
        dists = [abs(x - vi) for vi in self.values]
        ind = np.argmin(dists)
        return self.values[ind]


class ParameterNormalizer(object):
    """Normalizes continuous variables to +- 1.0
    """

    def __init__(self, min_val, max_val, rounders):
        self.offset = 0.5 * (min_val + max_val)
        self.scale = 0.5 * (max_val - min_val)
        if self.scale < 0:
            raise ValueError('max_val must be > min_val')
        self.min_val = min_val
        self.max_val = max_val
        self.rounders = rounders

    @property
    def integer_valued(self):
        return any([isinstance(ri, IntegerRounder) for ri in self.rounders])

    def check_bounds(self, x, normalized):
        if normalized:
            x = max(min(x, 1), -1)
        else:
            x = max(min(x, self.max_val), self.min_val)
        return x

    def normalize(self, x):
        for ri in self.rounders:
            x = ri.project(x)
        return (x - self.offset) / self.scale

    def unnormalize(self, x):
        x = (x * self.scale) + self.offset
        for ri in self.rounders:
            x = ri.project(x)
        return x
