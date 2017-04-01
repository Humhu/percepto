"""This module contains classes for normalizing and augmenting features online.
"""

import numpy as np


class OnlineMaxTracker(object):
    """Computes the maximum/min for a stream of vectors online.
    """

    def __init__(self, dim):
        self.mins = np.full(dim, float('inf'))
        self.maxs = np.full(dim, float('-inf'))
        self.count = 0

    def __len__(self):
        return self.count

    def update(self, v):
        lesser = v < self.mins
        self.mins[lesser] = v[lesser]
        greater = v > self.maxs
        self.maxs[greater] = v[greater]
        self.count += 1

    @property
    def scale(self):
        return (self.maxs - self.mins) / 2

    @property
    def offset(self):
        return (self.maxs + self.mins) / 2


class OnlineMomentTracker:
    '''Computes the first two moments for a stream of vectors online using the
    Welford online moment algorithm.
    
    Parameters
    ----------
    dim     : integer
        The input dimensionality
    num_sds : float
        The number of standard deviations to normalize to
    '''

    def __init__(self, dim, num_sds):
        self.M1 = np.zeros(dim)
        self.M2 = np.zeros(dim)
        self.count = 0
        self.num_sds = num_sds

    def __len__(self):
        return self.count

    def update(self, v):
        self.count += 1

        if self.count == 1:
            self.M1 = v
            return

        delta = v - self.M1
        self.M1 += delta / self.count
        self.M2 += delta * (v - self.M1)
        #n = self.count - 2.0
        #N = self.count - 1.0
        #self.M2 = (n / N ) * self.M2 + (v - self.M1) / N

    @property
    def offset(self):
        return self.M1

    @property
    def scale(self):
        var = self.M2 / (self.count - 1)
        return self.num_sds * np.sqrt(var)


class OnlineFeatureNormalizer(object):
    """Learns scales and offsets for multivariate feature streams online.

    Parameters
    ----------
    dim           : integer
        The feature dimensionality
    mode          : string (minmax or moments)
        Which normalization method to use
    min_samples   : positive integer
        Minimum number of samples required before outputing
    keep_updating : boolean
        Whether to keep updating the normalization online after min_samples obtained
    """

    def __init__(self, dim, mode, min_samples, keep_updating, **kwargs):
        if mode == 'minmax':
            self.tracker = OnlineMaxTracker(dim, **kwargs)
        elif mode == 'moments':
            self.tracker = OnlineMomentTracker(dim, **kwargs)
        else:
            raise ValueError('Unknown mode %s' % mode)

        self.min_samples = min_samples
        self.keep_updating = keep_updating

    def process(self, v):
        """Learn using a received sample and return the normalized output.
        """
        v = np.asarray(v)

        if len(self.tracker) < self.min_samples or self.keep_updating:
            self.tracker.update(v)

        if len(self.tracker) < self.min_samples:
            return None
        else:
            scale = self.tracker.scale
            offset = self.tracker.offset
            return (v - offset) / scale

class FeaturePolynomialAugmenter(object):
    def __init__(self, dim, max_order):
        self._dim = dim
        self.max_order = max_order

    def process(self, v):
        v = np.asarray(v)
        feats = np.array([v ** i for i in range(1, self.max_order+1)])
        return feats.flatten()

    @property
    def dim(self):
        return self.max_order * self._dim