"""This module contains classes and functions for actively sampling data to train on.
"""

import abc
import numpy as np
import optim

class DataSampler(object):
    """Base class for all data samplers.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def sample_data(self, n, data):
        """Returns indices for a subset of the data.
        """
        return []


class WeightedDataSampler(DataSampler):
    """Samples data that have non-negative weights associated with them.

    Parameters
    ----------
    weight_func : function that takes a datum and outputs a non-negative scalar
    sample_func : function with keywords n, weights
        Returns n indices given weights. Default implementation uses np.choice
    """

    def __init__(self, weight_func, sample_func=None):
        self._weight_func = weight_func

        if sample_func is None:
            def default_sampler(n, weights):
                weights = np.array(weights)
                weights = weights / np.sum(weights)
                return np.random.choice(len(weights), size=n,
                                        replace=False, p=weights)

            self._sample_func = default_sampler
        else:
            self._sample_func = sample_func

    def sample_data(self, n, data):
        weights = [self._weight_func(dat) for dat in data]
        return self._sample_func(n=n, weights=weights)

class OrdinalDataSampler(DataSampler):
    """Samples data that can be sorted, but not weighted.

    Parameters
    ----------
    sort_func   : function that returns indices corresponding to sorted data order
    sample_func : function with keywords n, order
        Returns n indices given ordering. Default implementation uses tournament selection
    """
    def __init__(self, sort_func, sample_func=None):
        self._sort_func = sort_func

        if sample_func is None:
            def default_sampler(n, order):
                order = -np.asarray(order)
                return optim.tournament_selection(N=n, weights=order)
            self._sample_func = default_sampler
        else:
            self._sample_func = sample_func

    def sample_data(self, n, data):
        order = self._sort_func(data)
        return self._sample_func(n=n, order=order)