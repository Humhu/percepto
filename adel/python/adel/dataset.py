"""Classes for sampling and accessing data
"""

import abc
import random
import copy
from sampling import create_sampler
from collections import deque


class DatasetInterface(object):
    """Base interface for generic multi-volume dataset objects. Keys sort
    data into different volumes and can be used for multiple classes, etc.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def report_data(self, key, data):
        """Reports a piece of data with a corresponding key
        """
        pass

    @abc.abstractmethod
    def get_volume(self, key):
        """Retrieves all data with a corresponding key
        """
        pass

    def get_volume_size(self, key):
        return len(self.get_volume(key))


class BasicDataset(DatasetInterface):
    """Basic deque-based implementation of database. 
    TODO Storage/retrieval
    """

    def __init__(self):
        DatasetInterface.__init__(self)
        self.volumes = {}

    def report_data(self, key, data):
        if key not in self.volumes:
            self.volumes[key] = []#deque()
        self.volumes[key].append(data)

    def get_volume(self, key):
        if key not in self.volumes:
            #return deque()
            return []
        else:
            return self.volumes[key]

    def clear(self, key=None):
        if key is None:
            self.volumes = {}
        else:
            self.volumes.pop(key)

    # def to_saveable(self):
    #     """Converts the dataset to a pickleable object
    #     """
    #     return (self.sars, self.terminals)

    # def from_saveable(self, obj, append=True):
    #     """Loads the dataset from a pickled object
    #     """
    #     sars, terminals = obj
    #     if append:
    #         self.sars += sars
    #         self.terminals += terminals
    #     else:
    #         self.sars = sars
    #         self.terminals = terminals

    # def save(self, path):
    #     """Saves the dataset to a pickle.

    #     Parameters
    #     ----------
    #     path : string
    #         File path to save to
    #     """
    #     with open(path, 'w') as f:
    #         pickle.dump(self.to_saveable(), f)

    # def load(self, path, append=True):
    #     """Loads and/or appends data from a pickle.

    #     Parameters
    #     ----------
    #     path   : string
    #         File path to load from
    #     append : bool (default True)
    #         Whether to append or overwrite data
    #     """
    #     with open(path, 'r') as f:
    #         obj = pickle.load(f)
    #     self.from_saveable(obj)


class SubindexedDataset(DatasetInterface):
    """Wrapper around a dataset that refers to a subset of that dataset
    using indices
    """

    def __init__(self, base):
        DatasetInterface.__init__(self)
        self.base = base
        self.sub_inds = {}
        self.terminal_inds = []

    def clear(self):
        self.base.clear()

    def set_inds(self, key, inds):
        self.sub_inds[key] = inds

    def report_data(self, key, data):
        if key not in self.sub_inds:
            self.sub_inds[key] = deque()
        self.sub_inds[key].append(self.base.num_data)
        self.base.report_data(key, data)

    def get_volume(self, key):
        vol = self.base.get_volume(key)
        return [vol[i] for i in self.sub_inds[key]]


class HoldoutWrapper(DatasetInterface):
    """Maintains a training and validation holdout dataset in parallel. Provides
    methods for splitting and accessing data.

    Parameters
    ----------
    training : DatasetInterface
        The training dataset object
    holdout : DatasetInterface
        The holdout dataset object
    sampler : Sampler object
        The sampler to use for online/offline splitting
    """

    def __init__(self, training, holdout, **kwargs):
        DatasetInterface.__init__(self)
        self.training = training
        self.holdout = holdout
        self.samplers = {}
        self.sampler_args = kwargs

    def clear(self):
        self.training.clear()
        self.holdout.clear()
        for s in self.samplers.itervalues():
            s.reset()

    def report_data(self, key, data):
        if key not in self.samplers:
            self.samplers[key] = create_sampler(**self.sampler_args)

        if self.samplers[key].sample():
            self.holdout.report_data(key, data)
        else:
            self.training.report_data(key, data)

    @property
    def get_volume(self, key):
        #out = deque(self.training.get_volume(key))
        #out.extend(self.holdout.get_volume(key))
        #return out
        return self.training.get_volume(key) + self.holdout.get_volume(key)


class DatasetSampler(DatasetInterface):
    """Wraps a dataset to provide methods for sampling tuples

    # NOTE Resamples on each call to get_volume
    # TODO More sampling methods
    """

    def __init__(self, base, k=1, method='uniform'):
        DatasetInterface.__init__(self)
        self.base = base
        self.method = method
        self.k = k
        self.cache = {}

    def clear(self):
        self.base.clear()

    def sample_data(self, key, k=None):
        """Samples k data
        """
        if k is None:
            k = self.k

        vsize = self.base.get_volume_size(key)
        if k > vsize:
            raise ValueError('Volume %s has %d but requested %d'
                             % (str(key), vsize, k))

        N = self.base.get_volume_size(key)
        if self.method == 'uniform':
            self.cache[key] = random.sample(range(N), k)
        else:
            raise ValueError('Invalid sampling strategy')

    def report_data(self, key, data):
        self.base.report_data(key, data)

    def get_volume(self, key):
        if key not in self.cache:
            self.sample_data(key=key)
        vol = self.base.get_volume(key)
        return [vol[i] for i in self.cache[key]]

class DatasetChunker(DatasetInterface):
    """Wraps a dataset to allow iterating through fixed-size blocks
    """
    def __init__(self, base, block_size):
        DatasetInterface.__init__(self)
        self.base = base
        self.block_size = block_size
        self.block_index = {}

    def reset(self, key):
        # TODO case where key=None?
        self.block_index[key] = 0

    def is_done(self, key):
        vsize = self.base.get_volume_size(key)
        return vsize < self.block_index[key] * self.block_size

    def report_data(self, key, data):
        self.base.report_data(key, data)

    def iter_volume(self, key):
        self.reset(key)
        while not self.is_done(key):
            yield self.get_volume(key)
        self.reset(key)

    def iter_subdata(self, key):
        self.reset(key)
        while not self.is_done(key):
            yield self.get_subdata(key)
        self.reset(key)

    def iter_chunk_sizes(self, key):
        self.reset(key)
        while not self.is_done(key):
            yield self.get_volume_size(key)
        self.reset(key)

    def get_subdata(self, key):
        if key not in self.block_index:
            self.block_index[key] = 0

        sub = SubindexedDataset(base=self.base)
        i = self.block_index[key] * self.block_size
        fin = min(i + self.block_size, self.base.get_volume_size(key))
        sub.set_inds(key, range(i, fin))
        self.block_index[key] += 1        
        return sub

    def get_volume(self, key):
        if key not in self.block_index:
            self.block_index[key] = 0
        
        i = self.block_index[key] * self.block_size
        vol = self.base.get_volume(key)[i:i+self.block_size]
        self.block_index[key] += 1
        return vol

    def get_volume_size(self, key):
        return len(self.get_volume(key))