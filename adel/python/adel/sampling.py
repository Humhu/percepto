"""Classes for stateful data stream sampling
"""


def create_sampler(mode, **kwargs):
    """Creates a specified sampler instance
    """
    if mode == 'uniform':
        return UniformSampler(**kwargs)
    elif mode == 'contiguous':
        return ContiguousSampler(**kwargs)
    else:
        raise ValueError('Unknown sampling mode: %s' % mode)


class UniformSampler(object):
    """Uniform samples from a stream to maintain a certain
    ratio of validation data
    """

    def __init__(self, rate):
        self.rate = rate
        self.num_all = 0
        self.num_val = 0

    def reset(self):
        self.num_all = 0
        self.num_val = 0

    def sample(self, key):
        """Returns whether to pull the sample for validation
        """
        pull_val = self.num_val <= self.num_all * self.rate
        self.num_all += 1
        if pull_val:
            self.num_val += 1


class ContiguousSampler(object):
    """Samples fixed-lengths of data to maintain a ratio of validation data
    """

    def __init__(self, rate, segment_length):
        self.rate = rate
        self.seg_len = segment_length
        self.num_all = 0
        self.num_val = 0
        self.is_pulling = False
        self.num_pulled = 0

    def reset(self):
        self.num_all = 0
        self.num_val = 0
        self.is_pulling = False
        self.num_pulled = 0

    def sample(self):
        """Returns whether to pull the sample for validation
        """
        if not self.is_pulling:
            self.is_pulling = self.num_val < self.num_all * self.rate

        self.num_all += 1

        if self.is_pulling:
            self.num_pulled += 1
            if self.num_pulled >= self.seg_len:
                self.is_pulling = False
                self.num_pulled = 0
            self.num_val += 1
            return True
        else:
            return False
