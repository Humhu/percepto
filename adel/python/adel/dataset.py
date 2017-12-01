"""Classes for aggregating, sampling, and accessing data
"""

import random
import copy
import cPickle as pickle

class EpisodeAggregator(object):
    """Wraps a dataset to form a stream of tuples into coherent tuples

    Parameters
    ----------
    base : SARSDataset
        The dataset to wrap and feed tuples into
    """
    def __init__(self, base):
        self.current_sar = None
        self.base = base

    def reset(self):
        """Resets the streaming state
        """
        self.current_sar = None

    def report_episode_step(self, s, a, r):
        """Adds a SARS tuple to the current episode
        """
        if self.current_sar is None:
            pass
        else:
            self.base.report_sars(self.current_sar[0],
                                  self.current_sar[1],
                                  self.current_sar[2],
                                  s)
        self.current_sar = (s, a, r)

    def report_episode_end(self, s):
        """Reports the end of an episode without a terminal state
        """
        self.base.report_sars(self.current_sar[0],
                              self.current_sar[1],
                              self.current_sar[2],
                              s)
        self.reset()

    def report_episode_terminal(self):
        """Reports a terminal condition, ending the current episode
        """
        self.base.report_terminal(self.current_sar[0],
                                  self.current_sar[1])

        self.reset()


class ValidationHoldout(object):
    """Maintains a training and validation holdout dataset in parallel. Provides
    methods for splitting and accessing data.

    Parameters
    ----------
    training : SARSDataset
        The training dataset
    holdout : SARSDataset
        The holdout dataset
    sampler : Sampler object
        The sampler to use for online/offline splitting
    """
    def __init__(self, training, holdout, sampler):
        self.training = training
        self.holdout = holdout
        self.sars_sampler = sampler
        # TODO Hack?
        self.term_sampler = copy.copy(sampler)

    def report_sars(self, s, a, r, sn):
        """Reports a SARS tuple
        """
        if self.sars_sampler.sample():
            self.holdout.report_sars( s, a, r, sn )
        else:
            self.training.report_sars( s, a, r, sn )

    def report_terminal(self, s, a):
        """Reports a terminal state-action
        """
        if self.term_sampler.sample():
            self.holdout.report_terminal( s, a )
        else:
            self.training.report_terminal( s, a )

class SamplingInterface(object):
    """Wraps a dataset to provide methods for sampling tuples
    # TODO More sampling methods
    """
    def __init__(self, base, method='uniform'):
        self.base = base
        self.method = method

    def sample_sars(self, k):
        """Samples from non-terminal SARS tuples
        """
        if self.method == 'uniform':
            return zip(*random.sample(self.base.sars, k))
        else:
            raise ValueError('Invalid sampling strategy')

    def sample_terminals(self, k):
        """Samples terminal states
        """
        if self.method == 'uniform':
            return zip(*random.sample(self.base.terminals, k))


class SARSDataset(object):
    """Stores and loads SARS data tuples.
    """

    def __init__(self):
        self.sars = []
        self.terminals = []

    def save(self, path):
        """Saves the dataset to a pickle.

        Parameters
        ----------
        path : string
            File path to save to
        """
        with open(path, 'w') as f:
            pickle.dump((self.sars, self.terminals), f)

    def load(self, path, append=True):
        """Loads and/or appends data from a pickle.

        Parameters
        ----------
        path   : string
            File path to load from
        append : bool (default True)
            Whether to append or overwrite data
        """
        with open(path, 'r') as f:
            sars, terminals = pickle.load(f)
        if append:
            self.sars += sars
            self.terminals += terminals
        else:
            self.sars = sars
            self.terminals = terminals

    @property
    def num_tuples(self):
        return len(self.sars)

    @property
    def num_terminals(self):
        return len(self.terminals)

    @property
    def all_states(self):
        return [sars[0] for sars in self.sars]

    @property
    def all_actions(self):
        return [sars[1] for sars in self.sars]

    @property
    def all_rewards(self):
        return [sars[2] for sars in self.sars]

    @property
    def all_next_states(self):
        return [sars[3] for sars in self.sars]

    @property
    def all_terminal_states(self):
        return [sa[0] for sa in self.terminals]

    @property
    def all_terminal_actions(self):
        return [sa[1] for sa in self.terminals]

    def report_sars(self, s, a, r, sn):
        """Reports a SARS tuple. Used for batch adding data.
        """
        self.sars.append((s, a, r, sn))

    def report_terminal(self, s, a):
        """Reports a terminal state-action. Used for batch adding data.
        """
        self.terminals.append((s, a))
