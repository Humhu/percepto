"""Classes for aggregating and sampling data
"""

import random
import cPickle as pickle

# Operations we will want to support:
# 1. Mixing states/actions to generate synthetic terminal SA tuples
# 2. Randomly selecting SARS for value training

class EpisodicSARSDataset(object):
    """Stores episodic SARS data tuples. Has interfaces for sequentially
    reporting episodes as well as directly reporting SARS tuples.

    Parameters
    ==========
    sampling_strategy : 'uniform'
        TODO Implement more
    """

    def __init__(self, sampling_strategy='uniform'):
        self.episode_lens = []
        self._ep_counter = 0
        self.sars_data = []
        self.terminals = []
        self.current_sar = None

        valid_ss = ['uniform']
        if sampling_strategy not in valid_ss:
            err = ('Unsupported sampling strategy %s. ' 
                   + 'Supported are %s') % (sampling_strategy, valid_ss)
            raise ValueError(err)
        self._sstrat = sampling_strategy

    # NOTE Doesn't save other state
    def save(self, path):
        with open(path, 'w') as f:
            pickle.dump((self.sars_data, self.terminals), f)

    def load(self, path):
        with open(path, 'r') as f:
            self.sars_data, self.terminals = pickle.load(f)
        self.reset()

    def reset(self):
        self.current_sar = None
        self._ep_counter = 0

    @property
    def num_episodes(self):
        return len(self.episode_lens)

    @property
    def num_tuples(self):
        return len(self.sars_data)

    @property
    def all_states(self):
        return [sars[0] for sars in self.sars_data]

    @property
    def all_actions(self):
        return [sars[1] for sars in self.sars_data]

    @property
    def all_rewards(self):
        return [sars[2] for sars in self.sars_data]

    @property
    def all_next_states(self):
        return [sars[3] for sars in self.sars_data]

    def report_sars(self, s, a, r, sn):
        """Reports a SARS tuple. Used for batch adding data.
        """
        self.sars_data.append((s,a,r,s))

    def report_terminal(self, s, a):
        """Reports a terminal state-action. Used for batch adding data.
        """
        self.terminals.append((s,a))

    def report_episode_step(self, s, a, r):
        """Adds a SARS tuple to the current episode. Used for sequentially
        constructing episodes.
        """
        if self.current_sar is None:
            #self.current_sar = (s,a,r)
            pass
        else:
            self.sars_data.append(self.current_sar + (s,))
        self.current_sar = (s,a,r)
        
        # Length bookkeeping
        self._ep_counter += 1

    def report_episode_end(self, s):
        """Reports the end of an episode without a terminal state. Used for
        sequentially constructing episodes.
        """
        self.sars_data.append(self.current_sar + (s,))

        # Length bookkeeping
        self.episode_lens.append(self._ep_counter)
        self.reset()

    def report_episode_terminal(self):
        """Reports a terminal condition, ending the current episode. Used for
        sequentially constructing episodes.
        """
        # self.terminals.append(s)
        # self.sars_data.append(self.current_sar + (s,))
        self.terminals.append(self.current_sar[0:2])

        # Length bookkeeping
        self.episode_lens.append(self._ep_counter)
        self.reset()

    def sample_sars(self, k):
        """Samples from non-terminal SARS tuples
        """
        if self._sstrat == 'uniform':
            return zip(*random.sample(self.sars_data, k))
        else:
            raise ValueError('Invalid sampling strategy')    

    def sample_terminals(self, k):
        """Samples terminal states
        """
        if self._sstrat == 'uniform':
            return zip(*random.sample(self.terminals, k))
