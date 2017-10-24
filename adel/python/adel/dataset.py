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

    def partition_validation(self, r, method='uniform'):
        k_sars = int(round(r * len(self.sars_data)))
        k_term = int(round(r * len(self.terminals))) 
        if method == 'uniform':
            d_inds = random.sample(range(len(self.sars_data)), k_sars)
            t_inds = random.sample(range(len(self.terminals)), k_term)
        elif method == 'contiguous':
            # For now, just use tail
            n = len(self.sars_data)
            d_inds = range(n-k_sars, n)
            m = len(self.terminals)
            t_inds = range(m-k_term, m)

        cinds = [i for i in range(len(self.sars_data)) if i not in d_inds]
        valid_sars = [self.sars_data[i] for i in d_inds]
        self.sars_data = [self.sars_data[i] for i in cinds]

        cinds = [i for i in range(len(self.terminals)) if i not in t_inds]
        valid_terminals = [self.terminals[i] for i in t_inds]
        self.terminals = [self.terminals[i] for i in cinds]

        return valid_sars, valid_terminals

    # NOTE Doesn't save other state
    def save(self, path):
        with open(path, 'w') as f:
            pickle.dump((self.sars_data, self.terminals), f)

    def load(self, path, append=True):
        with open(path, 'r') as f:
            sars_data, terminals = pickle.load(f)
        if append:
            self.sars_data += sars_data
            self.terminals += terminals
        else:
            self.sars_data = sars_data
            self.terminals = terminals

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
    def num_terminals(self):
        return len(self.terminals)

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
