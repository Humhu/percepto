"""Classes for aggregating and sampling data
"""

import random

# Operations we will want to support:
# 1. Mixing states/actions to generate synthetic terminal SA tuples
# 2. Randomly selecting SARS for value training

class SARSDataset(object):
    """Stores episodic SARS data tuples.
    """

    # TODO Different sampling strategies?
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

    def report_step(self, s, a, r):
        """Adds a SARS tuple to the current episode.
        """
        if self.current_sar is None:
            #self.current_sar = (s,a,r)
            pass
        else:
            self.sars_data.append(self.current_sar + (s,))
        self.current_sar = (s,a,r)
        
        # Length bookkeeping
        self._ep_counter += 1

    def report_terminal(self, s):
        """Reports a terminal state, ending the current episode.
        """
        self.terminals.append(s)
        self.sars_data.append(self.current_sar + (s,))
        self.current_sar = None

        # Length bookkeeping
        self.episode_lens.append(self._ep_counter)
        self._ep_counter = 0

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
            return random.sample(self.terminals, k)
