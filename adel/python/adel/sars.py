"""Classes for SARS tuples
"""
from dataset import DatasetInterface

class SARSDatasetTranslator(object):
    """Wraps a DatasetInterface object to provide SARS-specific methods
    """
    def __init__(self, base):
        self.base = base

    def report_sars(self, s, a, r, sn):
        """Reports a SARS tuple.
        """
        self.base.report_data(key=True, data=(s, a, r, sn))

    def report_terminal(self, s, a):
        """Reports a terminal state-action.
        """
        self.base.report_data(key=False, data=(s, a))

    @property
    def all_data(self):
        return self.all_sars, self.all_terminals

    @property
    def all_sars(self):
        return self.base.get_volume(key=True)

    @property
    def all_terminals(self):
        return self.base.get_volume(key=False)

    @property
    def num_sars(self):
        return len(self.all_sars)

    @property
    def num_terminals(self):
        return len(self.all_terminals)

    @property
    def all_states(self):
        return [i[0] for i in self.all_sars]

    @property
    def all_actions(self):
        return [i[1] for i in self.all_sars]
        
    @property        
    def all_rewards(self):
        return [i[2] for i in self.all_sars]

    @property
    def all_next_states(self):
        return [i[3] for i in self.all_sars]

    @property
    def all_terminal_states(self):
        return [i[0] for i in self.all_terminals]

    @property
    def all_terminal_actions(self):
        return [i[1] for i in self.all_terminals]

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