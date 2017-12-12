"""Classes for labelled datasets
"""
from dataset import DatasetInterface

class LabeledDatasetTranslator(object):
    """Wraps a DatasetInterface object to provide label-specific methods
    """
    def __init__(self, base):
        self.base = base

    def report_label(self, x, y):
        """Reports an input x and label y
        """
        self.base.report_data(key=None, data=(x,y))

    @property
    def all_data(self):
        return self.base.get_volume(key=None)

    @property
    def num_data(self):
        return len(self.base.get_volume(key=None))

    @property
    def all_inputs(self):
        if self.num_data == 0:
            return []
        return zip(*self.all_data)[0]

    @property
    def all_labels(self):
        if self.num_data == 0:
            return []
        return zip(*self.all_data)[1]
