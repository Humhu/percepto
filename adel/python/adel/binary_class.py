"""Classes for binary classification data
"""

from dataset import DatasetInterface

# NOTE Shouldn't this also implement DatasetInterface
class BinaryDatasetTranslator(object):
    """Wraps a DatasetInterface object to provide binary-specific methods
    """
    def __init__(self, base):
        self.base = base

    def report_positive(self, s):
        self.base.report_data(key=True, data=s)

    def report_negative(self, s):
        self.base.report_data(key=False, data=s)

    def report_data(self, s, c):
        self.base.report_data(key=c, data=s)

    @property
    def all_data(self):
        return zip(self.all_positives, [True] * self.num_positives) +\
               zip(self.all_negatives, [False] * self.num_negatives)

    @property
    def all_inputs(self):
        return self.all_positives, self.all_negatives

    @property
    def num_data(self):
        return self.num_positives + self.num_negatives

    @property
    def all_classes(self):
        return [True] * self.num_positives + [False] * self.num_negatives

    @property
    def all_positives(self):
        return self.base.get_volume(key=True)

    @property
    def all_negatives(self):
        return self.base.get_volume(key=False)

    @property
    def num_positives(self):
        return len(self.all_positives)

    @property
    def num_negatives(self):
        return len(self.all_negatives)
