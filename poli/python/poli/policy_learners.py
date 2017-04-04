import poli.gradient_estimators as pge
import optim


class PolicyLearner(object):
    def __init__(self, optimizer):
        self._optimizer = optimizer