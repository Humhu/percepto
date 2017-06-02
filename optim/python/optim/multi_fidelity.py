"""Multi-fidelity acquisition functions and reward models based on work of 
Kandasamy et al.
"""

import abc
import numpy as np
from optim.reward_models import *
from optim.bayesian_optimization import UCBAcquisition


def parse_mf_reward_model(spec):
    if 'type' not in spec:
        raise ValueError('Specification must include type!')

    model_type = spec.pop('type')
    lookup = {'tabular': TabularRewardModel,
              'gaussian_process': GaussianProcessRewardModel,
              'random_forest': RandomForestRewardModel}
    if model_type not in lookup:
        raise ValueError('Model type %s not valid type: %s' %
                         (model_type, str(lookup.keys())))

    num_fidelities = int(spec.pop('num_fidelities'))
    biases = spec.pop('biases')
    return MultiFidelityWrapperModel(num_fidelities=num_fidelities,
                                     biases=biases,
                                     base_model=lookup[model_type],
                                     **spec)


def pick_acquisition_mf(acq_func, optimizer, gammas, x_init):
    # 1. Find arm to pull
    x, acq = optimizer.optimize(x_init=x_init, func=acq_func)

    # 2. Determine what fidelity to operate at
    bounds = acq_func.get_bounds(x)
    gamma_aug = np.hstack((gammas, float('-inf')))
    print bounds
    print bounds > gamma_aug
    # NOTE This will always have at least one element since the last gamma
    # is -inf
    fid = np.where(bounds > gamma_aug)[0][0]
    return fid, x


class MultiFidelityRewardModel(RewardModel):
    """Base class for all multi-fidelity reward models.

    Note that fidelity index convention assumes fid = 0 corresponds
    to the lowest fidelity, with increasing index corresponding to
    increasingly higher fidelity.
    """
    __metaclass__ = abc.ABCMeta

    def report_sample(self, x, reward):
        """Updates the reward model assuming that the first
        element of x is the fidelity index
        """
        if x[0] != round(x[0]):
            raise ValueError('Fidelity index must be integer')
        fid = int(x[0])
        x_sub = x[1:]
        self.report_mf_sample(fid=fid, x=x_sub, reward=reward)

    @abc.abstractmethod
    def report_mf_sample(self, fid, x, reward):
        """Update the reward model with a test result.

        Parameters:
        -----------
        fid    : integer fidelity index
            Index corresponding to fidelity used
        x      : ?
        reward : numeric
            The received reward
        """
        return

    def predict(self, x, return_std=False):
        """Predict the reward assuming that the first element of
        x is the fidelity index
        """
        if x[0] != round(x[0]):
            raise ValueError('Fidelity index must be integer')
        fid = int(x[0])
        x_sub = x[1:]
        return self.predict_mf(fid=fid, x=x_sub, return_std=return_std)

    @abc.abstractmethod
    def predict_mf(self, fid, x, return_std=False):
        """Predict the reward and standard deviation of an input
        at a specified fidelity.
        """
        pass

    @abc.abstractproperty
    def num_fidelities(self):
        pass


class MultiFidelityWrapperModel(MultiFidelityRewardModel):
    """A set of base reward models corresponding to different levels of
    fidelity.

    Parameters
    ----------
    num_fidelities : positive integer
        The number of fidelity levels
    biases         : numeric or iterable of (num_fidelities -1) numerics
        The biases for all lower fidelities (not highest fidelity)
    base_model     : RewardModel class
        The base class to use for each fidelity level
    """

    def __init__(self, num_fidelities, biases, base_model, **kwargs):
        self._models = [base_model(**kwargs) for _ in range(num_fidelities)]

        if not np.iterable(biases):
            fids = np.arange(1, num_fidelities)[::-1]
            biases = fids * biases
        if len(biases) != num_fidelities - 1:
            raise ValueError('Must define %d biases for %d fidelities' %
                             (num_fidelities - 1, num_fidelities))
        self._biases = np.array(biases)

    @property
    def num_fidelities(self):
        return len(self._models)

    def clear(self):
        for m in self._models:
            m.clear()

    def batch_optimize(self, n_restarts=None):
        for m in self._models:
            m.batch_optimize(n_restarts)

    def report_mf_sample(self, fid, x, reward):
        self.__check_fid(fid)
        self._models[fid].report_sample(x=x, reward=reward)

    def predict_mf(self, fid, x, return_std=False):
        self.__check_fid(fid)
        return self._models[fid].predict(x=x, return_std=return_std)

    def __check_fid(self, fid):
        """Make sure that the requested fidelity is valid.
        """
        if fid >= self.num_fidelities:
            raise IndexError('Requested fidelity %d but only has %d!' %
                             (fid, self.num_fidelities))


class MultiFidelityUCBAcquisition(UCBAcquisition):
    def __init__(self, model):
        if not isinstance(model, MultiFidelityRewardModel):
            raise ValueError('model must be MultiFidelityRewardModel')

        super(MultiFidelityUCBAcquisition, self).__init__(model)

    def get_bounds(self, x):
        """Overrides UCBAcquisition.get_bounds to return a list of
        bounds at each level of fidelity
        """
        bounds = []
        for i in range(self.model.num_fidelities):
            x_aug = np.hstack((i, x))
            bound = super(MultiFidelityUCBAcquisition, self).get_bounds(x_aug)
            bounds.append(bound)
        return bounds

    def predict_mf(self, fid, x):
        x_aug = np.hstack((fid, x))
        return super(MultiFidelityUCBAcquisition, self).predict(x_aug)

    def __call__(self, x):
        ests = []
        for i in range(self.model.num_fidelities):
            x_aug = np.hstack((i, x))
            ucb = super(MultiFidelityUCBAcquisition, self).__call__(x_aug)
            ests.append(ucb)
        return np.min(ests)
