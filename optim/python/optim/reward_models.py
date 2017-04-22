"""Reward models (response surface models) for Bayesian optimization and bandit algorithms.
"""
import abc
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor as GPRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, Matern
from gp_extras.kernels import HeteroscedasticKernel


def parse_reward_model(spec):
    """Takes a specification dictionary and returns an initialized reward model.
    """
    if 'type' not in spec:
        raise ValueError('Specification must include type!')

    model_type = spec.pop('type')
    lookup = {'tabular': TabularRewardModel,
              'gaussian_process': GaussianProcessRewardModel,
              'random_forest': RandomForestRewardModel}
    if model_type not in lookup:
        raise ValueError('Model type %s not valid type: %s' %
                         (model_type, str(lookup.keys())))
    return lookup[model_type](**spec)


class RewardModel(object):
    """Base class for all reward models.

    Reward models predict the reward mean and standard deviation.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def report_sample(self, x, reward):
        """Update the reward model with a test result.

        Parameters:
        -----------
        x     : TODO (?)
            Tested input
        reward: numeric
            Received reward
        """
        return

    @abc.abstractmethod
    def predict(self, x):
        """Predict the reward and standard deviation of an input's rewards.

        Parameters:
        -----------
        x: hashable
            The input value to predict.

        Returns:
        --------
        mean: numeric
            The estimated x reward mean
        variance: numeric
            The estimated x reward standard deviation
        """
        return

    @abc.abstractmethod
    def clear(self):
        """Clears and resets the reward model.
        """
        pass

    def get_parameters(self):
        """Return this model's parameters. If it has no parameters, return None.
        """
        return None

    def set_parameters(self, t):
        """Set this model's parameters.
        """
        if t is not None:
            raise ValueError('This model has no parameters!')


class TabularRewardModel(RewardModel):
    """Models rewards with as a lookup table.

    Implemented as a dictionary that maps from hashable inputs
    to histories of rewards.

    Parameters:
    -----------
    default_mean: numeric (default 0)
        The mean estimate to return for inputs with no data.
    default_std: numeric (default inf)
        The standard deviation estimate to return for inputs with no data.
    """

    def __init__(self, default_mean=0, default_std=float('inf')):
        self._table = {}
        self._default_mean = default_mean
        self._default_std = default_std

    def report_sample(self, x, reward):
        try:
            self._table[x].append(reward)
        except KeyError:
            self._table[x] = []
            self._table[x].append(reward)

    def predict(self, x):
        if not hasattr(x, '__iter__'):
            x = [x]

        pred_means = []
        pred_stds = []
        for xi in x:
            if xi in self._table:
                y = self._table[xi]
                pred_means.append(np.mean(y))
                pred_stds.append(np.std(y))
            else:
                pred_means.append(self._default_mean)
                pred_stds.append(self._default_std)

    def clear(self):
        self._table = {}

    @property
    def num_inputs(self):
        return len(self._table)

    def num_samples(self, x):
        """Returns the number of samples held for the specified x.
        """
        if x not in self._table:
            return 0
        else:
            return len(self._table[x])


class GaussianProcessRewardModel(RewardModel):
    """
    Models rewards with a Gaussian process regressor.

    Implemented with a modified version of scikit-learn's Gaussian Process
    Regressor class.

    The GP is updated online as samples are added. As such, hyperparameters
    for the GP are fit in batch after a threshold number of samples are collected.
    The hyperparameters are then refined afterwards as more samples are added
    until the number of samples passes an upper threshold, after which the
    hyperparameters are no longer updated. This helps avoid highly expensive
    refinement which has computational complexity of O(N^3) in number of samples.

    Parameters:
    -----------
    min_samples: integer (default 100)
        The number of samples after which initial batch hyperparameter
        fitting is performed.
    batch_retries: integer (default 20)
        The number of random restarts for the initial hyperparameter fit.
    refine_ll_delta: numeric (default 1.0)
        The hyperparameters are refined after the average GP marginal
        log-likelihood decreases by this much since the last refinement.
    max_samples: integer (default 1000)
        The number of samples after which hyperparameters are no longer
        refined.

    Other Keyword Parameters:
    -------------------
    Refer to sklearn.gaussian_process.GaussianProcessRegressor's __init__
    """

    def __init__(self, min_samples=10, batch_retries=20, refine_ll_delta=1.0,
                 refine_retries=1, hetero_centers=[], hetero_gamma=5.0, **kwargs):

        if len(hetero_centers) == 0:
            noise = WhiteKernel(1.0, (1e-3, 1e3))
        else:
            noise = HeteroscedasticKernel.construct(hetero_centers,
                                                    1.0, (1e-3, 1e3),
                                                    gamma=hetero_gamma,
                                                    gamma_bounds='fixed')

        # TODO Option for Matern vs RBF
        self._kernel_base = ConstantKernel(
            1.0, (1e-3, 1e3)) * Matern(1.0, (1e-3, 1e3), nu=1.5)
        self._kernel_noisy = self._kernel_base + noise

        self.gp = GPRegressor(kernel=self._kernel_noisy,
                              kernel_noiseless=self._kernel_base,
                              **kwargs)
        self.hp_min_samples = min_samples
        self.hp_batch_retries = batch_retries
        self.hp_refine_ll_delta = refine_ll_delta
        self.hp_refine_retries = refine_retries
        self.hp_init = False
        self.last_ll = None

        # TODO Track the y mean and update it during batch operations

    def report_sample(self, x, reward):
        self.gp.add_data(x, reward, incremental=True)

        num_samples = len(self.gp.training_y)
        if not self.hp_init and num_samples > self.hp_min_samples:
            self.gp.batch_update(num_restarts=self.hp_batch_retries)
            self.hp_init = True
            self.last_ll = self.gp.log_marginal_likelihood()

        # Wait until we've initialized
        if not self.hp_init:
            return

        current_ll = self.gp.log_marginal_likelihood()
        if current_ll > self.last_ll:
            self.last_ll = current_ll
        elif current_ll < self.last_ll - self.hp_refine_ll_delta:
            self.gp.batch_update(num_restarts=self.hp_refine_retries)
            self.last_ll = self.gp.log_marginal_likelihood()

    def predict(self, x, return_std=False):
        x = np.atleast_2d(x)
        if len(x.shape) > 2:
            raise ValueError('x must be at most 2D')
        # TODO return-std might have a bug, use return_cov instead?
        pred_mean, pred_std = self.gp.predict(x, return_std=True)
        if return_std:
            return np.squeeze(pred_mean), np.squeeze(pred_std)
        else:
            return np.squeeze(pred_mean)

    def clear(self):
        # TODO
        pass

    def fit(self, X, y):
        """Initialize the model from lists of inputs and corresponding rewards.

        Parameters
        ----------
        X : Iterable of inputs
        Y : Iterable of corresponding rewards
        """
        if len(X) != len(y):
            raise RuntimeError('X and Y lengths must be the same!')

        self.gp.fit(X, y, num_restarts=self.hp_batch_retries,
                    optimize_hyperparams=True)
        self.last_ll = self.gp.log_marginal_likelihood()
        self.hp_init = True

    @property
    def kernel_noiseless(self):
        return self._kernel_base

    @property
    def kernel_noisy(self):
        return self._kernel_noisy

    @property
    def num_samples(self):
        return len(self.gp.training_X)

    def get_parameters(self):
        return self.gp.theta

    def set_parameters(self, t):
        self.gp.theta = t

    @property
    def model(self):
        return self.gp


class RandomForestRewardModel(RewardModel):
    """Models rewards with a random forest.

    Uses a modified version of scikit-learn which returns the predictions of all
    trees in the forest to predict both the mean and variance of the prediction.

    Parameters
    ----------
    incremental : boolean (default False)
        Whether to fit the forest incrementally
    inc_n_trees : integer (default 1)
        If incremental, the number of trees to add for each sample
    min_samples : integer (default 10)
        The minimum number of samples before the regressor is fitted
    """

    def __init__(self, incremental=False, inc_n_trees=1, min_samples=10, **kwargs):
        self._forest = RandomForestRegressor(warm_start=incremental, **kwargs)
        self._min_samples = min_samples
        self._inc_n_trees = inc_n_trees

        self._initialized = False
        self._X = []  # TODO Use a more efficient container?
        self._Y = []

    def report_sample(self, x, reward):
        x = np.atleast_1d(x)
        self._X.append(x)
        self._Y.append(reward)

        if self.num_samples < self._min_samples:
            return

        if self._forest.warm_start:
            self._forest.n_estimators += self._inc_n_trees

        self._forest.fit(self._X, self._Y)

    def predict(self, x):
        x = np.atleast_2d(x)
        if len(x.shape) > 2:
            raise ValueError('x must be at most 2D')

        outs = self._forest.retrieve(x)
        pred_mean = np.mean(outs, axis=0)
        pred_sd = np.std(outs, axis=0)
        return np.squeeze(pred_mean), np.squeeze(pred_sd)

    def clear(self):
        # TODO
        pass

    @property
    def num_samples(self):
        return len(self._X)
