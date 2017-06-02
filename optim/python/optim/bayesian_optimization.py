"""Bayesian optimization exploration rates and acquisition functions
"""
import numpy as np
import optim.reward_models as rewards
#from sklearn.neighbors import KernelDensity


def pick_acquisition(acq_func, optimizer, x_init):
    return optimizer.optimize(x_init=x_init,
                              func=acq_func)


class UCBExplorationRate(object):
    """Implements a common logarithmic exploration rate UCB
    of beta = alpha * dim * log( gamma * t )
    """

    def __init__(self, dim, alpha=0.2, gamma=2):
        self._dim = dim
        self._alpha = alpha
        self._gamma = gamma

    def __call__(self, t):
        beta = self._alpha * self._dim * np.log(t * self._gamma)
        return np.sqrt(beta)

# TODO Implement more acquisition functions!


class UCBAcquisition(object):
    """Implements the Upper Confidence Bound acquisition function.
    """

    def __init__(self, model):
        if not isinstance(model, rewards.RewardModel):
            raise ValueError('model must be a RewardModel')
        self.model = model
        self._exploration_rate = 1.0

    @property
    def exploration_rate(self):
        """Return the scale on the prediction uncertainty.
        """
        return self._exploration_rate

    @exploration_rate.setter
    def exploration_rate(self, b):
        if b < 0:
            raise RuntimeWarning('Negative value %f for exploration_rate' % b)
        self._exploration_rate = b

    def predict(self, x):
        return self.model.predict(x, return_std=True)

    def get_bounds(self, x):
        _, pred_sd = self.predict(x)
        return self.exploration_rate * pred_sd

    def __call__(self, x):
        pred_mean, pred_sd = self.predict(x)
        if np.isnan(pred_sd):
            raise ValueError('Nan SD %f' % pred_sd)
        return pred_mean + self.exploration_rate * pred_sd


class ContextualUCBAcquisition(UCBAcquisition):
    """An upper confidence bound on expected reward over empirical contexts.
    """

    def __init__(self, model, mode, contexts, **kwargs):
        super(ContextualUCBAcquisition, self).__init__(model)

        self.contexts = contexts  # reference
        self.mode = mode
        if self.mode == 'empirical':
            pass
        elif self.mode == 'kde':
            raise ValueError('kde mode not supported yet!')
            #self.kde = KernelDensity(**kwargs)
        else:
            raise ValueError('Unknown mode: %s' % self.mode)

    def predict(self, x):
        if self.mode == 'empirical':
            N = len(self.contexts)
            acts = np.tile(x, (N, 1))
            X = np.hstack((acts, self.contexts))
            pred_mean, pred_sd = self.model.predict(X, return_std=True)
        return np.mean(pred_mean), np.mean(pred_sd)

    def __call__(self, x):
        pred_mean, pred_sd = self.predict(x)
        return pred_mean + self.exploration_rate * pred_sd
