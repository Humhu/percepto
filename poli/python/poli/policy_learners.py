"""This module contains learner classes for bandit policies.
"""
import dill
import numpy as np
import random
from itertools import izip
from collections import deque
from math import sqrt
import policies as pp
import policy_gradient as ppg

from joblib import Parallel, delayed


def parse_learner(spec, policy):
    """Parses a dictionary to construct the appropriate learner object.
    """
    if 'type' not in spec:
        raise ValueError('Specification must include type!')
    learner_type = spec.pop('type')

    lookup = {'policy_gradient': PolicyGradientLearner}
    if learner_type not in lookup:
        raise ValueError('Learner type %s not valid type: %s' %
                         (learner_type, str(lookup.keys())))

    return lookup[learner_type](policy=policy, **spec)


def __compute_gradients(policy, traj):
    return [policy.gradient(s, a) for s, a, _, _ in traj]


class PolicyGradientLearner(object):
    """A bandit policy learner that uses importance sampling to reuse past data
    for estimating policy gradients and expected rewards.

    Parameters
    ----------
    policy               : Policy object
        The policy to modify and learn
    traj_mode       : string ('reinforce', 'pgt', 'gpomdp')
        Algorithm to use for processing trajectories
    batch_size           : integer (default 0)
        Number of samples to use per gradient calculation. 0 means all samples.
    buffer_size          : integer (default 0)
        Max number of samples to keep. 0 means keep all samples.
    use_log_probs        : boolean (default True)
        Whether or not to use log probabilities for importance sampling
    use_natural_gradient : boolean (default True)
        Whether or not to use the natural policy gradient
    fisher_offset        : float (default 1E-9)
        Offset added to the diagonal of the inverse Fisher matrix for conditioning
    seed                 : int (default None)
        Seed for the random number generator
    """

    def __init__(self, policy, traj_mode, batch_size=0, buffer_size=0,
                 use_natural_gradient=True, fisher_offset=1E-9, seed=None,
                 use_norm_sample=True, use_diag_fisher=False, use_baseline=True,
                 n_threads=4):

        if not isinstance(policy, pp.StochasticPolicy):
            raise ValueError(
                'Policy must implement the StochasticPolicy interface.')
        self._policy = policy

        # TODO Do we need to wrap random so its repeatable with threading?
        seed = int(seed)
        if seed is not None:
            random.seed(seed)

        self._traj_mode = traj_mode

        self._batch_size = int(batch_size)
        self._buffer_size = int(buffer_size)
        self._use_nat_grad = bool(use_natural_gradient)
        self._fish_off = float(fisher_offset)
        self._use_norm_sample = use_norm_sample
        self._use_diag_fisher = use_diag_fisher
        self._use_baseline = use_baseline
        self._pool = Parallel(n_jobs=n_threads)

        self.reset()

    def reset(self):
        """Clears the internal SAR buffer.
        """
        if self._buffer_size != 0:
            self._buffer = deque(maxlen=self._buffer_size)
        else:
            self._buffer = deque()

    @property
    def num_samples(self):
        """Returns the number of buffered SAR samples.
        """
        return len(self._buffer)

    def get_reward_trace(self):
        """Retrieve the history of rewards.
        """
        return [data[2] for data in self._buffer]

    def report_trajectory(self, states, actions, rewards, logprobs=None):
        """Adds a trajectory to the internal buffer.

        Parameters
        ----------
        states   : iterable of numpy 1D array
        actions  : iterable of numpy 1D array
        rewards  : iterable of float
        logprobs : iterable of float (optional, default None)
        """

        if logprobs is None:
            logprobs = [self._policy.logprob(s, a)
                        for s, a in izip(states, actions)]

        self._buffer.append(zip(states, actions, rewards, logprobs))

    def estimate_reward(self, x=None, sample=True):
        """Estimate the policy expected reward.
        """
        if x is not None:
            theta = self._policy.get_theta()
            self._policy.set_theta(x)

        samples = self.__sample_buffer(sample)
        ret = self.__estimate(samples, est_reward=True, est_grad=False)[0]

        if x is not None:
            self._policy.set_theta(theta)
        return ret

    def estimate_gradient(self, x=None, sample=True):
        """Estimate the policy gradient.
        """
        if x is not None:
            theta = self._policy.get_theta()
            self._policy.set_theta(x)

        samples = self.__sample_buffer(sample)
        ret = self.__estimate(samples, est_reward=False, est_grad=True)[1]

        if x is not None:
            self._policy.set_theta(theta)
        return ret

    def estimate_reward_and_gradient(self, x=None, sample=True):
        """Estimate both the expected reward and policy gradient.
        """
        if x is not None:
            theta = self._policy.get_theta()
            self._policy.set_theta(x)

        samples = self.__sample_buffer(sample)
        ret = self.__estimate(samples, est_reward=True, est_grad=True)

        if x is not None:
            self._policy.set_theta(theta)
        return ret

    def __sample_buffer(self, sample=True):
        """Returns a set of samples from the internal buffer, or override
        settings to use all samples.

        Returns None if there are not enough samples (num samples < batch size)
        """
        if self.num_samples < self._batch_size:
            return None

        sum_rewards = [np.sum([t[2] for t in traj]) for traj in self._buffer]
        pick_probs = sum_rewards / np.sum(sum_rewards)

        if self._batch_size != 0:

            inds = random.sample(range(self.num_samples), self._batch_size)
            # inds = np.random.choice(self.num_samples, size=self._batch_size,
            #                        p=pick_probs)

            samples = [self._buffer[i] for i in inds]
        else:
            samples = self._buffer

        return samples

    def __comp_gradients(self, traj):
        return [self._policy.gradient(s, a) for s, a, _, _ in traj]

    def __estimate(self, samples, est_reward, est_grad):
        """Perform importance sampling on a set of samples to estimate
        expected reward and gradient.
        """
        if samples is None:
            return None, None

        rewards, gradients, curr_probs, past_probs = zip(
            *self._pool(delayed(_process_traj)(self._policy, traj) for traj in samples))

        # rewards = [[r for s, a, r, p in traj] for traj in samples]
        # past_probs = [[p for s, a, r, p in traj] for traj in samples]
        # gradients = [[self._policy.gradient(
        #     s, a) for s, a, r, p in traj] for traj in samples]
        # curr_probs = [[self._policy.logprob(
        #     s, a) for s, a, r, p in traj] for traj in samples]

        rewb, gradb = ppg.importance_gpomdp(rewards=rewards,
                                            gradients=gradients,
                                            p_tar=curr_probs,
                                            p_gen=past_probs,
                                            use_baseline=self._use_baseline,
                                            use_natural_grad=self._use_nat_grad,
                                            fisher_diag=self._use_diag_fisher,
                                            normalize=self._use_norm_sample)

        return rewb, gradb


def _process_traj(policy, traj):
    s, a, r, p = zip(*traj)
    c, g = zip(*[(policy.logprob(s, a), policy.gradient(s, a))
                 for s, a in izip(s, a)])
    return r, g, c, p
