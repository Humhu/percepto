"""This module contains learner classes for bandit policies.
"""
import numpy as np
import random
from itertools import izip
from collections import deque

import poli.policies as pp
import poli.policy_gradient as ppg


def parse_learner(spec, policy):
    """Parses a dictionary to construct the appropriate learner object.
    """
    if 'type' not in spec:
        raise ValueError('Specification must include type!')
    learner_type = spec.pop('type')

    lookup = {'bandit_gradient': BanditPolicyGradientLearner}
    if learner_type not in lookup:
        raise ValueError('Learner type %s not valid type: %s' %
                         (learner_type, str(lookup.keys())))

    return lookup[learner_type](policy=policy, **spec)


class BanditPolicyGradientLearner(object):
    """A bandit policy learner that uses importance sampling to reuse past data
    for estimating policy gradients and expected rewards.

    Parameters
    ----------
    policy               : Policy object
        The policy to modify and learn
    batch_size           : integer (default 0)
        Number of samples to use per gradient calculation. 0 means all samples.
    buffer_size          : integer (default 0)
        Max number of samples to keep. 0 means keep all samples.
    use_log_probs        : boolean (default True)
        Whether or not to use log probabilities for importance sampling
    use_natural_gradient : boolean (default True)
        Whether or not to use the natural policy gradient
    inv_fisher_offset    : float (default 1E-9)
        Offset added to the diagonal of the inverse Fisher matrix for conditioning
    seed                 : int (default None)
        Seed for the random number generator
    """

    def __init__(self, policy, batch_size=0, buffer_size=0, use_log_probs=True,
                 use_natural_gradient=True, inv_fisher_offset=1E-9, seed=None):

        if not isinstance(policy, pp.StochasticPolicy):
            raise ValueError(
                'Policy must implement the StochasticPolicy interface.')
        self._policy = policy

        # TODO Do we need to wrap random so its repeatable with threading?
        seed = int(seed)
        if seed is not None:
            random.seed(seed)

        self._batch_size = int(batch_size)
        self._buffer_size = int(buffer_size)
        self._use_log_probs = bool(use_log_probs)
        self._use_nat_grad = bool(use_natural_gradient)
        self._inv_fish_off = float(inv_fisher_offset)

        self.reset()

    def reset(self):
        """Clears the internal SAR buffer.
        """
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

    def report_sample(self, state, action, reward):
        """Adds a SAR sample to the internal buffer.
        """
        if self._use_log_probs:
            prob = self._policy.logprob(state, action)
        else:
            prob = self._policy.prob(state, action)
        self._buffer.append((state, action, reward, prob))

        if self._buffer_size != 0:
            while self.num_samples > self._buffer_size:
                self._buffer.popleft()

    def estimate_reward(self, sample=True):
        """Estimate the policy expected reward.
        """
        samples = self.__sample_buffer(sample)
        return self.__estimate(samples, est_reward=True, est_grad=False)[0]

    def estimate_gradient(self, sample=True):
        """Estimate the policy gradient.
        """
        samples = self.__sample_buffer(sample)
        return self.__estimate(samples, est_reward=False, est_grad=True)[1]

    def estimate_reward_and_gradient(self, sample=True):
        """Estimate both the expected reward and policy gradient.
        """
        samples = self.__sample_buffer(sample)
        return self.__estimate(samples, est_reward=True, est_grad=True)

    # TODO Remove and require use of an optimization object externally instead?
    def step(self, sample=True):
        """Perform one iteration of optimization on the policy.
        """
        grad = self.estimate_gradient(sample)
        if grad is None:
            return

        if np.linalg.norm(grad) > 1:
            grad = grad / np.linalg.norm(grad)

        # TODO Hacked for now, generalize after testing
        theta_new = self._policy.get_theta() + 1E-1 * grad
        self._policy.set_theta(theta_new)

    def __sample_buffer(self, sample=True):
        """Returns a set of samples from the internal buffer, or override
        settings to use all samples.

        Returns None if there are not enough samples (num samples < batch size)
        """
        if self.num_samples < self._batch_size:
            return None

        if self._batch_size != 0:
            inds = random.sample(range(self.num_samples), self._batch_size)
            samples = [self._buffer[i] for i in inds]
        else:
            samples = self._buffer

        return samples

    def __estimate(self, samples, est_reward, est_grad):
        """Perform importance sampling on a set of samples to estimate
        expected reward and gradient.
        """
        if samples is None:
            return None, None
        states, actions, rewards, past_probs = zip(*samples)

        gradients = [self._policy.gradient(x, a)
                     for x, a in izip(states, actions)]
        if self._use_log_probs:
            curr_probs = [self._policy.logprob(
                x, a) for x, a in izip(states, actions)]
        else:
            curr_probs = [self._policy.prob(x, a)
                          for x, a in izip(states, actions)]

        return ppg.bandit_importance(rewards=rewards,
                                     gradients=gradients,
                                     curr_probs=curr_probs,
                                     past_probs=past_probs,
                                     log_prob=self._use_log_probs,
                                     use_natural_grad=self._use_nat_grad,
                                     inv_fisher_offset=self._inv_fish_off,
                                     est_reward=est_reward,
                                     est_grad=est_grad)
