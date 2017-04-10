"""This module contains learner classes for bandit policies.
"""
import dill
import numpy as np
import random
from itertools import izip
from collections import deque, namedtuple
import math
import scipy.stats as sps

import sampling as samp
import policies as pp
import policy_gradient as ppg
import optim

from joblib import Parallel, delayed

#SarTuple = namedtuple('SarTuple', ['state', 'action', 'reward', 'gen_logprob, curr_logprob'])


def parse_gradient_estimator(spec, **kwargs):
    """Parses a dictionary to construct the appropriate gradient estimator object.
    """
    if 'type' not in spec:
        raise ValueError('Specification must include type!')
    learner_type = spec.pop('type')

    lookup = {'episodic_policy_gradient': EpisodicPolicyGradientEstimator}
    if learner_type not in lookup:
        raise ValueError('Gradient estimator type %s not valid type: %s' %
                         (learner_type, str(lookup.keys())))

    for key, value in kwargs.iteritems():
        spec[key] = value
    return lookup[learner_type](**spec)


def __compute_gradients(policy, traj):
    return [policy.gradient(s, a) for s, a, _, _ in traj]


class EpisodicPolicyGradientEstimator(object):
    """A policy learner that uses importance sampling to reuse past episode
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
                 use_natural_gradient=True, fisher_offset=1E-9,
                 regularizer=None, sampler=None, use_norm_sample=True,
                 use_diag_fisher=False, use_baseline=True,
                 log_weight_lim=float('inf'), min_ess=float('-inf'),
                 max_grad_flip_prob=1.0, n_threads=4):

        if not isinstance(policy, pp.StochasticPolicy):
            raise ValueError(
                'Policy must implement the StochasticPolicy interface.')
        self._policy = policy

        if traj_mode == 'reinforce':
            self._grad_est = ppg.importance_reinforce
        elif traj_mode == 'gpomdp':
            self._grad_est = ppg.importance_gpomdp
        elif traj_mode == 'per':
            self._grad_est = ppg.importance_per_decision
        else:
            raise ValueError('Unsupported trajectory mode')

        self.regularizer = regularizer
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.buffer_size = int(buffer_size)
        self.use_nat_grad = bool(use_natural_gradient)
        self.fish_off = float(fisher_offset)
        self.use_norm_sample = use_norm_sample
        self.use_diag_fisher = use_diag_fisher
        self.use_baseline = use_baseline
        self.log_weight_lim = log_weight_lim

        self.min_ess = float(min_ess)
        self.max_grad_flip_prob = max_grad_flip_prob
        #self.max_dev_ratio = float(max_dev_ratio)
        #self.max_grad_dev = float(max_grad_dev)

        self._pool = Parallel(n_jobs=n_threads)

        self.reset()

    def reset(self):
        """Clears the internal SAR buffer.
        """
        if self.buffer_size != 0:
            self._buffer = deque(maxlen=self.buffer_size)
        else:
            self._buffer = deque()

    @property
    def num_samples(self):
        """Returns the number of buffered episodes.
        """
        return len(self._buffer)

    def get_reward_trace(self):
        """Retrieve the history of rewards.
        """
        return [data[2] for data in self._buffer]

    def report_episode(self, states, actions, rewards, logprobs=None):
        """Adds an episode to the internal buffer.

        Parameters
        ----------
        states   : iterable of numpy 1D array

        actions  : iterable of numpy 1D array

        rewards  : iterable of float

        logprobs : iterable of float (optional, default None)
            The generating log-probabilities for the episode. If None,
            the current policy is used to compute them.
        """

        if logprobs is None:
            logprobs = [self._policy.logprob(s, a)
                        for s, a in izip(states, actions)]

        # TODO Used a namedtuple?
        gradients = [self._policy.gradient(s, a)
                     for s, a in izip(states, actions)]
        self._buffer.append(zip(states, actions, rewards,
                                gradients, logprobs, logprobs))

    def estimate_reward_and_gradient(self, x=None):
        """Estimate both the expected reward and policy gradient.
        """
        if x is not None:
            theta = self._policy.get_theta()
            self._policy.set_theta(x)

        samples = self.__sample_buffer()
        obj, grad = self.__estimate(samples, est_reward=True, est_grad=True)
        if self.regularizer is not None:
            reg, dreg = self.regularizer.compute_objective_and_gradient()
            obj += reg
            grad += dreg

        if x is not None:
            self._policy.set_theta(theta)
        return obj, grad

    def remove_unlikely_trajectories(self, min_log_weight=None):
        if min_log_weight is None:
            min_log_weight = -self.log_weight_lim

        trajectories = list(self._buffer)
        self.reset()
        for traj in trajectories:
            s, a, _, g, q, p = izip(*traj)
            p = [self._policy.logprob(si, ai) for si, ai in izip(s, a)]
            p = np.sum(p)
            q = np.sum(q)
            log_weight = np.sum(p - q)
            if log_weight < min_log_weight:
                print 'Dropping trajectory with log_weight %f' % log_weight
            else:
                self._buffer.append(traj)

    def update_buffer(self):
        data = list(self._buffer)
        self.reset()
        for dat in data:
            s, a, r, g, q, p = izip(*dat)
            g = [self._policy.gradient(si, ai) for si, ai in izip(s, a)]
            p = [self._policy.logprob(si, ai) for si, ai in izip(s, a)]
            self._buffer.append(zip(s, a, r, g, q, p))

    def __sample_buffer(self):
        """Returns a set of sample indices.

        Returns None if there are not enough samples (num samples < batch size)
        """
        inds = range(self.num_samples)
        random.shuffle(inds)
        return inds

    def __estimate(self, inds, est_reward, est_grad):
        """Perform importance sampling on a set of samples to estimate
        expected reward and gradient.
        """
        # data = self._pool(delayed(_process_traj)(self._policy, traj)
        #                  for traj in self._buffer)

        if len(inds) < self.batch_size:
            return None, None

        data = [zip(*self._buffer[ind]) for ind in inds]
        _, _, rs, gs, qs, ps = zip(*data)
        rew, grad, ess, rvar, gvar = self._grad_est(rewards=rs,
                                                    gradients=gs,
                                                    p_tar=ps,
                                                    p_gen=qs,
                                                    use_baseline=self.use_baseline,
                                                    use_natural_grad=self.use_nat_grad,
                                                    fisher_diag=self.use_diag_fisher,
                                                    normalize=self.use_norm_sample,
                                                    log_weight_lim=self.log_weight_lim,
                                                    ret_diagnostics=True)

        # Confidence interval shrinks with root N
        gsd = np.sqrt(np.diag(gvar / ess))
        # sdr = gsd / np.abs(grad)
        print 'ESS: %f' % ess
        print 'Grad: %s' % np.array_str(grad)
        print 'Grad SD: %s' % np.array_str(gsd)

        pass_min_ess = ess >= self.min_ess
        #pass_max_dev = gsd <= self.max_grad_dev
        #pass_max_ratio = sdr <= self.max_dev_ratio

        cdf_vals = np.array([sps.norm(loc=abs(val), scale=sd).cdf(0)
                             for val, sd in izip(grad, gsd)])
        print 'Grad sign flip probs: %s' % np.array_str(cdf_vals)
        pass_cdf = np.all(cdf_vals <= self.max_grad_flip_prob)
        # Must pass one of two conditions
        if pass_min_ess or pass_cdf:
            # or np.all(np.logical_or(pass_max_dev, pass_max_ratio)):
            return rew, grad
        else:
            print 'Gradient does not pass tests, skipping...'
            return None, None

        # rs = []
        # gs = []
        # ps = []
        # qs = []
        # for ssize, next_ind in enumerate(inds):
        #     ssize += 1  # Starts at 0

        #     traj = self.__augment_trajectory(self._buffer[next_ind])
        #     _, _, r, g, q, p = zip(*traj)
        #     rs.append(r)
        #     gs.append(g)
        #     ps.append(p)
        #     qs.append(q)

        #     traj_ps = [np.sum(pi) for pi in ps]
        #     traj_qs = [np.sum(qi) for qi in qs]
        #     mw, ess = samp.importance_sample_ess(p_gen=traj_qs,
        #                                          p_tar=traj_ps,
        #                                          log_weight_lim=self.log_weight_lim)

        #     if self.batch_size == 0 and ssize < self.num_samples:
        #         continue
        #     elif self.batch_size > 0 and ess < self.batch_size:
        #         p = math.exp(np.sum(p))
        #         q = math.exp(np.sum(q))
        #         print 'Added w %f (p %f q %f)' % (p / q, p, q)
        #         print 'ESS %f < desired %f' % (ess, self.batch_size)
        #         continue

        #     print 'Using %d samples to achieve ESS %f' % (ssize, ess)
        #     return self._grad_est(rewards=rs,
        #                           gradients=gs,
        #                           p_tar=ps,
        #                           p_gen=qs,
        #                           use_baseline=self.use_baseline,
        #                           use_natural_grad=self.use_nat_grad,
        #                           fisher_diag=self.use_diag_fisher,
        #                           normalize=self.use_norm_sample,
        #                           log_weight_lim=self.log_weight_lim)

        # print 'Could not achieve desired sample size'
        # return None, None


def _process_traj(policy, traj):
    s, a, r, p = zip(*traj)
    c, g = zip(*[(policy.logprob(s, a), policy.gradient(s, a))
                 for s, a in izip(s, a)])
    return r, g, c, p
