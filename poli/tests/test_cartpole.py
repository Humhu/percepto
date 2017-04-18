import numpy as np
import scipy as sp
import gym
import poli
import optim
import math
from itertools import izip

import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')
env.reset()
a_scale = env.action_space.high - env.action_space.low


class DeterministicLinearPolicy(object):
    def __init__(self, indim, outdim):
        self.A = np.zeros((outdim, indim))

    def sample_action(self, state):
        return np.dot(self.A, state)

    def get_theta(self):
        return self.A.flatten()

    def set_theta(self, a):
        self.A = np.reshape(a, self.A.shape)

    def logprob(self, s, a):
        return 0


def execute(state, policy, enable_safety):
    x, xdot, th, thdot, _ = state
    # print '%f %f' % (th, thdot)
    action = policy.sample_action(state)
    if enable_safety:
        if thdot > 0.5 and action < 0:
            print 'Executing safety action since thdot > 0.5'
            return np.array([0]), True
        if thdot < -0.5 and action > 0:
            print 'Executing safety action since thdot < 0.5'
            return np.array([0]), True
    return action, False


def run_trial(policy, pfunc, continuous, max_len=200):

    observation = env.reset()
    observation = pfunc(observation)

    observations = []
    rewards = []
    actions = []
    logprobs = []
    num_safety = 0
    max_safety = 10
    for t in range(max_len):
        env.render()
        # action, used_safety = execute(observation,
        #                              policy,
        #                              enable_safety=num_safety < max_safety)
        action, used_safety = policy.sample_action(observation), False
        if used_safety:
            num_safety += 1
        action[action > 1] = 1
        action[action < -1] = -1

        actions.append(action)
        observations.append(observation)
        logprobs.append(policy.logprob(observation, action))

        observation, reward, done, info = env.step(a_scale * action)
        observation = pfunc(observation)
        # Reward should correspond to current action

        if continuous:
            if done:
                rewards.append(-10)
                print 'Episode terminated'
                break
            else:
                rewards.append(0.0)
        else:
            if done:
                rewards.append(0.0)
                print 'Episode terminated'
                break
            else:
                rewards.append(1.0)

    print 'Episode finished after %d timesteps' % (t + 1)
    return observations, actions, rewards, logprobs


def train_policy(policy, optimizer, estimator, continuous, n_iters, t_len):
    trials = []
    grads = []

    for i in range(n_iters):
        print 'Trial %d...' % i
        print 'A:\n%s' % np.array_str(policy.A)
        print 'B:\n%s' % np.array_str(policy.B)
        states, actions, rewards, logprobs = run_trial(policy,
                                                       preprocess,
                                                       max_len=t_len,
                                                       continuous=continuous)
        estimator.report_episode(states, actions, rewards, logprobs)

        trials.append(len(rewards))

        start_theta = policy.get_theta()
        theta, _ = optimizer.optimize(x_init=start_theta,
                                      func=estimator.estimate_reward_and_gradient)
        if np.any(theta != start_theta):
            policy.set_theta(theta)
            estimator.update_buffer()
            estimator.remove_unlikely_trajectories(-3)
            print '%d trajectories remaining' % estimator.num_samples

        if len(trials) > 3 and np.mean(trials[-3:]) >= t_len:
            print 'Convergence achieved'
            break

    return trials, grads


def train_policy_ppge(dist, policy, optimizer, estimator, continuous, n_iters, t_len):
    trials = []
    grads = []

    for i in range(n_iters):
        print 'Trial %d...' % i

        theta = dist.sample_action()
        logprob = dist.logprob(state=None, action=theta)
        policy.set_theta(theta)

        print 'Theta Mean:\n%s\nSDs:%s' % (np.array_str(dist.mean),
                                           np.array_str(dist.sds))
        print 'A:\n%s' % np.array_str(policy.A)

        _, _, rewards, _ = run_trial(policy,
                                     preprocess,
                                     max_len=t_len,
                                     continuous=continuous)

        estimator.report_episode(states=[1], actions=[theta],
                                 rewards=[rewards], logprobs=[logprob])
        trials.append(len(rewards))

        gamma_init = dist.get_theta()
        gamma, _ = optimizer.optimize(x_init=gamma_init,
                                      func=estimator.estimate_reward_and_gradient)
        if np.any(gamma != gamma_init):
            dist.set_theta(gamma)
            estimator.update_buffer()
            estimator.remove_unlikely_trajectories(-3)
            print '%d trajectories remaining' % estimator.num_samples

        if len(trials) > 3 and np.mean(trials[-3:]) >= t_len:
            print 'Convergence achieved'
            break

    return trials, grads


def pred_grad(policy, estimator, n):
    estimator.reset()
    for i in range(n):
        states, actions, rewards, logprobs = run_trial(policy, preprocess, 500)
        estimator.report_episode(states, actions, rewards, logprobs)
    return estimator.estimate_reward_and_gradient()


if __name__ == '__main__':

    # Create policy
    raw_xdim = env.observation_space.shape[0]

    augmenter = poli.PolynomialFeatureAugmenter(dim=raw_xdim, max_order=1)

    def preprocess(v):
        v = augmenter.process(v)
        return np.hstack((v, 1))

    xdim = augmenter.dim + 1
    adim = env.action_space.shape[0]
    A = np.zeros((adim, xdim))
    A[0, 2] = 1
    B = np.zeros((adim, xdim))
    B[:, -1] = -2

    policy = poli.FixedLinearPolicy(input_dim=xdim, output_dim=adim)
    policy.A = A
    policy.B[:] = -2
    # policy.B = B
    init_theta = policy.get_theta()

    dpolicy = DeterministicLinearPolicy(indim=xdim, outdim=adim)
    dpolicy.A = A

    # TODO How do we pick good step sizes?
    optimizer = optim.GradientDescent(mode='max', step_size=0.1,
                                      max_linf_norm=0.1, max_iters=1)

    sampling_args = {'log_weight_lim': 3,
                     'normalize': True}
    reward_args = {'gamma': 1.0,
                   'horizon': float('inf')}

    per_estimator = poli.EpisodicPolicyGradientEstimator(policy=policy,
                                                         traj_mode='per',
                                                         buffer_size=0,
                                                         use_natural_gradient=True,
                                                         use_baseline=True,
                                                         sampling_args=sampling_args,
                                                         reward_args=reward_args,
                                                         max_grad_flip_prob=0.1,
                                                         min_ess=float('inf'),
                                                         n_threads=1)

    gpomdp_estimator = poli.EpisodicPolicyGradientEstimator(policy=policy,
                                                            traj_mode='gpomdp',
                                                            buffer_size=0,
                                                            use_natural_gradient=True,
                                                            use_baseline=True,
                                                            sampling_args=sampling_args,
                                                            max_grad_flip_prob=0.1,
                                                            min_ess=0,
                                                            n_threads=1)

    reinforce_estimator = poli.EpisodicPolicyGradientEstimator(policy=policy,
                                                               traj_mode='reinforce',
                                                               buffer_size=0,
                                                               use_natural_gradient=True,
                                                               sampling_args=sampling_args,
                                                               use_baseline=True,
                                                               n_threads=1)

    ppge = poli.ParameterDistribution(len(dpolicy.get_theta()))
    ppge.mean = dpolicy.get_theta()
    ppge.sds = np.full(len(ppge.sds), 0.1)
    init_gamma = ppge.get_theta()
    ppge_estimator = poli.EpisodicPolicyGradientEstimator(policy=ppge,
                                                          traj_mode='ppge',
                                                          buffer_size=0,
                                                          batch_size=30,
                                                          use_natural_gradient=True,
                                                          fisher_diag=True,
                                                          sampling_args=sampling_args,
                                                          reward_args=reward_args,
                                                          use_baseline=True)

    # n_samples = 30
    # n_trials = 30
    # print 'Running no filter trials'
    # nobaseline_results = [pred_grad(policy, estimator, n_samples)
    #                       for i in range(n_trials)]
    # nobaseline_rewards, nobaseline_grads = izip(*nobaseline_results)

    # print 'Running filtered trials...'
    # estimator.log_weight_lim = 1
    # baseline_results = [pred_grad(policy, estimator, n_samples)
    #                     for i in range(n_trials)]
    # baseline_rewards, baseline_grads = izip(*baseline_results)

    num_trials = 10

    def training_test(est, continuous):
        est.batch_size = 30

        lens = []
        gs = []
        params = []
        for _ in range(num_trials):
            policy.set_theta(init_theta)
            est.reset()
            trace, grads = train_policy(policy=policy,
                                        optimizer=optimizer,
                                        estimator=est,
                                        continuous=continuous,
                                        n_iters=300,
                                        t_len=500)
            lens.append(trace)
            gs.append(grads)
            params.append(policy.get_theta())
        return lens, gs, params

    def training_ppge_test(continuous):
        lens = []
        gs = []
        params = []
        for _ in range(num_trials):
            ppge.set_theta(init_gamma)
            ppge_estimator.reset()
            trace, grads = train_policy_ppge(dist=ppge,
                                             policy=dpolicy,
                                             optimizer=optimizer,
                                             estimator=ppge_estimator,
                                             continuous=continuous,
                                             n_iters=300,
                                             t_len=500)
            lens.append(trace)
            gs.append(grads)
            params.append(ppge.get_theta())
        return lens, gs, params

    # rei_lens, rei_gs, rei_params = training_test(reinforce_estimator,
    # continuous=False)
    # gpo_lens, gpo_gs, gpo_params = training_test(gpomdp_estimator,
    # continuous=False)
    # per_lens, per_gs, per_params = training_test(per_estimator,
    #                                             continuous=False)
    ppge_lens, ppge_gs, ppge_params = training_ppge_test(continuous=False)
