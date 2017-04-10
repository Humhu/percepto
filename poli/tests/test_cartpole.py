import numpy as np
import scipy as sp
import gym
import poli
import optim
import math
from itertools import izip
from argus_utils import KalmanFilter

import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')
env.reset()
a_scale = env.action_space.high - env.action_space.low


def trial_sum_reward(datum):
    s, a, r, q, p = izip(*datum)
    return np.sum(r)


def trial_prob(datum):
    s, a, r, q, p = izip(*datum)
    return math.exp(np.sum(q))


def trial_expected_reward(datum):
    s, a, r, q, p = izip(*datum)
    return np.sum(r) * math.exp(np.sum(p))


def importance_weight(datum):
    s, a, r, q, p = izip(*datum)
    w = math.exp(np.sum(p) - np.sum(q))
    return w


def execute(state, policy, enable_safety):
    x, xdot, th, thdot, _ = state
    if enable_safety:
        if th > 0.1 and thdot > 0.1:
            print 'Executing safety action 1'
            return np.array([1.0]), True
        elif th < -0.1 and thdot < -0.1:
            print 'Executing safety action -1'
            return np.array([-1.0]), True

    return policy.sample_action(state), False


def run_trial(policy, pfunc, max_len=200):

    observation = env.reset()
    observation = pfunc(observation)

    observations = []
    rewards = []
    actions = []
    logprobs = []
    num_safety = 0
    max_safety = 3
    for t in range(max_len):
        env.render()
        # action, used_safety = execute(observation, policy,
        # enable_safety=num_safety < max_safety)
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
        rewards.append(float(reward) / max_len)

        if done:
            break
    print "Episode finished after %d timesteps" % (t + 1)
    return observations, actions, rewards, logprobs


def train_policy(policy, optimizer, estimator, n_iters, t_len):
    trials = []
    rews = []
    grads = []
    counter = 0

    for i in range(n_iters):
        print 'Trial %d...' % counter
        print 'A:\n%s' % np.array_str(policy.A)
        print 'B:\n%s' % np.array_str(policy.B)
        counter += 1
        states, actions, rewards, logprobs = run_trial(
            policy, preprocess, t_len)
        estimator.report_episode(states, actions, rewards, logprobs)

        trials.append(len(rewards))

        start_theta = policy.get_theta()
        theta, pred_rew = optimizer.optimize(x_init=start_theta,
                                             func=estimator.estimate_reward_and_gradient)
        if np.any(theta != start_theta):
            policy.set_theta(theta)
            estimator.update_buffer()
            #estimator.reset()
            estimator.remove_unlikely_trajectories(min_log_weight=-estimator.log_weight_lim - 1)

        if len(trials) > 3 and np.mean(trials[-3:]) == t_len:
            break

    return trials, rews, grads


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

    policy = poli.LinearPolicy(input_dim=xdim, output_dim=adim)
    policy.A = A
    #policy.b[:] = -2
    policy.B = B
    init_theta = policy.get_theta()

    # TODO How do we pick good step sizes?
    optimizer = optim.GradientDescent(mode='max', step_size=1E-1,
                                      max_linf_norm=0.1, max_iters=1)

    wds = poli.WeightedDataSampler(weight_func=importance_weight)

    estimator = poli.EpisodicPolicyGradientEstimator(policy=policy,
                                                     traj_mode='per',
                                                     buffer_size=0,
                                                     # sampler=wds,
                                                     use_natural_gradient=True,
                                                     use_norm_sample=True,
                                                     use_diag_fisher=False,
                                                     use_baseline=True,
                                                     n_threads=1)

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

    num_trials = 30
    estimator.log_weight_lim = 1
    estimator.max_dev_ratio = 0.25
    estimator.max_grad_dev = 0.1
    estimator.min_ess = 20

    def training_test():
        lens = []
        rs = []
        gs = []
        params = []
        for i in range(num_trials):
            policy.set_theta(init_theta)
            estimator.reset()
            trace, rews, grads = train_policy(policy=policy,
                                              optimizer=optimizer,
                                              estimator=estimator,
                                              n_iters=300,
                                              t_len=500)
            lens.append(trace)
            rs.append(rews)
            gs.append(grads)
            params.append(policy.get_theta())
        return lens, rs, gs, params

    estimator.batch_size = 20
    cont_lens, cont_rs, cont_gs, cont_params = training_test()

    #estimator.batch_size = 10
    #ess_lens, ess_rs, ess_gs, ess_params = training_test()

    # Result processing
    # cont_As = [tr

    # Plotting
    plt.ion()
    plt.figure()
