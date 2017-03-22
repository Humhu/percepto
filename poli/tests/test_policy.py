import numpy as np
import random
import math

from itertools import izip

import poli.policies as pp
import poli.bandit_learners as pbl

import matplotlib.pyplot as plt


def state_generator(dim):
    x = np.random.multivariate_normal(
        mean=np.zeros(dim - 1), cov=np.identity(dim - 1))
    return np.hstack((x, [1]))


def reward_generator(aoff, x, a):
    aeff = a - aoff
    xeff = x[:-1]
    err = np.linalg.norm(aeff - xeff)
    if err > 0.5:
        upper = -0.5
    else:
        upper = 0
    return np.random.uniform(low=-err, high=upper)


def train(policy):

    # Loop over trials
    converged = False
    iter_counter = 0
    max_iters = 1000
    estimator = pbl.BanditPolicyGradientLearner(policy=policy,
                                                batch_size=100,
                                                buffer_size=0,
                                                use_log_probs=True,
                                                use_natural_gradient=False,
                                                inv_fisher_offset=1E-9,
                                                seed=1)
    estimated_rewards = []
    while not converged:
        # Execute
        print 'Iteration %d...' % (iter_counter)
        state = state_func()
        action = policy.sample_action(state)
        reward = reward_func(state, action)
        print 'Mean: %s\nCov: %s' % (np.array_str(policy.mean), np.array_str(policy.cov))

        estimator.report_sample(state, action, reward)

        iter_counter += 1
        if iter_counter >= max_iters:
            converged = True

        estimator.step()
        rew = estimator.estimate_reward()
        if rew is not None:
            print 'Expected reward: %f' % rew
            estimated_rewards.append(rew)
        else:
            estimated_rewards.append(None)

        print 'A:\n%s' % np.array_str(policy.A, precision=3)
        print 'B:\n%s' % np.array_str(policy.B, precision=3)

    return estimator.get_reward_trace(), estimated_rewards


if __name__ == '__main__':

    # Initialization and params
    xdim = 3
    adim = xdim - 1
    aoff = np.random.rand(adim) - 0.5

    def reward_func(x, a): return reward_generator(aoff, x, a)

    def state_func(): return state_generator(xdim)

    # Create policy
    A = np.random.rand(adim, xdim) - 0.5
    B = np.zeros((adim, xdim))
    B[:, -1] = -1

    poli = pp.LinearPolicy(A, B)
    init_params = poli.get_theta()

    poli.set_theta(init_params)
    ur_rews, est_rews = train(poli)
    ur_A = poli.A
    ur_B = poli.B

    # Plotting
    plt.ion()
    plt.figure()
    plt.plot(ur_rews, 'gx', label='received rewards')
    plt.plot(est_rews, 'b-', label='expected reward')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.legend(loc='best')
    plt.title('Policy gradient test trace')

    # Visualize reward function
    n_samples = 100
    test_points = np.linspace(start=-1, stop=0, num=50)
    x = np.zeros(xdim)
    a_tests = np.tile(aoff, reps=(len(test_points), 1))
    a_tests[:, 0] += test_points
    test_out = [[reward_func(x, ai) for i in range(n_samples)]
                for ai in a_tests]
    test_out = np.array(test_out)
    test_means = np.mean(test_out, axis=1)
    test_sd = np.std(test_out, axis=1)

    rep_test_points = np.repeat(test_points, n_samples)
    plt.figure()
    plt.plot(rep_test_points, test_out.flatten(), 'g.')
    plt.plot(test_points, test_means, 'g-', linewidth=2)
    
    plt.xlabel('Action error')
    plt.ylabel('Received reward')
    plt.title('Reward function')

    print 'Initial A:\n%s' % np.array_str(A)
    print 'Initial B:\n%s' % np.array_str(B)
    print 'True action offset: %s' % np.array_str(aoff)
    print 'Final A:\n%s' % np.array_str(ur_A)
    print 'Final B:\n%s' % np.array_str(ur_B)

    print 'Press any key to close...'
    raw_input()
