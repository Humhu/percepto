import numpy as np
import gym
import poli
import optim
import math
from itertools import izip

env = gym.make('CartPole-v1')
env.reset()
a_scale = env.action_space.high - env.action_space.low

def trial_sum_reward(datum):
    s, a, r, l = izip(*datum)
    return np.sum(r)

def trial_prob(datum):
    s, a, r, l = izip(*datum)
    return math.exp(np.sum(l))

def run_trial(policy, pfunc, max_len=200):

    observation = env.reset()
    observation = pfunc(observation)

    observations = []
    rewards = []
    actions = []
    logprobs = []
    for t in range(max_len):
        env.render()
        action = policy.sample_action(observation)
        action[action > 1] = 1
        action[action < -1] = -1

        actions.append(action)
        observations.append(observation)
        logprobs.append(policy.logprob(observation, action))
        observation, reward, done, info = env.step(a_scale * action)
        observation = pfunc(observation)
        rewards.append(reward)  # Reward should correspond to current action

        if done:
            break
    print "Episode finished after %d timesteps" % (t + 1)
    return observations, actions, rewards, logprobs


if __name__ == '__main__':

    # Create policy
    raw_xdim = env.observation_space.shape[0]

    # Sort of initialize the preprocessor here
    min_preprocessor_samples = 30
    # preprocessor = poli.OnlineFeatureNormalizer(dim=raw_xdim, mode='moments',
    #                                             min_samples=min_preprocessor_samples,
    #                                             keep_updating=True,
    #                                             num_sds=2.0)
    augmenter = poli.PolynomialFeatureAugmenter(dim=raw_xdim, max_order=1)

    def preprocess(v):
        #v = preprocessor.process(v)
        # if v is None:
        #    return None
        v = augmenter.process(v)
        return np.hstack((v, 1))

    xdim = augmenter.dim + 1
    adim = env.action_space.shape[0]
    #A = np.random.rand(adim, xdim) - 0.5
    A = np.zeros((adim, xdim))
    #A[0, 2] = 1
    B = np.zeros((adim, xdim))
    B[:, -1] = -2

    policy = poli.LinearPolicy(input_dim=xdim, output_dim=adim)
    policy.A = A
    policy.B = B
    init_params = policy.get_theta()

    wds = poli.WeightedDataSampler(weight_func=trial_prob)

    batch_size = 30
    init_learn_size = 40
    estimator = poli.EpisodicPolicyGradientEstimator(policy=policy,
                                                     traj_mode='gpomdp',
                                                     batch_size=batch_size,
                                                     buffer_size=100,
                                                     sampler=wds,
                                                     use_natural_gradient=True,
                                                     use_norm_sample=True,
                                                     use_diag_fisher=False,
                                                     use_baseline=True,
                                                     seed=1)

    optimizer = optim.GradientDescent(mode='max', step_size=0.1,
                                      max_linf_norm=0.01, max_iters=10)

    # print 'Initializing feature preprocessor...'
    # for i in range(min_preprocessor_samples):
    #     preprocessor.process(env.reset())
    # for i in range(30):
    #     run_trial(policy, preprocess, 300)
    # print 'Initialization complete'
    # preprocessor.set_updating(False)

    trials = []
    counter = 0
    while True:
        print 'Trial %d...' % counter
        print 'A:\n%s' % np.array_str(policy.A)
        print 'B:\n%s' % np.array_str(policy.B)
        counter += 1
        states, actions, rewards, logprobs = run_trial(policy, preprocess, 500)
        estimator.report_episode(states, actions, rewards, logprobs)

        trials.append(len(rewards))

        if estimator.num_samples < init_learn_size:
            continue

        rew, grad = estimator.estimate_reward_and_gradient()
        print 'Est reward: %f grad: %s' % (rew, np.array_str(grad))
        theta, pred_rew = optimizer.optimize(x_init=policy.get_theta(),
                                             func=estimator.estimate_reward_and_gradient)
        policy.set_theta(theta)
