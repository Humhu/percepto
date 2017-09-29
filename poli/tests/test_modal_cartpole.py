import tensorflow as tf
import gym
import numpy as np
import random
import math
import time
import itertools

import cma
from bayes_opt import BayesianOptimization
import sklearn.cluster as skc

import optim
from adel import *

import matplotlib.pyplot as plt

if __name__ == '__main__':

    n_policy_modes = 4

    # Initialize cartpole problem
    env = gym.make('CartPole-v1')
    env.reset()
    a_scale = env.action_space.high - env.action_space.low
    a_offset = 0.5 * (env.action_space.high + env.action_space.low)
    a_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]

    # Inputs
    state_ph = tf.placeholder(tf.float32,
                              shape=[None, obs_dim],
                              name='state')
    action_ph = tf.placeholder(tf.float32,
                               shape=[None, a_dim],
                               name='action')
    state_action = tf.concat([state_ph, action_ph],
                             axis=-1,
                             name='state_action')

    next_state_ph = tf.placeholder(tf.float32,
                                   shape=[None, obs_dim],
                                   name='next_state')

    reward_ph = tf.placeholder(tf.float32,
                               shape=[None, 1],
                               name='reward')

    final_state_ph = tf.placeholder(tf.float32,
                                    shape=[None, obs_dim],
                                    name='final_state')
    final_action_ph = tf.placeholder(tf.float32,
                                     shape=[None, a_dim],
                                     name='final_action')
    final_state_action = tf.concat([final_state_ph, final_action_ph],
                                   axis=-1,
                                   name='final_state_action')

    library_ph = tf.placeholder(tf.float32,
                                shape=[n_policy_modes, a_dim],
                                name='library')
    targets_ph = tf.placeholder(tf.int32,
                                shape=[None],
                                name='targets')

    policy_training = tf.placeholder(tf.bool, name='policy_training')
    value_training = tf.placeholder(tf.bool, name='value_training')
    dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')

    ##########################################################################
    # Networks
    ##########################################################################
    policy_spec = {'n_layers': 2,
                   'n_units': 64,
                   'n_outputs': n_policy_modes,
                   'final_rect': None,
                   'scope': 'policy',
                   'use_batch_norm': True,
                   'training': policy_training,
                   'use_dropout': False,
                   'dropout_rate': dropout_rate}
    policy, policy_params, policy_updates = make_network(input=state_ph,
                                                         **policy_spec)
    policy_ind = tf.argmax(policy[-1], axis=-1)
    policy_out = tf.gather(params=library_ph, indices=policy_ind)
    policy_probs = tf.nn.softmax(policy[-1])

    value_spec = {'n_layers': 3,
                  'n_units': 256,
                  'n_outputs': 1,
                  'scope': 'value',
                  'final_rect': None,
                  'use_batch_norm': False,
                  'training': value_training,
                  'use_dropout': False,
                  'dropout_rate': dropout_rate}
    value, value_params, value_updates = make_network(input=state_action,
                                                      reuse=False,
                                                      **value_spec)

    next_policy, _, _ = make_network(input=next_state_ph,
                                     reuse=True,
                                     **policy_spec)
    next_policy_probs = tf.nn.softmax(next_policy[-1])

    def make_value(input):
        return make_network(input=input,
                            reuse=True,
                            **value_spec)[0][-1]

    value_combos = value_combinations(states=next_state_ph,
                                      make_value=make_value,
                                      actions=library_ph)
    expected_value = tf.reduce_sum(value_combos * next_policy_probs,
                                   axis=1, keep_dims=True)
    empirical_return = tf.reduce_mean(expected_value)

    # Loss to fix values for terminal states
    final_value, _, _ = make_network(input=final_state_action,
                                     reuse=True,
                                     **value_spec)
    value_obj, td_loss, drift_loss = anchored_td_loss(rewards=reward_ph,
                                                      values=value[-1],
                                                      next_values=expected_value,
                                                      terminal_values=final_value[-1],
                                                      gamma=0.99, dweight=10.0)

    demo_onehot = tf.one_hot(indices=targets_ph, depth=n_policy_modes)
    demo_advantages_ph = tf.placeholder(
        dtype=tf.float32, shape=[None], name='advantages')
    demo_loss = tf.losses.softmax_cross_entropy(onehot_labels=demo_onehot,
                                                logits=policy[-1])

    policy_optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
    with tf.control_dependencies(policy_updates):
        demo_train = policy_optimizer.minimize(
            demo_loss, var_list=policy_params)
        expect_train = policy_optimizer.minimize(
            -empirical_return, var_list=policy_params)
    policy_reset_op = optimizer_initializer(policy_optimizer, policy_params)

    value_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    with tf.control_dependencies(value_updates):
        value_train = value_optimizer.minimize(
            value_obj, var_list=value_params)
    value_reset_op = optimizer_initializer(value_optimizer, value_params)

    ##########################################################################
    # LEARNING TIME!!11one
    ##########################################################################
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    buffer = SARSDataset()

    training_batch_size = 100
    final_batch_size = 10
    min_lib_batch_size = 30
    max_buff_len = float('inf')

    def demo_policy(obs):
        K_demo = np.array([[-1, 0, 20, -2]])
        action = np.atleast_1d(np.dot(K_demo, obs.T))
        action[action > env.action_space.high] = env.action_space.high
        action[action < env.action_space.low] = env.action_space.low
        return action

    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))

    def learned_policy(obs, lib, p_policy):
        # er = sess.run(empirical_return, feed_dict={next_state_ph: obs,
                                                    #    dropout_rate: 0.0,
                                                    #    library_ph: lib,
                                                    #    value_training: False})
        # p_policy_eff = max((math.exp(-er*er_k), p_policy))
        if random.random() > p_policy:
            action = np.random.uniform(low=env.action_space.low,
                                       high=env.action_space.high,
                                       size=(1, a_dim))
        else:
            probs = sess.run(policy_probs,
                             feed_dict={state_ph: obs,
                                        library_ph: lib,
                                        dropout_rate: 0.0,
                                        value_training: False,
                                        policy_training: False})

            # As the predicted value goes down, get more random
            pick_ind = np.random.choice(len(library), p=np.squeeze(probs))
            action = [library[pick_ind]]
        return action

    def run_trial(policy, buffer, **kwargs):
        states = []
        rewards = []
        actions = []
        next_states = []

        obs = np.reshape(env.reset(), (1, -1))
        # NOTE the environment will stop at some point
        for t in range(env.spec.max_episode_steps):
            env.render()
            action = policy(obs, **kwargs)
            v = sess.run(value[-1], feed_dict={state_ph: obs,
                                               action_ph: action,
                                               dropout_rate: 0.0,
                                               value_training: False})[0]
            print 'obs: %s action: %s value: %f' % (str(obs[0]), str(action[0]), v)
            next_obs, reward, done, info = env.step(action[0])
            next_obs = np.reshape(next_obs, (1, -1))
            buffer.report_step(s=obs[0], a=action[0], r=np.atleast_1d(reward))

            # NOTE When the episode terminates due to achieving max steps,
            # the 'done' flag still gets set!
            if done and t < (env.spec.max_episode_steps - 1):
                print 'Episode terminating prematurely!'
                buffer.report_terminal(s=next_obs[0])
                break

            obs = next_obs

        print 'Episode terminated at step %d' % t

    def bo_opt(f):
        aux = optim.BFGSOptimizer(mode='max', num_restarts=5)
        aux.lower_bounds = env.action_space.low
        aux.upper_bounds = env.action_space.high
        reward_model = optim.GaussianProcessRewardModel(batch_retries=9,
                                                        enable_refine=True,
                                                        refine_period=5)
        acq_func = optim.UCBAcquisition(reward_model)

        init_samples = np.random.uniform(low=aux.lower_bounds,
                                         high=aux.upper_bounds,
                                         size=(20, a_dim))
        # TODO implement best_seen in reward model
        samples = []
        for x in init_samples:
            y = f(x)
            reward_model.report_sample(x=x, reward=y)
            samples.append((x, y))

        for _ in range(100):  # TODO
            x, acq_val = optim.pick_acquisition(acq_func=acq_func,
                                                optimizer=aux,
                                                x_init=a_offset)
            y = f(x)
            reward_model.report_sample(x=x, reward=y)
            samples.append((x, y))
        return max(samples, key=lambda x: x[1])  # Returns input, value

    def update_library(states, old_lib, k, mode='greedy'):
        '''Updates the library by dropping k random elements and running greedy set
        coverage.

        In greedy mode, maximizes the performance of an optimal greedy policy.

        In current mode, maximizes the worst-case improvement over the current
        policy using the modes induced by the current policy. If no improvement
        can be achieved, keeps the current policy actions. If the action to be
        replaced has no corresponding states, greedy mode will be used instead.
        '''
        lib = np.array(old_lib)

        def raw_value(s, a):
            vals = sess.run(value[-1], feed_dict={state_ph: s,
                                                  action_ph: np.tile(a, (len(s), 1)),
                                                  dropout_rate: 0.0,
                                                  value_training: False})
            return np.squeeze(vals)

        curr_values = np.vstack([raw_value(states, ai) for ai in old_lib])
        curr_inds = sess.run(policy_ind, feed_dict={state_ph: states,
                                                    dropout_rate: 0.0,
                                                    policy_training: False})

        def mean_max_value(a, i):
            '''Performance according to greedy optimal policy
            '''
            curr_values[i] = raw_value(states, a)
            return np.mean(np.max(curr_values, axis=0))

        def min_delta_value(s, a_new, v_old):
            '''Smallest change in performance compared to current policy
            '''
            delta = raw_value(s, a_new) - v_old
            return np.min(delta)

        # Replace k elements of the library
        inds = random.sample(range(len(lib)), k)
        curr_values[inds, :] = 0

        for i in inds:
            # NOTE We take the max over current tested action and library
            # members
            if mode == 'greedy':
                a_fin = bo_opt(lambda x: mean_max_value(x, i))[0]
            elif mode == 'current':
                active_states = [si for si, ii in zip(
                    states, curr_inds) if ii == i]
                a_old = old_lib[i]
                # If action to be replaced has no corresponding states, default
                # to greedy mode
                if len(active_states) == 0:
                    print 'No active states for element %d! Defaulting to greedy behavior' % i
                    a_fin = bo_opt(lambda x: mean_max_value(x, i))[0]
                else:
                    v_active = raw_value(active_states, a_old)
                    a_fin, v_fin = bo_opt(lambda x: min_delta_value(
                        active_states, x, v_active))

                    print 'New action %s has delta value %f compared to old action %s.' \
                        % (str(a_fin), v_fin, str(a_old))
                    if v_fin < 0:
                        print 'Keeping old!'
                        a_fin = a_old

            lib[i] = a_fin
            curr_values[inds] = raw_value(states, a_fin)

        inds = np.argmax(curr_values, axis=1)
        return lib, inds

    def train_policy_demo(sampler, lib, conv):
        conv.clear()
        # sess.run(policy_reset_op)
        l = float('inf')
        while not conv.check(l):
            s, i = sampler()
            l = sess.run([demo_loss, demo_train],
                         feed_dict={state_ph: s,
                                    targets_ph: i,
                                    library_ph: lib,
                                    demo_advantages_ph: [1.0],
                                    policy_training: True,
                                    dropout_rate: 0.1})[0]
            if conv.iter % 1000 == 0:
                print 'Policy training iter %d loss %f' % (conv.iter, l)

    def train_value_only(sampler, lib, conv):
        conv.clear()
        # sess.run(value_reset_op)
        td = float('inf')
        while not conv.check(td):
            s, a, r, sn, fs, fa = sampler()
            td, dl = sess.run([drift_loss, value_train],
                              feed_dict={library_ph: lib,
                                         state_ph: s,
                                         action_ph: a,
                                         reward_ph: r,
                                         next_state_ph: sn,
                                         final_state_ph: fs,
                                         final_action_ph: fa,
                                         policy_training: False,
                                         value_training: True,
                                         dropout_rate: 0.1})[0:2]
            if conv.iter % 1000 == 0:
                print 'Value training iter %d TD err: %f drift: %f' % (conv.iter, td, dl)

    def train_interleaved(sampler, lib, n, conv_p, conv_v):
        conv_p.clear()
        conv_v.clear()
        for _ in range(n):
            s, a, r, sn, fs, fa = sampler()
            td, dl = sess.run([td_loss, drift_loss, value_train],
                              feed_dict={library_ph: lib,
                                         state_ph: s,
                                         action_ph: a,
                                         reward_ph: r,
                                         next_state_ph: sn,
                                         final_state_ph: fs,
                                         final_action_ph: fa,
                                         policy_training: False,
                                         value_training: True,
                                         dropout_rate: 0.1})[0:2]

            if conv_v.iter % 1000 == 0:
                print 'Value training iter: %d TD: %f Drift: %f' % (conv_v.iter, td, dl)
            value_converged = conv_v.check(td)
            # if td > conv_v.tol:
            #     continue

            er = sess.run([empirical_return, expect_train],
                          feed_dict={state_ph: s,  # NOTE Not sure why we need this if using batch norm
                                     next_state_ph: s,
                                     library_ph: lib,
                                     policy_training: False,
                                     value_training: False,
                                     dropout_rate: 0.1})[0]
            if conv_p.iter % 1000 == 0:
                print 'Policy training iter: %d Expected return: %f' % (conv_p.iter, er)
            policy_converged = conv_p.check(er)

            if value_converged and policy_converged:
                return

    # First initialize policy
    n_init_demos = 10
    print 'Initializing policy with %d P-control demos...' % n_init_demos
    demos = []
    for i in range(n_init_demos):
        run_trial(demo_policy, buffer)

    # Run k-means on demo data to initialize library
    kmeans = skc.KMeans(n_clusters=n_policy_modes)
    a_inds = kmeans.fit_predict(buffer.all_actions)
    library = kmeans.cluster_centers_
    print 'Initial library:\n%s' % str(library)
    demo_buffer = zip(buffer.all_states, a_inds)

    demo_conv = Convergence(min_n=100, tol=0.1, max_iter=10000)
    policy_conv = Convergence(
        min_n=10, tol=0.5, max_iter=10000, use_delta=True)
    value_conv = Convergence(min_n=100, tol=0.1, max_iter=10000)

    def sample_policy_demo():
        return zip(*random.sample(demo_buffer, k=training_batch_size))
    print 'Training initial policy...'
    train_policy_demo(sampler=sample_policy_demo, lib=library, conv=demo_conv)
    print 'Initial training done'

    def sample_both_data():
        sars = buffer.sample_sars(training_batch_size)

        final_states = buffer.sample_terminals(final_batch_size)
        final_acts = buffer.sample_sars(final_batch_size)[1]
        terminals = zip(*itertools.product(final_states, final_acts))
        return sars + terminals

    print 'Initializing value function...'
    train_value_only(sampler=sample_both_data, lib=library, conv=value_conv)
    print 'Initialization done'

    trial_batch_size = 1
    init_p = 0.25
    final_p = 0.0
    final_iters = 100

    def get_p(i):
        i = max(min(i, final_iters), 0)
        return init_p + (final_p - init_p) * i / final_iters

    libraries = [library]
    start_time = time.time()
    for iter_i in range(100):

        # Run trial
        p = get_p(iter_i)
        print 'Trial %d using exploration p %f' % (iter_i, p)
        for _ in range(trial_batch_size):
            run_trial(learned_policy, buffer, lib=library, p_policy=1.0 - p)

        if buffer.num_tuples < training_batch_size or buffer.num_episodes < final_batch_size:
            continue

        # Now re-select library
        if buffer.num_episodes > min_lib_batch_size:  # \
                # and value_conv.last_converged \
                # and policy_conv.last_converged:
            lib_replace_k = 1
            print 'Replacing %d elements of library...' % lib_replace_k
            new_lib, new_inds = update_library(states=buffer.all_states,
                                               old_lib=library,
                                               k=lib_replace_k,
                                               mode='current')  # NOTE or greedy?
            print 'Old library:\n%s' % str(library)
            print 'New library:\n%s' % str(new_lib)

            library = np.array(new_lib)
            libraries.append(np.array(library))
            print 'Libraries:\n%s' % str(libraries)

            print 'Resetting policy and value optimizers...'
            sess.run(policy_reset_op)
            sess.run(value_reset_op)

        print 'Iterating policy and value function...'
        train_interleaved(sampler=sample_both_data, lib=library, n=5000,
                          conv_p=policy_conv, conv_v=value_conv)
        print 'Iterations complete. Policy converged: %s Value converged: %s' \
            % (policy_conv.last_converged, value_conv.last_converged)

        print 'Num tuples: %d' % buffer.num_tuples
    finish_time = time.time()

    plt.ion()
    plt.figure()
    plt.plot(trial_lens)
    plt.xlabel('Iteration')
    plt.ylabel('Number of steps')
    plt.title('Trial lengths over time')

    plt.figure()
    plt.plot(np.squeeze(libraries))
    plt.xlabel('Iteration')
    plt.ylabel('Force')
    plt.title('Primitives over time')
