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

    n_policy_modes = 2

    # Initialize cartpole problem
    env = gym.make('CartPole-v1')
    env.reset()
    a_scale = env.action_space.high - env.action_space.low
    a_offset = 0.5 * (env.action_space.high + env.action_space.low)
    a_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]

    ##########################################################################
    # Networks
    ##########################################################################
    def init_graph(sess, n_modes, old_sess=None, old_graph=None):
        with sess.graph.as_default():

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
                                        shape=[None, a_dim],
                                        name='library')
            targets_ph = tf.placeholder(tf.int32,
                                        shape=[None],
                                        name='targets')

            policy_training = tf.placeholder(tf.bool, name='policy_training')
            value_training = tf.placeholder(tf.bool, name='value_training')
            dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')

            policy_spec = {'n_layers': 2,
                           'n_units': 64,
                           'n_outputs': n_modes,
                           'final_rect': None,
                           'scope': 'policy',
                           'batch_training': policy_training}  # ,
            #'dropout_rate': dropout_rate}
            policy, policy_params, policy_state, policy_updates = make_network(input=state_ph,
                                                                               **policy_spec)
            policy_ind = tf.argmax(policy[-1], axis=-1)
            policy_out = tf.gather(params=library_ph, indices=policy_ind)
            policy_probs = tf.nn.softmax(policy[-1])

            # NOTE batch normalization breaks the residual gradient, probably because it
            # rescales everything to garbage. Meanwhile it works well for the policy since
            # that is simply a classification task?
            value_spec = {'n_layers': 3,
                          'n_units': 256,
                          'n_outputs': 1,
                          'scope': 'value',
                          'final_rect': None}  # ,
            #'batch_training': value_training}#,
            #'dropout_rate': dropout_rate}
            value, value_params, value_state, value_updates = make_network(input=state_action,
                                                                           reuse=False,
                                                                           **value_spec)

            next_policy, _, _, _ = make_network(input=next_state_ph,
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
            final_value, _, _, _ = make_network(input=final_state_action,
                                                reuse=True,
                                                **value_spec)
            value_obj, td_loss, drift_loss = anchored_td_loss(rewards=reward_ph,
                                                              values=value[-1],
                                                              next_values=expected_value,
                                                              terminal_values=final_value[-1],
                                                              gamma=0.99, dweight=10.0)

            demo_onehot = tf.one_hot(
                indices=targets_ph, depth=tf.shape(library_ph)[0])
            demo_loss = tf.losses.softmax_cross_entropy(onehot_labels=demo_onehot,
                                                        logits=policy[-1])

            policy_opt = tf.train.AdamOptimizer(learning_rate=1e-2)
            with tf.control_dependencies(policy_updates):
                demo_train = policy_opt.minimize(
                    demo_loss, var_list=policy_params)
                expect_train = policy_opt.minimize(-empirical_return,
                                                   var_list=policy_params)
            policy_reset_op = optimizer_initializer(policy_opt, policy_params)

            value_opt = tf.train.AdamOptimizer(learning_rate=1e-3)
            with tf.control_dependencies(value_updates):
                value_train = value_opt.minimize(
                    value_obj, var_list=value_params)
            value_reset_op = optimizer_initializer(value_opt, value_params)

            sess.run(tf.global_variables_initializer())

            if old_graph is not None:
                old_policy_vals = old_sess.run(old_graph['policy_state'])
                old_value_vals = old_sess.run(old_graph['value_state'])
                copy_expanded_params(sess, old_policy_vals, policy_state)
                copy_expanded_params(sess, old_value_vals, value_state)

            out = {'state': state_ph, 'action': action_ph, 'next_state': next_state_ph, 'reward': reward_ph,
                   'final_state': final_state_ph, 'final_action': final_action_ph, 'library': library_ph,
                   'targets': targets_ph, 'dropout_rate': dropout_rate, 'policy_training': policy_training,
                   'value_training': value_training,
                   'policy_opt_reset': policy_reset_op, 'value_opt_reset': value_reset_op,
                   'value': value, 'td_loss': td_loss, 'drift_loss': drift_loss, 'demo_loss': demo_loss,
                   'demo_train': demo_train, 'expect_train': expect_train, 'value_train': value_train,
                   'policy_probs': policy_probs, 'policy_ind': policy_ind, 'empirical_return': empirical_return,
                   'policy_params': policy_params, 'value_params': value_params,
                   'policy_state': policy_state, 'value_state': value_state}
        return out

    ##########################################################################
    # LEARNING TIME!!11one
    ##########################################################################
    sess = tf.Session(graph=tf.Graph())
    graph = init_graph(sess=sess, n_modes=n_policy_modes)

    buffer = SARSDataset()

    training_batch_size = 100
    final_batch_size = 10
    min_lib_batch_size = 15  # TODO For testing!
    max_num_modes = 4
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
            probs = sess.run(graph['policy_probs'],
                             feed_dict={graph['state']: obs,
                                        graph['library']: lib,
                                        graph['dropout_rate']: 0.0,
                                        graph['value_training']: False,
                                        graph['policy_training']: False})

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
            v = sess.run(graph['value'][-1], feed_dict={graph['state']: obs,
                                                        graph['action']: action,
                                                        graph['dropout_rate']: 0.0,
                                                        graph['value_training']: False})[0]
            print 'obs: %s action: %s value: %f' % (str(obs[0]), str(action[0]), v)
            next_obs, reward, done, info = env.step(action[0])
            next_obs = np.reshape(next_obs, (1, -1))
            buffer.report_step(s=obs[0], a=action[0], r=np.atleast_1d(reward))

            # NOTE When the episode terminates due to achieving max steps,
            # the 'done' flag still gets set!
            if done:
                if t < (env.spec.max_episode_steps - 1):
                    print 'Episode terminating prematurely!'
                    buffer.report_terminal(s=next_obs[0])
                else:
                    print 'Episode hit max length!'
                    buffer.report_episode_end(s=next_obs[0])
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

        def raw_value(s, a):
            vals = sess.run(graph['value'][-1], feed_dict={graph['state']: s,
                                                           graph['action']: np.tile(a, (len(s), 1)),
                                                           graph['dropout_rate']: 0.0,
                                                           graph['value_training']: False})
            return np.squeeze(vals)

        curr_inds = sess.run(graph['policy_ind'], feed_dict={graph['state']: states,
                                                             graph['dropout_rate']: 0.0,
                                                             graph['policy_training']: False})

        def mean_max_value(s, a_new, curr):
            '''Mean performance on active states assuming optimal policy after changing
            active action
            '''
            r = np.reshape(raw_value(s, a_new), (1, -1))
            all_vals = np.vstack((curr, r))
            opt_vals = np.max(all_vals, axis=0)
            return np.mean(opt_vals)

        def min_delta_value(s, a_new, v_old):
            '''Smallest change in performance on active states compared to current policy
            assuming only changing active action
            '''
            delta = raw_value(s, a_new) - v_old
            return np.min(delta)

        # Replace k elements of the library

        if mode == 'greedy':
            curr = [raw_value(states, ai) for ai in old_lib]
            curr_v = mean_max_value(states, old_lib[0], curr)
            a_fin, v_fin = bo_opt(lambda x: mean_max_value(states, x, curr))
            improvement = v_fin - curr_v
            print 'New action %s improves value from %f to %f' % (str(a_fin), curr_v, v_fin)
            lib = np.vstack([old_lib, np.reshape(a_fin, (1, -1))])

        elif mode == 'current':
            M = len(old_lib)
            inds = random.sample(range(M), k)
            for i in inds:
                # NOTE We take the max over current tested action and library
                # members
                active_states = [si for si, ii in zip(
                    states, curr_inds) if ii == i]
                a_old = old_lib[i]
                # If action to be replaced has no corresponding states, bail
                if len(active_states) == 0:
                    print 'No active states for element %d!' % i
                    a_fin = a_old
                else:

                    # curr = [raw_value(active_states, lib[j])
                    #         for j in range(M) if j != i]
                    # v_old = mean_max_value(active_states, a_old, curr)
                    # a_fin, v_fin = bo_opt(
                    #     lambda x: mean_max_value(active_states, x, curr))
                    # print 'New action %s has value %f compared to old action %s with value %f.' \
                    #     % (str(a_fin), v_fin, str(a_old), v_old)
                    # if v_fin < v_old:
                    #     print 'Keeping old!'
                    #     a_fin = a_old

                    v_active = raw_value(active_states, a_old)
                    a_fin, v_fin = bo_opt(
                        lambda x: min_delta_value(active_states, x, v_active))
                    print 'New action %s has delta %f' % (str(a_fin), v_fin)
                    if v_fin < 0:
                        print 'Keeping old!'
                        a_fin = a_old
            lib = np.array(old_lib)
            lib[i] = a_fin
        return lib

    def train_policy_demo(sampler, lib, conv):
        conv.clear()
        l = float('inf')
        while not conv.check(l):
            s, i = sampler()
            l = sess.run([graph['demo_loss'], graph['demo_train']],
                         feed_dict={graph['state']: s,
                                    graph['targets']: i,
                                    graph['library']: lib,
                                    graph['policy_training']: True,
                                    graph['dropout_rate']: 0.1})[0]
            if conv.iter % 1000 == 0:
                print 'Policy training iter %d loss %f' % (conv.iter, l)

    def train_value_only(sampler, lib, conv):
        conv.clear()
        td = float('inf')
        while not conv.check(td):
            s, a, r, sn, fs, fa = sampler()
            td, dl = sess.run([graph['td_loss'], graph['drift_loss'], graph['value_train']],
                              feed_dict={graph['library']: lib,
                                         graph['state']: s,
                                         graph['action']: a,
                                         graph['reward']: r,
                                         graph['next_state']: sn,
                                         graph['final_state']: fs,
                                         graph['final_action']: fa,
                                         graph['policy_training']: False,
                                         graph['value_training']: True,
                                         graph['dropout_rate']: 0.1})[0:2]
            if conv.iter % 1000 == 0:
                print 'Value training iter %d TD err: %f drift: %f' % (conv.iter, td, dl)

    def train_interleaved(sampler, lib, n, conv_p, conv_v):
        conv_p.clear()
        conv_v.clear()
        for _ in range(n):
            s, a, r, sn, fs, fa = sampler()
            td, dl = sess.run([graph['td_loss'], graph['drift_loss'], graph['value_train']],
                              feed_dict={graph['library']: lib,
                                         graph['state']: s,
                                         graph['action']: a,
                                         graph['reward']: r,
                                         graph['next_state']: sn,
                                         graph['final_state']: fs,
                                         graph['final_action']: fa,
                                         graph['policy_training']: False,
                                         graph['value_training']: True,
                                         graph['dropout_rate']: 0.1})[0:2]

            if conv_v.iter % 1000 == 0:
                print 'Value training iter: %d TD: %f Drift: %f' % (conv_v.iter, td, dl)
            value_converged = conv_v.check(td)

            er = sess.run([graph['empirical_return'], graph['expect_train']],
                          feed_dict={graph['state']: s,  # NOTE Not sure why we need this if using batch norm
                                     graph['next_state']: s,
                                     graph['library']: lib,
                                     graph['policy_training']: True,
                                     graph['value_training']: False,
                                     graph['dropout_rate']: 0.1})[0]
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

    init_conv = Convergence(min_n=100, max_iter=10000, min_iter=5000, test_bounds=False)
    policy_conv = Convergence(min_n=100, max_iter=10000, min_iter=1000, test_bounds=False)
    value_conv = Convergence(min_n=100, max_iter=10000, min_iter=1000, test_bounds=False)

    def sample_policy_demo():
        return zip(*random.sample(demo_buffer, k=training_batch_size))
    print 'Training initial policy...'
    train_policy_demo(sampler=sample_policy_demo, lib=library, conv=init_conv)
    print 'Initial training done'

    def sample_both_data():
        sars = buffer.sample_sars(training_batch_size)

        final_states = buffer.sample_terminals(final_batch_size)
        final_acts = buffer.sample_sars(final_batch_size)[1]
        terminals = zip(*itertools.product(final_states, final_acts))
        return sars + terminals

    print 'Initializing value function...'
    train_value_only(sampler=sample_both_data, lib=library, conv=init_conv)
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
        if buffer.num_episodes > min_lib_batch_size \
            and len(library) < max_num_modes \
            and policy_conv.last_converged \
            and value_conv.last_converged:
            
            lib_replace_k = 1
            print 'Replacing %d elements of library...' % lib_replace_k
            new_lib = update_library(states=buffer.all_states,
                                     old_lib=library,
                                     k=lib_replace_k,
                                     mode='greedy')
            print 'Old library:\n%s' % str(library)
            print 'New library:\n%s' % str(new_lib)

            library = np.array(new_lib)
            libraries.append(np.array(library))
            print 'Libraries:\n%s' % str(libraries)

            # print 'Resetting policy and value optimizers...'
            # sess.run(graph['policy_opt_reset'])
            # sess.run(graph['value_opt_reset'])

            print 'Recreating graph...'
            new_sess = tf.Session(graph=tf.Graph())
            new_graph = init_graph(sess=new_sess, n_modes=len(
                library), old_sess=sess, old_graph=graph)
            sess = new_sess
            graph = new_graph

        print 'Iterating policy and value function...'
        train_interleaved(sampler=sample_both_data, lib=library, n=5000,
                          conv_p=policy_conv, conv_v=value_conv)
        print 'Iterations complete. Policy converged: %s Value converged: %s' \
            % (policy_conv.last_converged, value_conv.last_converged)

        print 'Num tuples: %d' % buffer.num_tuples

    finish_time = time.time()

    plt.ion()
    plt.figure()
    plt.plot(buffer.episode_lens)
    plt.xlabel('Iteration')
    plt.ylabel('Number of steps')
    plt.title('Trial lengths over time')

    plt.figure()
    plt.plot(np.squeeze(libraries))
    plt.xlabel('Iteration')
    plt.ylabel('Force')
    plt.title('Primitives over time')
