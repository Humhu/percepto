import tensorflow as tf
import gym
import numpy as np
import random
import math
import time
from itertools import izip

import cma
from bayes_opt import BayesianOptimization
import sklearn.cluster as skc

import optim


class Convergence(object):
    def __init__(self, min_n, tol, max_iter, use_delta=False):
        self.min_n = min_n
        self.tol = tol
        self._hist = []
        self.iter = 0
        self.max_iter = max_iter
        self.use_delta = use_delta
        self.last_converged = False

    def clear(self):
        self._hist = []
        self.iter = 0
        self.last_converged = False

    def check(self, f):
        self._hist.append(f)
        self.iter += 1

        if len(self._hist) < self.min_n:
            return False
        if self.iter > self.max_iter:
            return True
        self._hist = self._hist[-self.min_n:]

        if self.use_delta:
            hist = np.diff(self._hist)
        else:
            hist = np.array(self._hist)

        self.last_converged = np.all(np.abs(hist) < self.tol)
        return self.last_converged


def make_network(input, n_layers, n_units, n_outputs, scope,
                 training=None, use_batch_norm=False,
                 dropout_rate=None, use_dropout=False,
                 reuse=False, final_rect=None):
    layers = []
    variables = []
    b_init = tf.constant_initializer(dtype=tf.float32, value=0.0)

    x = input
    with tf.variable_scope(scope, reuse=reuse):
        for i in range(n_layers):
            if i > 0:
                if use_batch_norm:
                    x = tf.layers.batch_normalization(inputs=x,
                                                      training=training,
                                                      name='batch_%d' % i,
                                                      reuse=reuse)
                    layers.append(x)
                if use_dropout:
                    x = tf.layers.dropout(inputs=x, rate=dropout_rate)
                    layers.append(x)

            if i == n_layers - 1:
                x = tf.layers.dense(inputs=x,
                                    units=n_outputs,
                                    activation=final_rect,
                                    bias_initializer=b_init,
                                    name='layer_%d' % i,
                                    reuse=reuse)
            else:
                x = tf.layers.dense(inputs=x,
                                    units=n_units,
                                    activation=tf.nn.leaky_relu,
                                    bias_initializer=b_init,
                                    name='layer_%d' % i,
                                    reuse=reuse)
            layers.append(x)
            variables += tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope='%s/layer_%d' % (scope, i))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
    return layers, variables, update_ops


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
                                name='target_inds')

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

    def initialize_policy(sess):
        sess.run([p.initializer for p in policy_params[-2:]])

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

    def initialize_value(sess):
        sess.run([p.initializer for p in value_params[-2:]])

    def decimate_params(params, p=0.1):
        for param in params:
            n = int(np.prod(param.shape))
            to_zero = int(math.ceil(n * p))
            inds = np.random.choice(n, size=to_zero, replace=False)
            values = sess.run(param)
            values.flat[inds] = 0
            sess.run(tf.assign(param, values))

    next_policy, _, next_policy_updates = make_network(input=next_state_ph,
                                                       reuse=True,
                                                       **policy_spec)
    next_policy_ind = tf.argmax(next_policy[-1], axis=-1)
    next_policy_out = tf.gather(params=library_ph, indices=next_policy_ind)
    next_state_action = tf.concat([next_state_ph, next_policy_out],
                                  axis=-1,
                                  name='next_state_action')

    next_value = make_network(input=next_state_action,
                              reuse=True,
                              **value_spec)[0]

    # To construct the optimal policy, we will tile [state] and [library]
    # as [state1, state1, state2, state2, ...] and [action1, action2, action
    # 1, action2, ...]
    # next_value_opt = tf.reduce_max(next_value_shaped, axis=1, keep_dims=True)

    # Optimal policy
    # value_ind_opt = tf.argmax(next_value_opt, axis=1)
    # action_opt = tf.gather(params=library_ph, indices=value_ind_opt)

    final_value = make_network(input=final_state_action,
                               reuse=True,
                               **value_spec)[0]

    # Computing expected policy return
    n_states = tf.shape(input=next_state_ph)[0]
    idx = tf.range(n_states)
    idx = tf.reshape(idx, [-1, 1])    # Convert to a len(yp) x 1 matrix.
    idx = tf.tile(idx, [1, n_policy_modes])  # Create multiple columns.
    idx = tf.reshape(idx, [-1])       # Convert back to a vector.
    tiled_states = tf.gather(next_state_ph, idx)

    jdx = tf.range(n_policy_modes)
    jdx = tf.tile(jdx, [n_states])
    tiled_library = tf.gather(library_ph, jdx)
    expect_state_action = tf.concat([tiled_states, tiled_library], axis=-1)
    expect_value = make_network(input=expect_state_action,
                                reuse=True,
                                **value_spec)[0]

    expect_value_shape = tf.stack([n_states, n_policy_modes])
    expect_value_shaped = tf.reshape(tensor=expect_value[-1],
                                     shape=expect_value_shape)
    next_policy_probs = tf.nn.softmax(next_policy[-1])
    expected_return = tf.reduce_sum(
        expect_value_shaped * next_policy_probs, axis=1, keep_dims=True)
    mean_expected_return = tf.reduce_mean(expected_return)

    # TD-Gradient error
    gamma = tf.constant(0.95, dtype=tf.float32, name='gamma')
    # td_err = reward_ph + gamma * next_value[-1] - value[-1]
    td_err = reward_ph + gamma * expected_return - value[-1]
    # td_err = reward_ph + gamma * next_value_opt - value[-1]
    td_loss = tf.reduce_mean(tf.nn.l2_loss(td_err))

    # Loss to fix values for terminal states
    drift_loss = tf.reduce_mean(tf.nn.l2_loss(final_value[-1]))
    drift_weight = tf.constant(10.0, dtype=tf.float32)

    demo_onehot = tf.one_hot(indices=targets_ph, depth=n_policy_modes)
    demo_advantages_ph = tf.placeholder(
        dtype=tf.float32, shape=[None], name='advantages')
    demo_loss = tf.losses.softmax_cross_entropy(onehot_labels=demo_onehot,
                                                logits=policy[-1])  # ,
    # weights=demo_advantages_ph)

    # NOTE Taken from StackOverflow
    def adam_variables_initializer(opt, var_list):
        adam_vars = [opt.get_slot(var, name)
                     for name in opt.get_slot_names()
                     for var in var_list]
        if isinstance(opt, tf.train.AdamOptimizer):
            adam_vars.extend(list(opt._get_beta_accumulators()))
        return tf.variables_initializer(adam_vars)

    policy_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    with tf.control_dependencies(policy_updates):
        demo_train = policy_optimizer.minimize(demo_loss,
                                               var_list=policy_params)
    with tf.control_dependencies(next_policy_updates):
        expect_train = policy_optimizer.minimize(-mean_expected_return,
                                                 var_list=policy_params)
    policy_reset_op = adam_variables_initializer(
        policy_optimizer, policy_params)

    value_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3, epsilon=0.1)
    with tf.control_dependencies(value_updates):
        value_train = value_optimizer.minimize(td_loss + drift_weight * drift_loss,
                                               var_list=value_params)
    value_reset_op = adam_variables_initializer(value_optimizer, value_params)

    ##########################################################################
    # LEARNING TIME!!11one
    ##########################################################################
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    buffer = []
    final_buffer = []

    training_batch_size = 100
    min_lib_batch_size = 1000
    max_buff_len = float('inf')

    # def get_indices(s, lib):
    #     N = len(s)

    #     def get_value(a):
    #         return sess.run(value[-1], feed_dict={state_ph: s,
    #                                               action_ph: np.tile(a, (N, 1)),
    #                                               value_training: False})
    #     values = np.array([np.squeeze(get_value(ai)) for ai in lib])
    #     # Indices of optimal policy
    #     best_inds = np.argmax(values, axis=0)

    #     # Indices of current policy
    #     curr_logits = sess.run(policy[-1], feed_dict={state_ph: s,
    #                                                   library_ph: lib})
    #     curr_probs = np.exp(curr_logits)
    #     curr_probs = (curr_probs.T / np.sum(curr_probs, axis=1))
    #     curr_expected_value = np.sum(values * curr_probs, axis=0)
    #     # values[curr_inds, range(N)]
    #     advantage = values[best_inds, range(N)] - curr_expected_value
    #     return best_inds, advantage

    def demo_policy(obs):
        K_demo = np.array([[-1, 0, 20, -2]])
        action = np.atleast_1d(np.dot(K_demo, obs.T))
        action[action > env.action_space.high] = env.action_space.high
        action[action < env.action_space.low] = env.action_space.low
        return action

    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))

    def learned_policy(obs, lib, p_policy, er_k):
        # er = sess.run(mean_expected_return, feed_dict={next_state_ph: obs,
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

    def run_trial(policy, buffer, final_buffer, **kwargs):
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
            buffer.append((obs[0],
                           action[0],
                           np.atleast_1d(reward),
                           next_obs[0]))

            # NOTE When the episode terminates due to achieving max steps,
            # the 'done' flag still gets set!
            # Thanks Python...
            if done and t < (env.spec.max_episode_steps - 1):
                # Return a bunch of examples to signify next_obs is a terminal
                # state
                print 'Episode terminating prematurely!'
                fake_action = np.random.uniform(low=env.action_space.low,
                                                high=env.action_space.high,
                                                size=(30, a_dim))
                final_buffer += [(next_obs[0], fi) for fi in fake_action]
                break
            obs = next_obs
        print 'Episode terminated at step %d' % t

    def bo_opt(f):
        aux = optim.BFGSOptimizer(mode='max', num_restarts=5)
        aux.lower_bounds = env.action_space.low
        aux.upper_bounds = env.action_space.high
        reward_model = optim.GaussianProcessRewardModel(enable_refine=True,
                                                        refine_period=2)
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
        print 'BO:\n%s' % (samples)
        return max(samples, key=lambda x: x[1])[0]

    def update_library(states, old_lib, k, mode='greedy'):
        '''Updates the library by dropping k random elements and
        running greedy set coverage. Maximizes the performance of an
        optimal greedy policy.
        '''
        lib = old_lib

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

        def mean_curr_value(s, a):
            '''Performance according to current policy
            '''
            return np.mean(raw_value(s, a))

        # Replace k elements of the library
        inds = random.sample(range(len(lib)), k)
        curr_values[inds, :] = 0

        for i in inds:
            # NOTE We take the max over current tested action and library
            # members
            if mode == 'greedy':
                a_fin = bo_opt(lambda x: mean_max_value(x, i))
            elif mode == 'current':
                active_states = [si for si, ii in izip(
                    states, curr_inds) if ii == i]
                # If action to be replaced has no corresponding states, default
                # to greedy mode
                if len(active_states) == 0:
                    print 'No active states for element %d! Defaulting to greedy behavior' % i
                    a_fin = bo_opt(lambda x: mean_max_value(x, i))
                else:
                    a_fin = bo_opt(lambda x: mean_curr_value(active_states, x))

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
            td, dl = sess.run([td_err, drift_loss, value_train],
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
            td = np.mean(np.abs(td))
            if conv.iter % 1000 == 0:
                print 'Value training iter %d TD err: %f drift: %f' % (conv.iter, td, dl)

    def train_interleaved(sampler, lib, n, conv_p, conv_v):
        conv_p.clear()
        conv_v.clear()
        # sess.run(policy_reset_op)
        # sess.run(value_reset_op)
        for _ in range(n):
            s, a, r, sn, fs, fa = sampler()
            td, dl = sess.run([td_err, drift_loss, value_train],
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

            td = np.mean(np.abs(td))
            if conv_v.iter % 1000 == 0:
                print 'Value training iter: %d TD: %f Drift: %f' % (conv_v.iter, td, dl)
            value_converged = conv_v.check(td)
            if td > conv_v.tol:
                continue

            er = sess.run([mean_expected_return, expect_train],
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
        run_trial(demo_policy, buffer, final_buffer)

    # Run k-means on demo data to initialize library
    demo_s, demo_a = zip(*buffer)[0:2]
    kmeans = skc.KMeans(n_clusters=n_policy_modes)
    a_inds = kmeans.fit_predict(demo_a)
    library = kmeans.cluster_centers_
    print 'Initial library:\n%s' % str(library)
    demo_buffer = zip(demo_s, a_inds)

    demo_conv = Convergence(min_n=100, tol=0.1, max_iter=10000)
    policy_conv = Convergence(
        min_n=10, tol=0.5, max_iter=10000, use_delta=True)
    value_conv = Convergence(min_n=100, tol=0.1, max_iter=10000)

    def sample_policy_demo():
        return zip(*random.sample(demo_buffer, k=training_batch_size))
    print 'Training initial policy...'
    train_policy_demo(sampler=sample_policy_demo, lib=library, conv=demo_conv)
    print 'Initial training done'

    def sample_value_data():
        s, a, r, sn = zip(*random.sample(buffer, k=training_batch_size))
        fs, fa = zip(*random.sample(final_buffer, k=training_batch_size))
        return s, a, r, sn, fs, fa
    print 'Initializing value function...'
    train_value_only(sampler=sample_value_data, lib=library, conv=value_conv)
    print 'Initialization done'

    trial_batch_size = 1
    init_p = 0.2
    final_p = 0.01
    final_iters = 50

    def get_p(i):
        i = max(min(i, final_iters), 0)
        return init_p + (final_p - init_p) * i / final_iters

    trial_lens = []
    libraries = [library]
    start_time = time.time()
    for iter_i in range(100):

        # Run trial
        p = get_p(iter_i)
        print 'Trial %d using exploration p %f' % (iter_i, p)
        for _ in range(trial_batch_size):
            l = len(buffer)
            run_trial(learned_policy, buffer, final_buffer,
                      lib=library, p_policy=0.9, er_k=0.2)  # Corresponds to 2% randomness for expected return 20
            trial_lens.append(len(buffer) - l)
            if len(buffer) > max_buff_len:
                buffer = buffer[-max_buff_len:]

        if len(buffer) < training_batch_size or len(final_buffer) < training_batch_size:
            continue

        # Now re-select library
        if len(buffer) > min_lib_batch_size \
                and value_conv.last_converged \
                and policy_conv.last_converged:
            lib_replace_k = 1
            print 'Replacing %d elements of library...' % lib_replace_k
            new_lib, new_inds = update_library(states=zip(*buffer)[-min_lib_batch_size:][0],
                                               old_lib=library,
                                               k=lib_replace_k,
                                               mode='current')  # NOTE or greedy?
            print 'New library:\n%s' % str(new_lib)

            library = new_lib
            libraries.append(np.array(library))

            print 'Resetting policy and value optimizers...'
            sess.run(policy_reset_op)
            sess.run(value_reset_op)

            # print 'Decimating policy and value function parameters...'
            # decimate_params(policy_params, p=0.1)
            # decimate_params(value_params, p=0.1)

            # print 'Partially reinitializing policy and value functions...'
            # initialize_policy(sess)
            # initialize_value(sess)

        def sample_both_data():
            s, a, r, sn = zip(*random.sample(buffer, k=training_batch_size))
            fs, fa = zip(*random.sample(final_buffer, k=training_batch_size))
            # inds, adv = get_indices(s, lib=library)
            return s, a, r, sn, fs, fa  # , inds, adv
        print 'Iterating policy and value function...'
        train_interleaved(sampler=sample_both_data, lib=library, n=5000,
                          conv_p=policy_conv, conv_v=value_conv)
        print 'Iterations complete. Policy converged: %s Value converged: %s' \
            % (policy_conv.last_converged, value_conv.last_converged)

        print 'Buff size %d' % len(buffer)
    finish_time = time.time()