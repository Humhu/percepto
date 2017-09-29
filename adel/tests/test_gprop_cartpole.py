import tensorflow as tf
import gym
import numpy as np
import random
import math

'''Creates a batch-normalized fully-connected network
'''


def leaky_relu(x):
    return tf.nn.relu(x) - 1e-3 * tf.nn.relu(-x)


def make_network(input, is_training, n_layers, n_units,
                 n_outputs, scope, reuse=False, final_rect=None):
    layers = []
    variables = []
    b_init = tf.constant_initializer(dtype=tf.float32, value=0.1)

    x = input
    with tf.variable_scope(scope, reuse=reuse):
        # Rectified layers
        for i in range(n_layers):
            x = tf.layers.batch_normalization(inputs=x,
                                              training=is_training,
                                              name='batch_%d' % i,
                                              reuse=reuse)
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
                                    activation=tf.nn.relu,
                                    bias_initializer=b_init,
                                    name='layer_%d' % i,
                                    reuse=reuse)
            layers.append(x)
            variables += tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope='%s/layer_%d' % (scope, i))
        return layers, variables


if __name__ == '__main__':

    # Initialize cartpole problem
    env = gym.make('CartPole-v1')
    env.reset()
    a_scale = env.action_space.high - env.action_space.low
    a_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]

    # Generate networks for gradient learning
    bsize = None
    # batch_size = tf.constant(bsize, dtype=tf.float32, name='batch_size')
    state_ph = tf.placeholder(tf.float32,
                              shape=[bsize, obs_dim],
                              name='state')
    next_state_ph = tf.placeholder(tf.float32,
                                   shape=[bsize, obs_dim],
                                   name='next_state')
    final_state_ph = tf.placeholder(tf.float32,
                                    shape=[None, obs_dim],
                                    name='final_state')
    action_ph = tf.placeholder(tf.float32,
                               shape=[bsize, a_dim],
                               name='action')
    next_action_ph = tf.placeholder(tf.float32,
                                    shape=[bsize, a_dim],
                                    name='next_action')
    final_action_ph = tf.placeholder(tf.float32,
                                     shape=[None, a_dim],
                                     name='final_action')
    reward_ph = tf.placeholder(tf.float32,
                               shape=[bsize, 1],
                               name='reward')
    epsilon_ph = tf.placeholder(tf.float32,
                                shape=[bsize, a_dim],
                                name='epsilon')

    state_action = tf.concat([state_ph, action_ph],
                             axis=-1,
                             name='state_action')
    next_state_action = tf.concat([next_state_ph, next_action_ph],
                                  axis=-1,
                                  name='next_state_action')
    final_state_action = tf.concat([final_state_ph, final_action_ph],
                                   axis=-1,
                                   name='final_state_action')
    is_training = tf.placeholder(tf.bool, name='mode')

    demo_action_ph = tf.placeholder(tf.float32,
                                    shape=[bsize, a_dim],
                                    name='demo_action')

    # Policy maps from state to action
    # TODO Finish updating
    policy, policy_params = make_network(input=state_ph,
                                         is_training=is_training,
                                         n_layers=1,
                                         n_units=8,
                                         n_outputs=a_dim,
                                         final_rect=tf.nn.tanh,
                                         scope='policy')
    policy_out = action_scale * policy[-1] + action_offset

    # Value (action-value) maps from state, action to value
    n_value_layers = 3
    n_value_units = 128
    value, value_params = make_network(input=state_action,
                                       is_training=is_training,
                                       n_layers=n_value_layers,
                                       n_units=n_value_units,
                                       n_outputs=1,
                                       scope='value',
                                       reuse=False)
    next_value, _ = make_network(input=next_state_action,
                                 is_training=is_training,
                                 n_layers=n_value_layers,
                                 n_units=n_value_units,
                                 n_outputs=1,
                                 scope='value',
                                 reuse=True)
    final_value, _ = make_network(input=final_state_action,
                                  is_training=is_training,
                                  n_layers=n_value_layers,
                                  n_units=n_value_units,
                                  n_outputs=1,
                                  scope='value',
                                  reuse=True)

    # Gradient maps from state, action to value's action gradient
    deviation, deviation_params = make_network(input=state_action,
                                               is_training=is_training,
                                               n_layers=3,
                                               n_units=128,
                                               n_outputs=a_dim,
                                               scope='deviation')

    def dot_product(a, b):
        '''Dot product of arrays of vectors
        '''
        return tf.reduce_sum(tf.multiply(a, b), axis=1, keep_dims=True)

    # Gradient computations
    # TD-Gradient error
    gamma = tf.constant(0.9, dtype=tf.float32, name='gamma')
    tdg_err = reward_ph \
        + gamma * next_value[-1] \
        - value[-1] \
        - dot_product(deviation[-1], epsilon_ph)

    # NOTE tf.gradients sums, so we have to:
    # 1. Unstack the different axes so we have vectors across samples
    # 2. Divide by number of samples to get mean gradient

    # Gradient of policy output i multiplied by deviation
    n_samples = tf.cast(tf.shape(state_ph)[0], dtype=tf.float32)
    policy_gradients = tf.gradients(ys=tf.unstack(policy[-1], axis=1),
                                    xs=policy_params,
                                    grad_ys=tf.unstack(-deviation[-1] / n_samples, axis=1))

    # For batch normalization
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                   scope='policy')
    with tf.control_dependencies(update_ops):
        policy_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        policy_train = policy_optimizer.apply_gradients(zip(policy_gradients,
                                                            policy_params))

        demo_loss = tf.losses.mean_squared_error(labels=demo_action_ph,
                                                 predictions=policy[-1])
        demo_train = policy_optimizer.minimize(demo_loss)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                   scope='value') \
        + tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                            scope='deviation')
    with tf.control_dependencies(update_ops):

        tdg_loss = tf.reduce_mean(tf.nn.l2_loss(tdg_err))
        drift_loss = tf.reduce_mean(tf.nn.l2_loss(final_value[-1]))
        drift_weight = tf.constant(1000.0, dtype=tf.float32)

        value_optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
        value_train = value_optimizer.minimize(tdg_loss + drift_weight * drift_loss,
                                               var_list=value_params + deviation_params)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    ##########################################################################
    # LEARNING TIME!
    ##########################################################################
    buffer = []
    final_buffer = []

    init_var = 3e-3
    final_var = 1e-3
    final_steps = 1000

    learn_steps = 100
    max_len = 1000
    batch_size = 500
    max_buff_len = float('inf')
    max_tdg_err = 1.0

    dvar = (final_var - init_var) / final_steps

    def get_var(i):
        return init_var + dvar * i

    def demo_trial():
        K_demo = np.array([-1, 0, 200, -2])

        states = []
        actions = []

        obs = env.reset()
        for t in range(max_len):
            env.render()
            action = np.atleast_1d(np.dot(K_demo, obs))
            action[action > 1] = 1
            action[action < -1] = -1
            next_obs, reward, done, info = env.step(action)

            states.append(obs)
            actions.append(action)
            obs = next_obs

            if done:
                break
        return states, actions

    def clip(a):
        a[a > env.action_space.high] = env.action_space.high
        a[a < env.action_space.low] = env.action_space.low
        return a

    def run_trial(final_buffer, eps_var):
        states = []
        rewards = []
        actions = []
        next_actions = []
        next_states = []
        epsilons = []

        obs = env.reset()
        for t in range(max_trial_len):
            env.render()
            eps = np.random.multivariate_normal(mean=np.zeros(a_dim),
                                                cov=np.diag(eps_var))
            mean_action = sess.run(policy_out, feed_dict={state_ph: obs,
                                                           is_training: False})[0]
            action = clip(mean_action + eps)
            next_obs, reward, done, info = env.step(action)

            next_obs = np.reshape(next_obs, (1, -1))
            next_action = sess.run(policy_out, feed_dict={state_ph: next_obs,
                                                          is_training: False})[0]

            states.append(obs[0])
            rewards.append(np.atleast_1d(reward))
            actions.append(np.atleast_1d(action))
            next_states.append(next_obs[0])
            next_actions.append(np.atleast_1d(next_action))
            epsilons.append(eps)

            if done:
                print 'Episode terminated early'
                final_buffer.append((next_obs[0], np.atleast_1d(next_action)))
                for _ in range(30):  # TODO
                    fake_action = np.random.uniform(low=-1, high=1)
                    final_buffer.append(
                        (next_obs[0], np.atleast_1d(fake_action)))
                break

            obs = next_obs

        return states, actions, rewards, next_states, next_actions, epsilons

    def sample_buffer():
        picks = random.sample(buffer, k=batch_size)
        s, a, r, sn, an, e = zip(*picks)
        return np.array(s), np.array(a), np.array(r), np.array(sn), np.array(an), np.array(e)

    # First initialize policy
    print 'Initializing policy with 10 P-control demos...'
    demos = []
    for i in range(10):
        s, a = demo_trial()
        demos += zip(s, a)
    for i in range(10000):
        sub = random.sample(demos, k=batch_size)
        s, a = zip(*sub)
        demo_mse = sess.run([demo_loss, demo_train], feed_dict={state_ph: np.array(s),
                                                                demo_action_ph: np.array(a),
                                                                is_training: True})[0]
        if i % 1000 == 0:
            print 'Demo MSE: %f' % demo_mse

    trial_i = 0
    policy_train_i = 0
    value_train_i = 0
    while True:
        # Run trial
        decayed_var = get_var(trial_i)
        print 'Running trial with variance %f' % decayed_var
        s, a, r, ns, na, e = run_trial(final_buffer, [decayed_var] * a_dim)
        g = sess.run(deviation[-1], feed_dict={state_ph: s,
                                               action_ph: a,
                                               is_training: False})
        for si, ai, ei, gi in zip(s, a, e, g):
            print 'obs: %s action: %s eps: %s dev: %s' % (str(si), str(ai), str(ei), str(gi))

        buffer += zip(s, a, r, ns, na, e)
        if len(buffer) > max_buff_len:
            buffer = buffer[-max_buff_len:]
        trial_i += 1

        if len(buffer) < batch_size:
            continue

        # Learn a bit
        print 'Training...'
        for _ in range(learn_steps):
            mean_abs_td = float('inf')
            while mean_abs_td > max_td_error or value_train_i < min_value_train:
                s, a, r, sn, an, e = sample_buffer()
                fs, fa = zip(*final_buffer)
                tdg, dl = sess.run([tdg_err, drift_loss, value_train],
                                   feed_dict={state_ph: s,
                                              action_ph: a,
                                              reward_ph: r,
                                              next_state_ph: sn,
                                              next_action_ph: an,
                                              final_state_ph: fs,
                                              final_action_ph: fa,
                                              epsilon_ph: e,
                                              is_training: True})[0:3]
                value_train_i += 1
                mean_abs_tdg = np.mean(np.abs(tdg))

        for _ in range(policy_learn_steps):
            grads = sess.run([policy_gradients, policy_train],
                             feed_dict={state_ph: s,
                                        action_ph: a,
                                        epsilon_ph: e,
                                        is_training: True})[0]
            policy_train_i += 1

        poli_pars = sess.run(policy_params)
        print 'Trained value %d times, policy %d times, buff size %d' % (value_train_i, policy_train_i, len(buffer))
        print 'Mean TDG Err: %f +- %f' % (mean_abs_tdg, np.std(tdg))
        print 'Drift err: %f' % np.mean(dl)
        print 'Gradients: %s' % str(grads)
        print 'Policy params: %s' % str(poli_pars)
