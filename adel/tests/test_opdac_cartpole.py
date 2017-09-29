import tensorflow as tf
import gym
import numpy as np
import random
import math

'''Creates a batch-normalized fully-connected network
'''


def make_network(input, is_training, n_layers, n_units,
                 n_outputs, scope, reuse=False, final_rect=None):
    layers = []
    variables = []
    b_init = tf.constant_initializer(dtype=tf.float32, value=0.0)

    x = input
    with tf.variable_scope(scope, reuse=reuse):
        # Rectified layers
        for i in range(n_layers):
            # x = tf.layers.batch_normalization(inputs=x,
            #                                   training=is_training,
            #                                   name='batch_%d' % i,
            #                                   reuse=reuse)
            # layers.append(x)
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
    # next_action_ph = tf.placeholder(tf.float32,
    #                                 shape=[bsize, a_dim],
    #                                 name='next_action')
    final_action_ph = tf.placeholder(tf.float32,
                                     shape=[None, a_dim],
                                     name='final_action')
    reward_ph = tf.placeholder(tf.float32,
                               shape=[bsize, 1],
                               name='reward')

    state_action = tf.concat([state_ph, action_ph],
                             axis=-1,
                             name='state_action')
    final_state_action = tf.concat([final_state_ph, final_action_ph],
                                   axis=-1,
                                   name='final_state_action')
    policy_training = tf.placeholder(tf.bool, name='policy_training')
    value_training = tf.placeholder(tf.bool, name='value_training')

    demo_action_ph = tf.placeholder(tf.float32,
                                    shape=[bsize, a_dim],
                                    name='demo_action')

    ##########################################################################
    # Networks
    ##########################################################################
    n_policy_layers = 2
    n_policy_units = 16
    policy, policy_params = make_network(input=state_ph,
                                         is_training=policy_training,
                                         n_layers=n_policy_layers,
                                         n_units=n_policy_units,
                                         n_outputs=a_dim,
                                         final_rect=tf.nn.tanh,
                                         scope='policy')
    action_scale = tf.constant((env.action_space.high - env.action_space.low) / 2.0,
                               dtype=tf.float32)
    action_offset = tf.constant((env.action_space.high + env.action_space.low) / 2.0,
                                dtype=tf.float32)
    policy_out = action_scale * policy[-1] + action_offset

    n_value_layers = 3
    n_value_units = 512
    value, value_params = make_network(input=state_action,
                                       is_training=value_training,
                                       n_layers=n_value_layers,
                                       n_units=n_value_units,
                                       n_outputs=1,
                                       scope='value',
                                       reuse=False)

    next_policy, _ = make_network(input=next_state_ph,
                                  is_training=policy_training,
                                  n_layers=n_policy_layers,
                                  n_units=n_policy_units,
                                  n_outputs=a_dim,
                                  final_rect=tf.nn.tanh,
                                  scope='policy',
                                  reuse=True)
    next_policy_out = action_scale * next_policy[-1] + action_offset
    next_state_action = tf.concat([next_state_ph, next_policy_out],
                                  axis=-1,
                                  name='next_state_action')
    next_value, _ = make_network(input=next_state_action,
                                 is_training=value_training,
                                 n_layers=n_value_layers,
                                 n_units=n_value_units,
                                 n_outputs=1,
                                 scope='value',
                                 reuse=True)

    final_value, _ = make_network(input=final_state_action,
                                  is_training=value_training,
                                  n_layers=n_value_layers,
                                  n_units=n_value_units,
                                  n_outputs=1,
                                  scope='value',
                                  reuse=True)

    # Gradient computations
    # TD-Gradient error
    gamma = tf.constant(0.9, dtype=tf.float32, name='gamma')
    #td_err = reward_ph + gamma * next_value[-1] - value[-1]
    td_err = reward_ph + gamma * next_value[-1] - value[-1]

    # NOTE tf.gradients assumes that what we really want is d/dx sum(ys)
    # This sum operates over all elements in each tensor in the list of ys!
    # Using the list for allows for different grad_ys premultiplies

    # Divide by number of samples to get mean
    n_samples = tf.cast(tf.shape(state_ph)[0], dtype=tf.float32)
    value_gradients = tf.gradients(ys=value[-1],
                                   xs=action_ph)[0]
    # NOTE Have to negative value gradients since tensorflow steps in negative
    # gradient direction
    policy_gradients = tf.gradients(ys=policy_out,
                                    xs=policy_params,
                                    grad_ys=-value_gradients / n_samples)

    # For batch normalization
    policy_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    value_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)

    # For policy demo learning, we don't need value updates
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                   scope='policy')
    with tf.control_dependencies(update_ops):
        demo_loss = tf.losses.mean_squared_error(labels=demo_action_ph,
                                                 predictions=policy_out)
        demo_train = policy_optimizer.minimize(demo_loss)

    # For policy gradient learning, need to update value batches
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        policy_train = policy_optimizer.apply_gradients(zip(policy_gradients,
                                                            policy_params))

        td_loss = tf.reduce_mean(tf.nn.l2_loss(td_err))
        drift_loss = tf.reduce_mean(tf.nn.l2_loss(final_value[-1]))
        drift_weight = tf.constant(1000.0, dtype=tf.float32)

        value_train = value_optimizer.minimize(td_loss + drift_weight * drift_loss,
                                               var_list=value_params)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    ##########################################################################
    # LEARNING TIME!
    ##########################################################################
    buffer = []
    final_buffer = []

    init_var = 1e-1
    final_var = 1e-3
    final_steps = 100

    value_prelearn_steps = 10000
    learn_steps = 100
    max_trial_len = 1000
    batch_size = 30
    max_buff_len = float('inf')

    dvar = (final_var - init_var) / final_steps

    def get_var(i):
        i = min(final_steps, i)
        return init_var + dvar * i

    def clip(a):
        a[a > env.action_space.high] = env.action_space.high
        a[a < env.action_space.low] = env.action_space.low
        return a

    def demo_trial():
        K_demo = np.array([-1, 0, 20, -2])

        states = []
        actions = []

        obs = env.reset()
        for t in range(max_trial_len):
            env.render()
            action = np.atleast_1d(np.dot(K_demo, obs))
            action = clip(action)
            next_obs, reward, done, info = env.step(action)

            states.append(obs)
            actions.append(action)
            obs = next_obs

            if done:
                break
        return states, actions

    def run_trial(final_buffer, eps_var):
        states = []
        rewards = []
        actions = []
        next_actions = []
        next_states = []

        obs = np.reshape(env.reset(), (1, -1))
        for t in range(max_trial_len):
            env.render()
            eps = np.random.multivariate_normal(mean=np.zeros(a_dim),
                                                cov=np.diag(eps_var))
            action = eps + sess.run(policy_out, feed_dict={state_ph: obs,
                                                           policy_training: False})[0]
            action = clip(action)
            next_obs, reward, done, info = env.step(action)

            next_obs = np.reshape(next_obs, (1, -1))
            next_action = sess.run(policy_out, feed_dict={state_ph: next_obs,
                                                          policy_training: False})[0]

            states.append(obs[0])
            rewards.append(np.atleast_1d(reward))
            actions.append(np.atleast_1d(action))
            next_states.append(next_obs[0])
            next_actions.append(np.atleast_1d(next_action))

            if done:
                print 'Episode terminated early'
                final_buffer.append((next_obs[0], np.atleast_1d(next_action)))
                for _ in range(30):  # TODO
                    fake_action = np.random.uniform(low=-1, high=1)
                    final_buffer.append(
                        (next_obs[0], np.atleast_1d(fake_action)))
                break

            obs = next_obs

        return states, actions, rewards, next_states, next_actions

    def sample_buffer():
        picks = random.sample(buffer, k=batch_size)
        s, a, r, sn, an = zip(*picks)
        return np.array(s), np.array(a), np.array(r), np.array(sn), np.array(an)

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
                                                                policy_training: True})[0]
        if i % 1000 == 0:
            print 'Demo MSE: %f' % demo_mse

    min_trials = 3
    trial_i = 0
    policy_train_i = 0
    value_train_i = 0
    max_td_error = 0.5
    min_value_train = 10000
    while True:
        # Run trial
        decayed_var = get_var(trial_i)
        print 'Running trial with variance %f' % decayed_var
        s, a, r, ns, na = run_trial(final_buffer, [decayed_var] * a_dim)
        g, v = sess.run([value_gradients, value[-1]],
                        feed_dict={state_ph: s,
                                   action_ph: a,
                                   value_training: False})
        g = np.squeeze(g)
        v = np.squeeze(v)
        for si, ai, gi, vi in zip(s, a, g, v):
            print 'obs: %s action: %s action grad: %s value: %s' % (str(si), str(ai), str(gi), str(vi))

        buffer += zip(s, a, r, ns, na)
        if len(buffer) > max_buff_len:
            buffer = buffer[-max_buff_len:]
        trial_i += 1

        if trial_i < min_trials or len(buffer) < batch_size:
            continue

        print 'Policy/value learning...'
        for _ in range(learn_steps):
            mean_abs_td = float('inf')
            while mean_abs_td > max_td_error or value_train_i < min_value_train:
                s, a, r, sn, an = sample_buffer()
                fs, fa = zip(*final_buffer)
                td, dl = sess.run([td_err, drift_loss, value_train],
                                  feed_dict={state_ph: s,
                                             action_ph: a,
                                             reward_ph: r,
                                             next_state_ph: sn,
                                             final_state_ph: fs,
                                             final_action_ph: fa})[0:2]
                mean_abs_td = np.mean(np.abs(td))
                value_train_i += 1
                if value_train_i % 100 == 0:
                    print 'Value training iter: %d policy: %d' % (value_train_i, policy_train_i)
            # mean_abs_td = np.mean(np.abs(td))
            # if mean_abs_td < max_td_error:
            pgs, vgs = sess.run([policy_gradients, policy_train],
                                feed_dict={state_ph: s,
                                           action_ph: a})[0:2]
            policy_train_i += 1

        poli_pars = sess.run(policy_params)
        print 'Buff size %d' % len(buffer)
        print 'Mean TD Err: %f +- %f' % (mean_abs_td, np.std(td))
        print 'Drift err: %f' % np.mean(dl)
        # print 'Value gradients: %s' % str(vgs)
        # print 'Policy gradients: %s' % str(pgs)
        # print 'Policy params: %s' % str(poli_pars)
