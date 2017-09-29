"""Software to facilitate value function learning.
"""

import tensorflow as tf


def value_combinations(states, actions, make_value):
    """Builds modules for computing values for state/action combinations.

    Assuming N different M-dimensional states
    and B different O-dimensional actions

    Parameters
    ==========
    states : tf.Tensor [N x M]
        States to evaluate at
    actions : tf.Tensor [B x O]
        Values of discrete actions
    make_value : function(input)
        Function to replicate the value network with a specified input
    """
    N = tf.shape(input=states)[0]
    B = tf.shape(input=actions)[0]

    # We will tile [state] and [actions]
    # as [state1, state1, state2, state2, ...] and
    # [action1, action2, action1, action2, ...]
    idx = tf.range(N)
    idx = tf.reshape(idx, [-1, 1])    # Convert to a len(yp) x 1 matrix.
    idx = tf.tile(idx, [1, B])  # Create multiple columns.
    idx = tf.reshape(idx, [-1])       # Convert back to a vector.
    tiled_states = tf.gather(states, idx)

    jdx = tf.range(B)
    jdx = tf.tile(jdx, [N])
    tiled_library = tf.gather(actions, jdx)
    tiled_state_action = tf.concat([tiled_states, tiled_library], axis=-1)

    tiled_value = make_value(input=tiled_state_action)
    tiled_value_shape = tf.stack([N, B])
    tiled_value_mat = tf.reshape(tensor=tiled_value,
                                 shape=tiled_value_shape)
    return tiled_value_mat

def anchored_td_loss(rewards, values, next_values, terminal_values,
                     gamma, dweight):
    """Builds modules for computing TD error

    Parameters
    ==========
    reward : tf.Tensor
        Reward samples
    gamma : tf.Tensor or float
        Discount factor
    curr_value : tf.Tensor
        Values for state/actions that generated rewards
    next_value : tf.Tensor
        Values for state/actions after generating rewards

    Returns
    =======
    loss : tf.Tensor
        Mean-squared TD error
    """
    if not isinstance(gamma, tf.Tensor):
        gamma = tf.constant(gamma, dtype=tf.float32, name='gamma')
    if not isinstance(dweight, tf.Tensor):
        dweight = tf.constant(dweight, dtype=tf.float32, name='dweight')

    td = rewards + gamma * next_values - values
    td_loss = tf.reduce_mean(tf.nn.l2_loss(td))
    drift_loss = tf.reduce_mean(tf.nn.l2_loss(terminal_values))
    
    return td_loss + dweight * drift_loss, td_loss, drift_loss