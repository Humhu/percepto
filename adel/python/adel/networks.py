"""Functions for constructing networks
"""

import tensorflow as tf
import numpy as np
from itertools import izip


def check_vector_arg(arg, n):
    """Check scalar or vector arguments
    """
    if not np.iterable(arg):
        return [arg] * n
    if arg is None:
        raise ValueError('Received None but need 1 or %d scalars' % n)
    if len(arg) != n:
        raise ValueError('Received %d args but need %d' % (len(arg), n))
    return arg


def parse_rect(s):
    """Converts from a string to a Tensorflow rectification class.

    Valid are: relu, tanh, sigmoid
    # NOTE leaky relu disappeared?
    """
    if not isinstance(s, str):
        return s

    lookup = {'relu': tf.nn.relu, #'leaky_relu': tf.nn.leaky_relu,
              'tanh': tf.nn.tanh, 'sigmoid': tf.nn.sigmoid}
    if s not in lookup:
        raise ValueError('Rectification %s not one of valid: %s' %
                         (s, lookup.keys()))
    return lookup[s]

def parse_pool1d(s):
    """Converts from a string to a Tensorflow 1D pooling class.

    Valid are: max, average
    """
    if not isinstance(s, str):
        return s

    lookup = {'max': tf.layers.max_pooling1d,
              'average': tf.layers.average_pooling1d}
    if s not in lookup:
        raise ValueError('1D pool %s not one of valid: %s' %
                         (s, lookup.keys()))
    return lookup[s]

def parse_pool2d(s):
    """Converts from a string to a Tensorflow 2D pooling class.

    Valid are: max, average
    """
    if not isinstance(s, str):
        return s

    lookup = {'max': tf.layers.max_pooling2d,
              'average': tf.layers.average_pooling2d}
    if s not in lookup:
        raise ValueError('2D pool %s not one of valid: %s' %
                         (s, lookup.keys()))
    return lookup[s]


def make_conv1d(input, n_layers, n_filters, filter_sizes, scope, conv_strides=1,
                reuse=False, rect=tf.nn.relu,
                pooling=tf.layers.max_pooling1d, pool_sizes=1, pool_strides=1,
                batch_training=None, dropout_rate=None,
                **kwargs):
    """Helper to create a 1D convolutional network with batch normalization,
    dropout, and proper scoping.

    Parameters
    ==========
    input : tf.Tensor object
        The input tensor to this network
    n_layers : int > 0
        The number of layers in this network
    n_filters : float or iterable of floats
        The number of filters in each layer. If float, replicated for each layer.
    filter_sizes : int or iterable of int
        The filter width and height. If int, replicated for each layer.
    conv_strides : int or iterable of int (default 1)
        The distance between filter applications. If int, replicated for each layer.
    scope : string
        The tf scope to put this layer in

    Keyword Parameters
    ==================
    reuse : bool (default False)
        Whether or not to reuse tf variables of the same name
    rect : tf Rectification class (default tf.nn.relu)
        The rectifier to use inbetween layers
    pooling : tf Pooling class (default tf.max_pooling2d)
        The pooling to use inbetween layers and on the output
    pool_sizes : int or iterable of int
        The block size over which to pool. If int, replicated for each layer. 
        Required if pooling is not None
    pool_strides : int or iterable of int
        The distance between pooling applications. If int, replicated for each layer.
        Required if pooling is not None
    batch_training : tf.Tensor object (default None)
        Boolean tensor for indicating training/inference mode to batch norm. If None,
        disables batch normalization.
    dropout_rate : bool (default None)
        Float tensor for indicating dropout rate. If None, disables dropout.

    NOTE: Any additional keyword parameters will be routed to the constructor
    for tf.layers.conv1d

    Returns
    =======
    layers : list of tf.Tensor objects
        Outputs for each layer, ordered from input to output
    train_variables : list of tf.Variable objects
        Trainable variables in the network
    state_variables: list of tf.Variable objects
        Non-trainable state variables in the network
    update_ops : list of tf operations
        If use_batch_norm is True, update dependencies for batch norm updating

    # TODO Variables for restoring batch norm state?
    """
    layers = []
    train_variables = []
    state_variables = []

    n_filters = check_vector_arg(n_filters, n_layers)
    filter_sizes = check_vector_arg(filter_sizes, n_layers)
    conv_strides = check_vector_arg(conv_strides, n_layers)
    rect = parse_rect(rect)
    if pooling is not None:
        pooling = parse_pool1d(pooling)
        pool_sizes = check_vector_arg(pool_sizes, n_layers)
        pool_strides = check_vector_arg(pool_strides, n_layers)

    # TODO Have b_init be an argument as well?
    b_init = tf.constant_initializer(dtype=tf.float32, value=0.0)

    x = input
    with tf.variable_scope(scope, reuse=reuse):
        for i in range(n_layers):

            # NOTE Only use dropout and batch norm in between layers
            if i > 0:
                if batch_training is not None:
                    x = tf.layers.batch_normalization(inputs=x,
                                                      training=batch_training,
                                                      name='batch_%d' % i,
                                                      reuse=reuse)
                    state_variables += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                         scope='%s/batch_%d' % (scope, i))
                    layers.append(x)
                if dropout_rate is not None:
                    x = tf.layers.dropout(inputs=x,
                                          rate=dropout_rate,
                                          name='dropout_%d' % i)
                    layers.append(x)

            x = tf.layers.conv1d(inputs=x,
                                 filters=n_filters[i],
                                 kernel_size=filter_sizes[i],
                                 strides=conv_strides[i],
                                 activation=rect,
                                 name='conv_%d' % i,
                                 **kwargs)
            layers.append(x)

            # Collect all trainable variables corresponding to conv layer
            train_variables += tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                                 scope='%s/conv_%d' % (scope, i))
            state_variables += tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES,
                                                 scope='%s/conv_%d' % (scope, i))

            if pooling is not None:
                x = pooling(inputs=x,
                            pool_size=pool_sizes[i],
                            strides=pool_strides[i],
                            name='pool_%d' % i)
                layers.append(x)
                train_variables += tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                                     scope='%s/pool_%d' % (scope, i))
                state_variables += tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES,
                                                     scope='%s/pool_%d' % (scope, i))

    # Collect all batch normalization update ops
    if batch_training is not None:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
    else:
        update_ops = []
    return layers, train_variables, state_variables, update_ops


def make_conv2d(input, n_layers, n_filters, filter_sizes, scope, conv_strides=1,
                reuse=False, rect=tf.nn.relu,
                pooling=tf.layers.max_pooling2d, pool_sizes=None, pool_strides=None,
                batch_training=None, dropout_rate=None,
                **kwargs):
    """Helper to create a convolutional network with batch normalization,
    dropout, and proper scoping.

    Parameters
    ==========
    input : tf.Tensor object
        The input tensor to this network
    n_layers : int > 0
        The number of layers in this network
    n_filters : float or iterable of floats
        The number of filters in each layer. If float, replicated for each layer.
    filter_sizes : int or iterable of int
        The filter width and height. If int, replicated for each layer.
    conv_strides : int or iterable of int (default 1)
        The distance between filter applications. If int, replicated for each layer.
    scope : string
        The tf scope to put this layer in

    Keyword Parameters
    ==================
    reuse : bool (default False)
        Whether or not to reuse tf variables of the same name
    rect : tf Rectification class (default tf.nn.relu)
        The rectifier to use inbetween layers
    pooling : tf Pooling class (default tf.max_pooling2d)
        The pooling to use inbetween layers and on the output
    pool_sizes : int or iterable of int
        The block size over which to pool. If int, replicated for each layer.
        Required if pooling is not None
    pool_strides : int or iterable of int
        The distance between pooling applications. If int, replicated for each layer.
        Required  if pooling is not None
    batch_training : tf.Tensor object (default None)
        Boolean tensor for indicating training/inference mode to batch norm. If None,
        disables batch normalization.
    dropout_rate : bool (default None)
        Float tensor for indicating dropout rate. If None, disables dropout.

    NOTE: Any additional keyword parameters will be routed to the constructor
    for tf.layers.conv2d

    Returns
    =======
    layers : list of tf.Tensor objects
        Outputs for each layer, ordered from input to output
    train_variables : list of tf.Variable objects
        Trainable variables in the network
    state_variables: list of tf.Variable objects
        Non-trainable state variables in the network
    update_ops : list of tf operations
        If use_batch_norm is True, update dependencies for batch norm updating

    # TODO Variables for restoring batch norm state?
    """
    layers = []
    train_variables = []
    state_variables = []

    n_filters = check_vector_arg(n_filters, n_layers)
    filter_sizes = check_vector_arg(filter_sizes, n_layers)
    conv_strides = check_vector_arg(conv_strides, n_layers)
    rect = parse_rect(rect)
    if pooling is not None:
        pooling = parse_pool2d(pooling)
        pool_sizes = check_vector_arg(pool_sizes, n_layers)
        pool_strides = check_vector_arg(pool_strides, n_layers)

    # TODO Have b_init be an argument as well?
    b_init = tf.constant_initializer(dtype=tf.float32, value=0.0)

    x = input
    with tf.variable_scope(scope, reuse=reuse):
        for i in range(n_layers):

            # NOTE Only use dropout and batch norm in between layers
            if i > 0:
                if batch_training is not None:
                    x = tf.layers.batch_normalization(inputs=x,
                                                      training=batch_training,
                                                      name='batch_%d' % i,
                                                      reuse=reuse)
                    state_variables += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                         scope='%s/batch_%d' % (scope, i))
                    layers.append(x)
                if dropout_rate is not None:
                    x = tf.layers.dropout(inputs=x,
                                          rate=dropout_rate,
                                          name='dropout_%d' % i)
                    layers.append(x)

            x = tf.layers.conv2d(inputs=x,
                                 filters=n_filters[i],
                                 kernel_size=(
                                     filter_sizes[i], filter_sizes[i]),
                                 strides=(conv_strides[i], conv_strides[i]),
                                 activation=rect,
                                 name='conv_%d' % i,
                                 **kwargs)
            layers.append(x)

            # Collect all trainable variables corresponding to conv layer
            train_variables += tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                                 scope='%s/conv_%d' % (scope, i))
            state_variables += tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES,
                                                 scope='%s/conv_%d' % (scope, i))

            if pooling is not None:
                x = pooling(inputs=x,
                            pool_size=(pool_sizes[i], pool_sizes[i]),
                            strides=(pool_strides[i], pool_strides[i]),
                            name='pool_%d' % i)
                layers.append(x)
                train_variables += tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                                     scope='%s/pool_%d' % (scope, i))
                state_variables += tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES,
                                                     scope='%s/pool_%d' % (scope, i))

    # Collect all batch normalization update ops
    if batch_training is not None:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
    else:
        update_ops = []
    return layers, train_variables, state_variables, update_ops


def make_fullycon(input, n_layers, n_units, n_outputs, scope,
                  reuse=False,
                  rect=tf.nn.relu, final_rect=None,
                  batch_training=None, dropout_rate=None,
                  **kwargs):
    """Helper to creates a fully connected network with batch normalization,
    dropout, proper scoping, and output rectification.

    Parameters
    ==========
    input : tf.Tensor object
        The input tensor to this network
    n_layers : int > 0
        The number of layers in this network
    n_units : int > 0
        The width of each layer in the network
    n_outputs : int > 0
        The final output dimensionality
    scope : string
        The tf scope to put this layer in

    Keyword Parameters
    ==================
    reuse : bool (default False)
        Whether or not to reuse tf variables of the same name
    rect : tf Rectification class (default tf.nn.relu)
        The rectifier to use inbetween layers
    final_rect : tf Rectification class or None (default None)
        The rectifier to use on the output layer if not None
    batch_training : tf.Tensor object (default None)
        Boolean tensor for indicating training/inference mode to batch norm.
        If None, disables batch normalization.
    dropout_rate : bool (default None)
        Float tensor for indicating dropout rate. If None, disables dropout.

    NOTE: Any additional keyword parameters will be routed to the constructor
    for tf.layers.dense

    Returns
    =======
    layers : list of tf.Tensor objects
        Outputs for each layer, ordered from input to output
    variables : list of tf.Variable objects
        Trainable variables in the network
    update_ops : list of ??? 
        If use_batch_norm is True, update dependencies for batch norm updating

    # TODO Variables for restoring batch norm state?
    """

    layers = []
    train_variables = []
    state_variables = []

    n_units = check_vector_arg(n_units, n_layers - 1)
    rect = parse_rect(rect)
    final_rect = parse_rect(final_rect)

    x = input
    with tf.variable_scope(scope, reuse=reuse):
        for i in range(n_layers):

            # NOTE Only use dropout and batch norm in between layers
            if i > 0:
                if batch_training is not None:
                    x = tf.layers.batch_normalization(inputs=x,
                                                      training=batch_training,
                                                      name='batch_%d' % i,
                                                      reuse=reuse)
                    state_variables += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                         scope='%s/batch_%d' % (scope, i))
                    layers.append(x)
                if dropout_rate is not None:
                    x = tf.layers.dropout(inputs=x,
                                          rate=dropout_rate,
                                          name='dropout_%d' % i)
                    layers.append(x)

            # Final output layer has different rectification
            if i == n_layers - 1:
                layer_rect = final_rect
                width = n_outputs
            else:
                layer_rect = rect
                width = n_units[i]

            x = tf.layers.dense(inputs=x,
                                units=width,
                                activation=layer_rect,
                                name='layer_%d' % i,
                                reuse=reuse,
                                **kwargs)
            layers.append(x)

            # Collect all trainable variables corresponding to dense layer
            train_variables += tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                                 scope='%s/layer_%d' % (scope, i))
            state_variables += tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES,
                                                 scope='%s/layer_%d' % (scope, i))

    # Collect all batch normalization update ops
    if batch_training is not None:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
    else:
        update_ops = []

    return layers, train_variables, state_variables, update_ops


def copy_expanded_params(sess, old_param_vals, new_params):
    """Copies parameters from an old network to an expanded version.
    """
    if len(old_param_vals) != len(new_params):
        raise ValueError('Expected %d params but got %d' %
                         (len(old_param_vals), len(new_params)))

    for old_vals, new in izip(old_param_vals, new_params):
        old_shape = old_vals.shape
        new_vals = sess.run(new)
        new_shape = new_vals.shape
        if np.any(old_shape > new_shape):
            raise ValueError('Old params have shape %s but new have smaller %s' %
                             (str(old_shape), str(new_shape)))
        ranges = [range(s) for s in old_shape]
        new_vals[np.ix_(*ranges)] = old_vals
        sess.run(tf.assign(new, new_vals))
