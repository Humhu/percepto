"""Functions for constructing networks
"""

import tensorflow as tf


def make_network(input, n_layers, n_units, n_outputs, scope,
                 reuse=False,
                 rect=tf.nn.relu, final_rect=None,
                 use_batch_norm=False, training=None,
                 use_dropout=False, dropout_rate=None,
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
    use_batch_norm : bool (default False)
        Whether or not to use batch normalization on each layer input
    training : tf.Tensor object (default None)
        Boolean tensor for indicating training/inference mode to batch norm
    use_dropout : bool (default False)
        Whether or not to use dropout between each layer
    dropout_rate : bool (default None)
        Float tensor for indicating dropout rate
    
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

    # Check arguments
    if use_batch_norm and training is None:
        raise ValueError(
            'Must specify training bool source if using batch norm')
    if use_dropout and dropout_rate is None:
        raise ValueError('Must specify dropout rate source if using dropout')

    layers = []
    variables = []
    
    # TODO Have b_init be an argument as well?
    b_init = tf.constant_initializer(dtype=tf.float32, value=0.0)

    x = input
    with tf.variable_scope(scope, reuse=reuse):
        for i in range(n_layers):

            # NOTE Only use dropout and batch norm in between layers
            if i > 0:
                if use_batch_norm:
                    x = tf.layers.batch_normalization(inputs=x,
                                                      training=training,
                                                      name='batch_%d' % i,
                                                      reuse=reuse)
                    layers.append(x)
                if use_dropout:
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
                width = n_units

            x = tf.layers.dense(inputs=x,
                                units=width,
                                activation=layer_rect,
                                name='layer_%d' % i,
                                reuse=reuse,
                                **kwargs)
            layers.append(x)

            # Collect all trainable variables corresponding to dense layer
            variables += tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope='%s/layer_%d' % (scope, i))

    # Collect all batch normalization update ops
    if use_batch_norm:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
    else:
        update_ops = []
    return layers, variables, update_ops
