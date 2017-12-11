"""Basic network architectures
"""

from networks import *
import tensorflow as tf
import numpy as np


def make_conv2d_fc_net(img_in, image_subnet, final_subnet, scope='',
                       dropout_rate=None, batch_training=None, reuse=False):
    """Creates an network that stacks a 2D convolution with a fully connected net.

    Parameters
    ----------
    img_in         : tensorflow 4D Tensor
    image_subnet   : dict
        Arguments to pass to image subnet constructor
    final_subnet   : dict
        Arguments to pass to fully connected subnet constructor
    scope          : string (default '')
        Scope prefix prepended to subnet scopes
    dropout_rate   : tensorflow bool Tensor (default None)
    batch_training : tensorflow bool Tensor (default None)
    reuse          : bool (default False)
        Whether or not to reuse existing variables
    """
    img_net, img_train, img_state, img_ups = make_conv2d(input=img_in,
                                                         scope='%sjoint_image' % scope,
                                                         dropout_rate=dropout_rate,
                                                         batch_training=batch_training,
                                                         reuse=reuse,
                                                         **image_subnet)
    flat_dim = int(np.prod(img_net[-1].shape[1:]))
    img_flat = tf.reshape(img_net[-1], (-1, flat_dim))
    fin_net, fin_train, fin_state, fin_ups = make_fullycon(input=img_flat,
                                                           scope='%sjoint_fc' % scope,
                                                           dropout_rate=dropout_rate,
                                                           batch_training=batch_training,
                                                           reuse=reuse,
                                                           **final_subnet)
    all_layers = img_net + [img_flat] + fin_net
    all_train = img_train + fin_train
    all_state = img_state + fin_state
    all_ups = img_ups + fin_ups
    return all_layers, all_train, all_state, all_ups


def make_conv2d_joint_net(img_in, vector_in, image_subnet, squeeze_subnet, vector_subnet,
                          final_subnet, scope='', dropout_rate=None,
                          batch_training=None, reuse=False):
    """Creates an network that combines a 2D convolution with a fully connected net and
    passes the flattened output through another fully connected network.

    Parameters
    ----------
    img_in         : tensorflow 4D Tensor
    vector_in      : tensorflow 2D Tensor
    image_subnet   : dict
        Arguments to pass to image subnet constructor
    vector_subnet  : dict
        Arguments to pass to vector subnet constructor
    final_subnet   : dict
        Arguments to pass to final subnet constructor
    scope          : string (default '')
        Scope prefix prepended to subnet scopes
    dropout_rate   : tensorflow bool Tensor (default None)
    batch_training : tensorflow bool Tensor (default None)
    reuse          : bool (default False)
        Whether or not to reuse existing variables
    """
    img_net, img_train, img_state, img_ups = make_conv2d(input=img_in,
                                                         scope='%sjoint_image' % scope,
                                                         dropout_rate=dropout_rate,
                                                         batch_training=batch_training,
                                                         reuse=reuse,
                                                         **image_subnet)
    vec_net, vec_train, vec_state, vec_ups = make_fullycon(input=vector_in,
                                                           scope='%sjoint_vector' % scope,
                                                           dropout_rate=dropout_rate,
                                                           batch_training=batch_training,
                                                           reuse=reuse,
                                                           **vector_subnet)
    flat_dim = int(np.prod(img_net[-1].shape[1:]))
    img_flat = tf.reshape(img_net[-1], (-1, flat_dim))
    sq_net, sq_train, sq_state, sq_ups = make_fullycon(inpu=img_flat,
                                                       scope='%sjoint_squeeze' % scope,
                                                       dropout_rate=dropout_rate,
                                                       batch_training=batch_training,
                                                       reuse=reuse,
                                                       **squeeze_subnet)
    joined = tf.concat([sq_net[-1], vec_net[-1]], axis=-1)
    fin_net, fin_train, fin_state, fin_ups = make_fullycon(input=joined,
                                                           scope='%sjoint_final' % scope,
                                                           dropout_rate=dropout_rate,
                                                           batch_training=batch_training,
                                                           reuse=reuse,
                                                           **final_subnet)
    all_layers = img_net + sq_net + vec_net + [img_flat, joined] + fin_net
    all_train = img_train + sq_train + vec_train + fin_train
    all_state = img_state + sq_state + vec_state + fin_state
    all_ups = img_ups + sq_ups + vec_ups + fin_ups
    return all_layers, all_train, all_state, all_ups
