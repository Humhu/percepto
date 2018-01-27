"""Basic network architectures
"""

import adel
from networks import *
import tensorflow as tf
import numpy as np

from tensorflow_vgg import Vgg16


def make_vgg_net(img_in, output_layer, post_pool=None, post_pool_size=-1, post_pool_stride=1,
                 reshape=True, build_fc=True, scale_output=1.0, **kwargs):
    """Creates an interface between an image and the VGG16 network
    """

    img = img_in
    layers = []
    if reshape:
        # First resize the image to 224
        img = tf.image.resize_images(images=img_in, size=[224, 224])
        layers.append(img)

    # Pad them to 3 channels
    img = tf.image.grayscale_to_rgb(images=img)
    layers.append(img)

    vgg = Vgg16(**kwargs)
    vgg.build(rgb=img, build_fc=build_fc)
    output = getattr(vgg, output_layer) * float(scale_output)
    layers.append(output)

    if post_pool is not None:
        pool_type = adel.parse_pool2d(post_pool)
        if post_pool_size < 0:
            post_pool_size = output.shape[1:3]
        else:
            post_pool_size = [post_pool_size, post_pool_size]
        pool = pool_type(inputs=output,
                         pool_size=post_pool_size,
                         strides=[post_pool_stride, post_pool_stride])
        layers.append(pool)
    return layers


def make_conv2d_fc_net(img_in, image_subnet, final_subnet, scope='', **kwargs):
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
    """
    image_subnet.update(kwargs)
    final_subnet.update(kwargs)
    img_net, img_train, img_state, img_ups = make_conv2d(input=img_in,
                                                         scope='%sjoint_image' % scope,
                                                         **image_subnet)
    flat_dim = int(np.prod(img_net[-1].shape[1:]))
    img_flat = tf.reshape(img_net[-1], (-1, flat_dim))
    fin_net, fin_train, fin_state, fin_ups = make_fullycon(input=img_flat,
                                                           scope='%sjoint_fc' % scope,
                                                           **final_subnet)
    all_layers = img_net + [img_flat] + fin_net
    all_train = img_train + fin_train
    all_state = img_state + fin_state
    all_ups = img_ups + fin_ups
    return all_layers, all_train, all_state, all_ups


def make_conv2d_joint_net(img_in, vector_in, image_subnet, squeeze_subnet, vector_subnet,
                          final_subnet, scope='', **kwargs):
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
    image_subnet.update(kwargs)
    vector_subnet.update(kwargs)
    squeeze_subnet.update(kwargs)
    final_subnet.update(kwargs)
    img_net, img_train, img_state, img_ups = make_conv2d(input=img_in,
                                                         scope='%sjoint_image' % scope,
                                                         **image_subnet)
    vec_net, vec_train, vec_state, vec_ups = make_fullycon(input=vector_in,
                                                           scope='%sjoint_vector' % scope,
                                                           **vector_subnet)
    flat_dim = int(np.prod(img_net[-1].shape[1:]))
    img_flat = tf.reshape(img_net[-1], (-1, flat_dim))
    sq_net, sq_train, sq_state, sq_ups = make_fullycon(input=img_flat,
                                                       scope='%sjoint_squeeze' % scope,
                                                       **squeeze_subnet)
    joined = tf.concat([sq_net[-1], vec_net[-1]], axis=-1)
    fin_net, fin_train, fin_state, fin_ups = make_fullycon(input=joined,
                                                           scope='%sjoint_final' % scope,
                                                           **final_subnet)
    all_layers = img_net + sq_net + vec_net + [img_flat, joined] + fin_net
    all_train = img_train + sq_train + vec_train + fin_train
    all_state = img_state + sq_state + vec_state + fin_state
    all_ups = img_ups + sq_ups + vec_ups + fin_ups
    return all_layers, all_train, all_state, all_ups


def make_conv2d_parallel_net(img1, img2, conv1, conv2, squeeze1, squeeze2, final,
                             scope='', **kwargs):
    """Creates a network that joins two convnets together
    """
    conv1.update(kwargs)
    conv2.update(kwargs)
    squeeze1.update(kwargs)
    squeeze2.update(kwargs)
    final.update(kwargs)

    net1, train1, state1, ups1 = make_conv2d(input=img1,
                                             scope='%s/sub1' % scope,
                                             **conv1)
    net2, train2, state2, ups2 = make_conv2d(input=img2,
                                             scope='%s/sub2' % scope,
                                             **conv2)
    flat_dim1 = int(np.prod(net1[-1].shape[1:]))
    flat1 = tf.reshape(net1[-1], (-1, flat_dim1))
    flat_dim2 = int(np.prod(net2[-1].shape[1:]))
    flat2 = tf.reshape(net2[-1], (-1, flat_dim2))

    sq1, sq_train1, sq_state1, sq_ups1 = make_fullycon(input=flat1,
                                                       scope='%s/squeeze1' % scope,
                                                       **squeeze1)
    sq2, sq_train2, sq_state2, sq_ups2 = make_fullycon(input=flat2,
                                                       scope='%s/squeeze2' % scope,
                                                       **squeeze2)

    joined = tf.concat([sq1[-1], sq2[-1]], axis=-1)
    fin_net, fin_train, fin_state, fin_ups = make_fullycon(input=joined,
                                                           scope='%s/final' % scope,
                                                           **final)
    all_layers = net1 + sq1 + [flat1] + net2 + \
        sq2 + [flat2] + [joined] + fin_net
    all_train = train1 + train2 + sq_train1 + sq_train2 + fin_train
    all_state = state1 + state2 + sq_state1 + sq_state2 + fin_state
    all_ups = ups1 + ups2 + sq_ups1 + sq_ups2 + fin_ups
    return all_layers, all_train, all_state, all_ups
