# encoding: utf-8

# tensorflow
import tensorflow as tf
import math

import dataset
# from model_part import conv2d
# from model_part import conv2d_
# from model_part import fc


def model(inputs, reuse=False, trainable=True, debug=False, keep_drop=0.5):
    """Create an upscaled Eigen coarse model.

    inputs.get_shape() = TensorShape([Dimension(8), Dimension(228),
                                      Dimension(304), Dimension(3)])
    """
    # Normalize by subtracting mean ~ 128.
    # inputs = tf.subtract(inputs, 128.0)
    net1 = conv2d('coarse1', inputs, [11, 11, 3, 96], [96], [1, 4, 4, 1],
                  padding='VALID', reuse=reuse, trainable=trainable)
    if debug:
        print('%s \t\t%s' % (net1.name, net1.get_shape()))
    net = tf.nn.max_pool(net1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='VALID', name='pool1')
    if debug:
        print('%s \t\t\t%s' % (net.name, net.get_shape()))

    net2 = conv2d('coarse2', net, [5, 5, 96, 256], [256], [1, 1, 1, 1],
                  padding='VALID', reuse=reuse, trainable=trainable)
    if debug:
        print('%s \t\t%s' % (net2.name, net2.get_shape()))
    net = tf.nn.max_pool(net2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool2')
    if debug:
        print('%s \t\t\t%s' % (net.name, net.get_shape()))
    net = conv2d('coarse3', net, [3, 3, 256, 384], [384], [1, 1, 1, 1],
                 padding='VALID', reuse=reuse, trainable=trainable)
    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))
    net4 = conv2d('coarse4', net, [3, 3, 384, 384], [384], [1, 1, 1, 1],
                  padding='VALID', reuse=reuse, trainable=trainable)
    if debug:
        print('%s \t\t%s' % (net4.name, net4.get_shape()))
    net = conv2d('coarse5', net4, [3, 3, 384, 256], [256], [1, 1, 1, 1],
                 padding='VALID', reuse=reuse, trainable=trainable)
    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))
    net = fc('coarse6', net, [6*10*256, 4096], [4096], reuse=reuse,
             trainable=trainable)

    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))
    net = tf.nn.dropout(net, keep_drop)

    if debug:
        print('%s \t\t\t%s' % (net.name, net.get_shape()))
    net = fc('coarse7', net, [4096, 6*10*256], [6*10*256], reuse=reuse,
             trainable=trainable)
    if debug:
            print('%s \t\t%s' % (net.name, net.get_shape()))
    net = tf.nn.dropout(net, keep_drop)

    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))
    net = tf.reshape(net, [-1, 6, 10, 256])

    if debug:
        print('%s \t\t\t%s' % (net.name, net.get_shape()))
    net = tf.image.resize_images(net, size=[8, 12],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    if debug:
        print('%s \t%s' % (net.name, net.get_shape()))
    net = conv2d('up1', net, [3, 3, 256, 384], [384], [1, 1, 1, 1],
                 padding='SAME', reuse=reuse, trainable=trainable)
    if debug:
        print('%s \t\t\t%s' % (net.name, net.get_shape()))

    # residual
    net = net + net4
    if debug:
        print('%s \t\t\t\t%s' % (net.name, net.get_shape()))
    net = tf.image.resize_images(net, size=[23, 32],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    if debug:
        print('%s \t%s' % (net.name, net.get_shape()))
    net = conv2d('up2', net, [3, 3, 384, 256], [256], [1, 1, 1, 1],
                 padding='SAME', reuse=reuse, trainable=trainable)
    if debug:
        print('%s \t\t\t%s' % (net.name, net.get_shape()))

    # residual
    net = net + net2

    if debug:
        print('%s \t\t\t%s' % (net.name, net.get_shape()))
    net = tf.image.resize_images(net, size=[55, 74],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    if debug:
        print('%s \t%s' % (net.name, net.get_shape()))
    net = conv2d('up3', net, [3, 3, 256, 96], [96], [1, 1, 1, 1],
                 padding='SAME', reuse=reuse, trainable=trainable)
    if debug:
        print('%s \t\t\t%s' % (net.name, net.get_shape()))

    # residual
    net = net + net1

    if debug:
        print('%s \t\t\t%s' % (net.name, net.get_shape()))
    net = tf.image.resize_images(net, size=[228, 304],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    if debug:
        print('%s \t%s' % (net.name, net.get_shape()))
    net = conv2d_('up4', net, [3, 3, 96, 1], [1], [1, 1, 1, 1],
                 padding='SAME', reuse=reuse, trainable=trainable)

    if debug:
        print('%s \t\t\t%s' % (net.name, net.get_shape()))

    return net


def loss(logits, depths, invalid_depths):
    """Calculate Eigen RMSE scale-invariant loss."""
    logits_flat = tf.reshape(logits, [-1, 228*304])
    depths_flat = tf.reshape(depths, [-1, 228*304])
    invalid_depths_flat = tf.reshape(invalid_depths, [-1, 228*304])

    predict = tf.multiply(logits_flat, invalid_depths_flat)
    target = tf.multiply(depths_flat, invalid_depths_flat)
    d = tf.subtract(predict, target)
    square_d = tf.square(d)
    sum_square_d = tf.reduce_sum(square_d, 1)
    sum_d = tf.reduce_sum(d, 1)
    square_sum_d = tf.square(sum_d)
    cost = tf.reduce_mean((sum_square_d / (228.0*304.0)) - (0.5*square_sum_d /
                          math.pow(228*304, 2)))
    tf.add_to_collection('losses', cost)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def scale_invariant_loss(logits, depths):
    """Calculate Eigen RMSE scale-invariant log-loss.

    logits: predictions
    depths: ground truth
    """
    n = dataset.IMAGE_HEIGHT * dataset.IMAGE_WIDTH
    logits = tf.reshape(logits, [-1, n])
    depths = tf.reshape(depths, [-1, n])

    # d = tf.subtract(tf.log(logits), tf.log(depths))
    d = tf.subtract(logits, depths)
    square_d = tf.square(d)
    sum_square_d = tf.reduce_sum(square_d, 1)
    sum_d = tf.reduce_sum(d, 1)
    square_sum_d = tf.square(sum_d)
    cost = tf.reduce_mean(((sum_square_d * n) - (0.5 * square_sum_d)) /
                          math.pow(n, 2))
    tf.add_to_collection('losses', cost)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op
