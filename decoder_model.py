# encoding: utf-8

# tensorflow
import tensorflow as tf
import math

import dataset
from model_part import conv2d
from model_part import fc


def model(inputs, is_training, reuse=False, keep_drop=0.5, trainable=True,
          debug=False):
    """Create an upscaled Eigen coarse model.

    inputs.get_shape() = TensorShape([Dimension(8), Dimension(228),
                                      Dimension(304), Dimension(3)])
    """
    # Normalize.
    inputs = (inputs - dataset.IMAGES_MEAN) * dataset.IMAGES_ISTD
    net = conv2d('conv1', inputs, [11, 11, 3, 96], [96], [1, 4, 4, 1],
                 padding='VALID', reuse=reuse, trainable=trainable, wd=0.0005,
                 activation=None)
    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))
    net = tf.layers.batch_normalization(net, axis=3, training=is_training,
                                        name='conv1_bn')
    if debug:
        print('%s%s' % (net.name, net.get_shape()))
    net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='VALID', name='pool1')
    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))
    net1 = tf.nn.relu(net, name='relu1')
    if debug:
        print('%s \t\t%s' % (net1.name, net1.get_shape()))

    net = conv2d('conv2', net1, [5, 5, 96, 256], [256], [1, 1, 1, 1],
                 padding='SAME', reuse=reuse, trainable=trainable, wd=0.0005,
                 activation=None)
    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))
    net = tf.layers.batch_normalization(net, axis=3, training=is_training,
                                        name='conv2_bn')
    if debug:
        print('%s%s' % (net.name, net.get_shape()))
    net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='VALID', name='pool2')
    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))
    net2 = tf.nn.relu(net, name='relu2')
    if debug:
        print('%s \t\t%s' % (net2.name, net2.get_shape()))

    net = conv2d('conv3', net2, [3, 3, 256, 384], [384], [1, 1, 1, 1],
                 padding='SAME', reuse=reuse, trainable=trainable, wd=0.0005,
                 activation=None)
    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))
    net = tf.layers.batch_normalization(net, axis=3, training=is_training,
                                        name='conv3_bn')
    if debug:
        print('%s%s' % (net.name, net.get_shape()))
    net = tf.nn.relu(net, name='relu3')
    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))

    net = conv2d('conv4', net, [3, 3, 384, 384], [384], [1, 1, 1, 1],
                 padding='SAME', reuse=reuse, trainable=trainable, wd=0.0005,
                 activation=None)
    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))
    net = tf.layers.batch_normalization(net, axis=3, training=is_training,
                                        name='conv4_bn')
    if debug:
        print('%s%s' % (net.name, net.get_shape()))
    net4 = tf.nn.relu(net, name='relu4')
    if debug:
        print('%s \t\t%s' % (net4.name, net4.get_shape()))

    net = conv2d('conv5', net4, [3, 3, 384, 256], [256], [1, 1, 1, 1],
                 padding='SAME', reuse=reuse, trainable=trainable, wd=0.0005,
                 activation=None)
    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))
    net = tf.layers.batch_normalization(net, axis=3, training=is_training,
                                        name='conv5_bn')
    if debug:
        print('%s%s' % (net.name, net.get_shape()))
    net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='VALID', name='pool5')
    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))
    net = tf.nn.relu(net, name='relu5')
    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))

    net = tf.layers.batch_normalization(net, axis=1, training=is_training,
                                        name='fc6_bn')
    if debug:
        print('%s%s' % (net.name, net.get_shape()))
    net = fc('fc6', net, [6*8*256, 4096], [4096], reuse=reuse,
             activation='relu', trainable=trainable)
    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))
    net = tf.nn.dropout(net, keep_drop)
    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))

    net = tf.layers.batch_normalization(net, axis=1, training=is_training,
                                        name='fc7_bn')
    if debug:
        print('%s%s' % (net.name, net.get_shape()))
    net = fc('fc7', net, [4096, 6*8*256], [6*8*256], reuse=reuse,
             activation='relu', trainable=trainable)
    if debug:
            print('%s \t\t%s' % (net.name, net.get_shape()))
    net = tf.nn.dropout(net, keep_drop)
    if debug:
        print('%s \t%s' % (net.name, net.get_shape()))

    net = tf.reshape(net, [-1, 6, 8, 256])
    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))

    net = tf.image.resize_images(net, size=[13, 17],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    if debug:
        print('%s %s' % (net.name, net.get_shape()))
    net = conv2d('up1', net, [3, 3, 256, 384], [384], [1, 1, 1, 1],
                 padding='SAME', reuse=reuse, trainable=trainable, wd=0.0005,
                 activation=None)
    # net = conv2d('up1_1', net, [3, 3, 384, 384], [384], [1, 1, 1, 1],
    #              padding='SAME', reuse=reuse, trainable=trainable, wd=0.0005,
    #              activation=None)
    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))
    net = tf.layers.batch_normalization(net, axis=3, training=is_training,
                                        name='up1_bn')
    if debug:
        print('%s%s' % (net.name, net.get_shape()))
    net = tf.nn.relu(net, name='up1_relu')
    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))

    # residual
    net = net + net4
    if debug:
        print('%s \t\t\t%s' % (net.name, net.get_shape()))
    # net = tf.image.resize_images(net, size=[23, 32],
    #                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # if debug:
    #     print('%s \t%s' % (net.name, net.get_shape()))
    net = conv2d('up2', net, [3, 3, 384, 256], [256], [1, 1, 1, 1],
                 padding='SAME', reuse=reuse, trainable=trainable, wd=0.0005,
                 activation=None)
    # net = conv2d('up2_1', net, [3, 3, 256, 256], [256], [1, 1, 1, 1],
    #              padding='SAME', reuse=reuse, trainable=trainable, wd=0.0005,
    #              activation=None)
    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))
    net = tf.layers.batch_normalization(net, axis=3, training=is_training,
                                        name='up2_bn')
    if debug:
        print('%s%s' % (net.name, net.get_shape()))
    net = tf.nn.relu(net, name='up2_relu')
    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))

    # residual
    net = net + net2

    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))
    net = tf.image.resize_images(net, size=[27, 36],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    if debug:
        print('%s %s' % (net.name, net.get_shape()))
    net = conv2d('up3', net, [3, 3, 256, 96], [96], [1, 1, 1, 1],
                 padding='SAME', reuse=reuse, trainable=trainable, wd=0.0005,
                 activation=None)
    # net = conv2d('up3_1', net, [3, 3, 96, 96], [96], [1, 1, 1, 1],
    #              padding='SAME', reuse=reuse, trainable=trainable, wd=0.0005,
    #              activation=None)
    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))
    net = tf.layers.batch_normalization(net, axis=3, training=is_training,
                                        name='up3_bn')
    if debug:
        print('%s%s' % (net.name, net.get_shape()))
    net = tf.nn.relu(net, name='up3_relu')
    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))

    # residual
    net = net + net1

    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))
        
    net = tf.image.resize_images(net, size=[55, 74],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    if debug:
        print('%s %s' % (net.name, net.get_shape()))
    net = conv2d('up4', net, [3, 3, 96, 48], [48], [1, 1, 1, 1],
                 padding='SAME', reuse=reuse, trainable=trainable, wd=0.0005,
                 activation=None)
    # net = conv2d('up4_1', net, [3, 3, 48, 48], [48], [1, 1, 1, 1],
    #              padding='SAME', reuse=reuse, trainable=trainable, wd=0.0005,
    #              activation=None)
    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))
    net = tf.layers.batch_normalization(net, axis=3, training=is_training,
                                        name='up4_bn')
    if debug:
        print('%s%s' % (net.name, net.get_shape()))
    net = tf.nn.relu(net, name='up4_relu')
    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))
        
    net = tf.image.resize_images(net, size=[111, 150],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    if debug:
        print('%s %s' % (net.name, net.get_shape()))
    net = conv2d('up5', net, [3, 3, 48, 24], [24], [1, 1, 1, 1],
                 padding='SAME', reuse=reuse, trainable=trainable, wd=0.0005,
                 activation=None)
    # net = conv2d('up5_1', net, [3, 3, 24, 24], [24], [1, 1, 1, 1],
    #              padding='SAME', reuse=reuse, trainable=trainable, wd=0.0005,
    #              activation=None)
    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))
    net = tf.layers.batch_normalization(net, axis=3, training=is_training,
                                        name='up5_bn')
    if debug:
        print('%s%s' % (net.name, net.get_shape()))
    net = tf.nn.relu(net, name='up5_relu')
    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))
    net = tf.image.resize_images(net, size=[228, 304],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    if debug:
        print('%s %s' % (net.name, net.get_shape()))
    net = conv2d('up6', net, [3, 3, 24, 1], [1], [1, 1, 1, 1],
                 padding='SAME', reuse=reuse, trainable=trainable, wd=0.0005,
                 activation=None)
    # net = conv2d('up6_1', net, [3, 3, 1, 1], [1], [1, 1, 1, 1],
    #              padding='SAME', reuse=reuse, trainable=trainable, wd=0.0005,
    #              activation=None)

    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))

    return net


def loss(logits, depths, invalid_depths):
    """Calculate Eigen RMSE scale-invariant log-loss.

    logits: predictions
    depths: ground truth
    invalid_depths: masked out missing depths
    """
    logits_flat = tf.reshape(logits, [-1, 55*74])
    depths_flat = tf.reshape(depths, [-1, 55*74])
    invalid_depths_flat = tf.reshape(invalid_depths, [-1, 55*74])

    predict = tf.multiply(logits_flat, invalid_depths_flat)
    target = tf.multiply(depths_flat, invalid_depths_flat)
    d = tf.subtract(predict, target)
    nvalid_pix = tf.reduce_sum(invalid_depths_flat)
    square_d = tf.square(d)
    sum_square_d = tf.reduce_sum(square_d, 1)
    sum_d = tf.reduce_sum(d, 1)
    sqare_sum_d = tf.square(sum_d)
    cost = tf.reduce_mean((sum_square_d * nvalid_pix - 0.5 * sqare_sum_d) /
                          tf.pow(nvalid_pix, 2))
    tf.summary.scalar('eigen_loss', cost)
    tf.add_to_collection('losses', cost)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def scale_invariant_loss(logits, depths):
    """Calculate Eigen RMSE scale-invariant loss.

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
    tf.summary.scalar('eigen_loss', cost)
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
