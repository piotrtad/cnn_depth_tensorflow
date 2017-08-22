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
    # inputs = (inputs - dataset.IMAGES_MEAN) * dataset.IMAGES_ISTD
    inputs = inputs - 128
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
        print('\n%s \t\t%s' % (net.name, net.get_shape()))
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
        print('\n%s \t\t%s' % (net.name, net.get_shape()))
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
        print('\n%s \t\t%s' % (net.name, net.get_shape()))
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
        print('\n%s \t\t%s' % (net.name, net.get_shape()))
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
        print('\n%s%s' % (net.name, net.get_shape()))
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
        print('\n%s%s' % (net.name, net.get_shape()))
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
                                 method=tf.image.ResizeMethod.BILINEAR)
    if debug:
        print('\n%s %s' % (net.name, net.get_shape()))

    # upscale conv layer 1
    net = conv2d('up1', net, [3, 3, 256, 384], [384], [1, 1, 1, 1],
                 padding='SAME', reuse=reuse, trainable=trainable, wd=0.0005,
                 activation=None)
    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))
    net = tf.layers.batch_normalization(net, axis=3, training=is_training,
                                        name='up1_bn')
    if debug:
        print('%s%s' % (net.name, net.get_shape()))
    unet1 = tf.nn.relu(net, name='up1_relu')
    if debug:
        print('%s \t\t%s' % (unet1.name, unet1.get_shape()))

    # residual
    net = net4 + unet1
    if debug:
        print('%s \t\t\t%s' % (net.name, net.get_shape()))

    # upscale conv layer 2
    net = conv2d('up2', net, [3, 3, 384, 256], [256], [1, 1, 1, 1],
                 padding='SAME', reuse=reuse, trainable=trainable, wd=0.0005,
                 activation=None)
    if debug:
        print('\n%s \t\t%s' % (net.name, net.get_shape()))
    net = tf.layers.batch_normalization(net, axis=3, training=is_training,
                                        name='up2_bn')
    if debug:
        print('%s%s' % (net.name, net.get_shape()))
    unet2 = tf.nn.relu(net, name='up2_relu')
    if debug:
        print('%s \t\t%s' % (unet2.name, unet2.get_shape()))

    # skips
    unet1_ = conv2d('up3_unet1_skip', unet1, [3, 3, 384, 256], [256],
                    [1, 1, 1, 1], padding='SAME', reuse=reuse,
                    trainable=trainable, wd=0.0005, activation=None)
    if debug:
        print('%s%s' % (unet1_.name, unet1_.get_shape()))

    # residual
    net = net2 + unet1_ + unet2

    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))
    net = tf.image.resize_images(net, size=[27, 36],
                                 method=tf.image.ResizeMethod.BILINEAR)
    if debug:
        print('%s %s' % (net.name, net.get_shape()))

    # upscale conv layer 3
    net = conv2d('up3', net, [3, 3, 256, 96], [96], [1, 1, 1, 1],
                 padding='SAME', reuse=reuse, trainable=trainable, wd=0.0005,
                 activation=None)
    if debug:
        print('\n%s \t\t%s' % (net.name, net.get_shape()))
    net = tf.layers.batch_normalization(net, axis=3, training=is_training,
                                        name='up3_bn')
    if debug:
        print('%s%s' % (net.name, net.get_shape()))
    unet3 = tf.nn.relu(net, name='up3_relu')
    if debug:
        print('%s \t\t%s' % (unet3.name, unet3.get_shape()))

    # skips
    unet1_ = tf.image.resize_images(unet1, size=[27, 36],
                                    method=tf.image.ResizeMethod.BILINEAR)
    unet1_ = conv2d('up4_unet1_skip', unet1_, [3, 3, 384, 96], [96],
                    [1, 1, 1, 1], padding='SAME', reuse=reuse,
                    trainable=trainable, wd=0.0005, activation=None)
    if debug:
        print('%s%s' % (unet1_.name, unet1_.get_shape()))
    unet2_ = tf.image.resize_images(unet2, size=[27, 36],
                                    method=tf.image.ResizeMethod.BILINEAR)
    unet2_ = conv2d('up4_unet2_skip', unet2_, [3, 3, 256, 96], [96],
                    [1, 1, 1, 1], padding='SAME', reuse=reuse,
                    trainable=trainable, wd=0.0005, activation=None)
    if debug:
        print('%s%s' % (unet2_.name, unet2_.get_shape()))

    # residual
    net = net1 + unet1_ + unet2_ + unet3

    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))

    net = tf.image.resize_images(net, size=[55, 74],
                                 method=tf.image.ResizeMethod.BILINEAR)
    if debug:
        print('%s %s' % (net.name, net.get_shape()))

    # upscale conv layer 4
    net = conv2d('up4', net, [3, 3, 96, 48], [48], [1, 1, 1, 1],
                 padding='SAME', reuse=reuse, trainable=trainable, wd=0.0005,
                 activation=None)
    if debug:
        print('\n%s \t\t%s' % (net.name, net.get_shape()))
    net = tf.layers.batch_normalization(net, axis=3, training=is_training,
                                        name='up4_bn')
    if debug:
        print('%s%s' % (net.name, net.get_shape()))
    unet4 = tf.nn.relu(net, name='up4_relu')
    if debug:
        print('%s \t\t%s' % (unet4.name, unet4.get_shape()))

    # skips
    unet1_ = tf.image.resize_images(unet1, size=[55, 74],
                                    method=tf.image.ResizeMethod.BILINEAR)
    unet1_ = conv2d('up5_unet1_skip', unet1_, [3, 3, 384, 48], [48],
                    [1, 1, 1, 1], padding='SAME', reuse=reuse,
                    trainable=trainable, wd=0.0005, activation=None)
    if debug:
        print('%s%s' % (unet1_.name, unet1_.get_shape()))
    unet2_ = tf.image.resize_images(unet2, size=[55, 74],
                                    method=tf.image.ResizeMethod.BILINEAR)
    unet2_ = conv2d('up5_unet2_skip', unet2_, [3, 3, 256, 48], [48],
                    [1, 1, 1, 1], padding='SAME', reuse=reuse,
                    trainable=trainable, wd=0.0005, activation=None)
    if debug:
        print('%s%s' % (unet2_.name, unet2_.get_shape()))
    unet3_ = tf.image.resize_images(unet3, size=[55, 74],
                                    method=tf.image.ResizeMethod.BILINEAR)
    unet3_ = conv2d('up5_unet3_skip', unet3_, [3, 3, 96, 48], [48],
                    [1, 1, 1, 1], padding='SAME', reuse=reuse,
                    trainable=trainable, wd=0.0005, activation=None)
    if debug:
        print('%s%s' % (unet3_.name, unet3_.get_shape()))

    # residual
    net = unet1_ + unet2_ + unet3_ + unet4
    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))

    net = tf.image.resize_images(net, size=[111, 150],
                                 method=tf.image.ResizeMethod.BILINEAR)
    if debug:
        print('%s %s' % (net.name, net.get_shape()))

    # upscale conv layer 5
    net = conv2d('up5', net, [3, 3, 48, 24], [24], [1, 1, 1, 1],
                 padding='SAME', reuse=reuse, trainable=trainable, wd=0.0005,
                 activation=None)
    if debug:
        print('\n%s \t\t%s' % (net.name, net.get_shape()))
    net = tf.layers.batch_normalization(net, axis=3, training=is_training,
                                        name='up5_bn')
    if debug:
        print('%s%s' % (net.name, net.get_shape()))
    unet5 = tf.nn.relu(net, name='up5_relu')
    if debug:
        print('%s \t\t%s' % (unet5.name, unet5.get_shape()))

    # skips
    unet1_ = tf.image.resize_images(unet1, size=[111, 150],
                                    method=tf.image.ResizeMethod.BILINEAR)
    unet1_ = conv2d('up6_unet1_skip', unet1_, [3, 3, 384, 24], [24],
                    [1, 1, 1, 1], padding='SAME', reuse=reuse,
                    trainable=trainable, wd=0.0005, activation=None)
    if debug:
        print('%s%s' % (unet1_.name, unet1_.get_shape()))
    unet2_ = tf.image.resize_images(unet2, size=[111, 150],
                                    method=tf.image.ResizeMethod.BILINEAR)
    unet2_ = conv2d('up6_unet2_skip', unet2_, [3, 3, 256, 24], [24],
                    [1, 1, 1, 1], padding='SAME', reuse=reuse,
                    trainable=trainable, wd=0.0005, activation=None)
    if debug:
        print('%s%s' % (unet2_.name, unet2_.get_shape()))
    unet3_ = tf.image.resize_images(unet3, size=[111, 150],
                                    method=tf.image.ResizeMethod.BILINEAR)
    unet3_ = conv2d('up6_unet3_skip', unet3_, [3, 3, 96, 24], [24],
                    [1, 1, 1, 1], padding='SAME', reuse=reuse,
                    trainable=trainable, wd=0.0005, activation=None)
    if debug:
        print('%s%s' % (unet3_.name, unet3_.get_shape()))
    unet4_ = tf.image.resize_images(unet4, size=[111, 150],
                                    method=tf.image.ResizeMethod.BILINEAR)
    unet4_ = conv2d('up6_unet4_skip', unet4_, [3, 3, 48, 24], [24],
                    [1, 1, 1, 1], padding='SAME', reuse=reuse,
                    trainable=trainable, wd=0.0005, activation=None)
    if debug:
        print('%s%s' % (unet4_.name, unet4_.get_shape()))

    # residual
    net = unet1_ + unet2_ + unet3_ + unet4_ + unet5
    if debug:
        print('%s \t\t%s' % (net.name, net.get_shape()))

    net = tf.image.resize_images(net, size=[228, 304],
                                 method=tf.image.ResizeMethod.BILINEAR)
    if debug:
        print('%s %s' % (net.name, net.get_shape()))

    # upscale conv layer 6
    net = conv2d('up6', net, [3, 3, 24, 1], [1], [1, 1, 1, 1],
                 padding='SAME', reuse=reuse, trainable=trainable, wd=0.0005,
                 activation=None)

    if debug:
        print('\n%s \t\t%s' % (net.name, net.get_shape()))
    return net


def errors(logits, depths):
    """Return evaluation metrics.

    logits: predictions
    depths: ground truth

    returns: [EigenError, MeanRelError, MeanLog10Error, RMSE, Accuracy]
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
    eigen_error = tf.reduce_mean(((sum_square_d * n) - (0.5 * square_sum_d)) /
                                 math.pow(n, 2))

    mean_rel_error = tf.reduce_mean(1.0/n * tf.reduce_sum(tf.abs(d)/depths))

    log_diff = tf.abs(tf.log(logits) - tf.log(depths)) / tf.log(10.0)
    mean_log10_error = tf.reduce_mean(1.0/n * tf.reduce_sum(log_diff))

    rmse = tf.reduce_mean(tf.sqrt(1.0/n * tf.reduce_sum(tf.pow(tf.abs(d),
                                                               2.0))))

    acc_with_threshold = []
    for t in [1.25, math.pow(1.25, 2), math.pow(1.25, 3)]:
        foo = logits/depths
        bar = depths/logits
        delta = tf.where(foo > bar, foo, bar)
        good = tf.where(delta < t, delta, tf.zeros_like(delta))
        nz = tf.count_nonzero(good, 1)
        accuracy = tf.reduce_mean(tf.cast(nz, tf.float32)/n)
        acc_with_threshold += [accuracy]
    return [eigen_error, mean_rel_error, mean_log10_error, rmse] + \
        acc_with_threshold


def error(logits, depths):
    """Calculate Eigen RMSE scale-invariant error.

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
    tf.summary.scalar('eigen_error', cost)
    tf.add_to_collection('losses', cost)
    return cost


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
