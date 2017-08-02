# encoding: utf-8

# tensorflow
import tensorflow as tf
import dataset
import math
from model_part import conv2d
from model_part import fc


def inference(images, is_training=False, keep_drop=1.0, reuse=False,
              trainable=True, debug=False):
    """Create Eigen's coarse model."""
    images = images - 128.0
    if debug:
        print('%s \t\t\t\t%s' % (images.name, images.get_shape()))
    net = conv2d('coarse_conv1', images, [11, 11, 3, 96], [96], [1, 4, 4, 1],
                 padding='VALID', reuse=reuse, trainable=trainable, wd=0.0005,
                 activation='relu')
    if debug:
        print('%s \t%s' % (net.name, net.get_shape()))
    net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='VALID', name='coarse_pool1')
    if debug:
        print('%s \t\t\t%s' % (net.name, net.get_shape()))
    net = tf.layers.batch_normalization(net, axis=3, training=is_training,
                                        name='coarse_pool1_bn')
    if debug:
        print('%s%s' % (net.name, net.get_shape()))
    net = conv2d('coarse_conv2', net, [5, 5, 96, 256], [256], [1, 1, 1, 1],
                 padding='SAME', reuse=reuse, trainable=trainable, wd=0.0005,
                 activation='relu')
    if debug:
        print('%s \t%s' % (net.name, net.get_shape()))
    net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='VALID', name='coarse_pool2')
    if debug:
        print('%s \t\t\t%s' % (net.name, net.get_shape()))
    net = tf.layers.batch_normalization(net, axis=3, training=is_training,
                                        name='coarse_pool2_bn')
    if debug:
        print('%s%s' % (net.name, net.get_shape()))
    net = conv2d('coarse_conv3', net, [3, 3, 256, 384], [384], [1, 1, 1, 1],
                 padding='SAME', reuse=reuse, trainable=trainable, wd=0.0005,
                 activation='relu')
    if debug:
        print('%s \t%s' % (net.name, net.get_shape()))
    net = conv2d('coarse_conv4', net, [3, 3, 384, 384], [384], [1, 1, 1, 1],
                 padding='SAME', reuse=reuse, trainable=trainable, wd=0.0005,
                 activation='relu')
    if debug:
        print('%s \t%s' % (net.name, net.get_shape()))
    net = conv2d('coarse_conv5', net, [3, 3, 384, 256], [256], [1, 1, 1, 1],
                 padding='SAME', reuse=reuse, trainable=trainable, wd=0.0005,
                 activation='relu')
    if debug:
        print('%s \t%s' % (net.name, net.get_shape()))
    net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='VALID', name='coarse_pool5')
    if debug:
        print('%s \t\t\t%s' % (net.name, net.get_shape()))
    net = fc('coarse_full1', net, [6*8*256, 4096], [4096], reuse=reuse,
             activation='relu', trainable=trainable, wd=0.0001)
    net = tf.nn.dropout(net, keep_drop)

    if debug:
        print('%s \t\t\t%s' % (net.name, net.get_shape()))
    net = fc('coarse_full2', net, [4096, 4070], [4070], reuse=reuse,
             activation=None, trainable=trainable, wd=0.0001)
    if debug:
        print('%s \t%s' % (net.name, net.get_shape()))
    net = tf.reshape(net, [-1, 55, 74, 1])
    if debug:
        print('%s \t\t\t%s' % (net.name, net.get_shape()))
    return net


def inference_refine(images, coarse7_output, reuse=False, trainable=True,
                     debug=False):
    """Create Eigen's refine model."""
    images = (images - dataset.IMAGES_MEAN) * dataset.IMAGES_ISTD
    if debug:
        print('%s \t\t\t\t%s' % (images.name, images.get_shape()))
    net = conv2d('fine_conv1', images, [9, 9, 3, 63], [63], [1, 2, 2, 1],
                 padding='VALID', reuse=reuse, trainable=trainable, wd=0.0001,
                 activation='relu')
    if debug:
        print('%s \t%s' % (net.name, net.get_shape()))
    net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='fine_pool1')
    if debug:
        print('%s \t\t\t%s' % (net.name, net.get_shape()))
    net = tf.concat([net, coarse7_output], 3)
    if debug:
        print('%s \t\t\t%s' % (net.name, net.get_shape()))
    net = conv2d('fine_conv2', net, [5, 5, 64, 64], [64], [1, 1, 1, 1],
                 padding='SAME', reuse=reuse, trainable=trainable, wd=0.0001,
                 activation='relu')
    if debug:
        print('%s \t%s' % (net.name, net.get_shape()))
    net = conv2d('fine_conv3', net, [5, 5, 64, 1], [1], [1, 1, 1, 1],
                 padding='SAME', reuse=reuse, trainable=trainable, wd=0.0001,
                 activation=None)
    if debug:
        print('%s \t%s' % (net.name, net.get_shape()))

    return net


def errors(logits, depths):
    """Return evaluation metrics.

    logits: predictions
    depths: ground truth

    returns: [EigenError, MeanRelError, MeanLog10Error, RMSE, Accuracy]
    """
    n = dataset.TARGET_HEIGHT * dataset.TARGET_WIDTH
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
    tf.summary.scalar('eigen_error', eigen_error)
    tf.add_to_collection('errors', eigen_error)
    # tf.add_to_collection('losses', cost)

    mean_rel_error = tf.reduce_mean(1.0/n * tf.reduce_sum(tf.abs(d)/depths))
    tf.summary.scalar('mean_rel_error', mean_rel_error)
    tf.add_to_collection('errors', mean_rel_error)

    log_diff = tf.abs(tf.log(logits) - tf.log(depths)) / tf.log(10.0)
    mean_log10_error = tf.reduce_mean(1.0/n * tf.reduce_sum(log_diff))
    tf.summary.scalar('mean_log10_error', mean_log10_error)
    tf.add_to_collection('errors', mean_log10_error)

    rmse = tf.reduce_mean(tf.sqrt(1.0/n * tf.reduce_sum(tf.pow(tf.abs(d),
                                                               2.0))))
    tf.summary.scalar('root_mean_square_error', rmse)
    tf.add_to_collection('errors', rmse)

    acc_with_threshold = []
    for t in [1.25, math.pow(1.25, 2), math.pow(1.25, 3)]:
        foo = logits/depths
        bar = depths/logits
        delta = tf.where(foo > bar, foo, bar)
        good = tf.where(delta < t, delta, tf.zeros_like(delta))
        # nz = tf.count_nonzero(good, [1, 2])
        nz = tf.count_nonzero(good, 1)
        accuracy = tf.reduce_mean(tf.cast(nz, tf.float32)/n)
        acc_with_threshold += [accuracy]
        tf.summary.scalar('mean_accuracy_with_threshold_%f' % t, accuracy)
    tf.add_to_collection('errors', acc_with_threshold)
    return [eigen_error, mean_rel_error, mean_log10_error, rmse] + \
        acc_with_threshold
    # return tf.get_collection('errors')


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


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op
