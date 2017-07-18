"""Evaluation for Eigen Depth.

Accuracy:
For NYUDepth, Eigen used the common distribution training set of 795 images.
They evaluate using several errors (see paper) but report a mean error of 0.304
for the log RMSE scale invariant loss.
Speed:
-
Usage:
-
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import math
import os
import sys
import time

import numpy as np
import tensorflow as tf

from dataset import DataSet
import decoder_model

FLAGS = tf.app.flags.FLAGS


def eval_once(saver, summary_writer, error, summary_op):
    """Run Eval once.

    Args:
    saver: Saver.
    summary_writer: Summary writer.
    error: Error op.
    summary_op: Summary op.
    """
    with tf.Session(config=tf.ConfigProto(
                    log_device_placement=FLAGS.log_device_placement,
                    gpu_options=tf.GPUOptions(visible_device_list='1',
                                              allow_growth=True))) as sess:

        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path \
                .split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            num_iter = int(math.ceil(FLAGS.num_examples
                                     / FLAGS.batch_size))
            total_loss = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([error])
                total_loss += np.sum(predictions)
                step += 1

            # Compute precision @ 1.
            precision = total_loss / total_sample_count

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1',
                              simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """Eval CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        dataset = DataSet(FLAGS.batch_size)
        images, depths, invalid_depths = dataset.csv_inputs(FLAGS.test_file,
                                                            target_size=[228,
                                                                         304])

        # Build a Graph that computes the logits predictions from the
        # inference model.
        # keep_conv = tf.placeholder(tf.float32)
        # keep_hidden = tf.placeholder(tf.float32)
        logits = decoder_model.model(images, trainable=False)

        # Calculate predictions.
        error = decoder_model.loss(logits, depths, invalid_depths)

        # Restore the moving average version of the learned variables for eval.
        # variable_averages = tf.train.ExponentialMovingAverage(
        #     train_operation.MOVING_AVERAGE_DECAY)
        # variables_to_restore = variable_averages.variables_to_restore()
        # saver = tf.train.Saver(variables_to_restore)
        # parameters
        variables_to_restore = {}

        for variable in tf.trainable_variables():
            variable_name = variable.name
            if variable_name.find("/") < 0 or variable_name.count("/") != 1:
                continue
            variables_to_restore[variable_name] = variable

        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, g)

        while True:
            eval_once(saver, summary_writer, error, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
    """Main."""
    if tf.gfile.Exists(FLAGS.test_dir):
        tf.gfile.DeleteRecursively(FLAGS.test_dir)
    tf.gfile.MakeDirs(FLAGS.test_dir)
    evaluate()


if __name__ == '__main__':
    today = datetime.strftime(datetime.now(), '%d%m%y')
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_device_placement', action='store_true',
                        help='Log device placement')  # default False
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--test_file', type=str, default='test.csv',
                        help='Test file')
    parser.add_argument('--test_dir', type=str,
                        default=os.path.join(today, 'test'),
                        help='Test directory')
    parser.add_argument('--checkpoint_dir', type=str,
                        default=("./300617/train/"),
                        help='Directory where to read model checkpoints.')
    parser.add_argument('--eval_interval_secs', default=60 * 5,
                        help='How often to run the eval.')
    parser.add_argument('--num_examples', type=int, default=654,
                        help='Number of examples to run.')
    parser.add_argument('--run_once', action='store_false',
                        help='Whether to run eval only once.')  # default True
    # parser.add_argument('--fine_tune', action='store_true',
    #                     help='Fine tune')  # stores False by default
    parser.add_argument('--log_dir', type=str,
                        default=os.path.join(today, 'test', 'logs'),
                        help='Log directory')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(today, 'output'),
                        help='Output prediction directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
