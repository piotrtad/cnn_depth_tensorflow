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

import tensorflow as tf

from dataset import DataSet
from dataset import output_predict
import decoder_model
import train_operation

FLAGS = tf.app.flags.FLAGS


def optimistic_restore(session, save_file):
    """Restore variables by fixing name mismatch."""
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in
                        tf.global_variables() if var.name.split(':')[0] in
                        saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x: x.name.split(':')[0],
                        tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    print('restore vars:\n%s' % '\n'.join(zip(*var_names)[1]))
    saver.restore(session, save_file)


def add_summary(sess, summary_writer, summary_str, tag, error, global_step):
    """Add summary."""
    with sess.as_default():
        summary = tf.Summary()
        summary.ParseFromString(summary_str)
        summary.value.add(tag=tag, simple_value=error)
        summary_writer.add_summary(summary, global_step)


def eval_once(saver, summary_writer, errors, summary_op, logits, images,
              depths):
    """Run Eval once.

    Args:
    saver: Saver.
    summary_writer: Summary writer.
    error: Error op.
    summary_op: Summary op.
    logits: Predictions.
    images: Input RGB.
    depths: Ground truth depth.
    """
    with tf.Session(config=tf.ConfigProto(
                    log_device_placement=FLAGS.log_device_placement,
                    gpu_options=tf.GPUOptions(visible_device_list=FLAGS.gpu,
                                              allow_growth=True))) as sess:

        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # optimistic_restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path \
                .split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        try:
            global_step = int(global_step)
        except ValueError as e:
            global_step = FLAGS.global_step

        # debug
        # exit(1)

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            num_iter = int(math.ceil(FLAGS.num_examples
                                     / FLAGS.batch_size))
            total_eigen_error = 0  # Accumulates the batch mean eigen error.
            total_rel_error = 0  # Accumulates the batch mean relative error.
            total_log10_error = 0  # Accumulates the batch mean log error.
            total_rmse_error = 0  # Accumulates the batch rmse error.
            total_accuracy1_error = 0  # Acc. the batch accuracy, d < 1.25.
            total_accuracy2_error = 0  # Acc. the batch accuracy, d < 1.25^2.
            total_accuracy3_error = 0  # Acc. the batch accuracy, d < 1.25^3.

            step = 0
            while step < num_iter and not coord.should_stop():
                summary_str, mean_errors, logits_val, images_val, depths_val = \
                    sess.run([summary_op, errors, logits, images, depths])

                # # debug
                # print(mean_errors)
                # coord.request_stop()
                # exit(1)

                mean_eigen_error, mean_rel_error, mean_log10_error, rmse,\
                    accuracy1, accuracy2, accuracy3 = mean_errors

                if step == num_iter:
                    last_batch_size = FLAGS.num_examples % FLAGS.batch_size
                    total_eigen_error += mean_eigen_error * last_batch_size
                    total_rel_error += mean_rel_error * last_batch_size
                    total_log10_error += mean_log10_error * last_batch_size
                    total_rmse_error += rmse * last_batch_size
                    total_accuracy1_error += accuracy1 * last_batch_size
                    total_accuracy2_error += accuracy2 * last_batch_size
                    total_accuracy3_error += accuracy3 * last_batch_size
                else:
                    total_eigen_error += mean_eigen_error * FLAGS.batch_size
                    total_rel_error += mean_rel_error * FLAGS.batch_size
                    total_log10_error += mean_log10_error * FLAGS.batch_size
                    total_rmse_error += rmse * FLAGS.batch_size
                    total_accuracy1_error += accuracy1 * FLAGS.batch_size
                    total_accuracy2_error += accuracy2 * FLAGS.batch_size
                    total_accuracy3_error += accuracy3 * FLAGS.batch_size
                step += 1

            # Mean eval set errors.
            eigen_error = total_eigen_error / FLAGS.num_examples
            rel_error = total_rel_error / FLAGS.num_examples
            log10_error = total_log10_error / FLAGS.num_examples
            rmse_error = total_rmse_error / FLAGS.num_examples
            accuracy1_error = total_accuracy1_error / FLAGS.num_examples
            accuracy2_error = total_accuracy2_error / FLAGS.num_examples
            accuracy3_error = total_accuracy3_error / FLAGS.num_examples

            stats = """%s: %s[global step]:
eigen error %f
relative error %f
log10 error %f
rmse error %f
accuracy (1.25) %f
accuracy (1.25^2) %f
accuracy (1.25^3) %f""" % (datetime.now(),
                           global_step,
                           eigen_error,
                           rel_error,
                           log10_error,
                           rmse_error,
                           accuracy1_error,
                           accuracy2_error,
                           accuracy3_error)
            print(stats)

            # Record summaries
            add_summary(sess, summary_writer, summary_str, 'eval eigen error',
                        eigen_error, global_step)
            add_summary(sess, summary_writer, summary_str, 'eval rel error',
                        rel_error, global_step)
            add_summary(sess, summary_writer, summary_str, 'eval log10 error',
                        log10_error, global_step)
            add_summary(sess, summary_writer, summary_str, 'eval rmse error',
                        rmse_error, global_step)
            add_summary(sess, summary_writer, summary_str,
                        'eval accuracy (delta < 1.25)',
                        accuracy1_error, global_step)
            add_summary(sess, summary_writer, summary_str,
                        'eval eval accuracy (delta < 1.25^2)',
                        accuracy2_error, global_step)
            add_summary(sess, summary_writer, summary_str,
                        'eval eval accuracy (delta < 1.25^3)',
                        accuracy3_error, global_step)

            output_predict(logits_val, images_val, depths_val,
                           os.path.join(FLAGS.output_dir, "predict_%s" %
                                        global_step))
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """Eval the model."""
    with tf.Graph().as_default() as g:
        # Get images and depths.
        dataset = DataSet(FLAGS.batch_size)
        images, depths, invalid_depths = dataset.csv_inputs(FLAGS.test_file,
                                                            target_size=[228,
                                                                         304])

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = decoder_model.model(images, is_training=False, keep_drop=1.0,
                                     trainable=False)

        # Calculate eval metrics.
        errors = decoder_model.errors(logits, depths)

        # Restore the moving average version of the learned variables for eval.
        decay = train_operation.MOVING_AVERAGE_DECAY
        variable_averages = tf.train.ExponentialMovingAverage(decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, g)

        while True:
            eval_once(saver, summary_writer, errors, summary_op, logits,
                      images, depths)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
    """Main."""
    if not tf.gfile.Exists(FLAGS.test_dir):
        tf.gfile.MakeDirs(FLAGS.test_dir)
    evaluate()


if __name__ == '__main__':
    today = datetime.strftime(datetime.now(), '%d%m%y')
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='1',
                        help='GPU to run i.e. "0", "1"')
    parser.add_argument('--log_device_placement', action='store_true',
                        help='Log device placement')  # default False
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Batch size')
    parser.add_argument('--test_file', type=str, default='test.csv',
                        help='Test file')
    parser.add_argument('--test_dir', type=str,
                        default=os.path.join(today, 'test'),
                        help='Test directory')
    parser.add_argument('--checkpoint_dir', type=str,
                        default=("./240717/trainsnapshot/"),
                        help='Directory where to read model checkpoints.')
    parser.add_argument('--eval_interval_secs', default=60 * 40,
                        help='How often to run the eval.')
    parser.add_argument('--num_examples', type=int, default=654,
                        help='Number of examples to run.')
    parser.add_argument('--run_once', action='store_true',
                        help='Whether to run eval only once.')  # default False
    parser.add_argument('--global_step', type=int, default=0,
                        help='Batch size')
    parser.add_argument('--log_dir', type=str,
                        default=os.path.join(today, 'logs', 'test'),
                        help='Log directory')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(today, 'test', 'output'),
                        help='Output prediction directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
