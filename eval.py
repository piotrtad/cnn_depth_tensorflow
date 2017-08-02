"""Run for main task."""
# encoding: utf-8

import argparse
from datetime import datetime
import math
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf
import dataset
import model
import os
import sys
import time
import train_operation


def add_summary(sess, summary_writer, summary_str, tag, error, global_step):
    """Add summary."""
    with sess.as_default():
        summary = tf.Summary()
        summary.ParseFromString(summary_str)
        summary.value.add(tag=tag, simple_value=error)
        summary_writer.add_summary(summary, global_step)


def eval_once(savers, summary_writer, errors, summary_op, logits, images,
              depths, model_params):
    """Run Eval once.

    Args:
    saver: List of savers i.e. [saver_coarse, saver_refine].
    summary_writer: Summary writer.
    error: Error op.
    summary_op: Summary op.
    logits: Predictions.
    images: Input images.
    depths: Ground truth depths.
    """
    # Session
    with tf.Session(config=tf.ConfigProto(
                      log_device_placement=FLAGS.log_device_placement,
                      gpu_options=tf.GPUOptions(visible_device_list=FLAGS.gpu,
                                                allow_growth=True))) as sess:
        # init_op = tf.global_variables_initializer()
        # sess.run(init_op)
        if FLAGS.refine_test:
            saver_coarse, saver_refine = savers
        else:
            saver_coarse = savers[0]
        is_training, keep_hidden = model_params

        if FLAGS.use_snapshot:
            path = os.path.join(FLAGS.coarse_dir, 'snapshot')
        else:
            path = os.path.join(FLAGS.coarse_dir, 'checkpoints')
        coarse_ckpt = tf.train.get_checkpoint_state(path)
        if coarse_ckpt and coarse_ckpt.model_checkpoint_path:
            print("Pretrained coarse Model Loading.")
            saver_coarse.restore(sess, coarse_ckpt.model_checkpoint_path)
            global_step = coarse_ckpt.model_checkpoint_path \
                .split('/')[-1].split('-')[-1]
            print("Pretrained coarse Model Restored: %s." %
                  coarse_ckpt.model_checkpoint_path)
        else:
            print("No Pretrained coarse Model found at %s." % path)
            return

        # if FLAGS.refine_test:
        #     if FLAGS.use_snapshot:
        #         path = os.path.join(FLAGS.refine_dir, 'snapshot')
        #     else:
        #         path = os.path.join(FLAGS.refine_dir, 'checkpoints')
        #     refine_ckpt = tf.train.get_checkpoint_state(path)
        #     if refine_ckpt and refine_ckpt.model_checkpoint_path:
        #         print("Pretrained refine Model Loading.")
        #         saver_refine.restore(sess,
        #                              refine_ckpt.model_checkpoint_path)
        #         global_step = refine_ckpt.model_checkpoint_path \
        #             .split('/')[-1].split('-')[-1]
        #         print("Pretrained refine Model Restored.")
        #     else:
        #         print("No Pretrained refine Model found at %s." % path)

        try:
            global_step = int(global_step)
        except ValueError as e:
            global_step = FLAGS.global_step
        # exit(1)

        # Start the queue runners
        coord = tf.train.Coordinator()
        try:
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))

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
                # sess.run([summary_op, error, logits, images, depths],
                #          feed_dict={is_training: False, keep_hidden: 1.0})

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

            # Mean eval set error.
            eigen_error = total_eigen_error / FLAGS.num_examples
            rel_error = total_rel_error / FLAGS.num_examples
            log10_error = total_log10_error / FLAGS.num_examples
            rmse_error = total_rmse_error / FLAGS.num_examples
            accuracy1_error = total_accuracy1_error / FLAGS.num_examples
            accuracy2_error = total_accuracy2_error / FLAGS.num_examples
            accuracy3_error = total_accuracy3_error / FLAGS.num_examples

            # print("%s: %s[global step]: test error %f" % (datetime.now(),
            #                                               global_step,
            #                                               eval_error))
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

            dataset.output_predict(logits_val, images_val, depths_val,
                                   os.path.join(FLAGS.output_dir, 'eval',
                                                "predict_%s" % global_step))

        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
        sess.close()


def evaluate():
    """Evaluate."""
    with tf.Graph().as_default() as g:
        # global_step = tf.Variable(0, trainable=False)
        ds = dataset.DataSet(FLAGS.batch_size)
        images, depths, invalid_depths = ds.csv_inputs(FLAGS.test_file,
                                                       target_size=[55, 74])
        # is_training = tf.placeholder(tf.bool)
        # keep_hidden = tf.placeholder(tf.float32)
        is_training = False
        keep_hidden = 1.0
        if FLAGS.refine_test:
            print("refine train.")
            coarse = model.inference(images, is_training, keep_hidden,
                                     trainable=False)
            logits = model.inference_refine(images, coarse, keep_hidden,
                                            trainable=False)
        else:
            print("coarse train.")
            logits = model.inference(images, is_training, keep_hidden,
                                     trainable=False)
        with tf.control_dependencies([tf.assert_non_negative(invalid_depths)]):
            # loss = model.loss(logits, depths, invalid_depths)
            errors = model.errors(logits, depths)
        tf.summary.image('images', images, max_outputs=3)
        tf.summary.image('depths', depths, max_outputs=3)
        tf.summary.image('logits', logits, max_outputs=3)

        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, g)

        # coarse_params_to_restore = ['coarse_conv1/biases',
        #                             'coarse_conv1/weights'
        #                             'coarse_conv2/biases:0',
        #                             'coarse_conv2/weights:0',
        #                             'coarse_conv3/biases:0',
        #                             'coarse_conv3/weights:0',
        #                             'coarse_conv4/biases:0',
        #                             'coarse_conv4/weights:0',
        #                             'coarse_conv5/biases:0',
        #                             'coarse_conv5/weights:0',
        #                             'coarse_full1/biases:0',
        #                             'coarse_full1/weights:0',
        #                             'coarse_full2/biases:0',
        #                             'coarse_full2/weights:0',
        #                             'coarse_pool1_bn/beta:0',
        #                             'coarse_pool1_bn/gamma:0',
        #                             'coarse_pool2_bn/beta:0',
        #                             'coarse_pool2_bn/gamma:0']

#        # parameters
#        coarse_params = {}
#        refine_params = {}

#        for variable in tf.global_variables():
#            variable_name = variable.name
#            print("parameter: %s" % (variable_name))
#            if variable_name.find("/") < 0 or \
#                    variable_name.count("/") != 1:
#                continue
#             if variable_name.find('coarse') >= 0 and \
#                 variable_name in coarse_params_to_restore:
#     if variable_name.find('coarse') >= 0:
#                coarse_params[variable_name.split(':')[0]] = variable
#            if variable_name.find('fine') >= 0:
#                refine_params[variable_name.split(':')[0]] = variable

        # define saver
#       print('\n\ncoarse params:%d\n%s\n\n' % (len(coarse_params), '\n'.join(coarse_params.keys())))

#        print('\n\nglobal params:%d\n%s\n\n' % (len(tf.global_variables()), '\n'.join([var.name for var in tf.global_variables()])))
#        saver_coarse = tf.train.Saver(coarse_params)

        decay = train_operation.MOVING_AVERAGE_DECAY
        variable_averages = tf.train.ExponentialMovingAverage(decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver_coarse = tf.train.Saver(variables_to_restore)

        savers = [saver_coarse]
        if FLAGS.refine_test:
            # saver_refine = tf.train.Saver(refine_params)
            # savers += [saver_refine]
            savers += [None]
        while True:
            eval_once(savers, summary_writer, errors,
                      merged, logits, images, depths, [is_training,
                                                       keep_hidden])
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):
    """Main."""
    if not gfile.Exists(FLAGS.coarse_dir):
        gfile.MakeDirs(FLAGS.coarse_dir)
    if not gfile.Exists(FLAGS.refine_dir):
        gfile.MakeDirs(FLAGS.refine_dir)
    evaluate()


if __name__ == '__main__':
    today = datetime.strftime(datetime.now(), '%d%m%y')
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='1',
                        help='GPU to run i.e. "0", "1"')
    parser.add_argument('--num_examples', type=int, default=654,
                        help='Max steps')
    parser.add_argument('--log_device_placement', action='store_true',
                        help='Log device placement')  # stores False by default
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--test_file', type=str, default='test.csv',
                        help='Test file')
    parser.add_argument('--coarse_dir', type=str,
                        default=os.path.join(today, 'coarse'),
                        help='Coarse directory')
    parser.add_argument('--refine_dir', type=str,
                        default=os.path.join(today, 'refine'),
                        help='Refine directory')
    parser.add_argument('--refine_test', action='store_true',
                        help='Refine train')  # stores False by default
    parser.add_argument('--eval_interval_secs', default=60 * 40,
                        help='How often to run the eval.')
    parser.add_argument('--run_once', action='store_true',
                        help='Whether to run eval only once.')  # default False
    parser.add_argument('--global_step', type=int, default=0,
                        help='Batch size')
    # parser.add_argument('--fine_tune', action='store_true',
    #                     help='Fine tune')  # stores False by default
    parser.add_argument('--use_snapshot', action='store_true',
                        help='Use snapshot to tune')  # stores False by default
    parser.add_argument('--log_dir', type=str,
                        default=os.path.join(today, 'logs'),
                        help='Log directory')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(today, 'output'),
                        help='Output prediction directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
