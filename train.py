"""Run for main task."""
# encoding: utf-8

import argparse
from datetime import datetime
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf
import dataset
import model
import os
import sys
import train_operation as op


def train():
    """Train."""
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        ds = dataset.DataSet(FLAGS.batch_size)
        images, depths, invalid_depths = ds.csv_inputs(FLAGS.train_file,
                                                       target_size=[55, 74])
        is_training = tf.placeholder(tf.bool)
        keep_conv = tf.placeholder(tf.float32)
        keep_hidden = tf.placeholder(tf.float32)
        if FLAGS.refine_train:
            print("refine train.")
            coarse = model.inference(images, is_training, keep_conv,
                                     trainable=False, debug=FLAGS.debug)
            logits = model.inference_refine(images, coarse, keep_conv,
                                            keep_hidden, debug=FLAGS.debug)
        else:
            print("coarse train.")
            logits = model.inference(images, is_training, keep_conv,
                                     keep_hidden, debug=FLAGS.debug)
        with tf.control_dependencies([tf.assert_non_negative(invalid_depths)]):
            loss = model.loss(logits, depths, invalid_depths)
        tf.summary.image('images', images, max_outputs=3)
        tf.summary.image('depths', depths, max_outputs=3)
        tf.summary.image('logits', logits, max_outputs=3)
        train_op = op.train(loss, global_step, FLAGS.batch_size)
        init_op = tf.global_variables_initializer()

        # Session
        sess = tf.Session(config=tf.ConfigProto(
                          log_device_placement=FLAGS.log_device_placement,
                          gpu_options=tf.GPUOptions(visible_device_list=
                                                    FLAGS.gpu,
                                                    allow_growth=True)))
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        sess.run(init_op)

        # coarse_params_to_restore = ['coarse_conv1/biases:0',
        #                             'coarse_conv1/weights:0'
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

        # parameters
        coarse_params = {}
        refine_params = {}
        if FLAGS.refine_train:
            for variable in tf.global_variables():
                variable_name = variable.name
                print("parameter: %s" % (variable_name))
                if variable_name.find("/") < 0 or \
                        variable_name.count("/") != 1:
                    continue
                # if variable_name.find('coarse') >= 0 and \
                #         variable_name in coarse_params_to_restore:
                if variable_name.find('coarse') >= 0:
                    coarse_params[variable_name.split(':')[0]] = variable
                if variable_name.find('fine') >= 0:
                    refine_params[variable_name.split(':')[0]] = variable
        else:
            for variable in tf.trainable_variables():  # change to global_...()
                variable_name = variable.name
                print("parameter: %s" % (variable_name))
                if variable_name.find("/") < 0 \
                        or variable_name.count("/") != 1:
                    continue
                if variable_name.find('coarse') >= 0:
                    coarse_params[variable_name.split(':')[0]] = variable
                if variable_name.find('fine') >= 0:
                    refine_params[variable_name.split(':')[0]] = variable
        # define saver
        # print('\n\ncoarse params:%s\n\n' % coarse_params)
        # print('\n\nrefine params:%s\n\n' % refine_params)
        saver_coarse = tf.train.Saver(tf.global_variables())
        restorer_coarse = tf.train.Saver(coarse_params)
        # if FLAGS.refine_train:
        #     saver_refine = tf.train.Saver(refine_params)
        # fine tune
        if FLAGS.fine_tune:
            if FLAGS.use_snapshot:
                path = os.path.join(FLAGS.coarse_checkpoint_dir, 'snapshot')
            else:
                path = os.path.join(FLAGS.coarse_checkpoint_dir, 'checkpoints')
            coarse_ckpt = tf.train.get_checkpoint_state(path)
            if coarse_ckpt and coarse_ckpt.model_checkpoint_path:
                print("Pretrained coarse Model Loading.")
                restorer_coarse.restore(sess,
                                        coarse_ckpt.model_checkpoint_path)
                print("Pretrained coarse Model Restored.")
            else:
                print("No Pretrained coarse Model.")
            # if FLAGS.refine_train:
            #     if FLAGS.use_snapshot:
            #       path = os.path.join(FLAGS.refine_dir, 'snapshot')
            #       path = os.path.join(FLAGS.coarse_checkpoint_dir, 'snapshot')
            #    else:
            #       path = os.path.join(FLAGS.refine_dir, 'checkpoints')
            #       path = os.path.join(FLAGS.coarse_checkpoint_dir, 'checkpoints')
            #    refine_ckpt = tf.train.get_checkpoint_state(path)
            #    if refine_ckpt and refine_ckpt.model_checkpoint_path:
            #        print("Pretrained refine Model Loading.")
            #        saver_refine.restore(sess,
            #                             refine_ckpt.model_checkpoint_path)
            #        print("Pretrained refine Model Restored.")
            #    else:
            #        print("No Pretrained refine Model.")

        # print('\n'.join(sorted([var.name for var in tf.global_variables()])))
        # exit(1)

        if FLAGS.refine_train:
            bn_is_training = False
        else:
            bn_is_training = True

        # train
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for step in xrange(FLAGS.max_steps):
            index = 0
            for i in xrange(1000):
                summary, _, loss_value, logits_val, images_val, depths_val = \
                    sess.run([merged, train_op, loss, logits, images, depths],
                             feed_dict={is_training: bn_is_training, keep_conv: 0.8,
                             keep_hidden: 0.5})
                if index % 10 == 0:
                    print("%s: %d[epoch]: %d[iteration]: train loss %f" % (
                        datetime.now(), step, index, loss_value))
                    summary_writer.add_summary(summary, 1000*step+i)
                    assert not np.isnan(
                        loss_value), 'Model diverged with loss = NaN'
                if index % 250 == 0:
                    if FLAGS.refine_train:
                        path = os.path.join(FLAGS.output_dir,
                                            "predict_refine_%05d_%03d" %
                                            (step, i))
                        dataset.output_predict(logits_val, images_val,
                                               depths_val, path)
                    else:
                        path = os.path.join(FLAGS.output_dir,
                                            "predict_%05d_%03d" %
                                            (step, i))
                        dataset.output_predict(logits_val, images_val,
                                               depths_val, path)
                index += 1

            if step % 5 == 0 or (step * 1) == FLAGS.max_steps:
                if FLAGS.refine_train:
                    refine_checkpoint_path = os.path.join(FLAGS.refine_dir,
                                                          'checkpoints',
                                                          'model.ckpt')
                    # saver_refine.save(sess, refine_checkpoint_path,
                    #                   global_step=step)
                    saver_coarse.save(sess, refine_checkpoint_path,
                                      global_step=step)
                else:
                    coarse_checkpoint_path = os.path.join(FLAGS.coarse_dir,
                                                          'checkpoints',
                                                          'model.ckpt')
                    saver_coarse.save(sess, coarse_checkpoint_path,
                                      global_step=step)
            if FLAGS.refine_train:
                refine_checkpoint_path = os.path.join(FLAGS.refine_dir,
                                                      'snapshot',
                                                      'model.ckpt')
                # saver_refine.save(sess, refine_checkpoint_path,
                #                   global_step=1000*step+i)
                saver_coarse.save(sess, refine_checkpoint_path,
                                  global_step=step)
            else:
                coarse_checkpoint_path = os.path.join(FLAGS.coarse_dir,
                                                      'snapshot',
                                                      'model.ckpt')
                saver_coarse.save(sess, coarse_checkpoint_path,
                                  global_step=1000*step+i)
        coord.request_stop()
        coord.join(threads)
        sess.close()


def main(argv=None):
    """Main."""
    if not gfile.Exists(FLAGS.coarse_dir):
        gfile.MakeDirs(FLAGS.coarse_dir)
    if not gfile.Exists(FLAGS.refine_dir):
        gfile.MakeDirs(FLAGS.refine_dir)
    train()


if __name__ == '__main__':
    today = datetime.strftime(datetime.now(), '%d%m%y')
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='1',
                        help='GPU to run i.e. "0", "1"')
    parser.add_argument('--max_steps', type=int, default=10000000,
                        help='Max steps')
    parser.add_argument('--log_device_placement', action='store_true',
                        help='Log device placement')  # stores False by default
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Batch size')
    parser.add_argument('--train_file', type=str, default='train.csv',
                        help='Train file')
    parser.add_argument('--coarse_dir', type=str,
                        default=os.path.join(today, 'coarse'),
                        help='Coarse directory')
    parser.add_argument('--coarse_checkpoint_dir', type=str,
                        default=os.path.join(today, 'coarse'),
                        help='Coarse checkpoint directory')
    parser.add_argument('--refine_dir', type=str,
                        default=os.path.join(today, 'refine'),
                        help='Refine directory')
    parser.add_argument('--refine_train', action='store_true',
                        help='Refine train')  # stores False by default
    parser.add_argument('--fine_tune', action='store_true',
                        help='Fine tune')  # stores False by default
    parser.add_argument('--use_snapshot', action='store_true',
                        help='Use snapshot to tune')  # stores False by default
    parser.add_argument('--log_dir', type=str,
                        default=os.path.join(today, 'logs'),
                        help='Log directory')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(today, 'output'),
                        help='Output prediction directory')
    parser.add_argument('--debug', action='store_true',
                        help='Debug network')  # stores False by default
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
