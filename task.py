"""Run for main task."""
# encoding: utf-8

import argparse
from datetime import datetime
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf
from dataset import DataSet
from dataset import output_predict
import model
import os
import sys
import train_operation as op


def train():
    """Train."""
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        dataset = DataSet(FLAGS.batch_size)
        images, depths, invalid_depths = dataset.csv_inputs(FLAGS.train_file)
        keep_conv = tf.placeholder(tf.float32)
        keep_hidden = tf.placeholder(tf.float32)
        if FLAGS.refine_train:
            print("refine train.")
            coarse = model.inference(images, keep_conv, trainable=False)
            logits = model.inference_refine(images, coarse, keep_conv,
                                            keep_hidden)
        else:
            print("coarse train.")
            logits = model.inference(images, keep_conv, keep_hidden)
        loss = model.loss(logits, depths, invalid_depths)
        train_op = op.train(loss, global_step, FLAGS.batch_size)
        init_op = tf.initialize_all_variables()

        # Session
        # sess = tf.Session(config=tf.ConfigProto(
        #                   log_device_placement=FLAGS.log_device_placement,
        #                   device_count={'GPU': 1}))
        sess = tf.Session(config=tf.ConfigProto(
                          log_device_placement=FLAGS.log_device_placement,
                          gpu_options=tf.GPUOptions(visible_device_list='1')
                          ))
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        sess.run(init_op)

        # parameters
        coarse_params = {}
        refine_params = {}
        if FLAGS.refine_train:
            for variable in tf.all_variables():
                variable_name = variable.name
                print("parameter: %s" % (variable_name))
                if variable_name.find("/") < 0 or variable_name.count("/") != 1:
                    continue
                if variable_name.find('coarse') >= 0:
                    coarse_params[variable_name] = variable
                print("parameter: %s" % (variable_name))
                if variable_name.find('fine') >= 0:
                    refine_params[variable_name] = variable
        else:
            for variable in tf.trainable_variables():
                variable_name = variable.name
                print("parameter: %s" % (variable_name))
                if variable_name.find("/") < 0 or variable_name.count("/") != 1:
                    continue
                if variable_name.find('coarse') >= 0:
                    coarse_params[variable_name] = variable
                if variable_name.find('fine') >= 0:
                    refine_params[variable_name] = variable
        # define saver
        print coarse_params
        saver_coarse = tf.train.Saver(coarse_params)
        if FLAGS.refine_train:
            saver_refine = tf.train.Saver(refine_params)
        # fine tune
        if FLAGS.fine_tune:
            coarse_ckpt = tf.train.get_checkpoint_state(FLAGS.coarse_dir)
            if coarse_ckpt and coarse_ckpt.model_checkpoint_path:
                print("Pretrained coarse Model Loading.")
                saver_coarse.restore(sess, coarse_ckpt.model_checkpoint_path)
                print("Pretrained coarse Model Restored.")
            else:
                print("No Pretrained coarse Model.")
            if FLAGS.refine_train:
                refine_ckpt = tf.train.get_checkpoint_state(FLAGS.refine_dir)
                if refine_ckpt and refine_ckpt.model_checkpoint_path:
                    print("Pretrained refine Model Loading.")
                    saver_refine.restore(sess,
                                         refine_ckpt.model_checkpoint_path)
                    print("Pretrained refine Model Restored.")
                else:
                    print("No Pretrained refine Model.")

        # train
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for step in xrange(FLAGS.max_steps):
            index = 0
            for i in xrange(1000):
                summary, _, loss_value, logits_val, images_val = sess.run(
                        [merged, train_op, loss, logits, images],
                        feed_dict={keep_conv: 0.8, keep_hidden: 0.5})
                if index % 10 == 0:
                    print("%s: %d[epoch]: %d[iteration]: train loss %f" % (
                        datetime.now(), step, index, loss_value))
                    summary_writer.add_summary(summary, 1000*step+i)
                    assert not np.isnan(
                        loss_value), 'Model diverged with loss = NaN'
                if index % 500 == 0:
                    if FLAGS.refine_train:
                        output_predict(logits_val, images_val,
                                       os.path.join(FLAGS.output_dir,
                                                    "predict_refine_%05d_%05d" %
                                                    (step, i)))
                    else:
                        output_predict(logits_val, images_val,
                                       os.path.join(FLAGS.output_dir,
                                                    "predict_%05d_%05d" %
                                                    (step, i)))
                index += 1

            if step % 5 == 0 or (step * 1) == FLAGS.max_steps:
                if FLAGS.refine_train:
                    refine_checkpoint_path = FLAGS.refine_dir + '/model.ckpt'
                    saver_refine.save(sess, refine_checkpoint_path,
                                      global_step=step)
                else:
                    coarse_checkpoint_path = FLAGS.coarse_dir + '/model.ckpt'
                    saver_coarse.save(sess, coarse_checkpoint_path,
                                      global_step=step)
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
    parser.add_argument('--gpu', type=str, default='/gpu:0',
                        help='GPU to run')
    parser.add_argument('--max_steps', type=int, default=10000000,
                        help='Max steps')
    parser.add_argument('--log_device_placement', action='store_false',
                        help='Log device placement')  # stores True by default
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--train_file', type=str, default='train.csv',
                        help='Train file')
    parser.add_argument('--coarse_dir', type=str,
                        default=os.path.join(today, 'coarse'),
                        help='Coarse directory')
    parser.add_argument('--refine_dir', type=str,
                        default=os.path.join(today, 'refine'),
                        help='Refine directory')
    parser.add_argument('--refine_train', action='store_true',
                        help='Refine train')  # stores False by default
    parser.add_argument('--fine_tune', action='store_true',
                        help='Fine tune')  # stores False by default
    parser.add_argument('--log_dir', type=str,
                        default=os.path.join(today, 'logs'),
                        help='Log directory')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(today, 'output'),
                        help='Output prediction directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
