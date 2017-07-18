"""Main task for Eigen model with resize convolution decoder."""
# encoding: utf-8

import argparse
from datetime import datetime
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf
from dataset import DataSet
from dataset import output_predict
import decoder_model
import os
import sys
import train_operation as op


def train():
    """Train."""
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        dataset = DataSet(FLAGS.batch_size)
        images, depths, invalid_depths = dataset.csv_inputs(FLAGS.train_file,
                                                            target_size=[228, 304])
        keep_conv = tf.placeholder(tf.float32)
        keep_hidden = tf.placeholder(tf.float32)
        logits = decoder_model.model(images, keep_conv, keep_hidden)
        loss = decoder_model.loss(logits, depths, invalid_depths)
        train_op = op.train(loss, global_step, FLAGS.batch_size)
        init_op = tf.initialize_all_variables()

        # Session
        # sess = tf.Session(config=tf.ConfigProto(
        #                   log_device_placement=FLAGS.log_device_placement,
        #                   device_count={'GPU': 1}))
        sess = tf.Session(config=tf.ConfigProto(
                          log_device_placement=FLAGS.log_device_placement,
                          gpu_options=tf.GPUOptions(visible_device_list='1',
						    allow_growth=True)
                          ))
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        sess.run(init_op)

        # parameters
        params = {}
        for variable in tf.trainable_variables():
            variable_name = variable.name
            print("parameter: %s" % (variable_name))
            if variable_name.find("/") < 0 or variable_name.count("/") != 1:
                continue
            params[variable_name] = variable
        # define saver
        print params
        saver = tf.train.Saver(params)
        # fine tune
        if FLAGS.fine_tune:
            ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print("Loading pretrained model...")
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Pretrained model restored.")
            else:
                print("No pretrained model.")

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
                    output_predict(logits_val, images_val,
                                       os.path.join(FLAGS.output_dir,
                                                    "predict_%05d_%05d" %
                                                    (step, i)))
                index += 1

            if step % 5 == 0 or (step * 1) == FLAGS.max_steps:
                checkpoint_path = FLAGS.train_dir + '/model.ckpt'
                saver.save(sess, checkpoint_path,
                                      global_step=step)
        coord.request_stop()
        coord.join(threads)
        sess.close()


def main(argv=None):
    """Main."""
    if not gfile.Exists(FLAGS.train_dir):
        gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    today = datetime.strftime(datetime.now(), '%d%m%y')
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=10000000,
                        help='Max steps')
    parser.add_argument('--log_device_placement', action='store_true',
                        help='Log device placement')  # stores True by default
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--train_file', type=str, default='train.csv',
                        help='Train file')
    parser.add_argument('--train_dir', type=str,
                        default=os.path.join(today, 'train'),
                        help='Train directory')
    parser.add_argument('--fine_tune', action='store_true',
                        help='Fine tune')  # stores False by default
    parser.add_argument('--log_dir', type=str,
                        default=os.path.join(today, 'train', 'logs'),
                        help='Log directory')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(today, 'output'),
                        help='Output prediction directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
