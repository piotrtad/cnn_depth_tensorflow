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
        images, depths, _ = dataset.csv_inputs(FLAGS.train_file,
                                               target_size=[228, 304])
        keep_drop = tf.placeholder(tf.float32)
        is_training = tf.placeholder(tf.bool)
        logits = decoder_model.model(images, is_training, keep_drop,
                                     debug=FLAGS.debug)
        loss = decoder_model.scale_invariant_loss(logits, depths)
        train_op = op.train(loss, global_step, FLAGS.batch_size)
        init_op = tf.global_variables_initializer()
        tf.summary.image('images', images, max_outputs=3)
        tf.summary.image('depths', depths, max_outputs=3)
        tf.summary.image('logits', logits, max_outputs=3)

        # Session
        sess = tf.Session(config=tf.ConfigProto(
                          log_device_placement=FLAGS.log_device_placement,
                          gpu_options=tf.GPUOptions(visible_device_list=
                                                    FLAGS.gpu,
                                                    allow_growth=True)))
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        sess.run(init_op)
        
        params_to_restore_once = [  'conv1_bn/beta:0',
                                    'conv1_bn/gamma:0',
                                    'conv1/biases:0',
                                    'conv1/weights:0',
                                    'conv2_bn/beta:0',
                                    'conv2/biases:0',
                                    'conv2/weights:0',
                                    'conv3_bn/beta:0',
                                    'conv3_bn/gamma:0',
                                    'conv3/biases:0',
                                    'conv3/weights:0',
                                    'conv4_bn/beta:0',
                                    'conv4_bn/gamma:0',
                                    'conv4/biases:0',
                                    'conv4/weights:0',
                                    'conv5_bn/beta:0',
                                    'conv5_bn/gamma:0',
                                    'conv5/biases:0',
                                    'conv5/weights:0',
                                    'fc6_bn/beta:0',
                                    'fc6_bn/gamma:0',
                                    'fc6/biases:0',
                                    'fc6/weights:0',
                                    'fc7_bn/beta:0',
                                    'fc7_bn/gamma:0',
                                    'fc7/biases:0',
                                    'fc7/weights:0',
                                    'up1_bn/beta:0',
                                    'up1_bn/gamma:0',
                                    'up1/biases:0',
                                    'up1/weights:0',
                                    'up2_bn/beta:0',
                                    'up2_bn/gamma:0',
                                    'up2/biases:0',
                                    'up2/weights:0',
                                    'up3_bn/beta:0',
                                    'up3_bn/gamma:0',
                                    'up3/biases:0',
                                    'up3/weights:0',
                                    'up4_bn/beta:0',
                                    'up4_bn/gamma:0',
                                    'up4/biases:0',
                                    'up4/weights:0',
                                    'up5_bn/beta:0',
                                    'up5_bn/gamma:0',
                                    'up5/biases:0',
                                    'up5/weights:0',
                                    'up6/biases:0',
                                    'up6/weights:0']

        # parameters
        params = {}
        for variable in tf.global_variables():
            variable_name = variable.name
            if variable_name not in params_to_restore_once or variable_name.find("/") < 0 or variable_name.count("/") != 1:
                continue
            params[variable_name] = variable

        # with open('trainable_params_1.txt', 'w') as param_file:
        #     param_file.write('\n'.join(params.keys()))
        
        print('params to restore:\n%s' % '\n'.join(params.keys()))

        # define saver
        restorer = tf.train.Saver(params)
        
        saver = tf.train.Saver(tf.global_variables())

        # fine tune
        if FLAGS.fine_tune:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print("Loading pretrained model...")
                restorer.restore(sess, ckpt.model_checkpoint_path)
                print("Pretrained model restored.")
            else:
                print("No pretrained model.")

        # train
        coord = tf.train.Coordinator()
        try:
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for step in xrange(FLAGS.max_steps):
                index = 0
                for i in xrange(1000):
                    summary, _, loss_value, logits_val, images_val, depths_val\
                        = sess.run([merged, train_op, loss, logits, images,
                                    depths], feed_dict={keep_drop: 0.5,
                                                        is_training: True})

                    if index % 10 == 0:
                        print("%s: %d[epoch]: %d[iteration]: train loss %f" % (
                            datetime.now(), step, index, loss_value))
                        summary_writer.add_summary(summary, 1000*step+i)
                        assert not np.isnan(
                            loss_value), 'Model diverged with loss = NaN'
                    if index % 250 == 0:
                        output_predict(logits_val, images_val, depths_val,
                                       os.path.join(FLAGS.output_dir,
                                                    "predict_%05d_%05d" %
                                                    (step, i)))
                        # Save snapshot after each epoch/step
                        checkpoint_path = FLAGS.train_dir + '/snapshot/model.ckpt'
                        saver.save(sess, checkpoint_path)
                    index += 1

                if step % 5 == 0 or (step * 1) == FLAGS.max_steps:
                    checkpoint_path = FLAGS.train_dir + '/ckpts/model.ckpt'
                    saver.save(sess, checkpoint_path, global_step=step)

                # Save snapshot after each epoch/step
                checkpoint_path = FLAGS.train_dir + '/snapshot/model.ckpt'
                saver.save(sess, checkpoint_path)
        except Exception as e:
            coord.request_stop(e)

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
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU to run i.e. "0", "1"')
    parser.add_argument('--max_steps', type=int, default=10000000,
                        help='Max steps')
    parser.add_argument('--log_device_placement', action='store_true',
                        help='Log device placement')  # stores False by default
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Batch size')
    parser.add_argument('--train_file', type=str, default='train.csv',
                        help='Train file')
    parser.add_argument('--checkpoint_dir', type=str, default='',
                        help='Checkpoint directory')
    parser.add_argument('--train_dir', type=str,
                        default=os.path.join(today, 'train'),
                        help='Train directory')
    parser.add_argument('--fine_tune', action='store_true',
                        help='Fine tune')  # stores False by default
    parser.add_argument('--log_dir', type=str,
                        default=os.path.join(today, 'logs', 'train'),
                        help='Log directory')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(today, 'train', 'output'),
                        help='Output prediction directory')
    parser.add_argument('--debug', action='store_true',
                        help='Debug network')  # stores False by default
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
