"""Test input queue feeder."""
import tensorflow as tf

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from dataset import DataSet  # noqa
import traceback  # noqa

with tf.Graph().as_default():
    dataset = DataSet(8)
    images, depths, invalid_depths = dataset.csv_inputs('train.csv',
                                                        target_size=[228, 304])
    print('csv_inputs over.')
    sess = tf.Session()
    coord = tf.train.Coordinator()
    try:
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        ims, deps = sess.run([images, depths])
        print(ims.shape)
        print(deps.shape)
    except Exception as e:
        coord.request_stop()
        traceback.print_exc()
    coord.request_stop()
    coord.join(threads)
    sess.close()
