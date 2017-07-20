"""Convert NYUv2 dataset from Matlab to usable input.

images: mat to jpg
depths: mat to npy
"""
# encoding: utf-8
import argparse
import os
import numpy as np
import h5py
from PIL import Image
import random
import sys
import tensorflow as tf

FLAGS = None


def main(unused_argv):
    """Main."""
    print("load dataset: %s" % (FLAGS.nyu_path))
    f = h5py.File(FLAGS.nyu_path)

    data_dir = os.path.join('data', 'nyu_datasets')

    if not tf.gfile.Exists(data_dir):
        tf.gfile.MakeDirs(data_dir)

    trains = []
    for i, (image, depth) in enumerate(zip(f['images'], f['depths'])):
        image = image.transpose(2, 1, 0)
        depth = depth.transpose(1, 0)
        # crop border
        image = image[45:471, 41:601, :]
        depth = depth[45:471, 41:601]
        image_pil = Image.fromarray(np.uint8(image))
        image_name = os.path.join(data_dir, "%05d.jpg" % (i))
        image_pil.save(image_name)
        depth_name = os.path.join(data_dir, "%05d.bin" % (i))
        # np.save(depth_name, depth)
        depth.tofile(depth_name)

        trains.append((image_name, depth_name))

        if i % 100 == 0:
            print('processed %d/%d...' % (i, len(f['images'])))

    print('processed %d/%d...' % (len(f['images']), len(f['images'])))
    random.shuffle(trains)

    with open('train.csv', 'w') as output:
        for (image_name, depth_name) in trains[:FLAGS.train_set_size]:
            output.write("%s,%s" % (image_name, depth_name))
            output.write("\n")

    with open('test.csv', 'w') as output:
        for (image_name, depth_name) in trains[FLAGS.train_set_size:]:
            output.write('%s,%s' % (image_name, depth_name))
            output.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nyu_path', type=str,
                        default='data/nyu_depth_v2_labeled.mat',
                        help='Path containing labelled NYUv2 dataset')
    parser.add_argument('--train_set_size', type=int,
                        default=795, help='Train set size')
    parser.add_argument('--test_set_size', type=int,
                        default=654, help='Test set size')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
