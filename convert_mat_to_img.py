"""Convert NYUv2 dataset to PNG images."""
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
        ra_image = image.transpose(2, 1, 0)
        ra_depth = depth.transpose(1, 0)
        re_depth = (ra_depth/np.max(ra_depth))*255.0
        image_pil = Image.fromarray(np.uint8(ra_image))
        depth_pil = Image.fromarray(np.uint8(re_depth))
        image_name = os.path.join(data_dir, "%05d.jpg" % (i))
        image_pil.save(image_name)
        depth_name = os.path.join(data_dir, "%05d.png" % (i))
        depth_pil.save(depth_name)

        trains.append((image_name, depth_name))

    random.shuffle(trains)

    with open('train.csv', 'w') as output:
        for (image_name, depth_name) in trains:
            output.write("%s,%s" % (image_name, depth_name))
            output.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nyu_path', type=str,
                        default='data/nyu_depth_v2_labeled.mat',
                        help='Path containing labelled NYUv2 dataset')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
