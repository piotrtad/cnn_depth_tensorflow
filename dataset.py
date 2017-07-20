"""Routine for decoding NYUDepth inputs."""
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from PIL import Image

IMAGE_HEIGHT = 228
IMAGE_WIDTH = 304
DEPTH_HEIGHT = 426
DEPTH_WIDTH = 560
TARGET_HEIGHT = 228
TARGET_WIDTH = 304


class DataSet:
    """DataSet class."""

    def __init__(self, batch_size):
        """Init."""
        self.batch_size = batch_size

    def csv_inputs(self, csv_file_path, image_size=[IMAGE_HEIGHT, IMAGE_WIDTH],
                   target_size=[TARGET_HEIGHT, TARGET_WIDTH]):
        """Create batches from csv."""
        filename_queue = tf.train.string_input_producer([csv_file_path],
                                                        shuffle=True)
        reader = tf.TextLineReader()
        _, serialized_example = reader.read(filename_queue)
        filename, depth_filename = tf.decode_csv(serialized_example, [["img"],
                                                 ["depth"]])
        # input
        jpg = tf.read_file(filename)
        image = tf.image.decode_jpeg(jpg, channels=3)
        image = tf.cast(image, tf.float32)

        # target
        depth = tf.read_file(depth_filename)
        depth = tf.decode_raw(depth, tf.float32)
        depth = tf.reshape(depth, [DEPTH_HEIGHT, DEPTH_WIDTH, 1])

        # resize
        image = tf.image.resize_images(image, image_size)
        depth = tf.image.resize_images(depth, target_size)
        invalid_depth = tf.sign(depth)
        # generate batch
        images, depths, invalid_depths = tf.train.batch(
            [image, depth, invalid_depth],
            batch_size=self.batch_size,
            num_threads=4,
            capacity=50 + 3 * self.batch_size,
        )
        return images, depths, invalid_depths


def output_predict(depths, images, output_dir):
    """Print predictions into directory."""
    print("output predict into %s" % output_dir)
    if not gfile.Exists(output_dir):
        gfile.MakeDirs(output_dir)
    for i, (image, depth) in enumerate(zip(images, depths)):
        pilimg = Image.fromarray(np.uint8(image))
        image_name = "%s/%05d_org.png" % (output_dir, i)
        pilimg.save(image_name)
        depth = depth.transpose(2, 0, 1)
        if np.max(depth) != 0:
            ra_depth = (depth/np.max(depth))*255.0
        else:
            ra_depth = depth*255.0
        depth_pil = Image.fromarray(np.uint8(ra_depth[0]), mode="L")
        depth_name = "%s/%05d.png" % (output_dir, i)
        depth_pil.save(depth_name)
        # np.save(("%s/%05d.npy" % (output_dir, i)), depth[0])
        depth[0].tofile("%s/%05d.bin" % (output_dir, i))
