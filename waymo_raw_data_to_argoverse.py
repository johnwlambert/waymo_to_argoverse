


import imageio
import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset



def main():
	""" """
	TFRECORD_DIR = '/export/share/Datasets/MSegV12/w_o_d/VAL_TFRECORDS'
	tfrecord_name = 'segment-967082162553397800_5102_900_5122_900_with_camera_labels.tfrecord'

	fpath = f'{TFRECORD_DIR}/{tfrecord_name}'

	dataset = tf.data.TFRecordDataset(fpath, compression_type='')
	for data in dataset:
	    frame = open_dataset.Frame()
	    frame.ParseFromString(bytearray(data.numpy()))

		for index, tf_cam_image in enumerate(frame.images):
			img = tf.image.decode_jpeg(tf_cam_image.image)
			print(img)
			print(img.dtype)
			print(img.shape)
			imageio.imwrite('sample.jpg', img)
			

if __name__ == '__main__':
	main()