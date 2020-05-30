


import imageio
import itertools
import json
import math
import numpy as np
import os
from pathlib import Path
import pdb
import tensorflow.compat.v1 as tf
from typing import Any, Dict, Union

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

"""
https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/dataset.proto

See paper:
https://arxiv.org/pdf/1912.04838.pdf
"""

def check_mkdir(dirpath):
	""" """
	if not Path(dirpath).exists():
		os.makedirs(dirpath, exist_ok=True)

def save_json_dict(json_fpath: Union[str, "os.PathLike[str]"], dictionary: Dict[Any, Any]) -> None:
	"""Save a Python dictionary to a JSON file.
	Args:
	json_fpath: Path to file to create.
	dictionary: Python dictionary to be serialized.
	"""
	with open(json_fpath, "w") as f:
		json.dump(dictionary, f)

def main():
	""" """
	TFRECORD_DIR = '/export/share/Datasets/MSegV12/w_o_d/VAL_TFRECORDS'
	tfrecord_name = 'segment-967082162553397800_5102_900_5122_900_with_camera_labels.tfrecord'

	fpath = f'{TFRECORD_DIR}/{tfrecord_name}'

	CAMERA_NAMES = [
	    'UNKNOWN', # 0
	    'FRONT', # 1
	    'FRONT_LEFT', # 2
	    'FRONT_RIGHT', # 3
	    'SIDE_LEFT', # 4
	    'SIDE_RIGHT', # 5
	]

	img_count = 0

	dataset = tf.data.TFRecordDataset(fpath, compression_type='')
	for data in dataset:
		frame = open_dataset.Frame()
		frame.ParseFromString(bytearray(data.numpy()))
		log_id = '967082162553397800_5102_900_5122_900'
		# 5 images per frame
		for index, tf_cam_image in enumerate(frame.images):

			# 4x4 row major transform matrix that tranforms 
			# 3d points from one frame to another.
			SE3_flattened = np.array(tf_cam_image.pose.transform)
			city_SE3_egovehicle = SE3_flattened.reshape(4,4)

			# microseconds
			timestamp_ms =  tf_cam_image.pose_timestamp
			timestamp_ns = timestamp_ms * 1000 # to nanoseconds
			# tf_cam_image.shutter
			# tf_cam_image.camera_trigger_time
			# tf_cam_image.camera_readout_done_time

			camera_name = CAMERA_NAMES[tf_cam_image.name]
			img = tf.image.decode_jpeg(tf_cam_image.image)
			
			img_save_fpath = f'logs/{log_id}/{camera_name}/{camera_name}_{timestamp_ns}.jpg'
			assert not Path(img_save_fpath).exists()
			check_mkdir(str(Path(img_save_fpath).parent))
			imageio.imwrite(img_save_fpath, img)
			img_count += 1
			if img_count % 1000 == 0:
				print(f"Saved {img_count}'th image for log = {log_id}")

			# pose_save_fpath = f'logs/{log_id}/poses/city_SE3_egovehicle_{timestamp_ns}.json'
			# assert not Path(pose_save_fpath).exists()
			# save_json_dict(pose_save_fpath)


if __name__ == '__main__':
	main()



