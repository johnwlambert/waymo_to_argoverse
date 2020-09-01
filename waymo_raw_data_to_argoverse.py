#!/usr/bin/env python3

import glob
import imageio
import itertools
import json
import math
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pdb
from typing import Any, Dict, Union
import uuid

from pyntcloud import PyntCloud
from scipy.spatial.transform import Rotation
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

from argoverse.utils.se3 import SE3
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

from waymo_data_splits import get_val_log_ids, get_test_log_ids
from transform_utils import yaw_to_quaternion3d

"""
Extract poses, images, and camera calibration from raw Waymo Open Dataset TFRecords.

See the Frame structure here:
https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/dataset.proto

See paper:
https://arxiv.org/pdf/1912.04838.pdf
"""

from transform_utils import (
	rotmat2quat,
	quat2rotmat
)


CAMERA_NAMES = [
    'unknown', # 0, 'UNKNOWN',
    'ring_front_center', # 1, 'FRONT'
    'ring_front_left', # 2, 'FRONT_LEFT',
    'ring_front_right', # 3, 'FRONT_RIGHT',
    'ring_side_left', # 4, 'SIDE_LEFT',
    'ring_side_right', # 5, 'SIDE_RIGHT'
]

LABEL_TYPES = [
	"OTHER_MOVER", #TYPE_UNKNOWN = 0
    "VEHICLE", # TYPE_VEHICLE = 1;
    "PEDESTRIAN", # TYPE_PEDESTRIAN = 2;
    "SIGN", # TYPE_SIGN = 3;
    "BICYCLIST", # TYPE_CYCLIST = 4;
]

RING_IMAGE_SIZES = {
	# width x height
	'ring_front_center': (1920, 1280),
	'ring_front_left':  (1920, 1280),
	'ring_side_left': (1920, 886),
	'ring_side_right': (1920,886)
}

track_id_dict = {}

def round_to_micros(t_nanos, base: int = 1000):
    """
    Round nanosecond timestamp to nearest microsecond timestamp
    """
    return base * round(t_nanos/base)


def test_round_to_micros():
    """
    test_round_to_micros()
    """
    t_nanos  = 1508103378165379072
    t_micros = 1508103378165379000

    assert t_micros == round_to_micros(t_nanos, base=1000)


def check_mkdir(dirpath: str):
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

def get_log_id_from_files(files, record_dir):
	log_ids = []
	for file in files:
		file = file.strip(record_dir)
		file = file.strip("segment-")
		file = file.strip("with_camera_labels.tfrecord")
		log_ids.append(file)
	return log_ids

def main(save_images: bool, save_poses: bool, save_calibration: bool, save_cloud: bool, save_labels:bool):
	""" """
	# TFRECORD_DIR = 'VAL_TFRECORDS'
	TFRECORD_DIR = '/srv/share3/hchittanuru3/waymo_training'
	ARGO_WRITE_DIR = '/srv/share3/hchittanuru3/testing_waymo2argo'

	val_log_ids = get_val_log_ids()
	test_log_ids = get_test_log_ids()

	img_count = 0
	files = glob.glob(f'{TFRECORD_DIR}/*.tfrecord')
	log_ids = get_log_id_from_files(files, TFRECORD_DIR)[:1]
	for log_id in log_ids:
		print(log_id)
		tfrecord_name = f'segment-{log_id}_with_camera_labels.tfrecord'
		tf_fpath = f'{TFRECORD_DIR}/{tfrecord_name}'
		dataset = tf.data.TFRecordDataset(tf_fpath, compression_type='')
		
		log_calib_json = None

		for data in dataset:
			frame = open_dataset.Frame()
			frame.ParseFromString(bytearray(data.numpy()))
			# discovered_log_id = '967082162553397800_5102_900_5122_900'
			assert log_id == frame.context.name
			# Frame start time, which is the timestamp of the first top lidar spin
			# within this frame, in microseconds
			timestamp_ms = frame.timestamp_micros
			timestamp_ns = int(timestamp_ms * 1000) # to nanoseconds
			SE3_flattened = np.array(frame.pose.transform)
			city_SE3_egovehicle = SE3_flattened.reshape(4,4)
			if save_poses:
				dump_pose(city_SE3_egovehicle, timestamp_ns, log_id, ARGO_WRITE_DIR)
			# Reading lidar data and saving it point cloud format
			(range_images,
			camera_projections,
			range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
			points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
										frame,
										range_images,
										camera_projections,
										range_image_top_pose,
										ri_index=1)
			points_all_ri2 = np.concatenate(points_ri2, axis=0)
			if save_cloud:
				dump_point_cloud(points_all_ri2, timestamp_ns, log_id, ARGO_WRITE_DIR)
			# Saving labels
			if save_labels:
    				dump_object_labels(frame.laser_labels, timestamp_ns, log_id, ARGO_WRITE_DIR)
			if save_calibration:
				calib_json = form_calibration_json(frame.context.camera_calibrations)
				if log_calib_json is None:
					log_calib_json = calib_json

					calib_json_fpath = f'{ARGO_WRITE_DIR}/{log_id}/vehicle_calibration_info.json'
					check_mkdir(str(Path(calib_json_fpath).parent))
					save_json_dict(calib_json_fpath, calib_json)
				else:
					assert calib_json == log_calib_json	

			# 5 images per frame
			for index, tf_cam_image in enumerate(frame.images):

				# 4x4 row major transform matrix that transforms 
				# 3d points from one frame to another.
				SE3_flattened = np.array(tf_cam_image.pose.transform)
				city_SE3_egovehicle = SE3_flattened.reshape(4,4)

				# in seconds
				timestamp_s =  tf_cam_image.pose_timestamp
				timestamp_ns = int(timestamp_s * 1e9) # to nanoseconds
				# tf_cam_image.shutter
				# tf_cam_image.camera_trigger_time
				# tf_cam_image.camera_readout_done_time
				if save_poses:
					dump_pose(city_SE3_egovehicle, timestamp_ns, log_id, ARGO_WRITE_DIR)

				if save_images:
					camera_name = CAMERA_NAMES[tf_cam_image.name]
					img = tf.image.decode_jpeg(tf_cam_image.image)
					img_save_fpath = f'{ARGO_WRITE_DIR}/{log_id}/{camera_name}/{camera_name}_{timestamp_ns}.jpg'
					#assert not Path(img_save_fpath).exists()
					check_mkdir(str(Path(img_save_fpath).parent))
					imageio.imwrite(img_save_fpath, img)
					img_count += 1
					if img_count % 100 == 0:
						print(f"\tSaved {img_count}'th image for log = {log_id}")

				
				# pose_save_fpath = f'logs/{log_id}/poses/city_SE3_egovehicle_{timestamp_ns}.json'
				# assert not Path(pose_save_fpath).exists()
				# save_json_dict(pose_save_fpath)


def form_calibration_json(calib_data):
	"""
	Argoverse expects to receive "egovehicle_T_camera", i.e. from camera -> egovehicle, with
		rotation parameterized as quaternion.
	Waymo provides the same SE(3) transformation, but with rotation parmaeterized as 3x3 matrix
	"""
	calib_dict = {
		'camera_data_': []
	}
	for camera_calib in calib_data:

		cam_name = CAMERA_NAMES[camera_calib.name]
		# They provide "Camera frame to vehicle frame."
		# https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/dataset.proto
		egovehicle_SE3_camera = np.array(camera_calib.extrinsic.transform).reshape(4,4)
		x, y, z = egovehicle_SE3_camera[:3,3]
		egovehicle_R_camera = egovehicle_SE3_camera[:3,:3]

		assert np.allclose( egovehicle_SE3_camera[3], np.array([0,0,0,1]) )
		egovehicle_q_camera = rotmat2quat(egovehicle_R_camera)
		qw, qx, qy, qz = egovehicle_q_camera
		f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3 = camera_calib.intrinsic

		cam_dict = {
			'key': 'image_raw_' + cam_name,
			'value': {
				'focal_length_x_px_': f_u,
				'focal_length_y_px_': f_v,
				'focal_center_x_px_': c_u,
				'focal_center_y_px_': c_v,
				'skew_': 0,
				'distortion_coefficients_': [0,0,0],
				'vehicle_SE3_camera_': {
					'rotation': {'coefficients': [qw, qx, qy, qz] },
					'translation': [x,y,z]
				}
			}
		}
		calib_dict['camera_data_'] += [cam_dict]

	return calib_dict


def dump_pose(city_SE3_egovehicle, timestamp, log_id, parent_path):
	""" """
	x,y,z = city_SE3_egovehicle[:3,3]
	R = city_SE3_egovehicle[:3,:3]
	assert np.allclose( city_SE3_egovehicle[3], np.array([0,0,0,1]) )
	q = rotmat2quat(R)
	w, x, y, z = q
	pose_dict = {
		'rotation': [w, x, y, z],
		'translation': [x,y,z]
	}
	json_fpath = f'{parent_path}/{log_id}/poses/city_SE3_egovehicle_{timestamp}.json'
	check_mkdir(str(Path(json_fpath).parent))
	save_json_dict(json_fpath, pose_dict)

def dump_point_cloud(points, timestamp, log_id, parent_path):
	data = {'x': points[:,0], 'y': points[:,1], 'z': points[:,2]}
	cloud = PyntCloud(pd.DataFrame(data))
	cloud_fpath = f'{parent_path}/{log_id}/lidar/PC_{timestamp}.ply'
	check_mkdir(str(Path(cloud_fpath).parent))
	cloud.to_file(cloud_fpath)

def dump_object_labels(labels, timestamp, log_id, parent_path):
	argoverse_labels = []
	for label in labels:
		argoverse_labels.append(build_label(label, timestamp))
	json_fpath = f'{parent_path}/{log_id}/per_sweep_annotations_amodal/tracked_object_labels_{timestamp}.json'
	check_mkdir(str(Path(json_fpath).parent))
	save_json_dict(json_fpath, argoverse_labels)

def build_label(label, timestamp):
	label_dict = {}
	label_dict["center"] = {}
	label_dict["center"]["x"] = label.box.center_x
	label_dict["center"]["y"] = label.box.center_y 
	label_dict["center"]["z"] = label.box.center_z
	label_dict["length"] = label.box.length
	label_dict["width"] = label.box.width
	label_dict["height"] = label.box.height
	label_dict["rotation"] = {}
	qx,qy,qz,qw = yaw_to_quaternion3d(label.box.heading)
	label_dict["rotation"]["x"] = qx
	label_dict["rotation"]["y"] = qy
	label_dict["rotation"]["z"] = qz
	label_dict["rotation"]["w"] = qw
	label_dict["label_class"] = LABEL_TYPES[label.type]
	label_dict["timestamp"] = timestamp
	if label.id not in track_id_dict.keys():
		track_id = uuid.uuid4().hex
		track_id_dict[label.id] = track_id
	else:
		track_id = track_id_dict[label.id]
	label_dict["track_label_uuid"] = track_id
	return label_dict

def rotX(deg: float):
    """
    Compute rotation matrix about the X-axis.
    Args:
    -   deg: in degrees
    
    rot_z = Rotation.from_euler('z', yaw).as_dcm()
    """
    t = np.deg2rad(deg)
    return Rotation.from_euler('x', t).as_dcm()

def rotZ(deg: float):
    """
    Compute rotation matrix about the Z-axis.
    Args
    -   deg: in degrees
    """
    t = np.deg2rad(deg)
    return Rotation.from_euler('z', t).as_dcm()

def rotY(deg: float):
    """
    Compute rotation matrix about the Y-axis.
    Args
    -   deg: in degrees
    """
    t = np.deg2rad(deg)
    return Rotation.from_euler('y', t).as_dcm()


if __name__ == '__main__':

	save_images = True
	save_poses = True
	save_calibration = True
	save_cloud = True
	save_labels = True

	main(save_images, save_poses, save_calibration, save_cloud, save_labels)



