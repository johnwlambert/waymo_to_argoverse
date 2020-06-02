#!/usr/bin/env python3

import glob
import json
import numpy as np
import os
from pathlib import Path
import pdb
from scipy.spatial.transform import Rotation
import time
from typing import Any, Dict, Union, Tuple

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2

from waymo_data_splits import get_val_log_ids, get_test_log_ids
from transform_utils import (
	se2_to_yaw,
	quaternion3d_to_yaw,
	yaw_to_quaternion3d,
)

"""
Given tracks in Argoverse format, convert them to Waymo submission format.
"""

OBJECT_TYPES = [
	'UNKNOWN', # 0
	'VEHICLE', # 1
	'PEDESTRIAN', # 2
	'SIGN', # 3
	'CYCLIST', # 4
]


def read_json_file(fpath: Union[str, "os.PathLike[str]"]) -> Any:
	"""Load dictionary from JSON file.
	Args:
		fpath: Path to JSON file.
	Returns:
		Deserialized Python dictionary.
	"""
	with open(fpath, "rb") as f:
		return json.load(f)


def create_submission(min_conf: float, min_hits: int):
	"""Creates a prediction objects file."""
	objects = metrics_pb2.Objects()

	split = 'test'
	exp_name = f'ab3dmot_tracks_conf{min_conf}_complete_sharded_{split}_minhits{min_hits}'
	TRACKER_OUTPUT_DATAROOT = f'{exp_name}/{split}-split-track-preds-maxage15-minhits{min_hits}-conf{min_conf}'
	if split == 'val':
		log_ids = get_val_log_ids()
	elif split == 'test':
		log_ids = get_test_log_ids()

	# loop over the logs in the split
	for i, log_id in enumerate(log_ids):
		print(f'On {i}th log {log_id}')
		start = time.time()
		# get all the per_sweep_annotations_amodal files
		json_fpaths = glob.glob(f'{TRACKER_OUTPUT_DATAROOT}/{log_id}/per_sweep_annotations_amodal/*.json')
		# for each per_sweep_annotation file
		for json_fpath in json_fpaths:
			timestamp_ns = int(Path(json_fpath).stem.split('_')[-1])
			timestamp_objs = read_json_file(json_fpath)
			# loop over all objects
			for obj_json in timestamp_objs:
				o = create_object_description(log_id, timestamp_ns, obj_json)
				objects.objects.append(o)
		end = time.time()
		duration = end - start
		print(f'\tTook {duration} sec')

	# Add more objects. Note that a reasonable detector should limit its maximum
	# number of boxes predicted per frame. A reasonable value is around 400. A
	# huge number of boxes can slow down metrics computation.

	# Write objects to a file.
	f = open(f'/w_o_d/{exp_name}.bin', 'wb')
	f.write(objects.SerializeToString())
	f.close()


def create_object_description(log_id, timestamp_ns, obj_json):
	""" """
	o = metrics_pb2.Object()
	# The following 3 fields are used to uniquely identify a frame a prediction
	# is predicted at. Make sure you set them to values exactly the same as what
	# we provided in the raw data. Otherwise your prediction is considered as a
	# false negative.
	o.context_name = log_id
	# The frame timestamp for the prediction. See Frame::timestamp_micros in
	# dataset.proto.
	invalid_ts = -1
	o.frame_timestamp_micros = int(timestamp_ns / 1000) # save as microseconds

	tx, ty, tz = obj_json["center"]["x"], obj_json["center"]["y"], obj_json["center"]["z"]
	qx, qy, qz, qw = obj_json["rotation"]["x"], obj_json["rotation"]["y"], obj_json["rotation"]["z"], obj_json["rotation"]["w"]
	q_argo = np.array([qw, qx, qy, qz])
	yaw = quaternion3d_to_yaw(q_argo)

	# Populating box and score.
	box = label_pb2.Label.Box()
	box.center_x = tx
	box.center_y = ty
	box.center_z = tz
	box.length = obj_json["length"]
	box.width = obj_json["width"]
	box.height = obj_json["height"]
	box.heading = yaw
	o.object.box.CopyFrom(box)
	# This must be within [0.0, 1.0]. It is better to filter those boxes with
	# small scores to speed up metrics computation.
	o.score = 0.5
	# For tracking, this must be set and it must be unique for each tracked
	# sequence.
	o.object.id = obj_json["track_label_uuid"]

	if obj_json["label_class"] == 'PEDESTRIAN':
		obj_type = label_pb2.Label.TYPE_PEDESTRIAN
	elif obj_json["label_class"] == 'CYCLIST':
		obj_type = label_pb2.Label.TYPE_CYCLIST
	elif obj_json["label_class"] == 'VEHICLE':
		obj_type = label_pb2.Label.TYPE_VEHICLE
	else:
		print('Unknown obj. type...')
		quit()

	# Use correct type.
	o.object.type = obj_type
	return o


if __name__ == '__main__':
	""" """
	create_submission(min_conf=0.475, min_hits=6)




