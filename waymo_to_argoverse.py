

"""
!rm -rf waymo-od > /dev/null
!git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od
!cd waymo-od && git branch -a
!cd waymo-od && git checkout remotes/origin/r2.0
!pip3 install --upgrade pip

pip install waymo-open-dataset-tf-2-1-0==1.2.0
!pip3 install waymo-open-dataset-tf-2-1-0==1.2.0

# from google.colab import files
# uploaded = files.upload()

tar -xvf
"""

import os
import tensorflow as tf
import math
import numpy as np
import itertools

tf.compat.v1.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.protos import metrics_pb2




fpath = '/mnt/beegfs/tier2/shared/Datasets/MSegV12/w_o_d/detection_3d_vehicle_detection_validation.bin'



"""
Example Object:

object {
  box {
    center_x: 67.52523040771484
    center_y: -1.3868849277496338
    center_z: 0.8951533436775208
    width: 0.8146794438362122
    length: 1.8189797401428223
    height: 1.790642261505127
    heading: -0.11388802528381348
  }
  type: TYPE_CYCLIST
}
score: 0.19764792919158936
context_name: "10203656353524179475_7625_000_7645_000"
frame_timestamp_micros: 1522688014970187
"""

# get all timestamps
# for each timestamp
	# save dict with "center": {"x", "y", "z", "rotation": "x", "y", "z", "w"}, "length", "width", "height", "track_label_uuid"
	# "timestamp", "label_class"


"""
https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/label.proto
// Box coordinates in vehicle frame.

// The heading of the bounding box (in radians).  The heading is the angle
// required to rotate +x to the surface normal of the SDC front face.

# context_name: "10203656353524179475_7625_000_7645_000"
"""

from google.colab import drive
drive.mount('/content/gdrive')

import json
import os
from typing import Any, Dict, Union

from pathlib import Path
from typing import Tuple
from scipy.spatial.transform import Rotation

def save_json_dict(json_fpath: Union[str, "os.PathLike[str]"], dictionary: Dict[Any, Any]) -> None:
    """Save a Python dictionary to a JSON file.
    Args:
        json_fpath: Path to file to create.
        dictionary: Python dictionary to be serialized.
    """
    with open(json_fpath, "w") as f:
        json.dump(dictionary, f)

def yaw_to_quaternion3d(yaw: float) -> Tuple[float,float,float,float]:
	"""
	Args:
	-   yaw: rotation about the z-axis
	Returns:
	-   qx,qy,qz,qw: quaternion coefficients
	"""
	qx,qy,qz,qw = Rotation.from_euler('z', yaw).as_quat()
	return qx,qy,qz,qw


from waymo_open_dataset.protos import metrics_pb2
# fpath = '/content/detection_3d_cyclist_detection_validation.bin'

DRIVE_DIR = '/content/gdrive/My Drive/WaymoOpenDatasetTracking'

bin_fnames = [
	'detection_3d_cyclist_detection_test.bin',
	'detection_3d_cyclist_detection_validation.bin',
	'detection_3d_pedestrian_detection_test.bin',
	'detection_3d_pedestrian_detection_validation.bin',
	'detection_3d_vehicle_detection_test.bin',
	'detection_3d_vehicle_detection_validation.bin',
]
SHARD_SZ = 500000
for bin_fname in bin_fnames:
	print(bin_fname)
	bin_fpath = f'{DRIVE_DIR}/{bin_fname}'
	shard_counter = 0
	json_fpath = f'{DRIVE_DIR}/{Path(bin_fname).stem}_shard_{shard_counter}.json'

	objects = metrics_pb2.Objects()

	f = open(bin_fpath, 'rb')
	objects.ParseFromString(f.read())
	f.close()

	OBJECT_TYPES = [
		'UNKNOWN', # 0
		'VEHICLE', # 1
		'PEDESTRIAN', # 2
		'SIGN', # 3
		'CYCLIST', # 4
	]

	gt_num_objs = len(objects.objects)
	print(f'num_objs={gt_num_objs}')
	tracked_labels = []
	for i, object in enumerate(objects.objects):
		if i % 50000 == 0:
			print(f'On {i}/{len(objects.objects)}')
		height = object.object.box.height
		width = object.object.box.width
		length = object.object.box.length
		score = object.score
		x = object.object.box.center_x
		y = object.object.box.center_y
		z = object.object.box.center_z
		ego_yaw_obj = object.object.box.heading

		qx,qy,qz,qw = yaw_to_quaternion3d(ego_yaw_obj)
		label_class = OBJECT_TYPES[object.object.type]

		tracked_labels.append({
			"center": {"x": -x, "y": -y, "z": -z},
			"rotation": {"x": qx , "y": qy, "z": qz , "w": qw},
			"length": length,
			"width": width,
			"height": height,
			"track_label_uuid": None,
			"timestamp": object.frame_timestamp_micros, # 1522688014970187
			"label_class": label_class,
			"score":  object.score, # float in [0,1]
			"context_name": object.context_name,
		})
		if len(tracked_labels) >= SHARD_SZ:
			save_json_dict(json_fpath, tracked_labels)
			tracked_labels = []
			shard_counter += 1
			json_fpath = f'{DRIVE_DIR}/{Path(bin_fname).stem}_shard_{shard_counter}.json'

		# label_dir = os.path.join(tracks_dump_dir, log_id, "per_sweep_annotations_amodal")    
		# check_mkdir(label_dir)
		# json_fname = f"tracked_object_labels_{current_lidar_timestamp}.json"
		# json_fpath = os.path.join(label_dir, json_fname) 
		
		# if Path(json_fpath).exists():
		# 	# accumulate tracks of another class together
		# 	prev_tracked_labels = read_json_file(json_fpath)
		# 	tracked_labels.extend(prev_tracked_labels)

	# ensure sharding correct
	print(f'Shard sz, {SHARD_SZ}, num_objs={gt_num_objs}')
	print(f'shard_counter={shard_counter}, len_tracked_labels{len(tracked_labels)}')
	assert gt_num_objs // SHARD_SZ == shard_counter 
	assert gt_num_objs % SHARD_SZ == len(tracked_labels)

	save_json_dict(json_fpath, tracked_labels)

# if too big, shard into pieces when exceed 500000 objects, reset a counter


