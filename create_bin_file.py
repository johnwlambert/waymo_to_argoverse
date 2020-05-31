# Lint as: python3
# Copyright 2020 The Waymo Open Dataset Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================*/
"""A simple example to generate a file that contains serialized Objects proto."""

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2


OBJECT_TYPES = [
	'UNKNOWN', # 0
	'VEHICLE', # 1
	'PEDESTRIAN', # 2
	'SIGN', # 3
	'CYCLIST', # 4
]



def test_quaternion3d_to_yaw():
	""" """
	yaw = 2.
	q = yaw_to_quaternion3d(yaw)
	new_yaw = quaternion3d_to_yaw(q)


def se2_to_yaw(B_SE2_A):
    """
    Computes the pose vector v from a homogeneous transform A.
    Args:
    -   B_SE2_A
    Returns:
    -   v
    """
    R = B_SE2_A.rotation
    theta = np.arctan2(R[1,0], R[0,0])
    return theta


def yaw_to_quaternion3d(yaw: float) -> Tuple[float,float,float,float]:
    """
    Args:
    -   yaw: rotation about the z-axis

    Returns:
    -   qx,qy,qz,qw: quaternion coefficients
    """
    qx,qy,qz,qw = Rotation.from_euler('z', yaw).as_quat()
    return qx,qy,qz,qw



def quaternion3d_to_yaw(q: np.ndarray) -> float:
	"""
	Args:
	-   q: qx,qy,qz,qw: quaternion coefficients

	Returns:
	-	yaw: float, rotation about the z-axis
	"""
	roll, pitch, yaw = Rotation.from_quat('zyx', yaw).as_euler()
	pdb.set_trace()
	return yaw


def _create_pd_file_example():
	"""Creates a prediction objects file."""
	objects = metrics_pb2.Objects()

	# loop over the logs in the split
		# get all the per_sweep_annotations_amodal files
		# for each per_sweep_annotation file
			# loop over all objects

				create_object_description()

	objects.objects.append(o)

	# Add more objects. Note that a reasonable detector should limit its maximum
	# number of boxes predicted per frame. A reasonable value is around 400. A
	# huge number of boxes can slow down metrics computation.

	# Write objects to a file.
	f = open('/export/share/Datasets/MSegV12/w_o_d/val_preds_v1.bin', 'wb')
	f.write(objects.SerializeToString())
	f.close()


def create_object_description(obj_json):
	""" """
	o = metrics_pb2.Object()
	# The following 3 fields are used to uniquely identify a frame a prediction
	# is predicted at. Make sure you set them to values exactly the same as what
	# we provided in the raw data. Otherwise your prediction is considered as a
	# false negative.
	o.context_name = ('context_name for the prediction. See Frame::context::name '
	                'in  dataset.proto.')
	# The frame timestamp for the prediction. See Frame::timestamp_micros in
	# dataset.proto.
	invalid_ts = -1
	o.frame_timestamp_micros = invalid_ts
	# This is only needed for 2D detection or tracking tasks.
	# Set it to the camera name the prediction is for.
	o.camera_name = dataset_pb2.CameraName.FRONT

	tx, ty, tz = obj_json["center"]["x"], obj_json["center"]["y"], obj_json["center"]["z"]
 	q = 
                "rotation": {"x": qx , "y": qy, "z": qz , "w": qw},

	yaw = quaternion3d_to_yaw(q)

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



if __name__ == '__main__':
	# create_pd_file_example()
	test_quaternion3d_to_yaw()


