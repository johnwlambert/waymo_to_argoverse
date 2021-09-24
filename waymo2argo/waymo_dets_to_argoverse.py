#!/usr/bin/env python3

import itertools
import json
import os

import math
import numpy as np
from pathlib import Path
import tensorflow as tf
from typing import Any, Dict, Union, Tuple

tf.compat.v1.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.protos import metrics_pb2

from transform_utils import yaw_to_quaternion3d

"""
Convert provided Waymo detections to Argoverse format.

Within Colab, download the detection files, and dump them to disk in Argoverse form.
To avoid exceeding Colab RAM, we shard the data.

If run in Colab, add the following:
!rm -rf waymo-od > /dev/null
!git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od
!cd waymo-od && git branch -a
!cd waymo-od && git checkout remotes/origin/r2.0
!pip3 install --upgrade pip

pip install waymo-open-dataset-tf-2-1-0==1.2.0
!pip3 install waymo-open-dataset-tf-2-1-0==1.2.0

# from google.colab import files
# uploaded = files.upload()

or 
from google.colab import drive
drive.mount('/content/gdrive')
"""


def round_to_micros(t_nanos, base=1000):
    """
    Round nanosecond timestamp to nearest microsecond timestamp
    """
    return base * round(t_nanos / base)


def test_round_to_micros():
    """
    test_round_to_micros()
    """
    t_nanos = 1508103378165379072
    t_micros = 1508103378165379000

    assert t_micros == round_to_micros(t_nanos, base=1000)


def save_json_dict(json_fpath: Union[str, "os.PathLike[str]"], dictionary: Dict[Any, Any]) -> None:
    """Save a Python dictionary to a JSON file.
    Args:
        json_fpath: Path to file to create.
        dictionary: Python dictionary to be serialized.
    """
    with open(json_fpath, "w") as f:
        json.dump(dictionary, f)


# DRIVE_DIR = '/content/gdrive/My Drive/WaymoOpenDatasetTracking'
DRIVE_DIR = "/srv/datasets/waymo_opendataset/waymo_open_dataset_v_1_0_0/training"

bin_fnames = [
    "detection_3d_cyclist_detection_test.bin",
    "detection_3d_cyclist_detection_validation.bin",
    "detection_3d_pedestrian_detection_test.bin",
    "detection_3d_pedestrian_detection_validation.bin",
    "detection_3d_vehicle_detection_test.bin",
    "detection_3d_vehicle_detection_validation.bin",
]
SHARD_SZ = 500000
for bin_fname in bin_fnames:
    print(bin_fname)
    bin_fpath = f"{DRIVE_DIR}/{bin_fname}"
    shard_counter = 0
    json_fpath = f"{DRIVE_DIR}/{Path(bin_fname).stem}_shard_{shard_counter}.json"

    objects = metrics_pb2.Objects()

    f = open(bin_fpath, "rb")
    objects.ParseFromString(f.read())
    f.close()

    OBJECT_TYPES = [
        "UNKNOWN",  # 0
        "VEHICLE",  # 1
        "PEDESTRIAN",  # 2
        "SIGN",  # 3
        "CYCLIST",  # 4
    ]

    gt_num_objs = len(objects.objects)
    print(f"num_objs={gt_num_objs}")
    tracked_labels = []
    for i, object in enumerate(objects.objects):
        if i % 50000 == 0:
            print(f"On {i}/{len(objects.objects)}")
        height = object.object.box.height
        width = object.object.box.width
        length = object.object.box.length
        score = object.score
        x = object.object.box.center_x
        y = object.object.box.center_y
        z = object.object.box.center_z

        # Waymo provides SE(3) transformation from
        # labeled_object->egovehicle like Argoverse
        obj_yaw_ego = object.object.box.heading

        qx, qy, qz, qw = yaw_to_quaternion3d(obj_yaw_ego)
        label_class = OBJECT_TYPES[object.object.type]

        tracked_labels.append(
            {
                "center": {"x": x, "y": y, "z": z},
                "rotation": {"x": qx, "y": qy, "z": qz, "w": qw},
                "length": length,
                "width": width,
                "height": height,
                "track_label_uuid": None,
                # TODO: write as int(nanoseconds) instead.
                "timestamp": object.frame_timestamp_micros,  # 1522688014970187
                "label_class": label_class,
                "score": object.score,  # float in [0,1]
                "context_name": object.context_name,
            }
        )
        if len(tracked_labels) >= SHARD_SZ:
            save_json_dict(json_fpath, tracked_labels)
            tracked_labels = []
            shard_counter += 1
            json_fpath = f"{DRIVE_DIR}/{Path(bin_fname).stem}_shard_{shard_counter}.json"

        # label_dir = os.path.join(tracks_dump_dir, log_id, "per_sweep_annotations_amodal")
        # check_mkdir(label_dir)
        # json_fname = f"tracked_object_labels_{current_lidar_timestamp}.json"
        # json_fpath = os.path.join(label_dir, json_fname)

        # if Path(json_fpath).exists():
        # 	# accumulate tracks of another class together
        # 	prev_tracked_labels = read_json_file(json_fpath)
        # 	tracked_labels.extend(prev_tracked_labels)

    # ensure sharding correct
    print(f"Shard sz, {SHARD_SZ}, num_objs={gt_num_objs}")
    print(f"shard_counter={shard_counter}, len_tracked_labels{len(tracked_labels)}")
    assert gt_num_objs // SHARD_SZ == shard_counter
    assert gt_num_objs % SHARD_SZ == len(tracked_labels)

    save_json_dict(json_fpath, tracked_labels)
