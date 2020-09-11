#!/usr/bin/env python3

import argparse
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
from typing import Any, Dict, List, Union
import uuid

import cv2
from pyntcloud import PyntCloud
from scipy.spatial.transform import Rotation
import tensorflow.compat.v1 as tf

tf.enable_eager_execution()

from argoverse.utils.json_utils import save_json_dict
from argoverse.utils.se3 import SE3
import waymo_open_dataset
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

from waymo2argo.transform_utils import (
    rotX,
    rotY,
    rotmat2quat,
    quat2rotmat,
    yaw_to_quaternion3d,
)

"""
Extract poses, images, and camera calibration from raw Waymo Open Dataset TFRecords.

See the Frame structure here:
https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/dataset.proto

See paper:
https://arxiv.org/pdf/1912.04838.pdf
"""

# Mapping from Argo Camera names to Waymo Camera names
# The indices correspond to Waymo's cameras
CAMERA_NAMES = [
    "unknown",  # 0, 'UNKNOWN',
    "ring_front_center",  # 1, 'FRONT'
    "ring_front_left",  # 2, 'FRONT_LEFT',
    "ring_front_right",  # 3, 'FRONT_RIGHT',
    "ring_side_left",  # 4, 'SIDE_LEFT',
    "ring_side_right",  # 5, 'SIDE_RIGHT'
]

# Mapping from Argo Label types to Waymo Label types
# The indices correspond to Waymo's label types
LABEL_TYPES = [
    "OTHER_MOVER",  # 0, TYPE_UNKNOWN
    "VEHICLE",  # 1, TYPE_VEHICLE
    "PEDESTRIAN",  # 2, TYPE_PEDESTRIAN
    "SIGN",  # 3, TYPE_SIGN
    "BICYCLIST",  # 4, TYPE_CYCLIST
]

RING_IMAGE_SIZES = {
    # width x height
    "ring_front_center": (1920, 1280),
    "ring_front_left": (1920, 1280),
    "ring_side_left": (1920, 886),
    "ring_side_right": (1920, 886),
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ("yes", "true", "t", "y", 1):
        return True
    elif v.lower() in ("no", "false", "f", "n", 0):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def round_to_micros(t_nanos: int, base: int = 1000) -> int:
    """
    Round nanosecond timestamp to nearest microsecond timestamp
    """
    return base * round(t_nanos / base)


def check_mkdir(dirpath: str) -> None:
    """ """
    if not Path(dirpath).exists():
        os.makedirs(dirpath, exist_ok=True)


def get_log_id_from_files(record_dir: str) -> List[str]:
    """Get the log IDs of the Waymo records from the directory
       where they are stored

    Args:
        record_dir: The path to the directory where the Waymo data
                    is stored
                    Example: "/path-to-waymo-data"
                    The args.waymo_dir is used here by default
    Returns:
        log_ids: A map of log IDs to tf records from the Waymo dataset
    """
    files = glob.glob(f"{record_dir}/*.tfrecord")
    log_ids = {}
    for i, file in enumerate(files):
        file = file.replace(record_dir, "")
        file = file.replace("/segment-", "")
        file = file.replace(".tfrecord", "")
        file = file.replace("_with_camera_labels", "")
        log_ids[file] = files[i]
    return log_ids


def main(args: argparse.Namespace) -> None:
    """ """
    TFRECORD_DIR = args.waymo_dir
    ARGO_WRITE_DIR = args.argo_dir
    track_id_dict = {}
    img_count = 0
    log_ids = get_log_id_from_files(TFRECORD_DIR)
    for log_id, tf_fpath in log_ids.items():
        dataset = tf.data.TFRecordDataset(tf_fpath, compression_type="")
        log_calib_json = None
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            # Checking if we extracted the correct log ID
            assert log_id == frame.context.name
            # Frame start time, which is the timestamp
            # of the first top lidar spin within this frame, in microseconds
            timestamp_ms = frame.timestamp_micros
            timestamp_ns = int(timestamp_ms * 1000)  # to nanoseconds
            SE3_flattened = np.array(frame.pose.transform)
            city_SE3_egovehicle = SE3_flattened.reshape(4, 4)
            if args.save_poses:
                dump_pose(city_SE3_egovehicle, timestamp_ns, log_id, ARGO_WRITE_DIR)
            # Reading lidar data and saving it in point cloud format
            # We are only using the first range image (Waymo provides two range images)
            # If you want to use the second one, you can change it in the arguments
            (
                range_images,
                camera_projections,
                range_image_top_pose,
            ) = frame_utils.parse_range_image_and_camera_projection(frame)
            if args.range_image == 1:
                (
                    points_ri,
                    cp_points_ri,
                ) = frame_utils.convert_range_image_to_point_cloud(
                    frame, range_images, camera_projections, range_image_top_pose
                )
            elif args.range_image == 2:
                (
                    points_ri,
                    cp_points_ri,
                ) = frame_utils.convert_range_image_to_point_cloud(
                    frame,
                    range_images,
                    camera_projections,
                    range_image_top_pose,
                    ri_index=1,
                )
            points_all_ri = np.concatenate(points_ri, axis=0)
            if args.save_cloud:
                dump_point_cloud(points_all_ri, timestamp_ns, log_id, ARGO_WRITE_DIR)
            # Saving labels
            if args.save_labels:
                dump_object_labels(
                    frame.laser_labels,
                    timestamp_ns,
                    log_id,
                    ARGO_WRITE_DIR,
                    track_id_dict,
                )
            if args.save_calibration:
                calib_json = form_calibration_json(frame.context.camera_calibrations)
                if log_calib_json is None:
                    log_calib_json = calib_json
                    calib_json_fpath = (
                        f"{ARGO_WRITE_DIR}/{log_id}/vehicle_calibration_info.json"
                    )
                    check_mkdir(str(Path(calib_json_fpath).parent))
                    save_json_dict(calib_json_fpath, calib_json)
                else:
                    assert calib_json == log_calib_json

            # 5 images per frame
            for index, tf_cam_image in enumerate(frame.images):
                # 4x4 row major transform matrix that transforms
                # 3d points from one frame to another.
                SE3_flattened = np.array(tf_cam_image.pose.transform)
                city_SE3_egovehicle = SE3_flattened.reshape(4, 4)
                # in seconds
                timestamp_s = tf_cam_image.pose_timestamp
                timestamp_ns = int(timestamp_s * 1e9)  # to nanoseconds
                if args.save_poses:
                    dump_pose(city_SE3_egovehicle, timestamp_ns, log_id, ARGO_WRITE_DIR)

                if args.save_images:
                    camera_name = CAMERA_NAMES[tf_cam_image.name]
                    img = tf.image.decode_jpeg(tf_cam_image.image)
                    new_img = undistort_image(
                        np.asarray(img),
                        frame.context.camera_calibrations,
                        tf_cam_image.name,
                    )
                    img_save_fpath = f"{ARGO_WRITE_DIR}/{log_id}/{camera_name}/{camera_name}_{timestamp_ns}.jpg"
                    # assert not Path(img_save_fpath).exists()
                    check_mkdir(str(Path(img_save_fpath).parent))
                    imageio.imwrite(img_save_fpath, new_img)
                    img_count += 1
                    if img_count % 100 == 0:
                        print(f"\tSaved {img_count}'th image for log = {log_id}")


def undistort_image(img: np.ndarray, calib_data: Any, camera_name: int):
    """Undistort the image from the Waymo dataset given camera calibration data"""
    for camera_calib in calib_data:
        if camera_calib.name == camera_name:
            f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3 = camera_calib.intrinsic
            # k1, k2 and k3 are the tangential distortion coefficients
            # p1, p2 are the radial distortion coefficients
            camera_matrix = np.array([[f_u, 0, c_u], [0, f_v, c_v], [0, 0, 1]])
            dist_coeffs = np.array([k1, k2, p1, p2, k3])
            return cv2.undistort(img, camera_matrix, dist_coeffs)


def form_calibration_json(calib_data):
    """
    Argoverse expects to receive "egovehicle_T_camera", i.e. from camera -> egovehicle, with
            rotation parameterized as quaternion.
    Waymo provides the same SE(3) transformation, but with rotation parmaeterized as 3x3 matrix
    """
    calib_dict = {"camera_data_": []}
    for camera_calib in calib_data:
        cam_name = CAMERA_NAMES[camera_calib.name]
        # They provide "Camera frame to vehicle frame."
        # https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/dataset.proto
        egovehicle_SE3_waymocam = np.array(camera_calib.extrinsic.transform).reshape(
            4, 4
        )
        standardcam_R_waymocam = rotY(-90).dot(rotX(90))
        standardcam_SE3_waymocam = SE3(
            rotation=standardcam_R_waymocam, translation=np.zeros(3)
        )
        egovehicle_SE3_waymocam = SE3(
            rotation=egovehicle_SE3_waymocam[:3, :3],
            translation=egovehicle_SE3_waymocam[:3, 3],
        )
        standardcam_SE3_egovehicle = standardcam_SE3_waymocam.right_multiply_with_se3(
            egovehicle_SE3_waymocam.inverse()
        )
        egovehicle_SE3_standardcam = standardcam_SE3_egovehicle.inverse()
        egovehicle_q_camera = rotmat2quat(egovehicle_SE3_standardcam.rotation)
        x, y, z = egovehicle_SE3_standardcam.translation
        qw, qx, qy, qz = egovehicle_q_camera
        f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3 = camera_calib.intrinsic
        cam_dict = {
            "key": "image_raw_" + cam_name,
            "value": {
                "focal_length_x_px_": f_u,
                "focal_length_y_px_": f_v,
                "focal_center_x_px_": c_u,
                "focal_center_y_px_": c_v,
                "skew_": 0,
                "distortion_coefficients_": [0, 0, 0],
                "vehicle_SE3_camera_": {
                    "rotation": {"coefficients": [qw, qx, qy, qz]},
                    "translation": [x, y, z],
                },
            },
        }
        calib_dict["camera_data_"] += [cam_dict]
    return calib_dict


def dump_pose(
    city_SE3_egovehicle: np.ndarray, timestamp: int, log_id: str, parent_path: str
) -> None:
    """Saves the SE3 transformation from city frame
        to egovehicle frame at a particular timestamp

    Args:
        city_SE3_egovehicle: A (4,4) numpy array representing the
                            SE3 transformation from city to egovehicle frame
        timestamp: Timestamp in nanoseconds when the lidar reading occurred
        log_id: Log ID that the reading belongs to
        parent_path: The directory that the converted data is written to
    """
    x, y, z = city_SE3_egovehicle[:3, 3]
    R = city_SE3_egovehicle[:3, :3]
    assert np.allclose(city_SE3_egovehicle[3], np.array([0, 0, 0, 1]))
    q = rotmat2quat(R)
    w, x, y, z = q
    pose_dict = {"rotation": [w, x, y, z], "translation": [x, y, z]}
    json_fpath = f"{parent_path}/{log_id}/poses/city_SE3_egovehicle_{timestamp}.json"
    check_mkdir(str(Path(json_fpath).parent))
    save_json_dict(json_fpath, pose_dict)


def dump_point_cloud(
    points: np.ndarray, timestamp: int, log_id: str, parent_path: str
) -> None:
    """Saves point cloud as .ply file extracted from Waymo's range images

    Args:
        points: A (N,3) numpy array representing the point cloud created from lidar readings
        timestamp: Timestamp in nanoseconds when the lidar reading occurred
        log_id: Log ID that the reading belongs to
        parent_path: The directory that the converted data is written to
    """
    data = {"x": points[:, 0], "y": points[:, 1], "z": points[:, 2]}
    cloud = PyntCloud(pd.DataFrame(data))
    cloud_fpath = f"{parent_path}/{log_id}/lidar/PC_{timestamp}.ply"
    check_mkdir(str(Path(cloud_fpath).parent))
    cloud.to_file(cloud_fpath)


def dump_object_labels(
    labels: List[waymo_open_dataset.label_pb2.Label],
    timestamp: int,
    log_id: str,
    parent_path: str,
    track_id_dict: Dict,
) -> None:
    """Saves object labels from Waymo dataset as json files

    Args:
        labels: A list of Waymo labels
        timestamp: Timestamp in nanoseconds when the lidar reading occurred
        log_id: Log ID that the reading belongs to
        parent_path: The directory that the converted data is written to
        track_id_dict: Dictionary to store object ID to track ID mappings
    """
    argoverse_labels = []
    for label in labels:
        if label.type != 3:
            argoverse_labels.append(build_argo_label(label, timestamp, track_id_dict))
    json_fpath = f"{parent_path}/{log_id}/per_sweep_annotations_amodal/"
    json_fpath += f"tracked_object_labels_{timestamp}.json"
    check_mkdir(str(Path(json_fpath).parent))
    save_json_dict(json_fpath, argoverse_labels)


def build_argo_label(
    label: waymo_open_dataset.label_pb2.Label, timestamp: int, track_id_dict: Dict
) -> Dict:
    """Builds a dictionary that represents an object detection in Argoverse format from a Waymo label

    Args:
        labels: A Waymo label
        timestamp: Timestamp in nanoseconds when the lidar reading occurred
        track_id_dict: Dictionary to store object ID to track ID mappings
    Returns:
        label_dict: A dictionary representing the object label in Argoverse format
    """
    label_dict = {}
    label_dict["center"] = {}
    label_dict["center"]["x"] = label.box.center_x
    label_dict["center"]["y"] = label.box.center_y
    label_dict["center"]["z"] = label.box.center_z
    label_dict["length"] = label.box.length
    label_dict["width"] = label.box.width
    label_dict["height"] = label.box.height
    label_dict["rotation"] = {}
    qx, qy, qz, qw = yaw_to_quaternion3d(label.box.heading)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-images",
        default=True,
        type=str2bool,
        help="whether to save images or not",
    )
    parser.add_argument(
        "--save-poses", default=True, type=str2bool, help="whether to save poses or not"
    )
    parser.add_argument(
        "--save-calibration",
        default=True,
        type=str2bool,
        help="whether to save camera calibration information or not",
    )
    parser.add_argument(
        "--save-cloud",
        default=True,
        type=str2bool,
        help="whether to save point clouds or not",
    )
    parser.add_argument(
        "--save-labels",
        default=True,
        type=str2bool,
        help="whether to save object labels or not",
    )
    parser.add_argument(
        "--range-image",
        default=1,
        type=int,
        choices=[1, 2],
        help="which range image to use from Waymo",
    )
    parser.add_argument(
        "--waymo-dir",
        type=str,
        required=True,
        help="the path to the directory where the Waymo data is stored",
    )
    parser.add_argument(
        "--argo-dir",
        type=str,
        required=True,
        help="the path to the directory where the converted data should be written",
    )
    args = parser.parse_args()
    main(args)
