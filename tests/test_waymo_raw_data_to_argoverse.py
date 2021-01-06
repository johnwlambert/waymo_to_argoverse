#!/usr/bin/env python3
import numpy as np
from unittest.mock import patch, Mock

from argoverse.utils.ply_loader import load_ply

from waymo2argo.waymo_raw_data_to_argoverse import (
    build_argo_label,
    dump_point_cloud,
    get_log_ids_from_files,
    round_to_micros
)

def test_round_to_micros():
    """
    test_round_to_micros()
    """
    t_nanos = 1508103378165379072
    t_micros = 1508103378165379000
    assert t_micros == round_to_micros(t_nanos, base=1000)


@patch("waymo2argo.waymo_raw_data_to_argoverse.glob")
def test_get_log_ids_from_files(mock_glob):
    mock_glob.glob.return_value = [
        "data/segment-123.tfrecord",
        "data/segment-456_with_camera_labels.tfrecord",
        "data/segment-789.tfrecord",
    ]
    actual_log_ids = get_log_ids_from_files("data")
    expected_log_ids = {
        "123": "data/segment-123.tfrecord",
        "456": "data/segment-456_with_camera_labels.tfrecord",
        "789": "data/segment-789.tfrecord",
    }
    assert actual_log_ids == expected_log_ids


def test_dump_point_cloud():
    points = np.array([[3, 4, 5], [2, 4, 1], [1, 5, 2], [5, 2, 1]])
    test_dir = "test_dir"
    timestamp = 0
    log_id = 123
    dump_point_cloud(points, timestamp, log_id, test_dir)
    file_name = "test_dir/123/lidar/PC_0.ply"
    ret_pts = load_ply(file_name)
    assert np.array_equal(points, ret_pts)


def test_build_argo_label():
    mock_label = Mock()
    mock_label.box.center_x = 5
    mock_label.box.center_y = 5
    mock_label.box.center_z = 5
    mock_label.box.length = 10
    mock_label.box.width = 10
    mock_label.box.height = 10
    mock_label.box.heading = 0
    mock_label.id = "100"
    mock_label.type = 1
    track_ids = {"100": "123"}
    timestamp = "55"
    expected_label = {
        "center": {"x": 5, "y": 5, "z": 5},
        "length": 10,
        "width": 10,
        "height": 10,
        "rotation": {"x": 0, "y": 0, "z": 0, "w": 1},
        "label_class": "VEHICLE",
        "timestamp": timestamp,
        "track_label_uuid": "123",
    }
    actual_label = build_argo_label(
        mock_label, timestamp, track_ids
    )
    assert actual_label == expected_label
