"""
Given sharded JSON files containing labeled objects or detections in random order, 
accumulate objects according to frame, at each nanosecond timestamp, and 
write them to disk in JSON again.

Also, writes corresponding dummy PLY files for each frame.
"""

import glob
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Union

from argoverse.utils.se3 import SE3


def round_to_micros(t_nanos, base=1000):
    """Round nanosecond timestamp to nearest microsecond timestamp."""
    return base * round(t_nanos / base)


def test_round_to_micros():
    """
    test_round_to_micros()
    """
    t_nanos = 1508103378165379072
    t_micros = 1508103378165379000

    assert t_micros == round_to_micros(t_nanos, base=1000)


def check_mkdir(dirpath):
    """ """
    if not Path(dirpath).exists():
        os.makedirs(dirpath, exist_ok=True)


def read_json_file(fpath: Union[str, "os.PathLike[str]"]) -> Any:
    """Load dictionary from JSON file.
    Args:
            fpath: Path to JSON file.
    Returns:
            Deserialized Python dictionary.
    """
    with open(fpath, "rb") as f:
        return json.load(f)


def save_json_dict(json_fpath: Union[str, "os.PathLike[str]"], dictionary: Dict[Any, Any]) -> None:
    """Save a Python dictionary to a JSON file.
    Args:
        json_fpath: Path to file to create.
        dictionary: Python dictionary to be serialized.
    """
    with open(json_fpath, "w") as f:
        json.dump(dictionary, f)


def main(verbose=False):
    """ """
    DETS_DATAROOT = "/Users/johnlamb/Downloads/waymo_logs_dets"
    RAW_DATAROOT = "/Users/johnlamb/Downloads/waymo_logs_raw_data"
    SHARD_DIR = "/Users/johnlamb/Downloads/waymo_pointpillars_detections"
    for split in ["validation", "test"]:
        print(split)
        for classname in ["cyclist", "pedestrian", "vehicle"]:
            print(f"\t{classname}")

            shard_fpaths = glob.glob(f"{SHARD_DIR}/{split}/detection_3d_{classname}*{split}_shard*.json")
            shard_fpaths.sort()
            for shard_fpath in shard_fpaths:

                log_to_timestamp_to_dets_dict = defaultdict(dict)
                print(f"\t\t{Path(shard_fpath).stem}")
                shard_data = read_json_file(shard_fpath)
                for i, det in enumerate(shard_data):

                    log_id = det["context_name"]
                    if i % 100000 == 0:
                        print(f"On {i}/{len(shard_data)}")
                    timestamp_ms = det["timestamp"]
                    timestamp_ns = int(1000 * timestamp_ms)

                    if log_id not in log_to_timestamp_to_dets_dict:
                        log_to_timestamp_to_dets_dict[log_id] = defaultdict(list)

                    log_to_timestamp_to_dets_dict[log_id][timestamp_ns].append(det)

                for log_id, timestamp_to_dets_dict in log_to_timestamp_to_dets_dict.items():
                    print(log_id)
                    for timestamp_ns, dets in timestamp_to_dets_dict.items():
                        sweep_json_fpath = (
                            f"{DETS_DATAROOT}/{log_id}/per_sweep_annotations/tracked_object_labels_{timestamp_ns}.json"
                        )
                        dummy_lidar_fpath = f"{RAW_DATAROOT}/{log_id}/lidar/PC_{timestamp_ns}.ply"

                        if Path(sweep_json_fpath).exists():
                            # accumulate tracks of another class together
                            prev_dets = read_json_file(sweep_json_fpath)
                            dets.extend(prev_dets)

                        check_mkdir(str(Path(sweep_json_fpath).parent))
                        save_json_dict(sweep_json_fpath, dets)

                        check_mkdir(str(Path(dummy_lidar_fpath).parent))
                        save_json_dict(dummy_lidar_fpath, {})

            if verbose:
                print("Shared timestamps:")
                #print(timestamps_counts)


if __name__ == "__main__":

    # test_transform()
    main()
