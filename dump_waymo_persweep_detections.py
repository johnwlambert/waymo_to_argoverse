

from collections import defaultdict
import glob
import json
import math
import numpy as np
import os
from pathlib import Path
import pdb
from typing import Any, Dict, Union, Tuple

from scipy.spatial.transform import Rotation
from argoverse.utils.se3 import SE3


def yaw_to_quaternion3d(yaw: float) -> Tuple[float,float,float,float]:
	"""
	Args:
	-   yaw: rotation about the z-axis
	Returns:
	-   qx,qy,qz,qw: quaternion coefficients
	"""
	qx,qy,qz,qw = Rotation.from_euler('z', yaw).as_quat()
	return qx,qy,qz,qw


def rotmat2quat(R: np.ndarray) -> np.ndarray:
	""" """
	q_scipy = Rotation.from_dcm(R).as_quat()
	x, y, z, w = q_scipy
	q_argo = w, x, y, z
	return q_argo


def quat2rotmat(q: np.ndarray) -> np.ndarray:
    """Convert a unit-length quaternion into a rotation matrix.
    Note that libraries such as Scipy expect a quaternion in scalar-last [x, y, z, w] format,
    whereas at Argo we work with scalar-first [w, x, y, z] format, so we convert between the
    two formats here. We use the [w, x, y, z] order because this corresponds to the
    multidimensional complex number `w + ix + jy + kz`.
    Args:
        q: Array of shape (4,) representing (w, x, y, z) coordinates
    Returns:
        R: Array of shape (3, 3) representing a rotation matrix.
    """
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-12)
    w, x, y, z = q
    q_scipy = np.array([x, y, z, w])
    return Rotation.from_quat(q_scipy).as_dcm()


def round_to_micros(t_nanos, base=1000):
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
	DETS_DATAROOT = '/Users/johnlamb/Downloads/waymo_logs_dets'
	RAW_DATAROOT = '/Users/johnlamb/Downloads/waymo_logs_raw_data'
	SHARD_DIR = '/Users/johnlamb/Downloads/waymo_pointpillars_detections'
	for split in ['validation', 'test']:
		print(split)
		for classname in ['cyclist','pedestrian','vehicle']:
			print(f'\t{classname}')
			
			shard_fpaths = glob.glob(f'{SHARD_DIR}/{split}/detection_3d_{classname}*{split}_shard*.json')
			shard_fpaths.sort()
			for shard_fpath in shard_fpaths:

				log_to_timestamp_to_dets_dict = defaultdict(dict)
				print(f'\t\t{Path(shard_fpath).stem}')
				shard_data = read_json_file(shard_fpath)
				for i, det in enumerate(shard_data):

					log_id = det['context_name']

					if log_id not in [
						'10868756386479184868_3000_000_3020_000',
						'9584760613582366524_1620_000_1640_000',
						'11450298750351730790_1431_750_1451_750'
					]:
						continue

					det['center']['x'] *= -1
					det['center']['y'] *= -1
					det['center']['z'] *= -1

					if i % 100000 == 0:
						print(f'On {i}/{len(shard_data)}')
					timestamp_ms = det['timestamp']
					timestamp_ns = int(1000 * timestamp_ms)


					if log_id not in log_to_timestamp_to_dets_dict:
						log_to_timestamp_to_dets_dict[log_id] = defaultdict(list)

					log_to_timestamp_to_dets_dict[log_id][timestamp_ns].append(det)
		
				for log_id, timestamp_to_dets_dict in log_to_timestamp_to_dets_dict.items():
					print(log_id)
					for timestamp_ns, dets in timestamp_to_dets_dict.items():
						sweep_json_fpath = f'{DETS_DATAROOT}/{log_id}/per_sweep_annotations/tracked_object_labels_{timestamp_ns}.json'
						dummy_lidar_fpath = f'{RAW_DATAROOT}/{log_id}/lidar/PC_{timestamp_ns}.ply'
						
						if Path(sweep_json_fpath).exists():
							# accumulate tracks of another class together
							prev_dets = read_json_file(sweep_json_fpath)
							dets.extend(prev_dets)

						check_mkdir(str(Path(sweep_json_fpath).parent))
						save_json_dict(sweep_json_fpath, dets)

						check_mkdir(str(Path(dummy_lidar_fpath).parent))
						save_json_dict(dummy_lidar_fpath, {})


			if verbose:
				print('Shared timestamps:')
				print(timestamps_counts)


def rotMatZ_3D(yaw):
	"""
	Args:
	-   tz
	Returns:
	-   rot_z
	"""
	c = np.cos(yaw)
	s = np.sin(yaw)
	rot_z = np.array(
	[
	[   c,-s, 0],
	[   s, c, 0],
	[   0, 0, 1 ]
	])
	return rot_z


def test_transform():
	""" """
	yaw = 90
	for yaw in np.random.randn(10) * 360:
		R = rotMatZ_3D(yaw)
		w, x, y, z = rotmat2quat(R)
		qx,qy,qz,qw = yaw_to_quaternion3d(yaw)

		print(w, qw)
		print(x, qx)
		print(y, qy)
		print(z, qz)
		assert np.allclose(w, qw)
		assert np.allclose(x, qx)
		assert np.allclose(y, qy)
		assert np.allclose(z, qz)



if __name__ == '__main__':

	#test_transform()
	main()





