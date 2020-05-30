

from collections import defaultdict
import glob
import json
import os
from pathlib import Path
import pdb
from typing import Any, Dict, Union

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
	DATAROOT = '/Users/johnlamb/Downloads/waymo_logs'
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
					if i % 100000 == 0:
						print(f'On {i}/{len(shard_data)}')
					timestamp = det['timestamp']
					log_id = det['context_name']
					if log_id not in log_to_timestamp_to_dets_dict:
						log_to_timestamp_to_dets_dict[log_id] = defaultdict(list)

					log_to_timestamp_to_dets_dict[log_id][timestamp].append(det)
		
				for log_id, timestamp_to_dets_dict in log_to_timestamp_to_dets_dict.items():
					print(log_id)
					for timestamp, dets in timestamp_to_dets_dict.items():
						sweep_json_fpath = f'{DATAROOT}/{log_id}/per_sweep_annotations/tracked_object_labels_{timestamp}.json'
						dummy_lidar_fpath = f'{DATAROOT}/{log_id}/lidar/PC_{timestamp}.ply'
						
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




if __name__ == '__main__':
	main()