

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
			timestamps_counts = defaultdict(int)

			shard_fpaths = glob.glob(f'{SHARD_DIR}/{split}/detection_3d_{classname}*{split}_shard*.json')
			shard_fpaths.sort()
			for shard_fpath in shard_fpaths:
				print(f'\t\t{Path(shard_fpath).stem}')
				shard_data = read_json_file(shard_fpath)
				for i, det in enumerate(shard_data):
					tracked_labels = []
					timestamp = det['timestamp']
					timestamps_counts[timestamp] += 1
					log_id = det['context_name']
					tracked_labels += [det]

					sweep_json_fpath = f'{DATAROOT}/{log_id}/per_sweep_annotations/tracked_object_labels_{timestamp}.json'
					if Path(sweep_json_fpath).exists():
						# accumulate tracks of another class together
						prev_tracked_labels = read_json_file(sweep_json_fpath)
						tracked_labels.extend(prev_tracked_labels)

					check_mkdir(str(Path(sweep_json_fpath).parent))
					save_json_dict(sweep_json_fpath, tracked_labels)

			if verbose:
				print('Shared timestamps:')
				print(timestamps_counts)




if __name__ == '__main__':
	main()