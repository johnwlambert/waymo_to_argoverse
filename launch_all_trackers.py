#!/usr/bin/env python3

from waymo_data_splits import get_val_log_ids, get_test_log_ids

from mseg_semantic.utils.subprocess_utils import run_command


def launch_all_trackers():
	""" """
	split = 'test' # 'val' #'val'
	dets_dataroot = '/export/share/Datasets/MSegV12/w_o_d/detections'

	#min_conf = 0.3
	for min_conf in [ 0.475]: # 0.50, , 0.525]: # 0.41, 0.42, 0.43, 0.44, 0.46, 0.47,  0.53, 0.54 ]:
		for min_hits in [6]: # 3,4,
		
			tracks_dump_dir = f'/export/share/Datasets/MSegV12/w_o_d/ab3dmot_tracks_conf{min_conf}_complete_sharded_{split}_minhits{min_hits}'
			
			if split == 'val':
				log_ids = get_val_log_ids()
			elif split == 'test':
				log_ids = get_test_log_ids()
			
			for log_id in log_ids:
				pose_dir = f'/export/share/Datasets/MSegV12/w_o_d/{split.upper()}_RAW_DATA_SHARDED/sharded_pose_logs/{split}_{log_id}'

				cmd = f'sbatch -p cpu -c 5'
				cmd += f' run_tracker.sh {split} {dets_dataroot} {pose_dir} {tracks_dump_dir} {min_conf} {min_hits}'
				print(cmd)
				run_command(cmd)


if __name__ == '__main__':
	launch_all_trackers()