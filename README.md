
Waymo Open Dataset -> Argoverse Converter

## Overview

Simple utility to convert Waymo Open Dataset raw data, ground truth, and detections to the Argoverse format [ [paper](https://arxiv.org/abs/1911.02620), [repo](https://github.com/argoai/argoverse-api) ], run an Argoverse-based tracker, and then submit to Waymo Open Dataset leaderboard.

Achieves the following on the Waymo 3d Tracking Leaderboard, using `run_ab3dmot.py` from my [argoverse_cbgs_kf_tracker](https://github.com/johnwlambert/argoverse_cbgs_kf_tracker) repo.

|    Model                | MOTA/L2    | 	MOTP/L2   | 	FP/L2	  |   Mismatch/L2	|   Miss/L2  |
| :---------------------: | :-------:  | :--------: | :--------:| :--------:    | :--------: |
| PPBA AB3DMOT (mine)     | 0.2914	   |  0.2696	  | 0.1714    |	0.0025 	      | 0.5347     |
| Waymo Baseline          |  0.2592	   | 0.1753	    | 0.0932    |	0.0020	      |  0.3122    |


## Usage Instructions

1. Download test split files (~150 logs) which include TFRecords
2. Download provided detections from PointPillars Progressive Population-Based Augmentation detector, as .bin files.
3. Convert to Argoverse format using scripts here
4. Run tracker
5. Convert track results to .bin file
6. Populate a `submission.txtpb` file with metadata describing your submission ([example here](https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/master/waymo_open_dataset/metrics/tools/submission.txtpb)).
7. Run `create_submission` binary to get tar.gz file. Binary is only compiled using Bazel. I used Google Colab. 
8. Submit to eval server.



Submission process overview is [here](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md#use-pre-compiled-pippip3-packages).
